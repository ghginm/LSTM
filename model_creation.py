import itertools
import math
import pickle
from copy import deepcopy
from typing import Literal

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sqlalchemy import create_engine
from torch.utils.data import Dataset, DataLoader


## Load a dataset

def get_data(db_url, data_path, sql_access=True):
    if sql_access:
        engine = create_engine(db_url)
        query = '''
                SELECT week, QuantityDal, ID FROM ml_data
                '''

        data = pd.read_sql(query, con=engine)
    else:
        with open(data_path, 'rb') as handle:
            data = pickle.load(handle)

    return data

## Time-series variables

def ts_var(data, data_vars, date_col, id_col, target_col):
    """Creating time-series variables: sine / cosine pairs. distant lags."""

    data['month'] = data[date_col].dt.month
    data['weekyear'] = [x.isocalendar()[1] for x in data[date_col]]

    for i in [1, 2, 12]:
        data[f'f_sin_52_{i}'] = np.sin((i * 2*np.pi * data['weekyear']) / 52)
        data[f'f_cos_52_{i}'] = np.cos((i * 2*np.pi * data['weekyear']) / 52)

    data = data.drop(['month', 'weekyear'], axis=1)

    for i in [26, 39, 51, 52, 53]:
        data[f'lag_{i}'] = data.groupby([id_col], observed=True,
                                         group_keys=False)[target_col].shift(i).fillna(0)

    if data_vars is not None:
        data = data.merge(data_vars, how='left', on=['ID', 'week'])

    return data

## Model post-proecessing

# Determining the number of epochs for training and the number of epochs to average parameters
def n_epoch_optim(loss_val_es_cv, loss_val_cv, final_epoch):
    """Searching for the optimal number of epochs `n` to be averaged.
    The value of `n` is determined based on minimising the recorded validation loss.

    Parameters
    ----------
    loss_val_es_cv : a list loss values for each validation data set used for early stopping (ES).
    loss_val_cv : a list loss values for each validation data set used for tuning other parameters.
    final_epoch : a list of the final number of epochs for each CV split, e.g. [100, 89, 120].
    """

    loss_val_es_cv, loss_val_cv = loss_val_es_cv, loss_val_cv
    n_epoch_final = math.ceil(np.mean(final_epoch))

    ratios = [x / 20 for x in range(1, 8)]

    loss_avg_epoch, ratio_avg_epoch = [], []

    for epoch_ratio in ratios:
        for (loss_es, loss_val) in zip(loss_val_es_cv, loss_val_cv):
            loss = np.average(np.array((loss_es, loss_val)), axis=0)
            n_epoch_avg = math.ceil(epoch_ratio * len(loss_es))
            loss_avg = np.mean(loss[-n_epoch_avg:])

            loss_avg_epoch.append(loss_avg)
            ratio_avg_epoch.append(epoch_ratio)

    df_avg = pd.DataFrame(zip(ratio_avg_epoch, loss_avg_epoch), columns=['ratio', 'loss'])
    df_avg = df_avg.groupby('ratio', as_index=False)['loss'].mean()
    epoch_ratio = df_avg.iloc[df_avg['loss'].idxmin(), 0]

    return n_epoch_final, math.ceil(epoch_ratio * n_epoch_final)


# Setting a model's parameters
def set_weights(model, weights):
    for name, param in model.named_parameters():
        # if name.endswith('weight'):
        param.data.copy_(weights[name])

## Data preparation and model initialisation

class LSTM_model(nn.Module):
    """A stateless LSTM neural network with return sequence=False.

    Parameters
    ----------
    input_size : the number of variables.
    hidden_size : the number of hidden units.
    n_layer : the number of stacked layers.
    dropout_prob : the dropout probability.
    device : train a model either on CPU or GPU.
    """

    def __init__(self, input_size, hidden_size, n_layer, dropout_prob, device):
        super().__init__()  # super(LSTM_model, self)
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.n_layer = n_layer
        self.dropout_prob = dropout_prob
        self.device = device

        # No dropout if a model has only one layer
        if n_layer == 1:
            self.lstm = nn.LSTM(self.input_size, self.hidden_size, self.n_layer, batch_first=True)
        else:
            self.lstm = nn.LSTM(self.input_size, self.hidden_size, self.n_layer,
                                batch_first=True, dropout=self.dropout_prob)

        self.output_layer = nn.Linear(self.hidden_size, 1)  # a fully connected layer (output)

    def forward(self, x):
        # Initial hidden state, initial cell state
        batch_size = x.size(0)
        h0 = torch.zeros(self.n_layer, batch_size, self.hidden_size).to(self.device)
        c0 = torch.zeros(self.n_layer, batch_size, self.hidden_size).to(self.device)

        out, _ = self.lstm(x, (h0, c0))  # i.e. stateful=False
        out = self.output_layer(out[:, -1, :])  # (batch_size, seq_length, hidden_size), i.e. return_sequences=False

        return out


class DatasetUtil(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.x)

    # Get the index
    def __getitem__(self, i):
        return self.x[i], self.y[i]


class PrepareData:
    """Transforming data to a supervised ML problem, scaling data, creating CV indices."""

    def __init__(self, data, id_col, var_cols, seq_len, sk_scaler_cv):
        self.data = data.copy()  # !!
        self.id_col = id_col
        self.var_cols = var_cols
        self.seq_len = seq_len
        self.sk_scaler_cv = sk_scaler_cv

        """
        Parameters
        ----------
        id_col : the name of the id column, e.g. 'id'.
        var_cols : a list containing variable names, e.g. ['x1', 'x2'].
        seq_len : the sliding window size, e.g. seq_len=10.
        sk_scaler_cv : sklearn estimator for data scaling, e.g. MinMaxScaler().
        """

    def ts_shape(self, col_vector, model_stage: Literal['cv', 'training', 'forecasting']):
        """
        For a given id, this method reshapes it using the sliding window approach.

        Example
        ----------
        The first column is the target variable (Y), the rest are exogenous variables (X):

        [[22, 1, 0],
         [33, 2, 0],
         [44, 3, 0],
         [55, 4, 0]]

        Transforming the dataset to a supervised problem assuming the seq_len=2:

        X_1 = np.array([[22, 1, 0], [33, 2, 0]]), Y_1 = [44]
        X_2 = np.array([[33, 2, 0], [44, 3, 0]]), Y_2 = [55]
        """

        if model_stage == 'forecasting':
            x = []

            window = col_vector[-self.seq_len:]
            x.append(window)
            x = np.array(x, dtype=np.float32)

            return x

        else:
            x, y = [], []

            if col_vector.ndim == 1:
                for i in range(len(col_vector) - self.seq_len):
                    window = col_vector[i:(i + self.seq_len)]
                    x.append(window)
                    y.append(col_vector[(i + self.seq_len)])

                x = np.array(x, dtype=np.float32)
                y = np.array(y, dtype=np.float32).reshape(-1, 1)

            else:
                for i in range(len(col_vector) - self.seq_len):
                    window = col_vector[i:(i + self.seq_len), :]
                    x.append(window)
                    y.append(col_vector[(i + self.seq_len), 0])  # keep only the 1st target

                x = np.array(x, dtype=np.float32)
                y = np.array(y, dtype=np.float32).reshape(-1, 1)  # reshape the 1st target

            return x, y

    def scale_fit_cv(self, idx_tr):
        scaler = self.sk_scaler_cv
        scaler.fit(self.data.loc[idx_tr, self.var_cols].to_numpy())

        return scaler

    def ts_shape_grouped(self, id_list, model_stage: Literal['cv', 'training', 'forecasting']):
        """Iterating through each id and applying `ts_shape`."""

        if model_stage == 'forecasting':  # (!! use only a subset of data !!)
            x, y = [], None

            for i in id_list:
                var_id = self.data.loc[self.data[self.id_col] == i, self.var_cols].to_numpy()
                x_id = self.ts_shape(col_vector=var_id, model_stage=model_stage)
                x.append(x_id)
            x = np.concatenate(x)
        else:
            x, y = [], []

            for i in id_list:
                var_id = self.data.loc[self.data[self.id_col] == i, self.var_cols].to_numpy()
                x_id, y_id = self.ts_shape(col_vector=var_id, model_stage=model_stage)
                x.append(x_id)
                y.append(y_id)
            x, y = np.concatenate(x), np.concatenate(y)

        return x, y

    def cv_idx(self, padded_len, n_split, val_size, val_size_es, idx_tr_scale=False):
        """This method creates indices for time-series CV.

        Parameters
        ----------
        padded_len : the maximum length after padding that any id can have.
        n_split : the number of CV splits.
        val_size : the size of a validation set for parameter tuning.
        val_size_es : the size of a validation set for early stopping.
        idx_tr_scale : if True, indices are determined for the original dataset to apply scaling properly within each
        CV loop. If False, indices are determined for the transformed dataset (reshaped for supervised learning).
        """

        n_ids = len(self.data[self.id_col].unique())

        if idx_tr_scale:
            idx_len = self.data.shape[0]
        else:
            idx_len = self.data.shape[0] - n_ids * self.seq_len

        idx_tr_val = []

        for split in range(n_split):
            idx_val_0, idx_val_1, idx_tr_end = [], [], []
            idx_cv_len = idx_len - (val_size_es * split)

            if idx_tr_scale:
                padded_len_cv = padded_len - (val_size_es * split) - 1
            else:
                padded_len_cv = padded_len - (val_size_es * split) - 1 - self.seq_len

            for x in range(idx_cv_len):
                if (x != 0) and (x % padded_len_cv == 0):
                    idx_u = x
                    idx_m = (idx_u - val_size_es) + 1
                    idx_l = (idx_u - (val_size + val_size_es)) + 1

                    idx_tr_end.append(idx_l - 1)
                    idx_val_0_ = list(range(idx_l, idx_m))
                    idx_val_0.append(idx_val_0_)
                    idx_val_1_ = list(range(idx_m, (idx_u + 1)))
                    idx_val_1.append(idx_val_1_)

            idx_val_0 = list(itertools.chain.from_iterable(idx_val_0))
            idx_val_1 = list(itertools.chain.from_iterable(idx_val_1))

            if idx_tr_scale:
                idx_tr = list(set(range(idx_cv_len)) - set(idx_val_0 + idx_val_1) - set(idx_tr_end))
                idx_tr_val.append(idx_tr)
            else:
                idx_tr = list(set(range(idx_cv_len)) - set(idx_val_0 + idx_val_1))
                idx_tr_val.append((idx_tr, idx_val_0, idx_val_1))

        return idx_tr_val


class EarlyStopping:
    def __init__(self, patience=1, min_delta=0.05):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = np.inf

    # Return True when validation loss is not decreased by the `min_delta` for `patience` times
    def early_stop_check(self, validation_loss):
        if ((validation_loss + self.min_delta * validation_loss) < self.min_validation_loss):
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif ((validation_loss + self.min_delta * validation_loss) > self.min_validation_loss):
            self.counter += 1

            if self.counter >= self.patience:
                print(43*'-')
                print(f'Loss has not improved for the last {self.patience} epochs')
                print(43*'-')

                return True

        print(f'Patience: {self.counter} / {self.patience}')

        return False


class LstmTrainingCv(PrepareData):
    def __init__(self, data, id_col, var_cols, seq_len, sk_scaler_cv,
                 id_list, loss_function, device):
        super().__init__(data, id_col, var_cols, seq_len, sk_scaler_cv)
        self.id_list = id_list
        self.loss_function = loss_function
        self.device = device

    def lstm_train_epoch(self, model, optimizer, loader_train, epoch):
        loss_cumsum = float(0)
        params_epoch = {}
        model.train(True)

        for idx, batch in enumerate(loader_train):
            x_batch, y_batch = batch[0].to(self.device), batch[1].to(self.device)

            output = model(x_batch)
            loss = self.loss_function(output, y_batch[:, 0].reshape(-1, 1))  # !!
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Monitoring total loss per epoch
            loss_cumsum += loss.item()

        avg_loss = loss_cumsum / len(loader_train)
        print(f'Epoch: {epoch + 1}')
        print('Avg_loss_tr: {0:0.10f}'.format(avg_loss))

        for key, value in model.named_parameters():
            params_epoch[key] = value

        return avg_loss, params_epoch

    def lstm_validate_epoch(self, model, loader_val, val_type=None):
        loss_cumsum = float(0)
        model.train(False)

        for idx, batch in enumerate(loader_val):
            x_batch, y_batch = batch[0].to(self.device), batch[1].to(self.device)

            with torch.no_grad():
                output = model(x_batch)
                loss = self.loss_function(output, y_batch[:, 0].reshape(-1, 1))  # !!

                # Monitoring total loss per epoch
                loss_cumsum += loss.item()

        avg_loss = loss_cumsum / len(loader_val)
        val_type = [f'_{x}' if x is not None else '' for x in [val_type]][0]
        print('Avg_loss_val{0}: {1:0.10f}'.format(val_type, avg_loss))

        return avg_loss

    def lstm_cv(self, input_size, hidden_size, n_layer, dropout_prob, l_rate, padded_len,
                n_epoch, batch_size, n_split, val_size, val_size_es, patience, min_delta):

        # Converting data to a supervised format
        prepare_data_lstm = super().ts_shape_grouped(id_list=self.id_list, model_stage='cv')
        data_shape = prepare_data_lstm[0].shape
        data_n_seq, data_seq_len, data_n_var = data_shape[0], data_shape[1], data_shape[2]

        x_tr_val = prepare_data_lstm[0].reshape(data_n_seq * data_seq_len, data_n_var)
        y_tr_val = prepare_data_lstm[1]
        dummy_matrix = np.zeros((data_n_seq, input_size - 1), dtype=np.float32)
        y_tr_val = np.column_stack((y_tr_val, dummy_matrix))

        # Getting cv indices
        cv_idx_scale = super().cv_idx(padded_len=padded_len, n_split=n_split,
                                      val_size=val_size, val_size_es=val_size_es, idx_tr_scale=True)

        cv_idx_tr_val = super().cv_idx(padded_len=padded_len, n_split=n_split,
                                       val_size=val_size, val_size_es=val_size_es, idx_tr_scale=False)

        # CV loop
        final_epoch, loss_final_epoch_val = [], []
        loss_val_es_cv, loss_val_cv, loss_tr_cv = [], [], []
        # param_all = []

        for i in range(n_split):
            print(f'CV iteration: {i + 1} / {n_split}')

            # Initialising an LSTM model
            model = LSTM_model(input_size=input_size, hidden_size=hidden_size,
                               n_layer=n_layer, dropout_prob=dropout_prob, device=self.device)

            model.to(self.device)
            optimizer = torch.optim.Adam(model.parameters(), lr=l_rate)

            # Preparing dataset: scaling, CV folds, loaders
            cv_idx_tr_scale = cv_idx_scale[i]
            cv_idx_tr = cv_idx_tr_val[i][0]
            cv_idx_val, cv_idx_val_es = cv_idx_tr_val[i][1], cv_idx_tr_val[i][2]

            scaler = super().scale_fit_cv(idx_tr=cv_idx_tr_scale)
            x_tr_val_fold = scaler.transform(x_tr_val)
            x_tr_val_fold = x_tr_val_fold.reshape(data_n_seq, data_seq_len, data_n_var)
            y_tr_val_fold = scaler.transform(y_tr_val)[:, 0].reshape(-1, 1)

            x_tr = torch.from_numpy(x_tr_val_fold[cv_idx_tr])
            y_tr = torch.from_numpy(y_tr_val_fold[cv_idx_tr])
            x_val = torch.from_numpy(x_tr_val_fold[cv_idx_val])
            y_val = torch.from_numpy(y_tr_val_fold[cv_idx_val])
            x_val_es = torch.from_numpy(x_tr_val_fold[cv_idx_val_es])
            y_val_es = torch.from_numpy(y_tr_val_fold[cv_idx_val_es])

            del x_tr_val_fold, y_tr_val_fold

            loader_tr = DataLoader(DatasetUtil(x=x_tr, y=y_tr), batch_size=batch_size, shuffle=True)
            loader_val = DataLoader(DatasetUtil(x=x_val, y=y_val), batch_size=batch_size, shuffle=False)
            loader_val_es = DataLoader(DatasetUtil(x=x_val_es, y=y_val_es), batch_size=batch_size, shuffle=False)

            # Training and validating. Gathering info (errors, parameters, epochs...)
            loss_epoch_val_es, loss_epoch_val, loss_epoch_tr = [], [], []
            # param_epoch = []

            es_criterion = EarlyStopping(patience=patience, min_delta=min_delta)

            for j in range(n_epoch):
                loss_tr, param = self.lstm_train_epoch(model=model, optimizer=optimizer, loader_train=loader_tr, epoch=j)
                loss_val = self.lstm_validate_epoch(model=model, loader_val=loader_val, val_type=None)
                loss_val_es = self.lstm_validate_epoch(model=model, loader_val=loader_val_es, val_type='es')

                loss_epoch_val_es.append(loss_val_es)
                loss_epoch_val.append(loss_val)
                loss_epoch_tr.append(loss_tr)
                # param_epoch.append(deepcopy(param))

                # Early stopping turns on after n epochs have passed, where n = patience
                if (j + 1) > patience:
                    if es_criterion.early_stop_check(validation_loss=loss_val_es):
                        final_epoch.append((j + 1) - patience)

                        loss_epoch_val_es = loss_epoch_val_es[0:(j - patience + 1)]
                        loss_final_val = loss_epoch_val[(j - patience)]
                        loss_epoch_val = loss_epoch_val[0:(j - patience + 1)]
                        loss_epoch_tr = loss_epoch_tr[0:(j - patience + 1)]
                        # param_epoch = param_epoch[0:(j - patience + 1)]

                        break

            # Results
            loss_val_es_cv.append(loss_epoch_val_es)
            loss_final_epoch_val.append(loss_final_val)
            loss_val_cv.append(loss_epoch_val)
            loss_tr_cv.append(loss_epoch_tr)
            # param_all.append(param_epoch)

        # Scores
        loss_avg = np.round(np.mean(loss_final_epoch_val), 10)
        loss_std = np.round(np.std(loss_final_epoch_val), 10)
        loss_coef_var = loss_std / loss_avg

        print(f'CV loss avg: {loss_avg}', f'CV loss std: {loss_std}',
              f'CV loss coef_var: {loss_coef_var}')

        return loss_val_es_cv, loss_val_cv, loss_final_epoch_val, final_epoch, loss_tr_cv  # param_all

    def lstm_train(self, input_size, hidden_size, n_layer, dropout_prob,
                   l_rate, n_epoch, batch_size):

        # Converting data to a supervised format
        prepare_data_lstm = super().ts_shape_grouped(id_list=self.id_list, model_stage='training')
        x_tr = torch.from_numpy(prepare_data_lstm[0])
        y_tr = torch.from_numpy(prepare_data_lstm[1])

        del prepare_data_lstm

        loader_tr = DataLoader(DatasetUtil(x=x_tr, y=y_tr), batch_size=batch_size, shuffle=True)

        # Initialising an LSTM model
        model = LSTM_model(input_size=input_size, hidden_size=hidden_size,
                           n_layer=n_layer, dropout_prob=dropout_prob, device=self.device)

        model.to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=l_rate)

        # Training
        param_all = []

        for x in range(n_epoch):
            loss_tr, param = self.lstm_train_epoch(model=model, optimizer=optimizer, loader_train=loader_tr, epoch=x)
            param_all.append(deepcopy(param))

        return model, param_all

    def lstm_forecast(self, model, data_vars, date_col, id_col, target_col, sk_scaler_fit):
        # Converting data to a supervised format
        prepare_data_lstm = super().ts_shape_grouped(id_list=self.id_list, model_stage='forecasting')
        x_fc = torch.from_numpy(prepare_data_lstm[0])

        # Forecasting
        preds = model(x_fc.to(self.device)).detach().cpu().numpy().flatten()

        # Adding predictions to the initial dataset (!! use a subset of data !!)
        dates = [self.data[date_col].max() + pd.to_timedelta(1, unit='W')] * len(preds)
        col_names = [id_col] + [date_col] + [target_col]
        preds_df = pd.DataFrame(zip(self.id_list, dates, preds), columns=col_names)
        preds_df = ts_var(data=preds_df, data_vars=data_vars, date_col=date_col,
                          id_col=id_col, target_col=target_col)

        input_col_new = [x for x in self.var_cols if target_col not in x]
        target_col_idx = self.var_cols.index(target_col)
        preds_df_scaled = np.delete(sk_scaler_fit.transform(preds_df[self.var_cols].to_numpy()), target_col_idx, 1)
        preds_df[input_col_new] = preds_df_scaled

        self.data = pd.concat([self.data, preds_df], axis=0)


if __name__ == '__main__':
    pass
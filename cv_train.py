import pickle
from pathlib import Path

import torch
from sklearn.preprocessing import MinMaxScaler

import model_creation as mc


## Utility functions

def avg_param(param_list):
    param_avg = dict.fromkeys(param_list[0].keys(), 0)
    n_avg = len(param_list)

    for i in range(n_avg):
        params_epoch = param_list[i]
        for key, value in params_epoch.items():
            param_avg[key] += (1 / n_avg) * value.detach()

    return param_avg

## Data

path_project = str(Path('__file__').absolute().parent)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Device:', device)

data = mc.get_data(db_url='mysql+mysqlconnector:...', data_path=f'{path_project}\\data\\data.pickle', sql_access=False)

data = data[data['week'] < '2023-06-19'].reset_index(drop=True)
data = data.sort_values(['ID', 'week']).reset_index(drop=True)

## Variables

with open(f'{path_project}\\data\\weather.pickle', 'rb') as handle:
    weather = pickle.load(handle)

data = mc.ts_var(data=data, data_vars=weather, date_col='week', id_col='ID', target_col='QuantityDal')

## Initial parameters

# Parameters
n_split = 4
val_size, val_size_es = 4, 4
padded_len = len(data['ID'][data['ID'] == data['ID'][0]])

input_col = (['QuantityDal', 'temp_day', 'temp_sat', 'temp_sun'] +
             [x for x in data.columns if 'f_' in x or 'lag' in x])

id_list = data['ID'].unique()
n_id = len(id_list)

sk_scaler = MinMaxScaler()
loss_func = torch.nn.MSELoss()

# Hyperparameters
seq_len = 15
input_size = len(input_col)
l_rate = 0.001
dropout_prob = 0.2
hidden_size = 2*32
n_layer = 2
batch_size = 10*64
n_epoch = 1000
patience = 35
min_delta = 0.04

# Modes
cv_mode = False
training_mode = True
n_models = 4

## CV

if cv_mode:
    lstm_train_cv = mc.LstmTrainingCv(data=data, id_col='ID', var_cols=input_col, seq_len=seq_len,
                                      sk_scaler_cv=sk_scaler, id_list=id_list, loss_function=loss_func, device=device)

    lstm_cv = lstm_train_cv.lstm_cv(input_size=input_size, hidden_size=hidden_size, n_layer=n_layer,
                                    dropout_prob=dropout_prob, l_rate=l_rate,
                                    padded_len=padded_len,
                                    n_epoch=n_epoch, batch_size=batch_size,
                                    n_split=n_split, val_size=val_size, val_size_es=val_size_es,
                                    patience=patience, min_delta=min_delta)

    loss_val_es_cv, loss_val_cv, loss_tr_cv = lstm_cv[0], lstm_cv[1], lstm_cv[4]
    loss_final_epoch_val, final_epoch = lstm_cv[2], lstm_cv[3]

    # Saving CV data
    with open(f'{path_project}\\model\\lstm_cv.pickle', 'wb') as handle:
        pickle.dump(lstm_cv, handle)

    n_epoch_final, n_epoch_avg = mc.n_epoch_optim(loss_val_es_cv=loss_val_es_cv, loss_val_cv=loss_val_cv,  final_epoch=final_epoch)
    if n_epoch_avg == 0:
        n_epoch_avg = -1
    param_epoch = {'n_epoch_final': n_epoch_final, 'n_epoch_avg': n_epoch_avg}

    # Saving epoch data
    with open(f'{path_project}\\model\\epoch.pickle', 'wb') as handle:
        pickle.dump(param_epoch, handle)
else:
    # Loading CV data
    with open(f'{path_project}\\model\\lstm_cv.pickle', 'rb') as handle:
        lstm_cv = pickle.load(handle)

    # Loading epoch data
    with open(f'{path_project}\\model\\epoch.pickle', 'rb') as handle:
        param_epoch = pickle.load(handle)

## Training

n_epoch_final, n_epoch_avg = param_epoch['n_epoch_final'], param_epoch['n_epoch_avg']
param_n_final_model, param_n_avg_model = [], []

if training_mode:
    scaler = sk_scaler.fit(data[input_col].to_numpy(), input_col)
    data[input_col] = scaler.transform(data[input_col].to_numpy())

    lstm_train_cv = mc.LstmTrainingCv(data=data, id_col='ID', var_cols=input_col, seq_len=seq_len,
                                      sk_scaler_cv=sk_scaler, id_list=id_list, loss_function=loss_func, device=device)

    for i in range(n_models):
        print(43 * '-')
        print(f'Model: {i}')
        print(43 * '-')

        final_model, param_final_model = lstm_train_cv.lstm_train(input_size=input_size, hidden_size=hidden_size,
                                                                  n_layer=n_layer, dropout_prob=dropout_prob, l_rate=l_rate,
                                                                  n_epoch=n_epoch_final, batch_size=batch_size)

        # Averaging parameters (equal weights)
        params_all_subset = param_final_model[-n_epoch_avg:]
        param_avg_epoch = avg_param(param_list=params_all_subset)
        param_n_avg_model.append(param_avg_epoch)
        torch.save(param_avg_epoch, f'{path_project}//model/final_model_{i}_param_avg.pt')
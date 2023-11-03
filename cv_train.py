import json
import pickle
from pathlib import Path

import torch
from sklearn.preprocessing import MinMaxScaler

import data_processing as dp
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

## Initial parameters

path_project = str(Path('__file__').absolute().parent)

with open(f'{path_project}//config.json', 'r') as config_file:
    config = json.load(config_file)

model_config, model_cv = config['model_configuration'], config['model_cv']
model_training, model_testing = config['model_training'], config['testing']
testing = model_testing['testing']

## Data

data = dp.get_data(db_url='mysql+mysqlconnector:...', data_path=f'{path_project}\\data\\data.pickle', sql_access=False)
data = dp.preprocess_data(data=data, date_col='week', id_col='ID', target_col='QuantityDal', empty_token=0)

if testing:
    data = dp.split_data(data=data, date_col='week', test_start='2023-06-19', test_end=None)[1]

## Variables

with open(f'{path_project}\\data\\weather.pickle', 'rb') as handle:
    weather = pickle.load(handle)

data = mc.ts_var(data=data, date_col='week', id_col='ID', target_col='QuantityDal',
                 sin_cos_vars=True, lag_vars=True, data_vars=weather)

## Script parameters

# Parameters
n_split = model_cv['n_split']
val_size, val_size_es = model_cv['val_size'], model_cv['val_size_es']
padded_len = len(data['ID'][data['ID'] == data['ID'][0]])

input_col = (['QuantityDal', 'temp_day', 'temp_sat', 'temp_sun'] +
             [x for x in data.columns if 'f_' in x or 'lag' in x])

id_list = data['ID'].unique()
n_id = len(id_list)

sk_scaler = MinMaxScaler()
loss_func = torch.nn.MSELoss()

# Hyperparameters
seq_len = model_config['seq_len']
input_size = len(input_col)
l_rate = model_config['l_rate']
dropout_prob = model_config['dropout_prob']
weight_decay = model_config['weight_decay']
hidden_size = model_config['hidden_size']
n_layer = model_config['n_layer']
batch_size = model_config['batch_size']
n_epoch = model_config['n_epoch']
patience = model_config['patience']
min_delta = model_config['min_delta']

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Modes
cv_mode = True
training_mode = True
n_models = model_training['n_models']

## CV

if cv_mode:
    lstm_train_cv = mc.LstmTrainingCv(data=data, id_col='ID', var_cols=input_col, seq_len=seq_len,
                                      sk_scaler_cv=sk_scaler, id_list=id_list, loss_function=loss_func, device=device)

    lstm_cv = lstm_train_cv.lstm_cv(input_size=input_size, hidden_size=hidden_size, n_layer=n_layer,
                                    dropout_prob=dropout_prob, weight_decay=weight_decay, l_rate=l_rate,
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
                                                                  n_layer=n_layer, dropout_prob=dropout_prob,
                                                                  weight_decay=weight_decay, l_rate=l_rate,
                                                                  n_epoch=n_epoch_final, batch_size=batch_size)

        # Averaging parameters (equal weights)
        params_all_subset = param_final_model[-n_epoch_avg:]
        param_avg_epoch = avg_param(param_list=params_all_subset)
        param_n_avg_model.append(param_avg_epoch)
        torch.save(param_avg_epoch, f'{path_project}//model/final_model_{i}_param_avg.pt')
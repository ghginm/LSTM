import json
import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler

import data_processing as dp
import model_creation as mc


## Initial parameters

path_project = str(Path('__file__').absolute().parent)

with open(f'{path_project}//config.json', 'r') as config_file:
    config = json.load(config_file)

model_config, model_cv = config['model_configuration'], config['model_cv']
model_training, model_testing, model_forecasting = config['model_training'], config['testing'], config['model_forecasting']
testing = model_testing['testing']

## Data

data = dp.get_data(db_url='mysql+mysqlconnector:...', data_path=f'{path_project}\\data\\data.pickle', sql_access=False)
data = dp.preprocess_data(data=data, date_col='week', id_col='ID', target_col='QuantityDal', empty_token=0)

if testing:
    data_test, data = dp.split_data(data=data, date_col='week', test_start='2023-06-19', test_end=None)

## Variables

with open(f'{path_project}\\data\\weather.pickle', 'rb') as handle:
    weather = pickle.load(handle)

data = mc.ts_var(data=data, date_col='week', id_col='ID', target_col='QuantityDal',
                 sin_cos_vars=True, lag_vars=True, data_vars=weather)

## Script parameters

# Parameters
input_col = (['QuantityDal', 'temp_day', 'temp_sat', 'temp_sun'] +
             [x for x in data.columns if 'f_' in x or 'lag' in x])

id_list = data['ID'].unique()

sk_scaler = MinMaxScaler()
loss_func = torch.nn.MSELoss()

if testing:
    h = len(data_test['week'].unique())
else:
    h = model_forecasting['h']

# Hyperparameters
seq_len = model_config['seq_len']
input_size = len(input_col)
dropout_prob = model_config['dropout_prob']
hidden_size = model_config['hidden_size']
n_layer = model_config['n_layer']

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

## Scaling data

scaler = sk_scaler.fit(data[input_col].to_numpy(), input_col)
data[input_col] = scaler.transform(data[input_col].to_numpy())

## Forecasting

model_name = os.listdir(f'{path_project}\\model')
model_name = [x for x in model_name if 'final_model' in x]
all_preds = []

for i in model_name:
    print(43*'-')
    print(f'Model: {i}')
    print(43 * '-')

    lstm_train_cv = mc.LstmTrainingCv(data=data, id_col='ID', var_cols=input_col, seq_len=seq_len,
                                      sk_scaler_cv=sk_scaler, id_list=id_list, loss_function=loss_func, device=device)

    final_model = mc.LSTM_model(input_size=input_size, hidden_size=hidden_size, n_layer=n_layer, dropout_prob=dropout_prob, device=device)
    final_model.load_state_dict(torch.load(f'{path_project}//model/{i}'))
    final_model.eval()

    for i in range(h):
        print(f'Forecasting week {i + 1}')
        lstm_train_cv.lstm_forecast(model=final_model, data_vars=weather, date_col='week', id_col='ID',
                                    target_col='QuantityDal', sk_scaler_fit=scaler)

    # Predictions
    data_pred = lstm_train_cv.data
    data_pred = data_pred[data_pred['week'] > data['week'].max()].copy()
    data_pred[input_col] = scaler.inverse_transform(data_pred[input_col])
    preds = data_pred['QuantityDal'].to_numpy()
    preds[preds < 0] = 0
    all_preds.append(preds)

# Filtering results
fc_total = np.sum(all_preds, axis=1)
fc_final_idx = []

for idx, x in enumerate(fc_total / np.median(fc_total)):
    if (x <= 1.1) and (x >= 0.9):
        fc_final_idx.append(idx)

all_preds = np.array(all_preds)[fc_final_idx]

# Saving results
if testing:
    preds_final = data_pred[['ID', 'week']].merge(data_test, how='left', on=['ID', 'week']).reset_index(drop=True)
    preds_final['LSTM'] = np.average(all_preds, axis=0)

    fc_version = (data['week'].max() + pd.to_timedelta(1, unit='W')).strftime('%Y-%m-%d')
    with open(f'{path_project}\\analysis\\fc_{fc_version}.pickle', 'wb') as handle:
        pickle.dump(preds_final, handle)  # load to SQL...
else:
    preds_final = data_pred[['ID', 'week']].reset_index(drop=True)
    preds_final['LSTM'] = np.average(all_preds, axis=0)

    fc_version = (data['week'].max() + pd.to_timedelta(1, unit='W')).strftime('%Y-%m-%d')
    with open(f'{path_project}\\output\\fc_{fc_version}.pickle', 'wb') as handle:
        pickle.dump(preds_final, handle)  # load to SQL...
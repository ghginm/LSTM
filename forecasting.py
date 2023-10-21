import os
import pickle
from pathlib import Path

import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler

import model_creation as mc


## Data

path_project = str(Path('__file__').absolute().parent)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Device:', device)

data = mc.get_data(db_url='mysql+mysqlconnector:...', data_path=f'{path_project}\\data\\data.pickle', sql_access=False)

data_test = data[data['week'] >= '2023-06-19'].reset_index(drop=True)
data = data[data['week'] < '2023-06-19'].reset_index(drop=True)
data = data.sort_values(['ID', 'week']).reset_index(drop=True)

## Variables

with open(f'{path_project}\\data\\weather.pickle', 'rb') as handle:
    weather = pickle.load(handle)

data = mc.ts_var(data=data, data_vars=weather, date_col='week', id_col='ID', target_col='QuantityDal')

## Initial parameters

# Parameters
input_col = (['QuantityDal', 'temp_day', 'temp_sat', 'temp_sun'] +
             [x for x in data.columns if 'f_' in x or 'lag' in x])

id_list = data['ID'].unique()

sk_scaler = MinMaxScaler()
loss_func = torch.nn.MSELoss()

h = len(data_test['week'].unique())

# Hyperparameters
seq_len = 15
input_size = len(input_col)
dropout_prob = 0.2
hidden_size = 2*32
n_layer = 2

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

preds_final = data_pred[['ID', 'week']].merge(data_test, how='left', on=['ID', 'week']).reset_index(drop=True)
preds_final['LSTM'] = np.average(all_preds, axis=0)

# Saving results
with open(f'{path_project}\\analysis\\lstm_fc.pickle', 'wb') as handle:
    pickle.dump(preds_final, handle)
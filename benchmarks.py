import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error
from statsforecast import StatsForecast
from statsforecast.models import AutoETS, AutoARIMA

from model_creation import get_data

## Utility functions

def model_evaluation(data, date_col, y_true, y_pred_list):
    data = data.copy()
    final_scores = []

    for i in data[date_col].unique():
        data_date = data[data[date_col] == i].copy()
        score_rmse, score_mae, score_fa, score_bias = [], [], [], []

        for j in y_pred_list:
            rmse = mean_squared_error(data_date[y_true], data_date[j], squared=False)
            score_rmse.append(round(rmse, 0))
            mae = mean_absolute_error(data_date[y_true], data_date[j])
            score_mae.append(round(mae, 0))

            data_date[f'delta_{j}'] = data_date[j] - data_date[y_true]
            data_date[f'delta_abs_{j}'] = np.abs(data_date[f'delta_{j}'])

        bias = [x for x in data_date.columns if 'delta' in x and 'abs' not in x]
        fa = [x for x in data_date.columns if 'delta_abs' in x]

        for fc, fa, bias in zip(y_pred_list, fa, bias):
            score_fa.append(round((1 - np.sum(data_date[fa]) / np.sum(data_date[fc]))*100, 2))
            score_bias.append(round((np.sum(data_date[bias]) / np.sum(data_date[fc]))*100, 2))

        df_scores = pd.DataFrame(zip(score_rmse, score_mae, score_fa, score_bias), index=y_pred_list,
                                 columns=['RMSE', 'MAE', 'Forecast accuracy, %', 'BIAS, %'])

        df_scores['date'] = i
        final_scores.append(df_scores)

    final_scores = pd.concat(final_scores)

    return final_scores

## Data

path_project = str(Path('__file__').absolute().parent)

data = get_data(db_url='mysql+mysqlconnector:...', data_path=f'{path_project}\\data\\data.pickle', sql_access=False)
data = data.rename(columns={'ID': 'unique_id', 'week': 'ds', 'QuantityDal': 'y'})

data = data[data['ds'] < '2023-06-19'].reset_index(drop=True)
data = data.sort_values(['unique_id', 'ds']).reset_index(drop=True)

## Initial parameters

h = 6

## Forecasting

models = [AutoETS(season_length=52, model='ZZZ'), AutoARIMA(season_length=52, max_d=2, max_D=2)]
sf = StatsForecast(models=models, freq='W', n_jobs=-1)

preds_final = sf.forecast(df=data, h=h).reset_index()

preds_final.loc[preds_final['AutoETS'] < 0, 'AutoETS'] = 0
preds_final.loc[preds_final['AutoARIMA'] < 0, 'AutoARIMA'] = 0

preds_final = preds_final.rename(columns={'unique_id': 'ID', 'ds': 'week'})
preds_final['week'] = preds_final['week'] + pd.to_timedelta(1, unit='d')

## Model evaluation

with open(f'{path_project}\\analysis\\lstm_fc.pickle', 'rb') as handle:
    lstm_fc = pickle.load(handle)

preds_final = lstm_fc.merge(preds_final, how='left', on=['ID', 'week'])

with open(f'{path_project}\\analysis\\fc.pickle', 'wb') as handle:
    pickle.dump(preds_final, handle)

final_scores = model_evaluation(data=preds_final, date_col='week', y_true='QuantityDal',
                                y_pred_list=['LSTM', 'AutoETS', 'AutoARIMA'])
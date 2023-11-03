import pickle
from pathlib import Path

import pandas as pd
from statsforecast import StatsForecast
from statsforecast.models import AutoETS, AutoARIMA

import data_processing as dp


## Initial parameters

path_project = str(Path('__file__').absolute().parent)

## Data
with open(f'{path_project}\\analysis\\lstm_fc.pickle', 'rb') as handle:
    lstm_fc = pickle.load(handle)

data = dp.get_data(db_url='mysql+mysqlconnector:...', data_path=f'{path_project}\\data\\data.pickle', sql_access=False)
data = dp.preprocess_data(data=data, date_col='week', id_col='ID', target_col='QuantityDal', empty_token=0)
data = dp.split_data(data=data, date_col='week', test_start=lstm_fc['week'].min(), test_end=None)[1]
data = data.rename(columns={'ID': 'unique_id', 'week': 'ds', 'QuantityDal': 'y'})

## Script parameters

h = len(lstm_fc['week'].unique())

## Forecasting

models = [AutoETS(season_length=52, model='ZZZ'), AutoARIMA(season_length=52, max_d=2, max_D=2)]
sf = StatsForecast(models=models, freq='W', n_jobs=-1)

preds_final = sf.forecast(df=data, h=h).reset_index()

preds_final.loc[preds_final['AutoETS'] < 0, 'AutoETS'] = 0
preds_final.loc[preds_final['AutoARIMA'] < 0, 'AutoARIMA'] = 0

preds_final = preds_final.rename(columns={'unique_id': 'ID', 'ds': 'week'})
preds_final['week'] = preds_final['week'] + pd.to_timedelta(1, unit='d')

## Model evaluation

preds_final = lstm_fc.merge(preds_final, how='left', on=['ID', 'week'])
final_scores = dp.model_evaluation(data=preds_final, date_col='week', y_true='QuantityDal',
                                   y_pred_list=['LSTM', 'AutoETS', 'AutoARIMA'])
import pickle

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sqlalchemy import create_engine


## Load and split a dataset

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


def split_data(data, date_col, test_start='2023-06-19', test_end=None):
    """Test / train splitting."""

    if test_end is None:
        data_test = data[data[date_col] >= test_start].reset_index(drop=True)
        data_train = data[data[date_col] < test_start].reset_index(drop=True)
    else:
        data_test = data[(data[date_col] >= test_start) & (data[date_col] <= test_end)].reset_index(drop=True)
        data_train = data[data[date_col] < test_start].reset_index(drop=True)

    return data_test, data_train


## Preprocess a dataset

def preprocess_data(data, date_col, id_col, target_col, empty_token=-1):
    """Data padding, recovering 0 sales."""

    data = data.set_index(list(set(data.columns) - {target_col}))[[target_col]].unstack().fillna(0).stack().reset_index()
    data['csum'] = data.groupby([id_col], observed=True)[target_col].cumsum()
    data[target_col] = [empty_token if x == 0 else y for x, y in zip(data['csum'], data[target_col])]
    data = data.drop('csum', axis=1).reset_index(drop=True)

    data = data.sort_values([id_col] + [date_col]).reset_index(drop=True)

    return data

## Accessing model performance

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


if __name__ == '__main__':
    pass
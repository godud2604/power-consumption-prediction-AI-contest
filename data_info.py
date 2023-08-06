# %%
import datetime as dt
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as pgo
from sklearn.preprocessing import OneHotEncoder

# %%
train = pd.read_csv('./data/train.csv')
test = pd.read_csv('./data/test.csv')
info = pd.read_csv('./data/building_info.csv')

# %%
drop_cols = ['num_date_time']
train.drop(drop_cols, axis=1, inplace=True)
test.drop(drop_cols, axis=1, inplace=True)

# %% 
use_cols = list(test)
tgt_col = ['전력소비량(kWh)']
target = train[tgt_col]

train = train[use_cols]
train = pd.concat([train, target], axis=1)

# %% 강수량 임의 처리
train.fillna(value=0, inplace=True)
test.fillna(value=0, inplace=True)
train

# %% 일시를 datetime 형태로 수정
date_format = "%Y%m%d %H"
train['일시'] = [datetime.strptime(date_string, date_format) for date_string in train['일시']]
test['일시'] = [datetime.strptime(date_string, date_format) for date_string in test['일시']]

# %% 건물번호를 인덱스로 변경
train.set_index(keys='건물번호', drop=True, inplace=True)
test.set_index(keys='건물번호', drop=True, inplace=True)
train

# %% 입력변수 내 시간정보 추가
train['month'] = train['일시'].dt.month
train['day'] = train['일시'].dt.day
train['hour'] = train['일시'].dt.hour
train['minute'] = train['일시'].dt.minute

test['month'] = test['일시'].dt.month
test['day'] = test['일시'].dt.day
test['hour'] = test['일시'].dt.hour
test['minute'] = test['일시'].dt.minute

# %% - 기호의 누락 기호를 0으로 대체
info.replace({'-' : 0}, inplace=True)
info

# %% ?
include_cols = ['건물유형', '연면적(m2)', '냉방면적(m2)', '태양광용량(kW)', 'ESS저장용량(kWh)', 'PCS용량(kW)']
for i in info['건물번호'].unique():
    train.loc[train.index == i, include_cols] = info.loc[info['건물번호'] == i, include_cols].to_numpy()
    test.loc[test.index == i, include_cols] = info.loc[info['건물번호'] == i, include_cols].to_numpy()

train
# %% 건물유형에 OneHotEncoder 적용
label_encoder = OneHotEncoder()
encoded_train_data = label_encoder.fit_transform(train['건물유형'].to_numpy().reshape(-1, 1))
encoded_test_data = label_encoder.transform(test['건물유형'].to_numpy().reshape(-1, 1))

for i in range(encoded_train_data.shape[1]):
    train[f"OH_{i}"] = encoded_train_data.toarray()[:, i]
    test[f"OH_{i}"] = encoded_test_data.toarray()[:, i]

train

# %% 
from pytimekr import pytimekr
train['휴일'] = 'W'
test['휴일'] = 'W'
for holiday in pytimekr.holidays(year=2022):
    condition = (train['일시'] >= pd.to_datetime(holiday)) & (train['일시'] < pd.to_datetime(holiday) + dt.timedelta(days=1))
    condition_weekday = (
        (train['일시'].dt.weekday == 5) |
        (train['일시'].dt.weekday == 6)
    )
    train.loc[condition, '휴일'] = 'H'
    train.loc[condition_weekday, '휴일'] = 'H'
    condition = (test['일시'] >= pd.to_datetime(holiday)) & (test['일시'] < pd.to_datetime(holiday) + dt.timedelta(days=1))
    condition_weekday = (
        (test['일시'].dt.weekday == 5) |
        (test['일시'].dt.weekday == 6)
    )
    test.loc[condition, '휴일'] = 'H'
    test.loc[condition_weekday, '휴일'] = 'H'

# %%
label_encoder = OneHotEncoder()
encoded_train_data = label_encoder.fit_transform(train['휴일'].to_numpy().reshape(-1, 1))
encoded_test_data = label_encoder.transform(test['휴일'].to_numpy().reshape(-1, 1))

for i in range(encoded_train_data.shape[1]):
    train[f"휴일{i}"] = encoded_train_data.toarray()[:, i]
    test[f"휴일{i}"] = encoded_test_data.toarray()[:, i]

train
# %% baseline model
drop_train_cols = ['일시', '전력소비량(kWh)', '건물유형', '휴일']
drop_test_cols = ['일시', '건물유형', '휴일']

train_x = train.drop(drop_train_cols, axis=1)
test_x = test.drop(drop_test_cols, axis=1)

for col in list(train_x):
    train_x[col] = train_x[col].astype('Float64')
    test_x[col] = test_x[col].astype('Float64')

# rf_reg = RandomForestRegressor(n_jobs=-1)
# rf_reg.fit(train_x, target)
# predicts = rf_reg.predict(test_x)

# %%
reference = pd.DataFrame(index=train_x.index, columns=['month', 'hour', '휴일0', '휴일1'])
reference['month'] = train_x['month']
reference['hour'] = train_x['hour']
reference['휴일0'] = train_x['휴일0']
reference['휴일1'] = train_x['휴일1']
reference['전력소비량(kWh)'] = train['전력소비량(kWh)']
reference

# %%
reference_test = pd.DataFrame(index=test_x.index, columns=['month', 'hour', '휴일0', '휴일1'])
reference_test['month'] = test_x['month']
reference_test['hour'] = test_x['hour']
reference_test['휴일0'] = test_x['휴일0']
reference_test['휴일1'] = test_x['휴일1']
reference_test

# %%
from sklearn.ensemble import RandomForestRegressor, StackingRegressor, HistGradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor

def forecast_stacker(
        train_features: np.ndarray, 
        train_targets: np.ndarray, 
        test_features: np.ndarray
    ):

    base_models = [
        ('HistGrad', HistGradientBoostingRegressor()),
        ('XGB', XGBRegressor()),
        ('MLP', MLPRegressor(hidden_layer_sizes=(256, 128, 64), max_iter=1000, early_stopping=True)),
        ('RANDOMFOREST', RandomForestRegressor(n_jobs=-1)),
        # ('LGBM', LGBMRegressor(n_jobs=-1)),
    ]

    stacker = StackingRegressor(
        estimators=base_models,
        final_estimator=XGBRegressor(),
        cv=5,
        n_jobs=-1,
    )

    stacker.fit(train_features, train_targets)
    predicts = stacker.predict(test_features)
    return predicts

# %% correct forecasting
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from yellowbrick.cluster import KElbowVisualizer
scaler = MinMaxScaler()
train_x = scaler.fit_transform(train_x)
test_x = scaler.transform(test_x)

k_viz = KElbowVisualizer(estimator=KMeans(n_init='auto', random_state=20230805), k=(10, 30))
k_viz.fit(train_x)
k_viz.show()

# %%
kmeans = KMeans(n_init='auto', random_state=20230805, n_clusters=k_viz.elbow_value_)
train_labels = kmeans.fit_predict(train_x)
test_labels = kmeans.predict(test_x)
reference['cluster'] = train_labels
reference

# %%
# 건물번호로 슬라이싱
partial_train = train.loc[train['일시'].dt.month == 8, :]
by_constructure = {}
for i in sorted(reference.loc[reference.month == 8, :].index.unique()):
    # 시간
    by_hour = {}
    for h in sorted(reference.loc[reference.month == 8, :]['hour'].unique()):
        by_holiday = {}
        # 휴일0
        for h0 in reference.loc[reference.month == 8, :]['휴일0'].unique():
            for h1 in reference.loc[reference.month == 8, :]['휴일1'].unique():
                mask = (
                    (reference.loc[reference.month == 8, :].index == i) &
                    (reference.loc[reference.month == 8, "hour"] == h) &
                    (reference.loc[reference.month == 8, '휴일0'] == h0) &
                    (reference.loc[reference.month == 8, '휴일1'] == h1)
                )
        
                if len(partial_train.loc[mask, '전력소비량(kWh)'].to_numpy()) > 0:
                    rep_consume = np.mean(partial_train.loc[mask, '전력소비량(kWh)'].to_numpy())
                else:
                    pass
        

                by_holiday[f'{h0}{h1}'] = rep_consume
        by_hour[h] = by_holiday
    by_constructure[i] =  by_hour


# %%
predicts = forecast_stacker(train_x, target, test_x)

# %%
for i in sorted(reference_test.index.unique()):
    # 시간
    by_hour = {}
    for h in sorted(reference_test['hour'].unique()):
        by_holiday = {}
        # 휴일0
        for h0 in reference_test['휴일0'].unique():
            for h1 in reference_test['휴일1'].unique():
                mask = (
                    (reference_test.index == i) &
                    (reference_test.hour == h) &
                    (reference_test['휴일0'] == h0) &
                    (reference_test['휴일1'] == h1)
                )
                condition1 = predicts[mask] < by_constructure[i][h][f'{h0}{h1}'] * 0.8
                condition2 = predicts[mask] > by_constructure[i][h][f'{h0}{h1}'] * 1.2
                replacement_value = by_constructure[i][h][f'{h0}{h1}']
                predicts[mask] = np.where(condition1 | condition2, replacement_value, predicts[mask])

# %%
submission = pd.read_csv('./data/sample_submission.csv')
submission['answer'] = predicts
submission.set_index(keys='num_date_time', inplace=True, drop=True)
submission.to_csv('./data/sample_submission.csv')
submission
# %%

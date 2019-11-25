#!/usr/bin/env python
# coding: utf-8

# ## Define a function to get the real distance between to lat/long points
# - Manhattan distance should be useful, but I think we can do better with real distance
# - Here we compare a manual calculation to the geopy library

# In[1]:


from math import sin, cos, sqrt, atan2, radians
import geopy.distance

def geo_manhattan_distance(lat1, lat2, long1, long2):
    """
    returns the manhattan distance between two geo points
    """
    return abs(lat2 - lat1) + abs(long2 - long1)

def geopy_dist(coord1, coord2):
    try:
        return geopy.distance.distance(coord1, coord2).kilometers
    except:
        return -1

def haversine(lat1, lon1, lat2, lon2, km_const=6371.0):
    lat1, lon1, lat2, lon2 = map(abs, [lat1, lon1, lat2, lon2])
    lat1, lon1, lat2, lon2 = map(np.deg2rad, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1 
    dlon = lon2 - lon1 
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a)) 
    mi = km_const * c
    return mi


# ## Import our dataset
# - our dataset consists of several files:
#     - train.csv: our training data
#     - test.csv: our testing data
#     - sample_submissions.csv: A sample submission file in the correct format (columns key and fare_amount). This dummy file 'predicts' fare_amount to be $11.35 for all rows, which is the mean fare_amount from the training set.

# In[2]:


import pandas as pd
import os
import sys
import numpy as np
import random

TOTAL_ROWS = 55423855

DATA_FILES_PATH = 'projectDataFiles/'

# training data types
TRAINING_TYPES = {
    'fare_amount': 'float32',
    'pickup_datetime': 'str',
    'pickup_longitude': 'float32',
    'pickup_latitude': 'float32',
    'dropoff_longitude': 'float32',
    'dropoff_latitude': 'float32',
    'passenger_count': 'uint8'
}

COLUMNS = list(TRAINING_TYPES.keys()) + ['real_dist']

FEATURES = [item for item in COLUMNS if item != 'fare_amount']

LABEL = 'fare_amount'

def import_training_dataset_limit(file_path, row_limit=100000):
    """
    function to import the dataset into a pandas dataframe.

    Takes a row limit to limit the number of rows read.
    """
    if row_limit:
        return pd.read_csv(file_path, nrows=row_limit)
    else:
        return pd.read_csv(file_path)


def get_df_list(file_path, chunksize=1000000):
    df_list = []
    pd.set_option('use_inf_as_na', True)
    for df_chunk in pd.read_csv(file_path, chunksize=chunksize, dtype=TRAINING_TYPES):
        df_chunk['pickup_datetime'] = df_chunk['pickup_datetime'].str.slice(0, 16)
        df_chunk['pickup_datetime'] = pd.to_datetime(df_chunk['pickup_datetime'], utc=True, format='%Y-%m-%d %H:%M')
        df_chunk['real_dist'] = haversine(df_chunk['pickup_latitude'], df_chunk['pickup_longitude'], df_chunk['dropoff_latitude'], df_chunk['dropoff_longitude'])
        df_chunk['hour'] = df_chunk['pickup_datetime'].dt.hour
        df_chunk['day'] = df_chunk['pickup_datetime'].dt.day
        df_chunk['month'] = df_chunk['pickup_datetime'].dt.month
        df_chunk['year'] = df_chunk['pickup_datetime'].dt.year
        add_col_hour = pd.get_dummies(df_chunk['hour'], prefix='hour')
        add_col_day = pd.get_dummies(df_chunk['day'], prefix='day')
        add_col_month = pd.get_dummies(df_chunk['month'], prefix='month')
        add_col_year = pd.get_dummies(df_chunk['year'], prefix='year')
        df_chunk = pd.concat([df_chunk, add_col_hour, add_col_day, add_col_month, add_col_year], axis=1)
        df_chunk.drop(['hour', 'day', 'month', 'year', 'key', 'pickup_datetime', 'pickup_longitude', 'pickup_latitude'], axis=1, inplace=True)
        df_chunk.dropna(axis=1, how='any', inplace=True)
        df_list.append(df_chunk)
    return df_list
        

def read_feathered_data(file_path):
    return pd.read_feather(file_path)

def feather_dataset(dataframe, file_out):
    dataframe.to_feather(file_out)

# import the dataset as a list of chunks, from here we can do our processing at a chunk level
print('Importing Datasets...')
DATA_LIST = get_df_list(f'{DATA_FILES_PATH}train.csv')

train_split = int(len(DATA_LIST) * 0.8)

random.shuffle(DATA_LIST)

TRAINING_LIST = DATA_LIST[:train_split]

TEST = pd.concat(DATA_LIST[train_split:])

TRAINING_LIST[0].head()
    


# In[3]:


TEST.head()


# ## Perform a SGD partial fit
# - SGD stands for stochastic gradient descent
# - Here we are feeding our chunks into the partial fit

# In[4]:


from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler

def sgd_train(chunk_list, loss="squared_loss"):
    my_sgd_regressor = SGDRegressor(loss=loss)
    my_sgd_regressor.n_iter = np.ceil(10**6 / len(TEST[LABEL]))
    scaler = StandardScaler()
    for chunk in chunk_list:
        X_train = chunk[chunk.columns.difference([LABEL])]
        scaler.fit(X_train)
        my_sgd_regressor.partial_fit(scaler.transform(X_train), chunk[LABEL])
    X_test = TEST[TEST.columns.difference([LABEL])]
    y_predict = my_sgd_regressor.predict(scaler.transform(X_test))
    return y_predict

print('Getting SGD predictions...')
Y_PREDICT_SGD = sgd_train(TRAINING_LIST)


# In[ ]:


from sklearn.ensemble import RandomForestRegressor
from functools import reduce

def gen_rf(X_train, y_train):
    rf = RandomForestRegressor(n_estimators = 2, verbose=3)
    rf.fit(X_train, y_train)
    return rf

def combine_rf(a, b):
    a.estimators_ += b.estimators_
    a.n_estimators = len(a.estimators_)
    return a

def rf_train(chunk_list):
    rf_list = [gen_rf(chunk[chunk.columns.difference([LABEL])], chunk[LABEL]) for chunk in chunk_list]
    rf_total = reduce(combine_rf, rf_list)
    y_predict = rf_total.predict(TEST[TEST.columns.difference([LABEL])])
    return y_predict

def rf_train_warm_start(chunk_list):
    """
    currently getting like the same rmse as linear with 1 estimator per million.

    is batching the issue?

    would be get a better value using random samples of the training set, maybe with bootstrapping?
    """
    rf = RandomForestRegressor(n_estimators = 1, verbose=3, warm_start=True)
    for chunk in chunk_list:
        X_train = chunk[chunk.columns.difference([LABEL])]
        rf.fit(X_train, chunk[LABEL])
        rf.n_estimators += 1
    X_test = TEST[TEST.columns.difference([LABEL])]
    y_predict = rf.predict(X_test)
    return y_predict
    

print('Getting RF Predictions...')
Y_PREDICT_RF = rf_train_warm_start(TRAINING_LIST)


# In[ ]:


from sklearn import metrics
import numpy as np

def calc_rmse(y_test, y_prediction):
    # Calculating "Mean Square Error" (MSE):
    mse = metrics.mean_squared_error(y_test, y_prediction)

    # Using numpy sqrt function to take the square root and calculate "Root Mean Square Error" (RMSE)
    return np.sqrt(mse)

print(f'SGB RMSE: {calc_rmse(TEST[LABEL], Y_PREDICT_SGD)}')
print(f'RF RMSE: {calc_rmse(TEST[LABEL], Y_PREDICT_RF)}')


# In[ ]:





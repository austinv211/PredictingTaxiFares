#!/usr/bin/env python
# coding: utf-8

# ## Define a function to get the real distance between to lat/long points
# - Manhattan distance should be useful, but I think we can do better with real distance
# - Here we compare a manual calculation to the geopy library

# In[5]:


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

# In[6]:


import pandas as pd
import os
import sys
import numpy as np

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
        add_col = pd.get_dummies(df_chunk['hour'], prefix='hour')
        df_chunk = pd.concat([df_chunk, add_col], axis=1)
        df_chunk.drop(['hour', 'key', 'pickup_datetime'], axis=1, inplace=True)
        df_chunk.dropna(axis=1, how='any', inplace=True)
        df_list.append(df_chunk)
    return df_list
        

def read_feathered_data(file_path):
    return pd.read_feather(file_path)

def feather_dataset(dataframe, file_out):
    dataframe.to_feather(file_out)

# import the dataset as a list of chunks, from here we can do our processing at a chunk level
print('Importing Datasets...')
TRAINING_LIST = get_df_list(f'{DATA_FILES_PATH}train.csv')
TEST = pd.concat(get_df_list(f'{DATA_FILES_PATH}test.csv'))

TRAINING_LIST[0].head()
    


# In[7]:


TEST.head()


# ## Perform a SGD partial fit
# - SGD stands for stochastic gradient descent
# - Here we are feeding our chunks into the partial fit

# In[8]:


from sklearn.linear_model import SGDRegressor

def fit_training_chunks(chunk_list):
    my_sgd_regressor = SGDRegressor()
    for chunk in chunk_list:
        print(chunk.columns.difference(['pickup_datetime', LABEL]))
        my_sgd_regressor.partial_fit(chunk[chunk.columns.difference(['pickup_datetime', LABEL])], chunk[LABEL])
    y_predict = my_sgd_regressor.predict(TEST[TEST.columns.difference(['pickup_datetime', LABEL])])
    return y_predict


fit_training_chunks(TRAINING_LIST)


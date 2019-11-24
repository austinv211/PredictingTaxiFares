#!/usr/bin/env python
# coding: utf-8

# ## Import our dataset
# - our dataset consists of several files:
#     - train.csv: our training data
#     - test.csv: our testing data
#     - sample_submissions.csv: A sample submission file in the correct format (columns key and fare_amount). This dummy file 'predicts' fare_amount to be $11.35 for all rows, which is the mean fare_amount from the training set.

# In[2]:


import pandas as pd
import os
from tqdm.notebook import tqdm
import mmap

DATA_FILES_PATH = 'projectDataFiles/'

def import_training_dataset_limit(file_path, row_limit=100000):
    """
    function to import the dataset into a pandas dataframe.

    Takes a row limit to limit the number of rows read.
    """
    if row_limit:
        return pd.read_csv(file_path, nrows=row_limit)
    else:
        return pd.read_csv(file_path)


def import_training_dataset_chunked(file_path, chunksize=1000000):
    """
    function to import the dataset into a pandas dataframe, reading the file in chunks and appending as we go
    """
    tqdm.pandas(desc="Applying Transformation")
    df = pd.DataFrame()
    counter = 0
    for chunk in pd.read_csv(file_path, chunksize=chunksize, low_memory=True):
        df = pd.concat([df, chunk])
        counter += 1
        print(f'completed # {counter}')
    return df

def read_feathered_data(file_path):
    return pd.read_feather(file_path)

def feather_dataset(dataframe, file_out):
    dataframe.to_feather(file_out)
        

# assign the dataset to the TRAIN Dataframe, right now we are only loading 1,000,000 rows (possibly chunk and feather to reduce loading time)
# TRAIN = import_training_dataset_limit(f'{DATA_FILES_PATH}train.csv')

# import the dataset in chunks
#TRAIN = import_training_dataset_chunked(f'{DATA_FILES_PATH}train.csv')

# feather the dataset
#feather_dataset(TRAIN, f'{DATA_FILES_PATH}train.feather')

# import the dataset from a feather
TRAIN = read_feathered_data(f'{DATA_FILES_PATH}train.feather')

# show the head of the the dataset to see its columns
TRAIN.head()
    


# ## Define a function to get the manhattan distance between two lat/long points
# - manhattan distance should be fairly relistic to new york because of the ways that streets work there, but we might want to use the real travel distance between locations somehow

# In[ ]:


def geo_manhattan_distance(lat1, lat2, long1, long2):
    """
    returns the manhattan distance between two geo points
    """
    return abs(lat2 - lat1) + abs(long2 - long1)


# test it out
geo_manhattan_distance(TRAIN['pickup_latitude'].iloc[0], TRAIN['dropoff_latitude'].iloc[0],TRAIN['pickup_longitude'].iloc[0], TRAIN['dropoff_longitude'].iloc[0])


# ## Define a function to get the real distance between to lat/long points
# - Manhattan distance should be useful, but I think we can do better with real distance
# - Here we compare a manual calculation to the geopy library

# In[ ]:


from math import sin, cos, sqrt, atan2, radians
import geopy.distance

def real_distance(lat1, lat2, long1, long2):
    """
    returns the real distance between two datapoints
    """
    R = 6373.0 #approximate radius of earth in km
    rad_lat1, rad_lat2, rad_long1, rad_long2 = (radians(abs(meas)) for meas in [lat1, lat2, long1, long2])
    long_dist = rad_long2 - rad_long1
    lat_dist = rad_lat2 - rad_lat1
    a = sin(lat_dist / 2)**2+ cos(rad_lat1) * cos(rad_lat2) * sin(long_dist / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    return R * c

def geopy_dist(coord1, coord2):
    return geopy.distance.distance(coord1, coord2).kilometers

# test out both functions
dist_test_1 = real_distance(TRAIN['pickup_latitude'].iloc[0], TRAIN['dropoff_latitude'].iloc[0],TRAIN['pickup_longitude'].iloc[0], TRAIN['dropoff_longitude'].iloc[0])

dist_test_2 = geopy_dist((TRAIN['pickup_latitude'].iloc[0], TRAIN['pickup_longitude'].iloc[0]), (TRAIN['dropoff_latitude'].iloc[0], TRAIN['dropoff_longitude'].iloc[0]))

print(f'Manual: {dist_test_1} km\nGeopy: {dist_test_2} km')


# In[ ]:





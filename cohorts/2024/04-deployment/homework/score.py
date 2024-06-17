#!/usr/bin/env python
# coding: utf-8

import os
import sys
import pickle
from datetime import datetime
import pandas as pd

with open('./cohorts/2024/04-deployment/homework/model.bin', 'rb') as f_in:
    dv, model = pickle.load(f_in)

categorical = ['PULocationID', 'DOLocationID']

def read_data(filename):
    df = pd.read_parquet(filename)
    
    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')
    
    return df

    

def prepare_dictionaries(df: pd.DataFrame):
    categorical = ['PULocationID', 'DOLocationID']
    df[categorical] = df[categorical].astype(str)
    
    df['PU_DO'] = df['PULocationID'] + '_' + df['DOLocationID']

    categorical = ['PU_DO']
    numerical = ['trip_distance']
    dicts = df[categorical + numerical].to_dict(orient='records')
    return dicts





def ride_duration_prediction(year, month):
    filename = 'https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2023-03.parquet'
    df = read_data(filename)
    dicts = df[categorical].to_dict(orient = 'records')
    X_val = dv.transform(dicts)
    y_pred = model.predict(X_val)
    df['ride_id'] = f'{year:04d}/{month:02d}_' + df.index.astype('str')
    result_df = pd.DataFrame()
    result_df['ride_id'] = df['ride_id']
    result_df['Prediction'] = y_pred
    result_df.to_parquet(
    'df_result',
    engine='pyarrow',
    compression=None,
    index=False
)



def run():
    year = 2023
    month = 3


    ride_duration_prediction(
        year = year,
        month = month
    )


if __name__ == '__main__':
    run()
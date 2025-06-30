#!/usr/bin/env python
# coding: utf-8

import sys
import os
import pickle
import pandas as pd
import argparse
import logging as logger

logger.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logger.INFO,
    datefmt='%Y-%m-%d %H:%M:%S'
)

def get_input_path(year, month):
    default_input_pattern = 'https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{year:04d}-{month:02d}.parquet'
    input_pattern = os.getenv('INPUT_FILE_PATTERN', default_input_pattern)
    return input_pattern.format(year=year, month=month)


def get_output_path(year, month):
    default_output_pattern = 's3://nyc-duration-prediction-alexey/taxi_type=fhv/year={year:04d}/month={month:02d}/predictions.parquet'
    output_pattern = os.getenv('OUTPUT_FILE_PATTERN', default_output_pattern)
    return output_pattern.format(year=year, month=month)

def read_data(filename, s3_endpoint_url):
    
    if s3_endpoint_url:
        options = {
            'client_kwargs': {
                'endpoint_url': s3_endpoint_url
            }
        }    
        
        df = pd.read_parquet(filename, storage_options=options)
    else:
        df = pd.read_parquet(filename)
        
    return df


def prepare_data(df, categorical):
    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')

    return df


def load_model(path):
    with open(path, "rb") as f_in:
        dv, model = pickle.load(f_in)
    return dv, model



def save_data(df, filename, s3_endpoint_url):
    if s3_endpoint_url:
        options = {
            'client_kwargs': {
                'endpoint_url': s3_endpoint_url
            }
        }    
        
        df.to_parquet(
            filename,
            engine='pyarrow',
            compression=None,
            index=False,
            storage_options=options
        )
        logger.info(f'Saved data to `{filename}` with S3 endpoint URL')
    else:
        df.to_parquet(
            filename,
            engine='pyarrow',
            compression=None,
            index=False
        )
        logger.info(f'Saved data to `{filename}` without S3 endpoint URL')
    

def main(year, month):
    input_file = get_input_path(year, month)
    output_file = get_output_path(year, month)
    s3_endpoint_url = os.getenv('S3_ENDPOINT_URL', None)

    logger.info(f'Reading data from `{input_file}`')

    dv, lr = load_model('model.bin')

    categorical = ['PULocationID', 'DOLocationID']

    df = read_data(input_file)
    df = prepare_data(df, categorical)
    
    df['ride_id'] = f'{year:04d}/{month:02d}_' + df.index.astype('str')

    dicts = df[categorical].to_dict(orient='records')
    X_val = dv.transform(dicts)
    y_pred = lr.predict(X_val)

    logger.info(f'Predicted mean duration: {y_pred.mean()}')

    df_result = pd.DataFrame()
    df_result['ride_id'] = df['ride_id']
    df_result['predicted_duration'] = y_pred

    save_data(df_result, output_file, s3_endpoint_url)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a model to predict taxi trip duration.')
    parser.add_argument('--year', type=int, required=True, help='Year of the data to train on')
    parser.add_argument('--month', type=int, required=True, help='Month of the data to train on')
    
    args = parser.parse_args()

    year = int(args.year)
    month = int(args.month)

    logger.info(f'Running scoring for {year}, {month}')

    main(year, month)

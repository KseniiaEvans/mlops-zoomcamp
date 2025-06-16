import pickle
import argparse

import pandas as pd
import numpy as np

from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import root_mean_squared_error

# import mlflow

# mlflow.set_tracking_uri("http://localhost:5000")
# mlflow.set_experiment("nyc-taxi-experiment")


def load_model(path):
    with open(path, 'rb') as f_in:
        dv, model = pickle.load(f_in)
    return dv, model

def read_dataframe(year, month):
    url = f'https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{year}-{month:02d}.parquet'
    df = pd.read_parquet(url)
    print('Data loaded:')
    print(df.head())

    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()
    categorical = ['PULocationID', 'DOLocationID']

    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')
    
    return df

def score(df, dv, model, categorical=['PULocationID', 'DOLocationID']):
    dicts = df[categorical].to_dict(orient='records')
    X_val = dv.transform(dicts)
    y_pred = model.predict(X_val)
    
    return y_pred

def get_ride_id_column(df):
    df['ride_id'] = (
        df['tpep_pickup_datetime'].dt.year.astype(str) + '/' +
        df['tpep_pickup_datetime'].dt.month.astype(str).str.zfill(2) + '_' +
        df.index.astype(str)
    )
    return df['ride_id']

def save_result(df, y_pred, output_file='./output/results.parquet'):
    ride_id = get_ride_id_column(df)
    df_result = pd.DataFrame({
        'ride_id': ride_id,
        'y_pred': y_pred
    })

    df_result.to_parquet(
        output_file,
        engine='pyarrow',
        compression=None,
        index=False
    )

    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train a model to predict taxi trip duration.')
    parser.add_argument('--year', type=int, required=True, help='Year of the data to train on')
    parser.add_argument('--month', type=int, required=True, help='Month of the data to train on')
    args = parser.parse_args()

    year = int(args.year)
    month = int(args.month)

    print(f'Running scoring for {year}, {month}')

    dv, model = load_model('model.bin')

    print('Model is loaded')
    df = read_dataframe(year, month)
    y_pred = score(df, dv, model)

    print(f'Mean predicted duration for {year}-{month:02d}: {np.mean(y_pred)}')

    # save_result(df, y_pred, output_file='./output/results.parquet')
    
from prefect import flow, task
from typing import List
import pandas as pd
import mlflow

from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression


def fit_dict_vectorizer(
        df: pd.DataFrame, 
        categorical: List[str], 
        numerical: List[str]
    ) -> (DictVectorizer, pd.DataFrame):
    dicts = df[categorical + numerical].to_dict(orient='records')
    dv = DictVectorizer()
    X_train = dv.fit_transform(dicts)
    return (dv, X_train)

@task
def read_dataframe(filename: str, categorical: List[str], target: str) -> pd.DataFrame:
    df = pd.read_parquet(filename)

    # Q3
    print(f"Question 3. Let's read the March 2023 Yellow taxi trips data. How many records did we load?")
    print(f"Answer: {df.shape[0]} records.")

    df[target] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df.duration = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)]

    df[categorical] = df[categorical].astype(str)
    
    # Q4
    print(f"Question 4. What's the size of the result after data preparation?")
    print(f"Answer: {df.shape[0]} records.")
    return df

    
@task
def train_model(
    df_train: pd.DataFrame, 
    categorical: List[str], 
    numerical: List[str], 
    target : str
    ) -> (DictVectorizer, LinearRegression):
    
    dv, X_train = fit_dict_vectorizer(df_train, categorical, numerical)
    y_train = df_train[target].values

    with mlflow.start_run():
        # mlflow.sklearn.autolog()
        lr = LinearRegression()
        lr.fit(X_train, y_train)

        y_pred = lr.predict(X_train)
        intercept = lr.intercept_
        mlflow.log_metric("intercept", intercept)

        # Q5
        print(f"Question 5. What's the intercept of the model?")
        print(f"Answer: {intercept}")
        
        # Q6
        print(f"Question 6. What's the size of the model? (model_size_bytes field):")
        print(f"Answer: 4508 bytes.")

        mlflow.sklearn.log_model(
            sk_model=lr,
            artifact_path="artifacts",
            registered_model_name="linear-regression-model",
        )

    return dv, lr


@flow(name="03-orchestration")
def main() -> List[str]:
    mlflow.set_tracking_uri('sqlite:///mlflow.sqlite')
    mlflow.set_experiment("03-linear-regression")
    
    filename = './data/yellow_tripdata_2023-03.parquet'

    categorical = ['PULocationID', 'DOLocationID']
    numerical = ['trip_distance']
    target = 'duration'

    print('Question 1. What\'s the name of the orchestrator you chose?')
    print('Answer: Prefect.')

    print('Question 2. What\'s the version of the orchestrator?')
    print('Answer: 2.20.18')

    # Step 1. Q3 and Q4
    df = read_dataframe(filename, categorical, target)

    # Step 2. Q5 and Q6
    dv, lr = train_model(df, categorical, numerical, target)


if __name__ == "__main__":
    main()
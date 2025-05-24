import os
import pickle
import click
import mlflow

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error


def load_pickle(filename: str):
    with open(filename, "rb") as f_in:
        return pickle.load(f_in)


# def run_train(data_path: str):
#     mlflow.sklearn.autolog()

#     X_train, y_train = load_pickle(os.path.join(data_path, "train.pkl"))
#     X_val, y_val = load_pickle(os.path.join(data_path, "val.pkl"))

#     rf = RandomForestRegressor(max_depth=10, random_state=0)
#     rf.fit(X_train, y_train)
#     y_pred = rf.predict(X_val)

#     rmse = mean_squared_error(y_val, y_pred, squared=False)

def run_train(data_path: str):
    # mlflow.sklearn.autolog()
    
    X_train, y_train = load_pickle(os.path.join(data_path, "train.pkl"))
    X_val, y_val = load_pickle(os.path.join(data_path, "val.pkl"))
    
    with mlflow.start_run():
        rf = RandomForestRegressor(max_depth=10, random_state=0)
        rf.fit(X_train, y_train)
        y_pred = rf.predict(X_val)

        with open('models/rand_forest.bin', 'wb+') as f_out:
            pickle.dump(rf, f_out)

        rmse = mean_squared_error(y_val, y_pred, squared=False)
        mlflow.log_metric("rmse", rmse)
        
        mlflow.log_artifact(local_path="models/rand_forest.bin", artifact_path="models_pickle")


if __name__ == '__main__':
    run_train()



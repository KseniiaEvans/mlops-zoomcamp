from prefect import flow, task
import os
from typing import List

from preprocess_data import run_data_prep
from train import run_train
from hpo import run_optimization
from register_model import run_register_model
import mlflow

#######
# Steps
#######

# 1. Preprocess data -> preprocess.py
# 2. Train model -> train.py
# 3. Tune model hyperparameters -> hpo.py
# 4. Promote the best model to the model registry -> register_model.py


@task
def preprocess_data(raw_data_path: str, dest_path: str, dataset: str) -> List[str]:
    # Fetch customer IDs from a database or API
    run_data_prep(raw_data_path, dest_path, dataset)

    
@task
def train_model(data_path: str) -> str:
    # Train the model using the preprocessed data
    run_train(data_path)
    return "Model training completed"

@task
def tune_model(data_path: str, num_trials: int=15) -> str:
    run_optimization(data_path, num_trials)
    return "Model tuning completed"


@flow(name="03-orchestration")
def main() -> List[str]:
    mlflow.set_tracking_uri('sqlite:///mlflow.sqlite')
    mlflow.set_experiment("random-forest-hyperopt")

    preprocess_data(raw_data_path = 'data',
                    dest_path = 'output', 
                    dataset = "green")
    
    train_model('./output')

    tune_model('./output', num_trials=15)

    mlflow.set_experiment('random-forest-best-models')
    run_register_model('./output', top_n=5)


if __name__ == "__main__":
    main()
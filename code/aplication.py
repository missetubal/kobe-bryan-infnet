
import pandas as pd
import numpy as np
import pycaret.classification as pc

import matplotlib.pyplot as plt

import mlflow
from mlflow.models.signature import infer_signature
from mlflow.tracking import MlflowClient


# Para usar o sqlite como repositorio
mlflow.set_tracking_uri("sqlite:///mlruns.db")

experiment_name = 'Kobe Bryant Shot Selection'
experiment = mlflow.get_experiment_by_name(experiment_name)
if experiment is None:
    experiment_id = mlflow.create_experiment(experiment_name)
    experiment = mlflow.get_experiment(experiment_id)
experiment_id = experiment.experiment_id



columns=['lat', 'lon', 'minutes_remaining', 'period', 'playoffs', 'shot_distance', 'shot_made_flag']

with mlflow.start_run(experiment_id=experiment_id, run_name = 'PipelineAplicacao'):

    model_uri = f"models:/model_kobe_bryant@staging"
    loaded_model = mlflow.sklearn.load_model(model_uri)
    data_prod = pd.read_parquet('../data/raw/base_prod.parquet')

    Y = loaded_model.predict_proba(data_prod[columns])[:,1]
    data_prod['predict_score'] = Y

    data_prod.to_parquet('../data/processed/prediction_prod.parquet')
    mlflow.log_artifact('../data/processed/prediction_prod.parquet')
    
    print(data_prod)
    
  


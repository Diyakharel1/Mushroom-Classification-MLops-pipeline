import mlflow
import os
def init_mlflow():
    # Centralize experiment and tracking server settings
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow-server:5001")
    experiment_name = os.getenv("MLFLOW_EXPERIMENT_NAME", "mushroom_classification_pipeline")

    mlflow.set_tracking_uri(tracking_uri)
    try:
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment is None:
            mlflow.create_experiment(experiment_name)
        mlflow.set_experiment(experiment_name)
        experiment_info = {'experiment_name': experiment_name}
        if experiment:
            experiment_info['new_experiment'] = True
        else:
            experiment_info['new_experiment'] = False
        return experiment_info    
    
    except Exception as e:
        return {'experiment_name':"Failure", 'new_experiment': False}
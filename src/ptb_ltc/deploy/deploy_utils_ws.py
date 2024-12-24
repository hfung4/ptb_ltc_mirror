"""
deploy_utils_ws.py: this module is to be used in Workspace Model Registry
"""

import mlflow
from typing import Dict, List
from ptb_ltc.logging import logger

from mlflow.tracking import MlflowClient
from mlflow.entities import Run
from mlflow.entities.model_registry import ModelVersion
from mlflow.entities.model_registry.registered_model import RegisteredModel



def register_model(model_name: str, run_id) -> None:
    """Register model in the workspace model registry

    Args:
        model_name (str): two-level namespace model name
        run_id (str): MLflow run ID of the model version to register
    """
    mlflow.register_model(
          model_uri = f"runs:/{run_id}/model",
          name=model_name)
    logger.info(f"Model {model_name} added to the Workspace Model Registry")



def get_registered_model(registry_model_name: str)->RegisteredModel:
    """
    Fetch the registered model by name from the MLflow Model Registry (Workspace Model Registry only).

    Args:
        registry_model_name (str): Name of the MLflow Registry Model.

    Returns:
        registered_model (RegisteredModel): The registered model object.

    Raises:
        RuntimeError: If no registered model is found.
    """
    client = MlflowClient()
    filter_string = f'name="{registry_model_name}"'
    registered_models = client.search_registered_models(filter_string=filter_string)
    
    if not registered_models:
        raise RuntimeError(f"No registered model found with name '{registry_model_name}'")
    
    return registered_models[0]


def get_model_version_by_stage_and_alias(registered_model: RegisteredModel, stage: str = "None", alias: str = None)-> ModelVersion:
    """
    For a given registered model object, fetch the latest model version for the given stage and alias (Workspace Model Registry only).

    Args:
        registered_model (RegisteredModel): The registered model object.
        stage (str): Stage for this model. One of "None", "Staging", or "Production".
        alias (str): Optional alias for this model. One of "Challenger" or "Champion".

    Returns:
        model_version (ModelVersion): The latest model version matching the stage and alias.

    Raises:
         ValueError: If the provided stage is not one of 'None', 'Staging', or 'Production'.
                    If the alias is provided but not one of 'Challenger' or 'Champion'.
                    If no model version is found that matches the specified stage and alias.
         
    """
    # Data validation
    if stage not in ["None", "Staging", "Production"]:
        raise ValueError("stage must be one of 'None', 'Staging', 'Production'")
    
    if alias and alias not in ["Challenger", "Champion"]:
        raise ValueError("alias must be one of 'Challenger' or 'Champion'")

    for model_version in registered_model.latest_versions:
        if model_version.current_stage == stage:
            # Check alias if provided
            if alias is None or model_version.tags.get('alias') == alias:
                return model_version
    
    raise ValueError(f"No model version found in stage '{stage}' with alias '{alias}'")

def get_run_from_run_id(run_id: str) -> Run:
    """
    Fetch the MLflow run object using the run ID.

    Args:
        run_id (str): The run ID of the model version.

    Returns:
        mlflow.entities.run.Run: The MLflow run object.
    
    Raises:
        RuntimeError: If the run cannot be fetched.
    """
    try:
        run = mlflow.get_run(run_id)
        return run
    except Exception as e:
        raise RuntimeError(f"Failed to retrieve run with run ID '{run_id}': {str(e)}")

    
def transition_model(model_version:ModelVersion, stage:str):
    """
    Transition a model to a specified stage in Workspace Model Registry using the associated 
    mlflow.entities.model_registry.ModelVersion object.

    Args:
        model_version (mlflow.entities.model_registry.ModelVersion. ModelVersion) object to transition
        stage (str): New desired stage for this model version. One of "Staging", "Production", "Archived" or "None"
    Returns:
        A single mlflow.entities.model_registry.ModelVersion object
    """
    client = MlflowClient()
    
    model_version = client.transition_model_version_stage(
        name=model_version.name,
        version=model_version.version,
        stage=stage,
        archive_existing_versions=True
    )

    # Set alias to Champion
    client.set_model_version_tag(model_version.name, model_version.version, "alias", "Champion")

    return model_version



def get_latest_best_run(experiment_id:str, parent_run_filter_string:str = '')->Dict[str,object]:
    """Get the best cv run (by median_cv_log_loss) from the latest parent run
       in the target experiment
    
    Args:
        experiment_id (str): id of the target experiment
        parent_run_filter_string (str): filter string for the parent run in CV experiments, default to empty string (no filtering)
    Returns:
        Dict: a dictionary containing the best_run id and the mlrun object
    """
    # Get the latest parent run
    latest_parent_run_id = mlflow.search_runs(
        experiment_ids = experiment_id,
        filter_string = parent_run_filter_string,
        order_by=["start_time DESC"]
        ).loc[0,"tags.mlflow.parentRunId"]
  
    # Get the best child (CV) run by metric_name
    best_run_id = mlflow.search_runs(
        experiment_ids = experiment_id,
        filter_string =f"status='FINISHED' and tags.status= 'OK' and tags.mlflow.parentRunId= '{latest_parent_run_id}'",
        order_by=[f"metrics.median_cv_log_loss ASC"],
        max_results=1
        ).run_id[0]
 
    return {"best_run_id": best_run_id,
            "best_run":mlflow.get_run(best_run_id)}
  

def get_overall_best_run(experiment_id:str, overall_run_filter_string:str)->Dict[str,object]:
    """Get the best overall cv run (by median_cv_log_loss) across all runs
       in the target experiment
    
    Args:
        experiment_id (str): id of the target experiment
        parent_run_filter_string (str): filter string for the parent run in CV experiments
    Returns:
        Dict: a dictionary containing the best_run id and the mlrun object
    """
    # Get the best runs across all runs under the experiment with experiment_id
    # by median_cv_log_loss
    best_run_id = mlflow.search_runs(
        experiment_ids = experiment_id,
        filter_string =overall_run_filter_string,
        order_by=["metrics.median_cv_log_loss ASC"]).run_id[0]
    
    return {"best_run_id": best_run_id,
            "best_run":mlflow.get_run(best_run_id)}
    


def get_comparison_metrics(run: object, metric_names: List[str], larger_is_better: List[bool]) -> Dict[str, float]:
    """
    Get a set of metrics to compare models against each other
    (e.g., challenger vs champion). If the metric is NOT larger the better,
    then negate the metric value.

    Args:
        run (mlflow.entities.run.Run): mlflow run object.
        metric_names (List[str]): A list of metric names.
        larger_is_better (List[bool]): A list of boolean flags indicating whether the metric is larger the better.

    Returns:
        Dict[str, float]: A dictionary where the key is the metric name and the value is the adjusted metric.
    """
    # Get metrics for the run
    metrics = [run.data.metrics[metric_name] for metric_name in metric_names]
    
    # Modify the metrics based on the larger_is_better flag. If false, negate the metric
    adjusted_metrics = {metric_name: (x if flag else -x) for metric_name, x, flag in zip(metric_names, metrics, larger_is_better)}
    
    return adjusted_metrics


def compare_metrics(challenger_metrics: Dict, champion_metrics: Dict) -> Dict:
    """
    Compare metrics from the challenger and champion models.

    Args:
        challenger_metrics (dict): Dictionary containing the challenger model's metrics.
        champion_metrics (dict): Dictionary containing the champion model's metrics.

    Returns:
        dict: A dictionary where the key is the metric name and the value is True if the
              challenger metric is larger than the champion metric, and False otherwise.
    """
    return {metric: challenger_metrics[metric] > champion_metrics[metric] for metric in challenger_metrics}

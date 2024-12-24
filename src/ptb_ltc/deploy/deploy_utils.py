"""
deploy_utils.py: this module is to be used in Unity Catalog enabled worksapce
"""

import mlflow
from typing import Dict, List
from ptb_ltc.logging import logger
from ptb_ltc.config.core import config

from mlflow.exceptions import RestException
from mlflow.tracking import MlflowClient
from mlflow.entities import Run
from mlflow.entities.model_registry import ModelVersion
from mlflow.entities.model_registry.registered_model import RegisteredModel



def register_model_version(run_id: str, model_uri: str, model_name: str) -> int:
    """Register model in Unity Catalog model registry (UC only)

    Args:
        run_id (str): model run ID
        model_uri (str): model URI
        model_name (str): 3-level model name

    Returns:
        int: model version
    """
    client = MlflowClient()
    # if model doesn't exist, then register
    try:
        client.get_registered_model(name=model_name)
    except RestException:
        client.create_registered_model(name=model_name)
        logger.info(f"Model {model_name} added to Unity Catalog")

    model_version = client.create_model_version(
        run_id=run_id,
        source=model_uri,
        name=model_name,
    )

    return model_version.version



def get_model_uri(model_name: str, alias: str) -> str:
    """Get model URI by model name and alias (UC only)

    Args:
        model_name (str): three level namespace model name
        alias (str): model alias

    Returns:
        str: model URI
    """
    client = MlflowClient()
    mv = client.get_model_version_by_alias(model_name, alias)
    return f"runs:/{mv.run_id}/model"


def get_latest_model_version(model_name:str)->object:
    """Get the latest version of the model from the model registry (UC only)
    Args:
        model_name (str): three-level namespace of the model
    Returns:
        latest_version: mlflow model version object
    """
    latest_version = 1
    client = MlflowClient()
    for mv in client.search_model_versions(f"name='{model_name}'"):
        version_int = int(mv.version)
        if version_int > latest_version:
            latest_version = version_int
    return latest_version


def get_run_by_name_and_alias(model_name: str, alias: str) -> Run:
    """Get run information (in mlflow format) based on model name and alias

    Args:
        model_name (str): 3-level model name
        alias (str): current alias

    Returns:
        Run: mlflow Run object
    """
    client = MlflowClient( )
    try:
        model_version = client.get_model_version_by_alias(model_name, alias).version
    except RestException:
        logger.warning("Model with given name and alias not found in registry")
        return None

    run_id = client.get_model_version(model_name, model_version).run_id
    run = client.get_run(run_id=run_id)

    return run



def transition_model(model_name: str, model_version: int, alias: str) -> None:
    """Transition given version of model to new alias (UC only)

    Args:
        model_name (str): three-level namespace model name
        model_version (int): version to transition
        alias (str): alias to assign to model
    """
    client = MlflowClient()

    client.set_registered_model_alias(
        name=model_name,
        version=model_version,
        alias=alias,
    )
    logger.info(f"Version {model_version} of {model_name} transitioned to {alias}")


def get_run_by_run_id(run_id: str) -> Run:
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
  

def get_overall_best_run(experiment_id:str, overall_run_filter_string:str = '')->Dict[str,object]:
    """Get the best overall cv run (by median_cv_log_loss) across all runs
       in the target experiment
    
    Args:
        experiment_id (str): id of the target experiment
        overall_run_filter_string (str): filter string for the overall runs in CV experiment, default to empty string (no filter)
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

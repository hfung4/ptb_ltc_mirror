from collections import ChainMap
from typing import Dict
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE, BorderlineSMOTE, ADASYN
import mlflow
from hyperopt import STATUS_OK, STATUS_FAIL
import signal

"""metrics"""
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
    average_precision_score,
)

"""models"""
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from ptb_ltc.config.core import config


# Define a custom exception for timeout
class TimeoutException(Exception):
    pass

def timeout_handler(signum, frame):
    raise TimeoutException("Trial timed out!")


def get_models(type: str) -> object:
    """Get sklearn model estimator and initailize with RANDOM_STATE

    Args:
        type (str): name of the model, corresponding to the key 'type' in the model_param_space dictionary

    Returns:
        object: sklearn estimator object
    """
    match type:
        case "KNeighborsClassifier":
            return KNeighborsClassifier()
        case "GaussianNB":
            return GaussianNB()
        case "DecisionTreeClassifier":
            return DecisionTreeClassifier(random_state=config.general.RANDOM_STATE)
        case "LogisticRegression":
            return LogisticRegression(random_state=config.general.RANDOM_STATE)
        case "RandomForestClassifier":
            return RandomForestClassifier(random_state=config.general.RANDOM_STATE)
        case "XGBClassifier":
            return XGBClassifier(verbosity=0, seed=config.general.RANDOM_STATE)
        case "LGBMClassifier":
            return LGBMClassifier(
                verbosity=-1, random_state=config.general.RANDOM_STATE
            )
        case "MLPClassifier":
            return MLPClassifier(
                verbose=False, random_state=config.general.RANDOM_STATE
            )


def get_samplers(type: str) -> object:
    """Get imbalanced-learn sampler and initialize with RANDOM_STATE

    Args:
        type (str): name of the sampler, corresponding to the key 'type' in the sampler_param_space dictionary

    Returns:
        object: imbalanced-learn sampler object
    """
    match type:
        case "SMOTE":
            return SMOTE(sampling_strategy=config.processing.SMOTE_OVER_SAMPLING_STRATEGY, random_state=config.general.RANDOM_STATE)
        case "BorderlineSMOTE":
            return BorderlineSMOTE(sampling_strategy=config.processing.SMOTE_OVER_SAMPLING_STRATEGY, random_state=config.general.RANDOM_STATE)
        case "ADASYN":
            return ADASYN(sampling_strategy=config.processing.SMOTE_OVER_SAMPLING_STRATEGY, random_state=config.general.RANDOM_STATE)


def optimize(
    args: Dict,
    X: pd.DataFrame,
    y: pd.Series,
    transform_pipeline: object,
    cv_experiment_name: str,
) -> Dict:
    """Defines the loss function to be optimized by hyperopt

    Args:
        args (Dict): an instance of the parameter space
        X (pd.DataFrame): train features
        y (pd.Series): train outcome variable
        transform_pipeline (object): processing pipeline
        cv_experiment_name (str): name of the cross-validation experiment

    Returns:
        Dict: dictionary containing the loss and status
    """

    # Timeout logic using the customized timeout handler
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(config.model.TIMEOUT_SECONDS)

    # Set experiment (need to set it in the optimize function again due to distributed nature of hyperopt SparkTrials)
    mlflow.set_experiment(
        f"{config.general.DATABRICKS_WORKSPACE_URL}{cv_experiment_name}"
    )

    mlflow.start_run(nested=True)

    try:
        # Initialize data pipeline by appending model to the transformation pipeline
        data_pipeline = Pipeline(
            [
                ("transformation", transform_pipeline),
                ("model", get_models(args["classifier"]["type"])),
            ]
        )

        # Set the hyperparameters of the pipeline
        # Model parameters will be set here
        data_pipeline.set_params(**args["classifier"]["params"])

        # Init a list of fold scores
        cv_scores = []
        fold_accuracies = []
        fold_precisions = []
        fold_recalls = []
        fold_f1s = []
        fold_roc_aucs = []
        fold_avg_precision = []
        fold_response_rate = []
        fold_avg_precision_lift = []

        # Init the cv folds
        skf = StratifiedKFold(
            n_splits=config.model.N_FOLDS,
            shuffle=True,
            random_state=config.general.RANDOM_STATE,
        )

        # Create a CV loop with an iterator instead of using cross_val_score() since only .fit() would trigger mlflow autologging
        for idx in skf.split(X=X, y=y):
            tr_idx, val_idx = idx[0], idx[1]
            X_tr = X.iloc[tr_idx]
            y_tr = y.iloc[tr_idx]
            X_val = X.iloc[val_idx]
            y_val = y.iloc[val_idx]

            # Fit pipeline on training data of each fold
            data_pipeline.fit(X_tr, y_tr)

            # Get predictions
            preds = data_pipeline.predict(X_val)

            # Get predictions probabilities
            pred_probas = data_pipeline.predict_proba(X_val)

            # Calculate cv log loss and other cv performance metrics
            cv_scores.append(log_loss(y_val, preds))
            fold_accuracies.append(accuracy_score(y_val, preds))
            fold_precisions.append(precision_score(y_val, preds, zero_division=0.0))
            fold_recalls.append(recall_score(y_val, preds, zero_division=0.0))
            fold_f1s.append(f1_score(y_val, preds, zero_division=0.0))
            fold_roc_aucs.append(roc_auc_score(y_val, pred_probas[:, 1]))
            fold_avg_precision.append(average_precision_score(y_val, pred_probas[:, 1]))
            fold_response_rate.append(y_val.mean())
            fold_avg_precision_lift.append(average_precision_score(y_val, pred_probas[:, 1])/y_val.mean() if y_val.mean() > 0 else 0)

        # Logging

        # Set tags
        mlflow.set_tags({
                "classifier_type": args["classifier"]["type"],
                "use_resampling_in_train_pipeline": config.model.USE_RESAMPLING_IN_TRAIN_PIPELINE,
                "random_state": config.general.RANDOM_STATE,
                "status":"OK"
                })

        # Hyperparameters
        mlflow.log_params(args)
        # Performance metrics
        loss = np.median(cv_scores)
        mlflow.log_metric("median_cv_log_loss", np.median(cv_scores))
        mlflow.log_metric("median_cv_accuracy", np.median(fold_accuracies))
        mlflow.log_metric("median_cv_precision", np.median(fold_precisions))
        mlflow.log_metric("median_cv_recall", np.median(fold_recalls))
        mlflow.log_metric("median_cv_f1", np.median(fold_f1s))
        mlflow.log_metric("median_cv_roc_auc", np.median(fold_roc_aucs))
        mlflow.log_metric("median_cv_avg_precision", np.median(fold_avg_precision))
        mlflow.log_metric("median_cv_response_rate", np.median(fold_response_rate))
        mlflow.log_metric("median_cv_avg_precision_lift", np.median(fold_avg_precision_lift))
        
        # Set to STATUS_OK if no timeout exception
        status = STATUS_OK

    except TimeoutException:
        # Assign a high loss for timeout failure
        loss = float("inf")
        status = STATUS_FAIL
        mlflow.set_tag("status", "FAIL")
        mlflow.log_metric("loss", loss)  
    except Exception as e:
        # Assign a high loss for general failure
        loss = float("inf")  
        # Mark as fail for general exceptions
        status = STATUS_FAIL
        mlflow.set_tag("status", "FAIL")  
        mlflow.log_metric("loss", loss)
        # Re-raise the exception to see the traceback
        raise  
    finally:
        # disable alarm after functions successfully runs or timedout
        signal.alarm(0)
        mlflow.end_run()
        

    return {"loss": loss, "status": status}


def optimize_with_resampling(
    args: Dict,
    X: pd.DataFrame,
    y: pd.Series,
    transform_pipeline: object,
    undersample_transformer: object,
    cv_experiment_name: str,
) -> Dict:
    """Defines the loss function with resampling steps to be optimized by hyperopt

    Args:
        args (Dict): an instance of the parameter space
        X (pd.DataFrame): train features
        y (pd.Series): train outcome variable
        transform_pipeline (object): processing pipeline
        undersample_transformer (object): undersample transformer
        cv_experiment_name (str): name of the cross-validation experiment

    Returns:
        Dict: dictionary containing the loss and status
    """

    # Timeout logic using the customized timeout handler
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(config.model.TIMEOUT_SECONDS)


    # Set experiment (need to set it in the optimize function again due to distributed nature of hyperopt SparkTrials)
    mlflow.set_experiment(
        f"{config.general.DATABRICKS_WORKSPACE_URL}{cv_experiment_name}"
    )

    mlflow.start_run(nested=True)

    try:
        # Initialize data pipeline by appending model to the transformation pipeline
        data_pipeline = ImbPipeline(
            [
                ("transformation", transform_pipeline),
                ("oversample", get_samplers(args["sampler"]["type"])),
                ("undersample", undersample_transformer),
                ("model", get_models(args["classifier"]["type"])),
            ]
        )

        # Set the hyperparameters of the pipeline
        # Model + processing parameters will be set here
        # Use ChainMap() to combine two sets of **kwargs (one for model hyperparameters, the other for transformation pipeline params)
        data_pipeline.set_params(
            **ChainMap(
                args["classifier"]["params"],
                args["sampler"]["params"],
            )
        )

        # Init a list of fold scores
        cv_scores = []
        fold_accuracies = []
        fold_precisions = []
        fold_recalls = []
        fold_f1s = []
        fold_roc_aucs = []
        fold_avg_precision = []
        fold_response_rate = []
        fold_avg_precision_lift = []

        # Init the cv folds
        skf = StratifiedKFold(
            n_splits=config.model.N_FOLDS,
            shuffle=True,
            random_state=config.general.RANDOM_STATE,
        )

        # Create a CV loop with an iterator instead of using cross_val_score() since only .fit() would trigger mlflow autologging
        for idx in skf.split(X=X, y=y):
            tr_idx, val_idx = idx[0], idx[1]
            X_tr = X.iloc[tr_idx]
            y_tr = y.iloc[tr_idx]
            X_val = X.iloc[val_idx]
            y_val = y.iloc[val_idx]

            # Fit pipeline on training data of each fold
            data_pipeline.fit(X_tr, y_tr)

            # Get predictions
            preds = data_pipeline.predict(X_val)

            # Get predictions probabilities
            pred_probas = data_pipeline.predict_proba(X_val)

            # Calculate cv log loss and other cv performance metrics
            cv_scores.append(log_loss(y_val, preds))
            fold_accuracies.append(accuracy_score(y_val, preds))
            fold_precisions.append(precision_score(y_val, preds, zero_division=0.0))
            fold_recalls.append(recall_score(y_val, preds, zero_division=0.0))
            fold_f1s.append(f1_score(y_val, preds, zero_division=0.0))
            fold_roc_aucs.append(roc_auc_score(y_val, pred_probas[:, 1]))
            fold_avg_precision.append(average_precision_score(y_val, pred_probas[:, 1]))
            fold_response_rate.append(y_val.mean())
            fold_avg_precision_lift.append(
                average_precision_score(y_val, pred_probas[:, 1]) / y_val.mean()
                if y_val.mean() > 0
                else 0
            )


        # Logging

        # Set tags
        mlflow.set_tags({
                "classifier_type": args["classifier"]["type"],
                "use_resampling_in_train_pipeline": config.model.USE_RESAMPLING_IN_TRAIN_PIPELINE,
                "random_state": config.general.RANDOM_STATE,
                "status": "OK"})

        # Hyperparameters
        mlflow.log_params(args)
        # Performance metrics
        loss = np.median(cv_scores)
        mlflow.log_metric("median_cv_log_loss", np.median(cv_scores))
        mlflow.log_metric("median_cv_accuracy", np.median(fold_accuracies))
        mlflow.log_metric("median_cv_precision", np.median(fold_precisions))
        mlflow.log_metric("median_cv_recall", np.median(fold_recalls))
        mlflow.log_metric("median_cv_f1", np.median(fold_f1s))
        mlflow.log_metric("median_cv_roc_auc", np.median(fold_roc_aucs))
        mlflow.log_metric("median_cv_avg_precision", np.median(fold_avg_precision))
        mlflow.log_metric("median_cv_response_rate", np.median(fold_response_rate))
        mlflow.log_metric(
            "median_cv_avg_precision_lift", np.median(fold_avg_precision_lift)
        )
        status = STATUS_OK

    except TimeoutException:
        # Assign a high loss for timeout failure
        loss = float("inf")
        # Mark as fail for timeout exceptions
        status = STATUS_FAIL
        mlflow.set_tag("status", "FAIL")
        mlflow.log_metric("loss", loss)  
    except Exception as e:
        # Assign a high loss for general failure
        loss = float("inf")  
        # Mark as fail for general exceptions
        status = STATUS_FAIL
        mlflow.set_tag("status", "FAIL")
        mlflow.log_metric("loss", loss)  
        # Re-raise the exception to see the traceback
        raise  
    finally:
        signal.alarm(0)
        mlflow.end_run()
 
    return {"loss": loss, "status": status}

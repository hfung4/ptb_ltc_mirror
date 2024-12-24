"""
Train pipeline
- Identify the best cv run (with the best median cv log loss), fit the associated 
ML pipeline with the full train set, register the model in the workspace (later on UC) model registry. 
- Assign model alias as "challenger"
"""

from ptb_ltc.config.core import config

if config.model.USE_RESAMPLING_IN_TRAIN_PIPELINE:
    # Need to install treadpoolctl==3.1.0 to avoid error when using calibrated classifier and imblearn
    import subprocess
    import sys

    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "threadpoolctl==3.1.0"]
    )

    # Import modules specific to using resampling in the train pipeline
    from collections import ChainMap
    from imblearn.pipeline import Pipeline as ImbPipeline
    from ptb_ltc.optimize.loss_function import get_samplers
    from ptb_ltc.pipeline import get_undersample_transformer

import ast
from argparse import ArgumentParser
import pandas as pd
import sklearn
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import make_scorer, roc_auc_score
import mlflow
from mlflow.tracking import MlflowClient
from ptb_ltc.evaluate.evaluation_metrics import generate_test_metrics
from ptb_ltc.evaluate.visualization import (
    generate_calibration_curve,
    generate_confusion_matrix_plot,
    generate_learning_curves,
    generate_prc_plot,
    generate_roc_plot,
    generate_shapley_values_plot,
)
from ptb_ltc.optimize.loss_function import get_models
from ptb_ltc.pipeline import get_transform_pipeline
import ptb_ltc.deploy.deploy_utils_ws as du
import ptb_ltc.data.data_manager as dm

# Set up scikit-learn to output pandas DataFrames
from sklearn import set_config

set_config(transform_output="pandas")

# Get user inputs-- environment (env), experiment name containing runs
# for refitting the optimized model, and git source info
parser = ArgumentParser()
for arg in ["--env", "--experiment_name", "--git_source_info"]:
    parser.add_argument(arg, type=str, required=True)
args = parser.parse_args()

# environment (dev, staging, or prod)
env = args.env

# Get schema and table names
schema_table_name_dict = dm.get_schema_table_names(env=env)


# Import data
selected_columns = (
    config.model.SELECTED_FEATURES + [config.model.OUTCOME_VARIABLE] + ["set_type"]
)
train_test_df = (
    spark.table(
        f"{schema_table_name_dict['gold_schema']}.{schema_table_name_dict['gold_table']}"
    )
    .select(*selected_columns)
    # remove rows with missing outcome variable
    .dropna(subset=[config.model.OUTCOME_VARIABLE])
    .toPandas()
)

train_test_features_df = train_test_df.drop(
    columns=[config.model.OUTCOME_VARIABLE, "set_type"]
)

# Get numeric and categorical features
numeric_features = train_test_features_df.select_dtypes("number").columns.tolist()
categorical_features = train_test_features_df.select_dtypes("object").columns.tolist()

# Train/test split
train_df = train_test_df.loc[train_test_df.set_type == "train", :].drop(
    columns=["set_type"]
)
test_df = train_test_df.loc[train_test_df.set_type == "test", :].drop(
    columns=["set_type"]
)

X_train = train_df.drop(columns=config.model.OUTCOME_VARIABLE)
y_train = train_df.loc[:, config.model.OUTCOME_VARIABLE]

X_test = test_df.drop(columns=config.model.OUTCOME_VARIABLE)
y_test = test_df.loc[:, config.model.OUTCOME_VARIABLE]


# Optional calibration set
if config.model.PERFORM_CALIBRATION:
    X_test, X_calib, y_test, y_calib = train_test_split(
        X_test,
        y_test,
        test_size=config.model.CALIBRATION_SIZE,
        random_state=config.general.RANDOM_STATE,
        stratify=y_test,
    )

# MLFlow setup
mlflow.login()
mlflow.set_tracking_uri("databricks")
mlflow.set_registry_uri("databricks")
mlflow.set_experiment(args.experiment_name)
client = MlflowClient()

# Get CV best run info
cv_experiment_name = f"ds_cv_{config.model.MODEL_NAME}"
experiment_id = mlflow.search_experiments(
    filter_string=f"name LIKE '{config.general.DATABRICKS_WORKSPACE_URL}{cv_experiment_name}'",
    order_by=["last_update_time DESC"],
)[0].experiment_id

# filter string for parent run: get runs that are completed and have specific status (same as config variable) on
# the "use_resampling_in_train_pipeline" tag
parent_run_filter_string = f"status = 'FINISHED' and tags.use_resampling_in_train_pipeline = '{config.model.USE_RESAMPLING_IN_TRAIN_PIPELINE}'"
overall_run_filter_string = f"status = 'FINISHED' and tags.status= 'OK' and tags.use_resampling_in_train_pipeline = '{config.model.USE_RESAMPLING_IN_TRAIN_PIPELINE}'"

best_run_info = (
    du.get_latest_best_run(experiment_id, parent_run_filter_string)
    if config.model.SELECT_FROM_LATEST_CV_RUNS
    else du.get_overall_best_run(experiment_id, overall_run_filter_string)
)
best_run = best_run_info["best_run"]

# Extract best model hyperparameters
best_model_params = ast.literal_eval(best_run.data.params["classifier"])["params"]
best_model_type = ast.literal_eval(best_run.data.params["classifier"])["type"]

# Initiate the transform pipeline
transform_pipeline = get_transform_pipeline(numeric_features, categorical_features)

# We are tuning the SMOTE variants to use, so we need to extract the associated hyperparameters
# and the undersample _transformer
if config.model.USE_RESAMPLING_IN_TRAIN_PIPELINE:
    best_sampler_params = ast.literal_eval(best_run.data.params["sampler"])["params"]
    best_sampler_type = ast.literal_eval(best_run.data.params["sampler"])["type"]
    undersample_transformer = get_undersample_transformer()

# Model signature
signature = mlflow.models.infer_signature(X_train, y_train)


# Initialize and set up best pipeline
if config.model.USE_RESAMPLING_IN_TRAIN_PIPELINE:
    # Pipeline with resampling steps
    best_pipeline = ImbPipeline(
        [
            ("transformation", transform_pipeline),
            ("oversample", get_samplers(best_sampler_type)),
            ("undersample", undersample_transformer),
            ("model", get_models(best_model_type)),
        ]
    )
    best_pipeline.set_params(**ChainMap(best_model_params, best_sampler_params))
else:
    # Pipeline with no resampling steps
    best_pipeline = Pipeline(
        [("transformation", transform_pipeline), ("model", get_models(best_model_type))]
    )
    best_pipeline.set_params(**best_model_params)


mlflow.end_run()  # Allow rerun
mlflow.autolog(disable=True)

# Fit and evaluate models
mlflow.start_run()
if not config.model.PERFORM_CALIBRATION:
    mlflow.autolog(log_input_examples=True, log_models=False, silent=True)

best_pipeline.fit(X_train, y_train)
y_pred, y_pred_proba = best_pipeline.predict(X_test), best_pipeline.predict_proba(
    X_test
)

if config.model.PERFORM_CALIBRATION:

    if config.model.USE_RESAMPLING_IN_TRAIN_PIPELINE:
        # Pipeline with resampling steps and  calibration
        best_pipeline_calib = ImbPipeline(
            [
                ("transformation", transform_pipeline),
                ("oversample", get_samplers(best_sampler_type)),
                ("undersample", undersample_transformer),
                (
                    "model",
                    CalibratedClassifierCV(
                        best_pipeline[-1],
                        method=config.model.CALIBRATION_METHOD,
                        cv="prefit",
                    ),
                ),
            ]
        )
    else:
        # Pipeline with no resampling steps
        best_pipeline_calib = Pipeline(
            [
                ("transformation", best_pipeline[:-1]),
                (
                    "model",
                    CalibratedClassifierCV(
                        best_pipeline[-1],
                        method=config.model.CALIBRATION_METHOD,
                        cv="prefit",
                    ),
                ),
            ]
        )

    mlflow.autolog(log_input_examples=True, log_models=False, silent=True)
    best_pipeline_calib.fit(X_calib, y_calib)
    y_pred, y_pred_proba = best_pipeline_calib.predict(
        X_test
    ), best_pipeline_calib.predict_proba(X_test)

# Log models and metrics

# Get the processed or resampled X_train data
if config.model.USE_RESAMPLING_IN_TRAIN_PIPELINE:
    mlflow.autolog(disable=True)
    resampled_data = best_pipeline[:-1].fit_resample(X_train, y_train)
    y_train_processed, X_train_processed = resampled_data[1], resampled_data[0]
else:
    X_train_processed = best_pipeline[:-1].transform(X_train)


# Log models
full_model_name = f"{env}_{config.model.MODEL_NAME}"

mlflow.sklearn.log_model(
    sk_model=best_pipeline_calib if config.model.PERFORM_CALIBRATION else best_pipeline,
    pyfunc_predict_fn="predict_proba",
    artifact_path="model",
    signature=signature,
    registered_model_name=full_model_name,
)

# Set model tags and alias
latest_version_info = client.get_latest_versions(full_model_name, stages=["None"])
latest_version = latest_version_info[0].version
client.set_model_version_tag(full_model_name, latest_version, "alias", "Challenger")

# Log characteristics of the processed and resampled data
mlflow.log_param("number_of_samples_X_train_processed", X_train_processed.shape[0])
mlflow.log_param("number_of_samples_X_train", X_train.shape[0])

# Parameters specific to the pipeline with resampling steps
if config.model.USE_RESAMPLING_IN_TRAIN_PIPELINE:
    mlflow.log_param(
        "sample_difference_in_X_train_processed",
        X_train_processed.shape[0] - X_train.shape[0],
    )
    mlflow.log_param(
        "resampled_negative_class_proportion",
        y_train_processed.value_counts(normalize=True).iloc[0],
    )
    mlflow.log_param(
        "resampled_positive_class_proportion",
        y_train_processed.value_counts(normalize=True).iloc[1],
    )

# Log test metrics and visualizations
test_metrics = generate_test_metrics(y_test, y_pred, y_pred_proba)
mlflow.log_metrics(test_metrics)

mlflow.log_figure(
    generate_roc_plot(y_test=y_test, y_scores=y_pred_proba), "test_roc.png"
)
mlflow.log_figure(
    generate_prc_plot(y_test=y_test, y_scores=y_pred_proba), "test_prc.png"
)
mlflow.log_figure(
    generate_confusion_matrix_plot(
        y_test=y_test, y_pred=y_pred, classes=["negative", "positive"], normalize=False
    ),
    "test_confusion_matrix.png",
)
mlflow.log_figure(
    generate_calibration_curve(y_test, y_pred_proba), "calibration_curves.png"
)

if config.model.PLOT_SHAP_VALUES and best_model_type not in [
    "MLPClassifier",
    "KNeighborsClassifier",
]:
    mlflow.log_figure(
        generate_shapley_values_plot(
            model=best_pipeline[-1], X_processed=X_train_processed
        ),
        "test_shapley_values.png",
    )

if config.model.PLOT_LEARNING_CURVES:
    roc_auc_scorer = make_scorer(
        roc_auc_score,
        greater_is_better=True,
        needs_proba=True if sklearn.__version__ == "1.3.0" else "predict_proba",
    )
    mlflow.log_figure(
        generate_learning_curves(
            best_pipeline, roc_auc_scorer, "roc_auc", X_train, y_train
        ),
        "learning_curves.png",
    )

mlflow.set_tags(
    {
        "outcome_variable": config.model.OUTCOME_VARIABLE,
        "model_type": best_model_type,
        "performed_calibration": config.model.PERFORM_CALIBRATION,
        "use_resampling_in_train_pipeline": config.model.USE_RESAMPLING_IN_TRAIN_PIPELINE,
    }
)

# Additional logging
mlflow.log_param("train_outcome_prevalence", y_train.mean())
mlflow.log_param("test_outcome_prevalence", y_test.mean())
mlflow.log_param("number_of_features", X_train.shape[1])
mlflow.log_param("selected_features", X_train.columns.tolist())
mlflow.log_param("performed_calibration", config.model.PERFORM_CALIBRATION)
mlflow.log_param("deployment_target", env)
mlflow.log_param("git_source_info", args.git_source_info)
mlflow.end_run()

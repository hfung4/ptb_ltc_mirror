# Databricks notebook source
# MAGIC %md ## Model tuning and hyperparameters tuning
# MAGIC - Output: An experiments with runs using different models and sets of processing/model hyperparameters
# MAGIC - This notebook should not be run in production-- once tuning is completed, I will simply get the optimized pipeline from the MLFlow experiment, and refit with the entire train set 

# COMMAND ----------

from datetime import datetime
import numpy as np
import pandas as pd

"""processing"""
from functools import partial

from sklearn.model_selection import train_test_split

"""hyperparameter tuning"""
from hyperopt import SparkTrials, Trials, fmin, space_eval, tpe

"""mlflow"""
import mlflow

# Ensure all transformers output pd.DataFrame
from sklearn import set_config
set_config(transform_output="pandas") 

from ptb_ltc.config.core import config
from ptb_ltc.optimize.param_space import param_space
from ptb_ltc.pipeline import get_transform_pipeline
import ptb_ltc.data.data_manager as dm
#from databricks.connect import DatabricksSession

# COMMAND ----------

# Ensure all transformers output pd.DataFrame
from sklearn import set_config

set_config(transform_output="pandas")

# COMMAND ----------

# MAGIC %md ### Import data

# COMMAND ----------

# Get schema and table names (hp_tuning.py will always be run in dev)
schema_table_name_dict = dm.get_schema_table_names(env="dev")

# COMMAND ----------

train_test_df = (spark.table(f"{schema_table_name_dict['gold_schema']}.{schema_table_name_dict['gold_table']}")
                 .toPandas()
                 .loc[:,list(config.model.SELECTED_FEATURES) + [config.model.OUTCOME_VARIABLE] + ["set_type"]]
                 # remove rows with missing outcome variable
                 .dropna(subset=[config.model.OUTCOME_VARIABLE]).reset_index(drop=True))

# COMMAND ----------

train_test_df.head()

# COMMAND ----------

### Get numeric and categorical features
train_test_features_df = train_test_df.drop(columns=[config.model.OUTCOME_VARIABLE, 'set_type'])
numeric_features = train_test_features_df.select_dtypes("number").columns.tolist()
categorical_features = train_test_features_df.select_dtypes("object").columns.tolist()

# COMMAND ----------

# MAGIC %md ### Get train data
# MAGIC - Filter by set_type column

# COMMAND ----------

train_df = train_test_df.query("set_type == 'train'").drop(columns=["set_type"])
test_df = train_test_df.query("set_type == 'test'").drop(columns=["set_type"])

# COMMAND ----------

X_train = train_df.drop(columns=config.model.OUTCOME_VARIABLE)
y_train = train_df.loc[:,config.model.OUTCOME_VARIABLE]

# COMMAND ----------

# Get transform pipeline
transform_pipeline = get_transform_pipeline(numeric_features, categorical_features)

# If we use resampling as part of train pipeline, get the undersample_transformer also
if config.model.USE_RESAMPLING_IN_TRAIN_PIPELINE:
  from ptb_ltc.pipeline import get_undersample_transformer
  undersample_transformer = get_undersample_transformer()

# COMMAND ----------

# MAGIC %md ### MLflow setup

# COMMAND ----------

mlflow.login()

# Requires users to configure that Databricks CLI. Three pieces of information needed:
# 1) Databricks Host (url of workspace), 2) user name (surgo email), 3) token (Databricks personal access token,
# generated in settings)
mlflow.set_tracking_uri("databricks")

# Create experiment for CV
cv_experiment_name = f"ds_cv_{config.model.MODEL_NAME}"
xp_path = f"{config.general.DATABRICKS_WORKSPACE_URL}{cv_experiment_name}"
mlflow.set_experiment(xp_path)

# COMMAND ----------

# If we use resampling in the train pipeline, we use the `optimize_with_resampling` loss function;
# otherwise, use `optimize`

if config.model.USE_RESAMPLING_IN_TRAIN_PIPELINE:
    from ptb_ltc.optimize.loss_function import optimize_with_resampling
    # Create a partial function with resampling steps, we preset X and y with the train data
    optimization_function = partial(
        optimize_with_resampling,
        X=X_train,
        y=y_train,
        transform_pipeline=transform_pipeline,
        undersample_transformer=undersample_transformer,
        cv_experiment_name=cv_experiment_name,
    )
else:
    from ptb_ltc.optimize.loss_function import optimize
    # Create a partial function, we preset X and y with the train data
    optimization_function = partial(
        optimize,
        X=X_train,
        y=y_train,
        transform_pipeline=transform_pipeline,
        cv_experiment_name=cv_experiment_name,
)

# COMMAND ----------

# MAGIC %md ### Model training: call the fmin() function from hyperopt to optimize the hyperparameters 

# COMMAND ----------

# fmin() requires the objective function, param_space, and other settings.
# All values of the hyperparameters and scores for each iteration (trials) are stored in the Trials object

# Use SparkTrials if uploaded and running on Databricks, otherwise use Trials
if config.general.RUN_ON_DATABRICKS_WS:
    trials = SparkTrials(parallelism=config.model.PARALLELISM)
else:
    trials = Trials()

run_name = f'trial_{datetime.now().strftime("%m-%d-%Y_%H:%M:%S")}'

# Start the mlflow run (training and hyperparameters tuning)
with mlflow.start_run(run_name = run_name):
    result = fmin(
        fn=optimization_function,
        space=param_space,
        algo=tpe.suggest,
        max_evals=config.model.N_ITER,
        trials=trials,
        rstate=np.random.default_rng(seed=config.general.RANDOM_STATE),
    )

    # Set tags
    mlflow.set_tags({
            "use_resampling_in_train_pipeline": config.model.USE_RESAMPLING_IN_TRAIN_PIPELINE,
            "random_state": config.general.RANDOM_STATE})

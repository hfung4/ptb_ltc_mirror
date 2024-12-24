"""
Batch Deployment
- Get the current champion model (in Staging) from the Workspace Model Registry (future: UC Model Registry)
- Create spark udf from the model
- Read serving data from hive metastore (future: UC)
- Perform inference on the serving data and generate spark inference dataframe
- Materalize inference dataframe to a delta table 
"""

from ptb_ltc.config.core import config
import ptb_ltc.deploy.deploy_utils_ws as du
import ptb_ltc.data.data_manager as dm
from argparse import ArgumentParser
import mlflow
from pyspark.sql.types import DoubleType
import pandas as pd
from pathlib import Path
from pyspark.sql.functions import pandas_udf, struct

parser = ArgumentParser()

# Get user inputs for target deployment environment
parser.add_argument(
    "--env",
    type=str,
    required=True,
    choices=["dev", "test", "staging", "prod"],
    help="The deploy environment name (e.g., 'dev', 'test', 'staging', 'prod').",
)
args = parser.parse_args()

# Get schema and table names
schema_table_name_dict = dm.get_schema_table_names(args.env)


# Create pandas_udf for performing inference (predicted probabilities)
# REF: https://medium.com/expedia-group-tech/lightning-fast-ml-predictions-with-pyspark-354c8c5abe83
@pandas_udf(returnType=DoubleType())
def predict_pandas_udf(*feature_names):
    """Executes the prediction using numpy arrays
     Args:
      *features: Model features names
    Returns:
      pd.Series of predicted probabilities
    """
    # need a multi-dimensional numpy array for sklearn models
    X = pd.concat(feature_names, axis=1)
    y = model.predict_proba(X)[:, 1]  # vectorized
    return pd.Series(y)


# Get the registered model
model = du.get_registered_model(config.model.MODEL_NAME)
# Get model version (stage = 'Staging'; alias = 'Champion)
champion_mv = du.get_model_version_by_stage_and_alias(model, "Staging", "Champion")
# Get model uri
model_uri = f"models:/{config.model.MODEL_NAME}/{champion_mv.version}"

# Load the model from its model uri
model = mlflow.sklearn.load_model(model_uri)
# Get a list of feature names
feature_names = model.feature_names_in_.tolist()

# Read serving data
serving_df = spark.table(
    f"{schema_table_name_dict['gold_schema']}.{schema_table_name_dict['gold_table_serving']}"
)


# Perform inference with the pandas udf
predictions_df = serving_df.select(
    config.model.CLIENT_UNIQUE_IDENTIFER,
    config.model.TIME_PERIOD_VARIABLE,
    *config.model.SELECTED_FEATURES,
    predict_pandas_udf(struct(*feature_names)).alias("predicted_probabilities"),
)


# Materialize inference table (containing client ID, features values, predicted probabilities)

# Write table to data storage
(
    predictions_df.write.mode("overwrite")
    .format("delta")
    .option("overwriteSchema", True)
    .option(
        "path",
        f"{schema_table_name_dict['model_assets_data_storage_path']}/{schema_table_name_dict['inference_table']}",
    )
    .saveAsTable(
        f"{schema_table_name_dict['model_assets_schema']}.{schema_table_name_dict['inference_table']}"
    )
)

from pyspark.sql import DataFrame as PySparkDataFrame
from pyspark.sql import functions as F
from pyspark.sql import SparkSession
import os

from ptb_ltc.config.core import config


if config.general.RUN_ON_DATABRICKS_WS:
    spark = SparkSession.builder.getOrCreate()


"""
Schema and Table names in the Hive Meta Store (to be changed/replaced once the project is migrated to UC sandbox)
"""

DATA_STORAGE_EXTERNAL_LOCATION = (
    "s3://thrivent-prd-datalake-analytics-workspace-east/DSI"
)


# Bronze data
BRONZE_SCHEMA_DICT = {
    env: f"aw_ds_ptb_bronze_{env}" for env in ["dev", "staging", "test", "prod"]
}

BRONZE_DATA_STORAGE_PATH_DICT = {
    env: f"{DATA_STORAGE_EXTERNAL_LOCATION}/{BRONZE_SCHEMA_DICT[env]}"
    for env in ["dev", "staging", "test", "prod"]
}

BRONZE_ALL_MEMBERS_TABLE = "all_members_monthly_vw"
BRONZE_CHOREOGRAPH_TABLE = "choreograph"
BRONZE_RESPONDER_TABLE = "ltc_product_purchased"


# Bronze Train Data
# BRONZE_TRAIN_ALL_MEMBERS_SCHEMA = "prep_do_all_members"
# BRONZE_TRAIN_ALL_MEMBERS_TABLE = "all_members_monthly_vw"

# BRONZE_TRAIN_CHOREOGRAPH_SCHEMA = "prep_dsi_feature_store"
# BRONZE_TRAIN_CHOREOGRAPH_TABLE = "demo_choreo_cif_dly_hst"

# BRONZE_TRAIN_RESPONDER_SCHEMA = "aw_dsi_feature_tables"
# BRONZE_TRAIN_RESPONDER_TABLE = "ltc_product_purchased_interim


# Bronze Serving Data (to be changed in the future when we have separated serving data from DEME)
# BRONZE_SERVING_ALL_MEMBERS_SCHEMA = "prep_do_all_members"
# BRONZE_SERVING_ALL_MEMBERS_TABLE = "all_members_monthly_vw"

# BRONZE_SERVING_CHOREOGRAPH_SCHEMA = "prep_dsi_feature_store"
# BRONZE_SERVING_CHOREOGRAPH_TABLE = "demo_choreo_cif_dly_hst"

# BRONZE_SERVING_RESPONDER_SCHEMA = "aw_dsi_feature_tables"
# BRONZE_SERVING_RESPONDER_TABLE = "ltc_product_purchased_interim"


# Silver Data
SILVER_SCHEMA_DICT = {
    env: f"aw_ds_ptb_silver_{env}" for env in ["dev", "staging", "test", "prod"]
}
SILVER_DATA_STORAGE_PATH_DICT = {
    env: f"{DATA_STORAGE_EXTERNAL_LOCATION}/{SILVER_SCHEMA_DICT[env]}"
    for env in ["dev", "staging", "test", "prod"]
}

SILVER_ALL_MEMBERS_TABLE = f"all_members_{config.model.MODEL_NAME}"
SILVER_CHOREOGRAPH_TABLE = f"choreograph_{config.model.MODEL_NAME}"
SILVER_RESPONDER_TABLE = f"responder_{config.model.MODEL_NAME}"

# Serving data (TODO: to be remove once we can use feature tore)
SILVER_ALL_MEMBERS_TABLE_SERVING = f"all_members_{config.model.MODEL_NAME}_serving"
SILVER_CHOREOGRAPH_TABLE_SERVING = f"choreograph_{config.model.MODEL_NAME}_serving"


# SILVER_TRAIN_SCHEMA = "aw_ds_ptb_silver_train"
# SILVER_TRAIN_DATA_STORAGE_PATH = (
#    f"s3://thrivent-prd-datalake-analytics-workspace-east/DSI/{SILVER_TRAIN_SCHEMA}"
# )

# SILVER_TRAIN_ALL_MEMBERS_TABLE = f"all_members_{config.model.MODEL_NAME}"
# SILVER_TRAIN_CHOREOGRAPH_TABLE = f"choreograph_{config.model.MODEL_NAME}"
# SILVER_TRAIN_RESPONDER_TABLE = f"responder_{config.model.MODEL_NAME}"


# Silver Serving Data
# SILVER_SERVING_SCHEMA = "aw_ds_ptb_silver_serving"
# SILVER_SERVING_DATA_STORAGE_PATH = (
#    f"s3://thrivent-prd-datalake-analytics-workspace-east/DSI/{SILVER_SERVING_SCHEMA}"
# )

# SILVER_SERVING_ALL_MEMBERS_TABLE = f"all_members_{config.model.MODEL_NAME}"
# SILVER_SERVING_CHOREOGRAPH_TABLE = f"choreograph_{config.model.MODEL_NAME}"
# SILVER_SERVING_RESPONDER_TABLE = f"responder_{config.model.MODEL_NAME}"


# Gold Data
GOLD_SCHEMA_DICT = {
    env: f"aw_ds_ptb_gold_{env}" for env in ["dev", "staging", "test", "prod"]
}
GOLD_DATA_STORAGE_PATH_DICT = {
    env: f"{DATA_STORAGE_EXTERNAL_LOCATION}/{GOLD_SCHEMA_DICT[env]}"
    for env in ["dev", "staging", "test", "prod"]
}
GOLD_TABLE = config.model.MODEL_NAME

# Serving data (TODO: to be remove once we can use feature tore)
GOLD_TABLE_SERVING = f"{config.model.MODEL_NAME}_serving"


# GOLD_TRAIN_SCHEMA = "aw_ds_ptb_gold_train"
# GOLD_TRAIN_DATA_STORAGE_PATH = (
#    f"s3://thrivent-prd-datalake-analytics-workspace-east/DSI/{GOLD_TRAIN_SCHEMA}"
# )
# GOLD_TRAIN_TABLE = config.model.MODEL_NAME

# Gold Serving Data
# GOLD_SERVING_SCHEMA = "aw_ds_ptb_gold_serving"
# GOLD_SERVING_DATA_STORAGE_PATH = (
#    f"s3://thrivent-prd-datalake-analytics-workspace-east/DSI/{GOLD_SERVING_SCHEMA}"
# )
# GOLD_SERVING_TABLE = config.model.MODEL_NAME

# Model asset schema (stores feautres, inference, and metric tables as well as models (when we use UC), Volumes, and functions)
MODEL_ASSETS_SCHEMA_DICT = {
    env: f"aw_ds_ptb_ml_assets_{env}" for env in ["dev", "staging", "test", "prod"]
}

MODEL_ASSETS_DATA_STORAGE_PATH_DICT = {
    env: f"{DATA_STORAGE_EXTERNAL_LOCATION}/{MODEL_ASSETS_SCHEMA_DICT[env]}"
    for env in ["dev", "staging", "test", "prod"]
}
INFERENCE_TABLE = f"inference_{config.model.MODEL_NAME}"
# METRICS_TABLE


"""
Create schemas
"""

# Bronze
for env in ["dev", "staging", "test", "prod"]:
    spark.sql(
        f"CREATE DATABASE IF NOT EXISTS {BRONZE_SCHEMA_DICT[env]} LOCATION '{BRONZE_DATA_STORAGE_PATH_DICT[env]}'"
    )


# Bronze Train
# spark.sql(f"CREATE DATABASE IF NOT EXISTS {BRONZE_TRAIN_ALL_MEMBERS_SCHEMA}")
# spark.sql(f"CREATE DATABASE IF NOT EXISTS {BRONZE_TRAIN_CHOREOGRAPH_SCHEMA}")
# spark.sql(f"CREATE DATABASE IF NOT EXISTS {BRONZE_TRAIN_RESPONDER_SCHEMA}")

# Bronze Serving
# spark.sql(f"CREATE DATABASE IF NOT EXISTS {BRONZE_SERVING_ALL_MEMBERS_SCHEMA}")
# spark.sql(f"CREATE DATABASE IF NOT EXISTS {BRONZE_SERVING_CHOREOGRAPH_SCHEMA}")
# spark.sql(f"CREATE DATABASE IF NOT EXISTS {BRONZE_SERVING_RESPONDER_SCHEMA}")


# Silver
for env in ["dev", "staging", "test", "prod"]:
    spark.sql(
        f"CREATE DATABASE IF NOT EXISTS {SILVER_SCHEMA_DICT[env]} LOCATION '{SILVER_DATA_STORAGE_PATH_DICT[env]}'"
    )


# Silver Train
# spark.sql(
#    f"CREATE DATABASE IF NOT EXISTS {SILVER_TRAIN_SCHEMA} LOCATION '{SILVER_TRAIN_DATA_STORAGE_PATH}'"
# )

# Silver Serving
# spark.sql(
#    f"CREATE DATABASE IF NOT EXISTS {SILVER_SERVING_SCHEMA} LOCATION '{SILVER_SERVING_DATA_STORAGE_PATH}'"
# )


# Gold
for env in ["dev", "staging", "test", "prod"]:
    spark.sql(
        f"CREATE DATABASE IF NOT EXISTS {GOLD_SCHEMA_DICT[env]} LOCATION '{GOLD_DATA_STORAGE_PATH_DICT[env]}'"
    )

# Gold Train
# spark.sql(
#    f"CREATE DATABASE IF NOT EXISTS {GOLD_TRAIN_SCHEMA} LOCATION '{GOLD_TRAIN_DATA_STORAGE_PATH}'"
# )

# Gold Serving
# spark.sql(
#    f"CREATE DATABASE IF NOT EXISTS {GOLD_SERVING_SCHEMA} LOCATION '{GOLD_SERVING_DATA_STORAGE_PATH}'"
# )


# Model Assets
for env in ["dev", "staging", "test", "prod"]:
    spark.sql(
        f"CREATE DATABASE IF NOT EXISTS {MODEL_ASSETS_SCHEMA_DICT[env]} LOCATION '{MODEL_ASSETS_DATA_STORAGE_PATH_DICT[env]}'"
    )


# Inference
# spark.sql(
#    f"CREATE DATABASE IF NOT EXISTS {INFERENCE_SCHEMA} LOCATION '{INFERENCE_DATA_STORAGE_PATH}'"
# )


def get_schema_table_names(env: str) -> None:

    schema_table_name_dict = {
        "bronze_schema": BRONZE_SCHEMA_DICT[env],
        "bronze_all_members_table": BRONZE_ALL_MEMBERS_TABLE,
        "bronze_choreograph_table": BRONZE_CHOREOGRAPH_TABLE,
        "bronze_responder_table": BRONZE_RESPONDER_TABLE,
        "silver_schema": SILVER_SCHEMA_DICT[env],
        "silver_data_storage_path": SILVER_DATA_STORAGE_PATH_DICT[env],
        "silver_all_members_table": SILVER_ALL_MEMBERS_TABLE,
        "silver_choreograph_table": SILVER_CHOREOGRAPH_TABLE,
        "silver_all_members_table_serving": SILVER_ALL_MEMBERS_TABLE_SERVING,
        "silver_choreograph_table_serving": SILVER_CHOREOGRAPH_TABLE_SERVING,
        "silver_responder_table": SILVER_RESPONDER_TABLE,
        "gold_schema": GOLD_SCHEMA_DICT[env],
        "gold_data_storage_path": GOLD_DATA_STORAGE_PATH_DICT[env],
        "gold_table": GOLD_TABLE,
        "gold_table_serving": GOLD_TABLE_SERVING,
        "model_assets_schema": MODEL_ASSETS_SCHEMA_DICT[env],
        "model_assets_data_storage_path": MODEL_ASSETS_DATA_STORAGE_PATH_DICT[env],
        "inference_table": INFERENCE_TABLE,
    }

    return schema_table_name_dict


# def get_schema_table_names(pipeline_type: bool, env: str) -> None:
#   if pipeline_type == "feature":
#       schema_table_name_dict = {
#           "bronze_all_members_schema": BRONZE_TRAIN_ALL_MEMBERS_SCHEMA,
#           "bronze_all_members_table": BRONZE_TRAIN_ALL_MEMBERS_TABLE,
#           "bronze_choreograph_schema": BRONZE_TRAIN_CHOREOGRAPH_SCHEMA,
#           "bronze_choreograph_table": BRONZE_TRAIN_CHOREOGRAPH_TABLE,
#           "bronze_responder_schema": BRONZE_TRAIN_RESPONDER_SCHEMA,
#           "bronze_responder_table": BRONZE_TRAIN_RESPONDER_TABLE,
#           "silver_data_storage_path": SILVER_TRAIN_DATA_STORAGE_PATH,
#           "silver_schema": SILVER_TRAIN_SCHEMA,
#           "silver_all_members_table": SILVER_TRAIN_ALL_MEMBERS_TABLE,
#           "silver_choreograph_table": SILVER_TRAIN_CHOREOGRAPH_TABLE,
#           "silver_responder_table": SILVER_TRAIN_RESPONDER_TABLE,
#           "gold_data_storage_path": GOLD_TRAIN_DATA_STORAGE_PATH,
#           "gold_schema": GOLD_TRAIN_SCHEMA,
#           "gold_table": GOLD_TRAIN_TABLE,
#       }
#   else:
#       schema_table_name_dict = {
#           "bronze_all_members_schema": BRONZE_SERVING_ALL_MEMBERS_SCHEMA,
#           "bronze_all_members_table": BRONZE_SERVING_ALL_MEMBERS_TABLE,
#           "bronze_choreograph_schema": BRONZE_SERVING_CHOREOGRAPH_SCHEMA,
#           "bronze_choreograph_table": BRONZE_SERVING_CHOREOGRAPH_TABLE,
#           "bronze_responder_schema": BRONZE_SERVING_RESPONDER_SCHEMA,
#           "bronze_responder_table": BRONZE_SERVING_RESPONDER_TABLE,
#           "silver_data_storage_path": SILVER_SERVING_DATA_STORAGE_PATH,
#           "silver_schema": SILVER_SERVING_SCHEMA,
#           "silver_all_members_table": SILVER_SERVING_ALL_MEMBERS_TABLE,
#           "silver_choreograph_table": SILVER_SERVING_CHOREOGRAPH_TABLE,
#           "silver_responder_table": SILVER_SERVING_RESPONDER_TABLE,
#           "gold_data_storage_path": GOLD_SERVING_DATA_STORAGE_PATH,
#           "gold_schema": GOLD_SERVING_SCHEMA,
#           "gold_table": GOLD_SERVING_TABLE,
#           "inference_data_storage_path": INFERENCE_DATA_STORAGE_PATH,
#           "inference_schema": INFERENCE_SCHEMA,
#           "inference_table": INFERENCE_TABLE,
#       }
#
#   return schema_table_name_dict

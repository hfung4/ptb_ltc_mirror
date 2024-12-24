from argparse import ArgumentParser
from pathlib import Path

from pyspark.sql import functions as F
from pyspark.sql import types as T

from ptb_ltc.processing.processing_utils import (
    snake_case_column_names,
    input_args_time_periods_validation,
    get_validated_time_period_inference_pipeline,
    filter_after_a_given_date,
    create_adjusted_effective_start_date,
)
import ptb_ltc.data.data_manager as dm
from ptb_ltc.config.core import config


# Get user inputs-- train and test, or serving time period start dates
parser = ArgumentParser()
for arg in ["--serving_start_dates"]:
    parser.add_argument(arg, type=str, nargs="*", required=False)

# Get user inputs for target deployment environment
parser.add_argument(
    "--env",
    type=str,
    required=True,
    choices=["dev", "test", "staging", "prod"],
    help="The deploy environment name (e.g., 'dev', 'test', 'staging', 'prod').",
)

args = parser.parse_args()


# Define the dictionary for holding time periods
time_period_dict = {
    "serving_start_dates": args.serving_start_dates,
}

# Get pipeline specific variables  -----------------------------------------------------------------------------
validated_time_period_dict = get_validated_time_period_inference_pipeline(
    time_period_dict
)

# Get schema and table names
schema_table_name_dict = dm.get_schema_table_names(args.env)

# Get a list of datetime.date objects from the validated time_period_dict
validated_time_period_list = sorted(
    [
        d
        for dates in validated_time_period_dict.values()
        if dates is not None
        for d in dates
    ]
)


# ETL for responder data -------------------------------------------------------------------------------------
silver_responder = (
    spark.table(
        f"{schema_table_name_dict['bronze_schema']}.{schema_table_name_dict['bronze_responder_table']}"
    )
    # transform column names to snake case
    .transform(snake_case_column_names)
    # cast from timestamp to date DataType for purchase_period_begin_timestamp
    .withColumn(
        "purchase_period_begin_timestamp",
        F.col("purchase_period_begin_timestamp").cast("date"),
    )
    # trim and cast customer id number
    .withColumn("cifidnbr", F.trim(F.col("customer_id_number")).cast(T.StringType()))
    # Create adjusted effective start date column
    .transform(
        lambda df_: create_adjusted_effective_start_date(
            df_, "purchase_period_begin_timestamp", validated_time_period_list
        )
    )
    # Donwsample the gold data if option flag is set
    .transform(
        lambda df_: (
            resample_responder_df(df_, config.processing.NON_RESPONSE_RATIO)
            if config.processing.PERFORM_INITIAL_DOWNSAMPLING
            else df_
        )
    )
)

# Write data to hive metastore
(
    silver_responder.write.mode("overwrite")
    .format("delta")
    .option("overwriteSchema", True)
    .option(
        "path",
        f"{schema_table_name_dict['silver_data_storage_path']}/{schema_table_name_dict['silver_responder_table']}",
    )
    .saveAsTable(
        f"{schema_table_name_dict['silver_schema']}.{schema_table_name_dict['silver_responder_table']}"
    )
)

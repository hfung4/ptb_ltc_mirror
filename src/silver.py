"""
silver.py: Perform ETL and data wrangling to create the silver table from the bronze (raw) data
"""

from argparse import ArgumentParser
from pathlib import Path

from pyspark.sql import functions as F
from pyspark.sql import types as T

from ptb_ltc.processing.processing_utils import (
    snake_case_column_names,
    input_args_time_periods_validation,
    get_validated_time_period_inference_pipeline,
    get_validated_time_period_feature_pipeline,
    filter_after_a_given_date,
    create_adjusted_effective_start_date,
    create_set_type_for_train_test_data,
    resample_responder_df,
)
import ptb_ltc.data.data_manager as dm
from ptb_ltc.config.core import config

# Get user inputs-- train and test, or serving time period start dates
parser = ArgumentParser()
for arg in ["--train_start_dates", "--test_start_dates", "--serving_start_dates"]:
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


# Compute time period columns -----------------------------------------------------------------------------

# Define the dictionary for holding time periods
time_period_dict = {
    "train_start_dates": args.train_start_dates,
    "test_start_dates": args.test_start_dates,
    "serving_start_dates": args.serving_start_dates,
}
# Check to see if the input is for the feature pipeline or the inference pipeline
pipeline_type = "inference" if time_period_dict["serving_start_dates"] else "feature"

# Set task values so that pipeline_type can be read by other tasks in the workflow (e.g., gold or predict)
dbutils.jobs.taskValues.set(key="pipeline_type", value=pipeline_type)

# input_args_time_periods_validation
input_args_time_periods_validation(time_period_dict)

# Get pipeline specific variables  -----------------------------------------------------------------------------

# Get time periods dictionary
if pipeline_type == "feature":
    validated_time_period_dict = get_validated_time_period_feature_pipeline(
        time_period_dict
    )
else:
    validated_time_period_dict = get_validated_time_period_inference_pipeline(
        time_period_dict
    )

# Get schema and table names
schema_table_name_dict = dm.get_schema_table_names(env=args.env)

# Get a list of datetime.date objects from the validated time_period_dict
validated_time_period_list = sorted(
    [
        d
        for dates in validated_time_period_dict.values()
        if dates is not None
        for d in dates
    ]
)

# ETL for All Members data ------------------------------------------------------------------------------
silver_all_members = (
    spark.table(
        f"{schema_table_name_dict['bronze_schema']}.{schema_table_name_dict['bronze_all_members_table']}"
    )
    # transform column names to snake case
    .transform(snake_case_column_names)
    # filter rows after the first period's start date
    .transform(
        lambda df_: (
            filter_after_a_given_date(df_, "eff_beg_dt", validated_time_period_list[0])
            if pipeline_type == "feature"
            else df_
        )
    )
    # trim and cast the cifidnbr (client ID)
    .withColumn("cifidnbr", F.trim(F.col("cifidnbr")).cast(T.StringType()))
    # Create adjusted effective start date column
    .transform(
        lambda df_: create_adjusted_effective_start_date(
            df_, "eff_beg_dt", validated_time_period_list
        )
    )
    # Create set_type column
    .transform(
        lambda df_: (
            create_set_type_for_train_test_data(
                df_, config.model.TIME_PERIOD_VARIABLE, validated_time_period_dict
            )
            if pipeline_type == "feature"
            else df_
        )
    ).transform(
        lambda df_: (
            df_.withColumn("set_type", F.lit("serving"))
            if pipeline_type == "inference"
            else df_
        )
    )
)


# Get table name for the silver all members data
silver_all_members_table_name = (
    schema_table_name_dict["silver_all_members_table"]
    if pipeline_type == "feature"
    else schema_table_name_dict["silver_all_members_table_serving"]
)

(
    silver_all_members.write.mode("overwrite")
    .format("delta")
    .option("overwriteSchema", True)
    .option(
        "path",
        f"{schema_table_name_dict['silver_data_storage_path']}/{silver_all_members_table_name}",
    )
    .saveAsTable(
        f"{schema_table_name_dict['silver_schema']}.{silver_all_members_table_name}"
    )
)


# ETL for Choerograph data ---------------------------------------------------------------------------
silver_choreo = (
    spark.table(
        f"{schema_table_name_dict['bronze_schema']}.{schema_table_name_dict['bronze_choreograph_table']}"
    )
    # transform column names to snake case
    .transform(snake_case_column_names)
    # filter rows after the first period's start date
    .transform(
        lambda df_: (
            filter_after_a_given_date(df_, "eff_beg_dt", validated_time_period_list[0])
            if pipeline_type == "feature"
            else df_
        )
    )
    # Create adjusted effective start date column
    .transform(
        lambda df_: create_adjusted_effective_start_date(
            df_, "eff_beg_dt", validated_time_period_list
        )
    )
    # Create set_type column
    .transform(
        lambda df_: (
            create_set_type_for_train_test_data(
                df_, config.model.TIME_PERIOD_VARIABLE, validated_time_period_dict
            )
            if pipeline_type == "feature"
            else df_
        )
    ).transform(
        lambda df_: (
            df_.withColumn("set_type", F.lit("serving"))
            if pipeline_type == "inference"
            else df_
        )
    )
)

# Get table name for the silver choreograph data
silver_choreo_table_name = (
    schema_table_name_dict["silver_choreograph_table"]
    if pipeline_type == "feature"
    else schema_table_name_dict["silver_choreograph_table_serving"]
)

(
    silver_choreo.write.mode("overwrite")
    .format("delta")
    .option("overwriteSchema", True)
    .option(
        "path",
        f"{schema_table_name_dict['silver_data_storage_path']}/{silver_choreo_table_name}",
    )
    .saveAsTable(
        f"{schema_table_name_dict['silver_schema']}.{silver_choreo_table_name}"
    )
)

# ETL for responder data -------------------------------------------------------------------------------------
if pipeline_type == "feature":
    # ETL for responder data
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
        # filter rows after the first period's start date
        .transform(
            lambda df_: filter_after_a_given_date(
                df_, "purchase_period_begin_timestamp", validated_time_period_list[0]
            )
        )
        # trim and cast customer id number
        .withColumn(
            "cifidnbr", F.trim(F.col("customer_id_number")).cast(T.StringType())
        )
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
        # Create set_type column
        .transform(
            lambda df_: create_set_type_for_train_test_data(
                df_, config.model.TIME_PERIOD_VARIABLE, validated_time_period_dict
            )
        )
    )

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

"""
gold.py: Further perform data processing, feature engineering to create data that is ready for modeling. In the future
we will write to feature tables in the feature store. 
"""

import subprocess

subprocess.run(["pip", "install", "great_expectations"])


from ptb_ltc.processing.processing_utils import (
    add_age_features,
    add_social_security_retirement_age,
    add_pct_features,
    transform_household_features,
    transform_and_add_membership_tenure_features,
    snake_case_column_names,
    combine_rare_levels,
    convert_decimal_to_double,
    convert_integer_to_double,
)

from ptb_ltc.config.core import config
import ptb_ltc.data.data_manager as dm

from pyspark.sql import functions as F
from pathlib import Path
from argparse import ArgumentParser


# Get input argument for testing mode
parser = ArgumentParser(description="Run the script with optional testing mode.")

# Get user inputs for target deployment environment
parser.add_argument(
    "--env",
    type=str,
    required=True,
    choices=["dev", "test", "staging", "prod"],
    help="The deploy environment name (e.g., 'dev', 'test', 'staging', 'prod').",
)

args = parser.parse_args()

# Get pipeline_type set in silver.py
pipeline_type = dbutils.jobs.taskValues.get(
    taskKey="silver", key="pipeline_type", debugValue="feature"
)
# Get schema and table names
schema_table_name_dict = dm.get_schema_table_names(args.env)

# Derive new features ------------------------------------------------------------------------------------------------------

# Get table name for the silver all members data
silver_all_members_table_name = (
    schema_table_name_dict["silver_all_members_table"]
    if pipeline_type == "feature"
    else schema_table_name_dict["silver_all_members_table_serving"]
)

# Transform All Members
all_members_df = (
    spark.table(
        f"{schema_table_name_dict['silver_schema']}.{silver_all_members_table_name}"
    )
    # add age features
    .transform(
        lambda _df: add_age_features(
            _df, config.processing.IMPUTE_CUSTAGE, config.processing.MAX_AGE
        )
    )
    # add social security retirement age
    .transform(lambda _df: add_social_security_retirement_age(_df))
    # add pct features
    .transform(lambda _df: add_pct_features(_df))
    # transform the household features
    .transform(
        lambda _df: transform_household_features(_df, config.processing.MAX_ASSETS)
    )
    # transform and add the membership tenure features
    .transform(
        lambda _df: transform_and_add_membership_tenure_features(
            _df, config.processing.IMPUTE_YEARSMEMBER, config.processing.MAX_AGE
        )
    )
    # Replace "AGE_UNKNOWN" in segment with None
    .withColumn(
        "segment",
        F.when(F.col("segment").isin("AGE UNKNOWN"), None).otherwise(F.col("segment")),
    )
    # Replace "UNKNOWN" in age_segment with None
    .withColumn(
        "age_segment",
        F.when(F.col("age_segment").isin("UNKNOWN"), None).otherwise(
            F.col("age_segment")
        ),
    )
    # Handling rare levels: lump cateogrical columns with less than X% of samples to 'Others'
    .transform(
        lambda _df: combine_rare_levels(
            _df,
            ["segment", "age_segment"],
            config.processing.RARE_LEVEL_MIN_PCT_THRESHOLD,
        )
    )
)


# Get table name for the silver choreograph data
silver_choreo_table_name = (
    schema_table_name_dict["silver_choreograph_table"]
    if pipeline_type == "feature"
    else schema_table_name_dict["silver_choreograph_table_serving"]
)

# Transform Choreograph
choreo_df = spark.table(
    f"{schema_table_name_dict['silver_schema']}.{silver_choreo_table_name}"
)

# Transform Responder
if pipeline_type == "feature":
    responder_df = spark.table(
        f"{schema_table_name_dict['silver_schema']}.{schema_table_name_dict['silver_responder_table']}"
    )


# Create gold table ----------------------------------------------------------------------------------------

combined_df_interim = (
    all_members_df
    # left join to choreo
    .join(
        choreo_df,
        on=[config.model.CLIENT_UNIQUE_IDENTIFER, config.model.TIME_PERIOD_VARIABLE],
        how="left",
    )
    # drop set_type duplicated column
    .drop(choreo_df.set_type)
)

# Apply left join if pipeline type is 'feature'
combined_df = (
    combined_df_interim.join(
        responder_df,
        on=[config.model.CLIENT_UNIQUE_IDENTIFER, config.model.TIME_PERIOD_VARIABLE],
        how="left",
    )
    if pipeline_type == "feature"
    else combined_df_interim
)

# Set selected columns based on pipeline type
selected_columns = [
    config.model.CLIENT_UNIQUE_IDENTIFER,
    *config.model.SELECTED_FEATURES,
    config.model.OUTCOME_VARIABLE if pipeline_type == "feature" else None,
    config.model.TIME_PERIOD_VARIABLE,
    combined_df_interim.set_type if pipeline_type == "feature" else None,
]
selected_columns = [x for x in selected_columns if x is not None]

# Create gold table
gold_df = (
    combined_df.select(*selected_columns)
    .transform(snake_case_column_names)
    .filter(
        F.col(config.model.OUTCOME_VARIABLE).isNotNull()
        if pipeline_type == "feature"
        else F.lit(True)
    )
    .transform(convert_decimal_to_double)
    .transform(convert_integer_to_double)
)
if args.env == "test":
    from ptb_ltc.data.gx_manager import validate_gold_data

    # Validate gold data
    validation_results = validate_gold_data(gold_df, pipeline_type)
    assert (
        validation_results.schema_check
    ), "Great Expectations Validation failed: schema check"
    assert (
        validation_results.numerical_ranges
    ), "Great Expectations Validation failed: numerical range"
    assert (
        validation_results.categorical_options
    ), "Great Expectations Validation failed: categorical options"

    if pipeline_type == "feature":
        assert (
            validation_results.missingness_in_response
        ), "Great Expectations Validation failed: missingness in response"

# Write table to data storage

# Get table name for the gold data
gold_table_name = (
    schema_table_name_dict["gold_table"]
    if pipeline_type == "feature"
    else schema_table_name_dict["gold_table_serving"]
)

(
    gold_df.write.mode("overwrite")
    .format("delta")
    .option("overwriteSchema", True)
    .option(
        "path",
        f"{schema_table_name_dict['gold_data_storage_path']}/{gold_table_name}",
    )
    .saveAsTable(f"{schema_table_name_dict['gold_schema']}.{gold_table_name}")
)

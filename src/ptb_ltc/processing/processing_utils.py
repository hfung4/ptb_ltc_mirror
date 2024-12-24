import re
from ptb_ltc.logging import logger
from pyspark.sql import DataFrame as PySparkDataFrame
from pyspark.sql import functions as F
from pyspark.sql import types as T
from pyspark.sql.types import DateType, DoubleType

from itertools import chain
import os
import math
from collections import ChainMap
from datetime import date, datetime
from typing import List, Tuple, Dict, Union
from ptb_ltc.config.core import config


def apply_sql_col_rules(var_list: list[str]) -> list[str]:
    """Checks if variable names are SQL compatible

    Args:
        var_list (list[str]): list of variable names to be checked

    Returns:
        list[str]: results of variable name checks
    """

    ret = []
    for name in var_list:
        new_name = re.sub(r"[^A-Za-z0-9_]+", "_", name).strip()
        if name != new_name:
            logger.info(f"{name} renamed to {new_name} for SQL compatibility")
        ret.append(new_name)
    return ret


def snake_case_column_names(df: PySparkDataFrame) -> PySparkDataFrame:
    """
    Snake and lower case all column names and converts all non-alpha-numeric characters to underscores (e.g., MY_COLUMN_NAME to my_column_name).

    Parameters:
      - df (PySparkDataFrame) : The input dataframe.

    Returns:
      - (PySparkDataFrame)    : The output dataframe with formatted column names.
    """

    # create a map that maps the old columns to the new columns (which are lower cased and snake cased)
    column_map = {k: re.sub(r"\W+", "_", k.lower()) for k in df.columns}

    # return dataframe with changed column names
    return df.withColumnsRenamed(column_map)


def map_column(
    df: PySparkDataFrame, col_name: str, map: dict, new_col_name: str = None
) -> PySparkDataFrame:
    """
    Map a column in a PySpark DataFrame from one value to another (as provided by the map dictionary).
    Note: This is the recommended way on Stack Overflow.

    Parameters:
      - df (PySparkDataFrame) : DataFrame to be modified
      - col_name (str)        : Column to be modified (overwritten if new_col_name is None)
      - map (dict)            : Mapping dictionary from old value to new value
      - new_col_name (str)    : Name of the new column (defaults to None, in which case the column is overwritten)

    Returns:
      - (PySparkDataFrame)      : DataFrame with the modified column

    """

    # overwrite the column if new_col_name is none
    if new_col_name is None:
        new_col_name = col_name

    # create mapping expression from map
    mapping_expr = F.create_map([F.lit(x) for x in chain(*map.items())])

    # return PySparkDataFrame with new column
    return df.withColumn(new_col_name, mapping_expr[F.col(col_name)])


def add_age_features(
    df: PySparkDataFrame, impute_custage: int = 40, max_age: int = 99
) -> PySparkDataFrame:
    """
    Takes a PySparkDataFrame and adds three age featues: CUSTAGE, age_cust, and age_segment.

    Parameters:
    - df (PySparkDataFrame) : Input DataFrame to which the new columns are added.
    - impute_custage (int)  : The value to impute for CUSTAGE if it is null.
    - max_age (int)         : The maximum age to impose.

    Returns:
    - (PySparkDataFrame)    : DataFrame with new columns added.
    """

    df = (
        df.withColumn(
            "CUSTAGE",
            F.when(
                F.col("CUSTAGE").isNull(),
                F.when(F.col("EMBRTYP") == "JUV", F.lit(10)).otherwise(
                    F.lit(impute_custage)
                ),
            ).otherwise(F.col("CUSTAGE")),
        )
        # Constrain age to be between 0 and max_age input.
        .withColumn(
            "age_cust",
            F.array_max(
                F.array(
                    F.lit(0), F.array_min(F.array(F.lit(max_age), F.col("CUSTAGE")))
                )
            ),
        )
    )

    # Create age segements
    age_segment_tuples = (
        (1945, "F Seniors"),
        (1955, "E LeadBoomers"),
        (1964, "D TailBoomers"),
        (1974, "C GenX"),
        (1990, "B Millennial"),
        (2010, "A GenZ"),
    )

    age_segment_column = F.when(F.col("BIRTHDTE").isNull(), "UNKNOWN")

    for year, segment in age_segment_tuples:
        age_segment_column = age_segment_column.when(
            F.year(F.col("BIRTHDTE")) <= year, segment
        )

    df = df.withColumn("age_segment", age_segment_column.otherwise("UNKNOWN"))

    return df


def add_social_security_retirement_age(df: PySparkDataFrame) -> PySparkDataFrame:
    """
    Adds the Social Security Full Retirement Age column based on a client's birthdate.

    Parameters:
    - df (PySparkDataFrame) : Input DataFrame to which the column is added.

    Returns:
    - (PySparkDataFrame)   : DataFrame with the new column added.
    """

    retirement_age_tuples = (
        (1942, 65.0),
        (1954, 66.0),
        (1955, 66.1667),
        (1956, 66.3333),
        (1957, 66.5),
        (1958, 66.6667),
        (1959, 66.8333),
    )

    socsec_full_retire_age = F.when(F.col("BIRTHDTE").isNull(), None)

    for year, retirement_age in retirement_age_tuples:
        socsec_full_retire_age = socsec_full_retire_age.when(
            F.year(F.col("BIRTHDTE")) <= year, F.lit(retirement_age)
        )

    return df.withColumn(
        "socsec_full_retire_age", socsec_full_retire_age.otherwise(F.lit(67.0))
    )


def add_pct_features(df: PySparkDataFrame) -> PySparkDataFrame:
    """
    Adds the pct features for Proprietary Annual Premium Bill Amount and Proprietary Life Face Amount features.

    Parameters:
    - df (PySparkDataFrame) : Input DataFrame to which the columns are added.

    Returns:
    - (PySparkDataFrame)    : DataFrame with the new columns added.
    """
    return df.withColumn(
        "pct_CVLPREM_TOTPREM",
        F.when(F.coalesce(F.col("TOTPREM"), F.lit(0)) < 1, F.lit(0.0)).otherwise(
            F.array_min(
                F.array(
                    F.lit(1),
                    (F.col("TRADPREM") + F.col("ULPREM") + F.col("VULPREM"))
                    / F.col("TOTPREM"),
                )
            )
        ),
    ).withColumn(
        "pct_CVLSA_TOTSA",
        F.when(F.coalesce(F.col("TOTSA"), F.lit(0)) < 1, F.lit(0)).otherwise(
            F.array_min(
                F.array(
                    F.lit(1),
                    F.coalesce(
                        F.col("TRADSA"),
                        F.lit(0)
                        + F.coalesce(F.col("ULSA"), F.lit(0))
                        + F.coalesce(F.col("VULSA"), F.lit(0)),
                    )
                    / F.col("TOTSA"),
                )
            )
        ),
    )


def transform_household_features(
    df: PySparkDataFrame, max_assets: int = 1_000_000
) -> PySparkDataFrame:
    """
    Convert House HOld features into an integer and cap it at max_assets.

    Parameters:
    - df (PySparkDataFrame) : Input DataFrame to which the columns is added.
    - max_assets (int)      : Maximum assets amount with which to cap the feature.

    Returs:
    - (PySparkDataFrame)    : DataFrame with the new column added.
    """
    return (
        df.withColumn("HH_AUA", (F.col("HHAUA") > 0).cast("int"))
        .withColumn("HHAUA", F.array_min(F.array(F.lit(max_assets), F.col("HHAUA"))))
        .withColumn(
            "HH_OWN_WRAP", (F.coalesce(F.col("HHWRAPOWN"), F.lit(0)) > 0).cast("int")
        )
    )


def transform_YEARSMEMBER_feature(
    df: PySparkDataFrame, impute_yearsmember: int = 25, max_age: int = 99
) -> PySparkDataFrame:
    """
    Transform the YEARSMEMBER feature to the input dataframe by capping at max_assets.

    Parameters:
    - df (PySparkDataFrame)    : Input DataFrame to which the column is added.
    - impute_yearsmember (int) : Value to impute if YEARSMEMBER is null.
    - max_age (int)            : Maximum age for YEARSMEMBER.

    Returns:
    - (PySparkDataFrame)    : DataFrame with the new column added.
    """
    return df.withColumn(
        "YEARSMEMBER",
        F.when(
            F.col("YEARSMEMBER").isNull(),
            F.when(F.col("YRSPURCH") > 0, F.col("YRSPURCH")).otherwise(
                F.lit(impute_yearsmember)
            ),
        ).otherwise(F.col("YEARSMEMBER")),
    ).withColumn(
        "YEARSMEMBER", F.array_min(F.array(F.lit(max_age), F.col("YEARSMEMBER")))
    )


def add_tot_AUM_per_yrmbr_feature(df: PySparkDataFrame) -> PySparkDataFrame:
    """
    Add the tot_AUM_per_yrmbr feature to the input dataframe.

    Parameters:
    - df (PySparkDataFrame) : Input DataFrame to which the column is added.

    Returns:
    - (PySparkDataFrame)    : DataFrame with the new column added.
    """
    return df.withColumn(
        "tot_AUM_per_yrmbr",
        F.when(F.col("YEARSMEMBER").isNull(), F.lit(0)).otherwise(
            (F.col("PROPASSETS"))
            / F.array_max(F.array(F.lit(0.5), F.col("YEARSMEMBER")))
        ),
    )

def transform_and_add_membership_tenure_features(df: PySparkDataFrame, impute_yearsmember: int = 25, max_age: int = 99) -> PySparkDataFrame:
  """
  Impute the YEARSMEMBER feature if needed, cap the feature at max_age, and calculate the number of years the client has been a member

  Parameters:
  - df (PySparkDataFrame)    : Input DataFrame to which the columns is added.
  - impute_yearsmember (int) : Value to impute if YEARSMEMBER is null.
  - max_age (int)            : Maximum age with which to cap the feature.

  Returns:
  - (PySparkDataFrame)       : DataFrame with the new columns added.
  """

  return (
    df
    .withColumn(
      'YEARSMEMBER',
      F.when(
        F.col('YEARSMEMBER').isNull(),
        F.when(
          F.col('YRSPURCH') > 0, 
          F.col('YRSPURCH')
        )
        .otherwise(F.lit(impute_yearsmember))
      )
      .otherwise(F.col('YEARSMEMBER'))
    )
    .withColumn(
      'YEARSMEMBER', 
      F.array_min(
        F.array(
          F.lit(max_age), 
          F.col('YEARSMEMBER')
        )
      )
    )
    # Calculate the number of years - with decimals - the client has been a member.
    .withColumn(
      'MBR_yrs', 
      F.when(
        F.col('MBRDT').isNotNull(),
        F.array_max(
          F.array(
            F.lit(0),
            F.datediff(
              F.col('eff_beg_dt'),
              F.col('MBRDT')
            ) / F.lit(365.25)
          )
        )
      )
      .otherwise(F.col('YEARSMEMBER'))
    )
    .withColumn(
        'tot_AUM_per_yrmbr',
        F.when(
            F.col('YEARSMEMBER').isNull(),
            F.lit(0)
        )
        .otherwise(
            (
              F.col('PROPASSETS')
            ) / F.array_max(
                F.array(
                    F.lit(0.5), 
                    F.col('YEARSMEMBER')
                )
            )
        )
    )
  )

def resample_responder_df(
    df: PySparkDataFrame, non_response_ratio: int = 4
) -> PySparkDataFrame:
    """
    Subsample the responder df on the target variable to achieve a non-response ratio.

    Parameters:
    - non_response_ratio (int) : Desired ratio of non-responders to responders.

    Returns:
    - (PySparkDataFrame)       : DataFrame that is resampled to achieve the desired ratio.
    """
    resp_count = df.filter("Target == 1").count()
    no_resp_count = df.filter("Target == 0").count()

    no_resp_frac = (resp_count * non_response_ratio) / (no_resp_count + resp_count)

    return df.sampleBy("Target", fractions={1: 1.0, 0: no_resp_frac}, seed=0)


def input_args_time_periods_validation(
    time_periods:Dict[str, List[str]]
)->None:
    """Check if the user input for time periods is valid. The user must provide either both train and test dates or only serving dates.

    Args:
        time_periods (time_periods): a named tuple of time periods
    """

    # Input validation: serving_start_dates can only be provided when both train and test dates are None
    if time_periods["serving_start_dates"] and (
        time_periods["train_start_dates"] is not None or time_periods["test_start_dates"] is not None
    ):
        raise ValueError(
            "You cannot provide --serving_start_dates together with --train_start_dates or --test_start_dates."
        )
    elif not ((time_periods["train_start_dates"] and time_periods["test_start_dates"]) or time_periods["serving_start_dates"]):
        raise ValueError(
            "You must provide either both --train_start_dates and --test_start_dates, or only --serving_start_dates."
        )


def parse_dates(date_input: Union[str, List[str]]) -> List[date]:
    """parse input string dates to standard date format, validate the format and return a list of date objects

    Args:
        date_input (Union[str, List[str]]): A list or single string of dates in YYYY-MM-DD format

    Raises:
        ValueError: Error message if the date_str is not in the correct format

    Returns:
        period_dates (List[date]): A list of date objects
    """
    # Check to see if the input is None
    if date_input is None:
        return None

    period_dates = []

    # If needed, convert single string to a list for uniform processing
    if isinstance(date_input, str):
        date_input = [date_input]

    for date_str in date_input:
        try:
            # Parse date and add to the list if it's in YYYY-MM-DD format
            period_dates.append(datetime.strptime(date_str, "%Y-%m-%d").date())
        except ValueError:
            # Raise a custom error message if date_str is not in the correct format
            raise ValueError(
                f"Invalid date format for '{date_str}'. Expected format is YYYY-MM-DD."
            )
    # sort the list to ensure that the output dates are in ascending order
    period_dates.sort()

    return period_dates

def check_ordering_of_train_test_periods(
    train_start_dates: List[str], test_start_dates: List[str]
) -> None:
    """check the ordering of the train and test periods to ensure that the test peri0ds do not precede the train periods

    Args:
        train_start_dates (List[str]): starting dates of train periods
        test_start_dates (List[str]): starting dates of test periods

    Raises:
        ValueError: if the test periods precede the train periods, raise error
    """
    if max(train_start_dates) >= min(test_start_dates):
        raise ValueError(
            "Time periods for testing cannot precede time periods for training."
        )


def check_consistent_time_period_cadences(period_start_dates: List[str]) -> None:
    # Only need to check time periods where there are two dates (we are pooling 
    # three or more waves of measurements for train/test/serving data)
    if len(period_start_dates) >= 3:
        # Calculate period lengths in days
        period_lengths = [
            (period_start_dates[i] - period_start_dates[i - 1]).days
            for i in range(1, len(period_start_dates))
        ]

        # Check if the difference between maximum and minimum period lengths exceeds the threshold
        if max(period_lengths) - min(period_lengths) > 5:
            raise ValueError(
                f"Irregular cadence detected: the difference between the longest and shortest period lengths must not exceed {5} days."
            )

def get_validated_time_period_feature_pipeline(
    time_period_dict: Dict[str, List[str]]
) -> Dict[str, List[date]]:
    """Validate train and test time periods and return a dictionary of datetime objects

    Args:
        time_period_dict (Dict[str,List[str]]): a dictionary of user input time periods

    Returns:
        Dict[str,List[date]]: validated and processed dictionary of time periods
    """
    for key in ["train_start_dates", "test_start_dates"]:
        time_period_dict[key] = parse_dates(date_input=time_period_dict[key])

    # Validate train and test periods to ensure the test periods is not before the train periods
    check_ordering_of_train_test_periods(
        train_start_dates=time_period_dict["train_start_dates"],
        test_start_dates=time_period_dict["test_start_dates"],
    )

    # Check for consistent cadences in the train and test periods
    train_test_periods = (
        time_period_dict["train_start_dates"] + time_period_dict["test_start_dates"]
    )
    check_consistent_time_period_cadences(train_test_periods)

    return time_period_dict


def get_validated_time_period_inference_pipeline(
    time_period_dict: Dict[str, List[str]]
) -> Dict[str, List[date]]:
    """Validate time periods for serving data and return a dictionary of datetime objects

    Returns:
        Dict[str,List[date]]: validated and processed dictionary of time periods
    """    
    time_period_dict["serving_start_dates"] = parse_dates(
        date_input=time_period_dict["serving_start_dates"]
    )
    # Check for consistent cadences in the train and test periods
    check_consistent_time_period_cadences(time_period_dict["serving_start_dates"])

    return time_period_dict


def filter_after_a_given_date(
    df: PySparkDataFrame, date_col: str, min_date: str
) -> PySparkDataFrame:
    """Filter the DataFrame to include only dates on or after the specified date.

    Args:
        df (PySparkDataFrame): Input DataFrame.
        date_col (str): Name of the date column to filter.
        min_date (str): The date in 'YYYY-MM-DD' format to filter from.

    Returns:
        PySparkDataFrame: Filtered DataFrame with dates on or after time_period.
    """
    # Validate if the date_col column is in DateType
    if not isinstance(df.schema[date_col].dataType, DateType):
        raise TypeError(
            f"The column {date_col} must be of DateType. Please check the column type."
        )

    # Convert the min_date string to a date format
    min_date = F.to_date(F.lit(min_date), "yyyy-MM-dd")

    # Filter the DataFrame based on the date condition
    df_filtered = df.filter(F.col(date_col) >= min_date)

    return df_filtered


def get_closest_date_mapping(
    unique_effective_dates: List[date], time_periods: List[date]
) -> Dict[str, str]:
    """Find the closest date from a set of effective start dates (from the raw data)
    for each time period start dates that I define (for train, test, or serving).
   
    Args:
        unique_effective_dates (List[date]): List of effective dates in 'YYYY-MM-DD' format (from data)
        time_periods (List[date]): List of time periods in 'YYYY-MM-DD' format (defined by user)

    Returns:
        Dict[str, str]: Dictionary with time_periods as keys and closest dates as values.
    """
    closest_dates = {}
    used_dates = set()  # Keep track of used closest dates

    # Iterate through time_periods except the last element
    for time_period in time_periods:
        # Find the closest date from unique_effective_dates
        closest_date = min(
            (
                eff_date
                for eff_date in unique_effective_dates
                if eff_date not in used_dates
            ),
            key=lambda eff_date: abs((eff_date - time_period).days),
            default=None,  # Default to None in case no effective dates are left
        )

        if closest_date:
            # Convert back to string in the closest_date dictionary
            closest_dates[closest_date.strftime("%Y-%m-%d")] = time_period.strftime(
                "%Y-%m-%d"
            )
            used_dates.add(closest_date)  # Mark this date as used

    return closest_dates


def create_adjusted_effective_start_date(
    df: PySparkDataFrame,
    date_col: str,
    time_periods: List[datetime],
) -> PySparkDataFrame:
    """Create a column "adjusted_effective_start_date" by finding the closest date in date_col
     to user defined time periods (for training, testing, and serving).

    Args:
        df (PySparkDataFrame): input dataframe
        date_col (str): Name of the date column containing effective start dates of waves of measurement
        time_periods(List[datetime]): List of time periods in 'YYY-MM-DD' format that is defined by user
    
    Returns:
        processed (PySparkDataFrame): processed datafraome
    """
    # Get a set of unique effective start dates from the date_col in the dataset
    unique_dates_str = sorted(
        (
            df.select(F.col(date_col))
            .distinct()
            .rdd.flatMap(lambda x: x)
            .map(lambda x: str(x))
            .collect()
        )
    )
    # Convert the list of strings to a set of datetime.date objects
    unique_effective_start_dates = {datetime.strptime(date_str, "%Y-%m-%d").date() for date_str in unique_dates_str}

    # Get a mapping of the unique effective start dates in the data their the closest time periods of interest
    closest_date_mapping = get_closest_date_mapping(
        unique_effective_start_dates, time_periods
    )

    # Create a adjusted_effective_date column by mapping from the dates in date_col using cloest_date_mapping

    # Create PySpark mapping using closest_date_mapping
    mapping_expr = F.create_map(
        [F.lit(x) for item in closest_date_mapping.items() for x in item]
    )

    # Apply the mapping to the DataFrame by creating a new column "adjusted_effective_date"
    processed = (
        df
        # create column
        .withColumn("adjusted_effective_date", mapping_expr[F.col(date_col)])
        # filter out null values in adjusted_effective_date
        # (dates that are not the closest to time periods of interest)
        .filter(F.col("adjusted_effective_date").isNotNull())
    )

    return processed

def _get_time_periods_label_mapping_feature_pipeline(time_period_dict:Dict):
    """Create mapping for each user time period (for train and test) to a label (train or test)
    based on time_period_dict

    Args:
        time_period_dict (Dict): A dictionary containing the time periods for train and test, 
        where train and test are keys, and date objects are values

    Returns:
        Dict: A dictionary containing the time periods as keys and their corresponding labels as values
    """    
    set_type_mapping_for_train = {
        dt.strftime("%Y-%m-%d"): "train" for dt in time_period_dict["train_start_dates"]
    }

    set_type_mapping_for_test = {
        dt.strftime("%Y-%m-%d"): "test" for dt in time_period_dict["test_start_dates"]
    }

    return dict(ChainMap(set_type_mapping_for_test, set_type_mapping_for_train))


def create_set_type_for_train_test_data(
    df: PySparkDataFrame,
    adj_date_col: str,
    time_period_dict: Dict,
) -> PySparkDataFrame:
    """Create a column "set_type" by mapping the time periods to their respective labels. Associates each
    adjusted effective start dates as train or test 

    Args:
      df (PySparkDataFrame): input train_test dataframe
      adj_date_col (str): Name of the column containing adjusted effective start dates of waves of measurement
      time_period_dict (Dict): A dictionary containing the time periods for train and test, 
      where train and test are keys, and date objects are values
    
    Returns:
      processed (PySparkDataFrame): processed datafraome
    """
    # Create a dictionary mapping time periods to their labels
    mapping = _get_time_periods_label_mapping_feature_pipeline(time_period_dict)

    # Create PySpark mapping from dictionary
    mapping_expr = F.create_map([F.lit(x) for item in mapping.items() for x in item])

    # Create set_type column by mapping the adjusted_effective_date column to a label
    processed = df.withColumn("set_type", mapping_expr[F.col(adj_date_col)])

    return processed


def combine_rare_levels(
    df: PySparkDataFrame, categorical_columns: List[str], pct_threshold: float
) -> PySparkDataFrame:
    """
    Replaces levels in a categorical column that appear below a percentage threshold with 'Others'.

    Args:
        df (PySparkDataFrame): The input PySpark DataFrame.
        categorical_column (List[str]): The column name(s) to be processed.
        pct_threshold (float): The minimum frequency a level must have to remain unchanged.
                           Levels with frequency less than this are replaced with 'Others'.

    Returns:
        PySparkDataFrame: A new DataFrame with levels below the threshold replaced by 'Others'.
    """
    processed_df = df

    # Calculate the frequency of each level in the column
    total_count = processed_df.count()
    for col in categorical_columns:
        freq_df = (
            processed_df.groupBy(col)
            .count()
            .withColumn("frequency", 100 * F.col("count") / total_count)
            .filter(F.col("frequency") < pct_threshold)
            .select(col)
        )

        # Collect the rare levels to use in the transformation
        rare_levels = [row[col] for row in freq_df.collect()]
        # Replace rare levels with 'Others'
        processed_df = processed_df.withColumn(
            col, F.when(F.col(col).isin(rare_levels), "Others").otherwise(F.col(col))
        )

    return processed_df


def convert_decimal_to_double(df: PySparkDataFrame) -> PySparkDataFrame:
    """Convert all decimal columns to double type in Pyspark (float64 in pandas). This is to faciliate quicker conversion
    of the spark dataframe to pandas dataframe since conversion of DecimalType columns
    is inefficient and may take a long time

    Args:
      df (PySparkDataFrame): input dataframe
    Returns:
      (PySparkDataFrame): processed datafraome
    """
    # Find all decimal columns in the DataFrame
    decimals_cols = [
        c
        for c in df.columns
        if "Decimal" in str(df.schema[c].dataType)
        and c != config.model.OUTCOME_VARIABLE
    ]

    # Convert all decimal columns to double
    for column in decimals_cols:
        df = df.withColumn(column, df[column].cast(DoubleType()))

    return df


def convert_integer_to_double(df: PySparkDataFrame) -> PySparkDataFrame:
    """
    Convert all integer columns in the given PySpark DataFrame to double,
    as any integers with NA values will be represented as floats in Python.
    During inference, if the integer column contains NA values while there were none
    during training, it could lead to a schema enforcement error.

    Args:
      df (PySparkDataFrame): The input PySpark DataFrame.

    Returns:
      (PySparkDataFrame): A new DataFrame with integer columns converted to double.
    """
    # Find all integer columns in the DataFrame
    int_cols = [
        c
        for c in df.columns
        if "Integer" in str(df.schema[c].dataType)
        and c != config.model.OUTCOME_VARIABLE
    ]

    # Convert all integer columns to doubles
    for column in int_cols:
        df = df.withColumn(column, df[column].cast(DoubleType()))

    return df

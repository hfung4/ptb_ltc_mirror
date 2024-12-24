import pytest

from pyspark.sql import types as T
from pyspark.sql import functions as F
from pyspark.sql import DataFrame as PySparkDataFrame
import datetime
from ptb_ltc.config.core import config
from pyspark.sql import SparkSession
from datetime import date
from decimal import Decimal
from pyspark.sql.types import StructType, StructField, StringType, DateType, DecimalType, DoubleType, IntegerType
from pyspark.sql import Row

from src.ptb_ltc.processing.processing_utils import (
    snake_case_column_names,
    map_column,
    add_social_security_retirement_age,
    transform_YEARSMEMBER_feature,
    add_tot_AUM_per_yrmbr_feature,
    resample_responder_df,
    input_args_time_periods_validation,
    parse_dates,
    check_ordering_of_train_test_periods,
    check_consistent_time_period_cadences,
    get_validated_time_period_feature_pipeline,
    get_validated_time_period_inference_pipeline,
    get_closest_date_mapping,
    create_adjusted_effective_start_date,
    filter_after_a_given_date,
    _get_time_periods_label_mapping_feature_pipeline,
    create_set_type_for_train_test_data,
    combine_rare_levels,
    convert_decimal_to_double,
    convert_integer_to_double
)

from src.ptb_ltc.config.core import config

from pyspark.sql import SparkSession


if config.general.RUN_ON_DATABRICKS_WS:
  spark = SparkSession.builder.getOrCreate()


@pytest.mark.processing_utils
@pytest.mark.parametrize(
  'input_columns, expected_columns',
  [
    # Test with column names that do not require renaming
    (['column_a', 'column_b'], ['column_a', 'column_b']),
    # Test with columns that need underscore renaming
    (['column a', 'column b'], ['column_a', 'column_b']),
    # Test with columns that need underscore renaming and special characters
    (['column #a', 'column ?b'], ['column_a', 'column_b']),
    # Test with columns that have multiple spaces
    (['column     a', 'column    b'], ['column_a', 'column_b']),
    # Test with columns with numbers
    (['column 1', 'column 2'], ['column_1', 'column_2']),
  ]
)
def test_snake_case_column_names(input_columns: list[str], expected_columns: list[str]):
  """
  Test that the snake_case_column_names function correctly transforms column names.
  Take an list of input columns and generate a fake dataframe with no data. Transform that dataframe and compare to expectation.

  Parameters:
    - input_columns (list[str])   : A list of column names to be transformed
    - expected_columns (list[str]): A list of expected column names to be transformed
  """
  # generate a fake schema based on input_columns
  schema = T.StructType([T.StructField(c, T.StringType(), True) for c in input_columns])
  # create the dataframe from the fake schema
  df = spark.createDataFrame(data = [], schema = schema)
  # assert equivalence between the fake dataframe's transformed columns and the expected columns
  assert df.transform(snake_case_column_names).columns == expected_columns



@pytest.fixture
def input_dataframe_for_map_column() -> PySparkDataFrame:
  """
  Create the input DataFrame for the test_map_column function.

  Returns:
  (PySparkDataFrame) - Mock data.
  """
  data = [
    ("James", "Strongly Agree"),
    ("Michael", "Disagree"),
    ("Anna", None)
  ]

  schema = T.StructType([
    T.StructField("name",T.StringType(),True),
    T.StructField("comment",T.StringType(),True)
  ])
  
  return spark.createDataFrame(data=data,schema=schema)

@pytest.mark.processing_utils
def test_map_column(input_dataframe_for_map_column):
  """
  Test that the map_column function correctly maps values from one value to another, according to a dictionary.
 
  Parameters:
    - input_dataframe_for_map_column (PySparkDataFrame) : Mock data.
  """

  # Test data type of input
  assert type(input_dataframe_for_map_column) == PySparkDataFrame, "Input must be PySparkDataFrame"

  # Test a typical map overwritting the test_column (str)
  test_map = {
    "Strongly Agree": "Positive",
    "Disagree": "Negative",
  }

  expected_result = ["Positive", "Negative", None]

  test_column = "comment"

  map_result_list = input_dataframe_for_map_column.transform(lambda _df: map_column(_df, test_column, test_map)).select(test_column).collect()

  assert [x[test_column] for x in map_result_list] == expected_result, "map_column function failed"

  # Test a typical map creating a new column new_column (str)
  test_map = {
    "Strongly Agree": "Positive",
    "Disagree": "Negative",
  }

  expected_result = ["Positive", "Negative", None]

  test_column = "comment"
  new_column = "sentiment"

  map_result_list = input_dataframe_for_map_column.transform(lambda _df: map_column(_df, test_column, test_map, new_column)).select(new_column).collect()

  assert [x[new_column] for x in map_result_list] == expected_result, "map_column function failed"

  # Test a map that does not contain values from the source column
  test_map = {
    # "Strongly Agree": "Positive", # I comment out this value, which is in the test data
    "Disagree": "Negative",
  }

  expected_result = [None, "Negative", None]

  test_column = "comment"
  new_column = "sentiment"

  map_result_list = input_dataframe_for_map_column.transform(lambda _df: map_column(_df, test_column, test_map, new_column)).select(new_column).collect()

  assert [x[new_column] for x in map_result_list] == expected_result, "map_column function failed"



@pytest.fixture
def input_dataframe_for_social_security_retirement_age() -> PySparkDataFrame:
    """
    Create the input DataFrame for the test_add_social_security_retirement_age function.

    Returns:
    (PySparkDataFrame) - Mock data.
    """
    data = [
        ("Sarah", datetime.date(1900, 1, 1)),
        ("David", datetime.date(1941, 1, 1)),
        ("Emily", datetime.date(1942, 1, 1)),
        ("John", datetime.date(1943, 1, 1)),
        ("Olivia", datetime.date(1953, 1, 1)),
        ("Michael", datetime.date(1954, 1, 1)),
        ("Jessica", datetime.date(1955, 1, 1)),
        ("James", datetime.date(1956, 1, 1)),
        ("Laura", datetime.date(1957, 1, 1)),
        ("Robert", datetime.date(1958, 1, 1)),
        ("Anna", datetime.date(1959, 1, 1)),
        ("William", datetime.date(1960, 1, 1)),
        ("Chloe", datetime.date(1960, 1, 1)),
        ("Harry", datetime.date(2000, 1, 1)),
        ("Juan", None)
    ]

    schema = T.StructType([
        T.StructField("name", T.StringType(), True),
        T.StructField("BIRTHDTE", T.DateType(), True)
    ])

    return spark.createDataFrame(data=data, schema=schema)

@pytest.mark.processing_utils
def test_add_social_security_retirement_age(input_dataframe_for_social_security_retirement_age):
  """
  Test that the add_social_security_retirement_age function correctly add's a client's estimated retirement age from their birthdate.
 
  Parameters:
    - input_dataframe_for_social_security_retirement_age (PySparkDataFrame) : Mock data.
  """

  # Test data type of input
  assert type(input_dataframe_for_social_security_retirement_age) == PySparkDataFrame, "Input must be PySparkDataFrame"

  # Test a function on mock data

  retirement_ages = input_dataframe_for_social_security_retirement_age.transform(add_social_security_retirement_age).select('socsec_full_retire_age').collect()

  expectation = [65.0, 65.0, 65.0, 66.0, 66.0, 66.0, 66.1667, 66.3333, 66.5, 66.6667, 66.8333, 67.0, 67.0, 67.0, None]

  assert [x['socsec_full_retire_age'] for x in retirement_ages] == expectation, "add_social_security_retirement_age function failed"



@pytest.fixture
def input_dataframe_for_transform_YEARSMEMBER_feature() -> PySparkDataFrame:

  impute_yearsmember = 20
  max_age = 99

  data = (
    [None, 100, max_age],
    [None, -100, impute_yearsmember],
    [200, 13, max_age],
  )

  schema = T.StructType([
    T.StructField("YEARSMEMBER", T.IntegerType(), True),
    T.StructField("YRSPURCH", T.IntegerType(), True),
    T.StructField("EXPECTED_YEARSMEMBER", T.IntegerType(), True)
  ])

  return spark.createDataFrame(data=data, schema=schema)

@pytest.mark.processing_utils
def test_transform_YEARSMEMBER_feature(input_dataframe_for_transform_YEARSMEMBER_feature):
  """
  Test that the transform_YEARSMEMBER_feature function correctly creates imputes and caps the YEARSMEMBER column.
 
  Parameters:
    - input_dataframe_for_transform_YEARSMEMBER_feature (PySparkDataFrame) : Mock data.
  """

  # Test data type of input
  assert type(input_dataframe_for_transform_YEARSMEMBER_feature) == PySparkDataFrame, "Input must be PySparkDataFrame"

  # Test a function on mock data
  YEARSMEMBERs = [x.YEARSMEMBER for x in input_dataframe_for_transform_YEARSMEMBER_feature.transform(transform_YEARSMEMBER_feature, impute_yearsmember=20, max_age=99).select('YEARSMEMBER').collect()]

  EXPECTED_YEARSMEMBERs = [x.EXPECTED_YEARSMEMBER for x in input_dataframe_for_transform_YEARSMEMBER_feature.select('EXPECTED_YEARSMEMBER').collect()]

  assert YEARSMEMBERs == EXPECTED_YEARSMEMBERs, "transform_YEARSMEMBER_feature function failed"


@pytest.fixture
def input_dataframe_for_add_tot_AUM_per_yrmbr_feature() -> PySparkDataFrame:

  data = (
    [None, 13.0, 0.0],
    [1.0, 13.0, 13.0],
    [0.25, 13.0, 26.0]
  )

  schema = T.StructType([
    T.StructField("YEARSMEMBER", T.DoubleType(), True),
    T.StructField("PROPASSETS", T.DoubleType(), True),
    T.StructField("EXPECTED_tot_AUM_per_yrmbr", T.DoubleType(), True)
  ])
 
  return spark.createDataFrame(data=data, schema=schema)

@pytest.mark.processing_utils
def test_add_tot_AUM_per_yrmbr_feature(input_dataframe_for_add_tot_AUM_per_yrmbr_feature):
  """
  Test that the add_tot_AUM_per_yrmbr_feature function correctly creates imputes and caps the MBR_yrs column.
 
  Parameters:
    - input_dataframe_for_add_tot_AUM_per_yrmbr_feature (PySparkDataFrame) : Mock data.
  """

  # Test data type of input
  assert type(input_dataframe_for_add_tot_AUM_per_yrmbr_feature) == PySparkDataFrame, "Input must be PySparkDataFrame"

  # Test a function on mock data
  tot_AUM_per_yrmbr = [x.tot_AUM_per_yrmbr for x in input_dataframe_for_add_tot_AUM_per_yrmbr_feature.transform(add_tot_AUM_per_yrmbr_feature).select('tot_AUM_per_yrmbr').collect()]

  EXPECTED_tot_AUM_per_yrmbr = [x.EXPECTED_tot_AUM_per_yrmbr for x in input_dataframe_for_add_tot_AUM_per_yrmbr_feature.select('EXPECTED_tot_AUM_per_yrmbr').collect()]

  assert tot_AUM_per_yrmbr == EXPECTED_tot_AUM_per_yrmbr, "add_tot_AUM_per_yrmbr_feature function failed"



@pytest.fixture
def input_dataframe_for_resample_responder_df() -> PySparkDataFrame:

  # Create responder mock data with 10 negatives and 2 positives
  data = ([[0]] * 100) + ([[1]] * 2)

  schema = T.StructType([
    T.StructField("Target", T.IntegerType(), True)
  ])

  return spark.createDataFrame(data=data, schema=schema)

@pytest.mark.processing_utils
def test_resample_responder_df(input_dataframe_for_resample_responder_df):
  """
  Test that the resample_responder_df function correctly resamples input dataframe.
 
  Parameters:
    - input_dataframe_for_resample_responder_df (PySparkDataFrame) : Mock data.
  """

  # Test data type of input
  assert type(input_dataframe_for_resample_responder_df) == PySparkDataFrame, "Input must be PySparkDataFrame"

  # Test a function on mock data

  non_responder_ratio = 4

  resampled_df = input_dataframe_for_resample_responder_df.transform(resample_responder_df, non_responder_ratio).groupBy("Target").count()

  responder_count = resampled_df.filter(resampled_df.Target == 1).collect()[0]['count']
  non_responder_count = resampled_df.filter(resampled_df.Target == 0).collect()[0]['count']

  EXPECTED_responder_count = input_dataframe_for_resample_responder_df.filter(input_dataframe_for_resample_responder_df.Target == 1).count()
  EXPECTED_non_responder_count = non_responder_ratio * EXPECTED_responder_count

  assert (responder_count == EXPECTED_responder_count) and (pytest.approx(non_responder_count, 1) == EXPECTED_non_responder_count), "resample_responder_df function failed"


@pytest.mark.processing_utils
@pytest.mark.parametrize(
  "time_periods, expected_exception",
  [
    # Test case where serving_start_dates is provided along with train_start_dates
    (
      {
        "serving_start_dates": ["2023-01-01"],
        "train_start_dates": ["2022-01-01"],
        "test_start_dates": None,
      },
      ValueError,
    ),
    # Test case where serving_start_dates is provided along with test_start_dates
    (
      {
        "serving_start_dates": ["2023-01-01"],
        "train_start_dates": None,
        "test_start_dates": ["2022-01-01"],
      },
      ValueError,
    ),
    # Test case where neither train_start_dates nor test_start_dates nor serving_start_dates are provided
    (
      {
        "serving_start_dates": None,
        "train_start_dates": None,
        "test_start_dates": None,
      },
      ValueError,
    ),
    # Test case where both train_start_dates and test_start_dates are provided
    (
      {
        "serving_start_dates": None,
        "train_start_dates": ["2022-01-01"],
        "test_start_dates": ["2023-01-01"],
      },
      None,
    ),
    # Test case where only serving_start_dates is provided
    (
      {
        "serving_start_dates": ["2023-01-01"],
        "train_start_dates": None,
        "test_start_dates": None,
      },
      None,
    ),
  ],
)

def test_input_args_time_periods_validation(time_periods, expected_exception):
  """
  Test the input_args_time_periods_validation function with various scenarios.
  """
  if expected_exception:
    with pytest.raises(expected_exception):
      input_args_time_periods_validation(time_periods)
  else:
    input_args_time_periods_validation(time_periods)


# Test for parse_dates function
@pytest.mark.processing_utils
def test_parse_dates():
  # Test with a single valid date string
  date_input = "2023-01-01"
  expected_output = [date(2023, 1, 1)]
  assert parse_dates(date_input) == expected_output

  # Test with a list of valid date strings
  date_input = ["2023-01-01", "2023-02-01", "2023-03-01"]
  expected_output = [date(2023, 1, 1), date(2023, 2, 1), date(2023, 3, 1)]
  assert parse_dates(date_input) == expected_output

  # Test with a list of valid date strings in unsorted order
  date_input = ["2023-03-01", "2023-01-01", "2023-02-01"]
  expected_output = [date(2023, 1, 1), date(2023, 2, 1), date(2023, 3, 1)]
  assert parse_dates(date_input) == expected_output

  # Test with None input
  date_input = None
  expected_output = None
  assert parse_dates(date_input) == expected_output

  # Test with invalid date format
  date_input = "2023/01/01"
  with pytest.raises(ValueError, match="Invalid date format for '2023/01/01'. Expected format is YYYY-MM-DD."):
    parse_dates(date_input)

  # Test with a list containing an invalid date format
  date_input = ["2023-01-01", "2023/02/01"]
  with pytest.raises(ValueError, match="Invalid date format for '2023/02/01'. Expected format is YYYY-MM-DD."):
    parse_dates(date_input)

  # Test with a list containing a mix of valid and invalid date formats
  date_input = ["2023-01-01", "invalid-date"]
  with pytest.raises(ValueError, match="Invalid date format for 'invalid-date'. Expected format is YYYY-MM-DD."):
    parse_dates(date_input)


@pytest.mark.processing_utils
def test_check_ordering_of_train_test_periods():
  """
  Test that the check_ordering_of_train_test_periods function correctly validates the ordering of train and test periods.
  """

  # Test case where test periods do not precede train periods
  train_start_dates = ["2023-01-01", "2023-02-01"]
  test_start_dates = ["2023-03-01", "2023-04-01"]
  try:
    check_ordering_of_train_test_periods(train_start_dates, test_start_dates)
  except ValueError:
    pytest.fail("check_ordering_of_train_test_periods raised ValueError unexpectedly!")

  # Test case where test periods precede train periods
  train_start_dates = ["2023-03-01", "2023-04-01"]
  test_start_dates = ["2023-01-01", "2023-02-01"]
  with pytest.raises(ValueError, match="Time periods for testing cannot precede time periods for training."):
    check_ordering_of_train_test_periods(train_start_dates, test_start_dates)

  # Test case where test periods overlap with train periods
  train_start_dates = ["2023-01-01", "2023-02-01"]
  test_start_dates = ["2023-02-01", "2023-03-01"]
  with pytest.raises(ValueError, match="Time periods for testing cannot precede time periods for training."):
    check_ordering_of_train_test_periods(train_start_dates, test_start_dates)

  # Test case where train and test periods are the same
  train_start_dates = ["2023-01-01", "2023-02-01"]
  test_start_dates = ["2023-01-01", "2023-02-01"]
  with pytest.raises(ValueError, match="Time periods for testing cannot precede time periods for training."):
    check_ordering_of_train_test_periods(train_start_dates, test_start_dates)


@pytest.mark.processing_utils
def test_check_consistent_time_period_cadences():
  # Test with consistent cadences
  period_start_dates = [
    datetime.date(2023, 1, 1),
    datetime.date(2023, 2, 1),
    datetime.date(2023, 3, 1),
    datetime.date(2023, 4, 1)
  ]
  try:
    check_consistent_time_period_cadences(period_start_dates)
  except ValueError:
    pytest.fail("check_consistent_time_period_cadences raised ValueError unexpectedly!")

  # Test with inconsistent cadences
  period_start_dates = [
    datetime.date(2023, 1, 1),
    datetime.date(2023, 2, 1),
    datetime.date(2023, 3, 1),
    datetime.date(2023, 5, 1)
  ]
  with pytest.raises(ValueError, match="Irregular cadence detected: the difference between the longest and shortest period lengths must not exceed 5 days."):
    check_consistent_time_period_cadences(period_start_dates)

  # Test with less than 3 dates (should not raise an error)
  period_start_dates = [
    datetime.date(2023, 1, 1),
    datetime.date(2023, 2, 1)
  ]
  try:
    check_consistent_time_period_cadences(period_start_dates)
  except ValueError:
    pytest.fail("check_consistent_time_period_cadences raised ValueError unexpectedly!") 


@pytest.mark.processing_utils
def test_get_validated_time_period_feature_pipeline():
  """
  Test the get_validated_time_period_feature_pipeline function to ensure it correctly validates and processes time periods.
  """
  # Test with valid input
  time_period_dict = {
    "train_start_dates": ["2023-01-01", "2023-02-01"],
    "test_start_dates": ["2023-03-01", "2023-04-01"]
  }

  expected_output = {
    "train_start_dates": [date(2023, 1, 1), date(2023, 2, 1)],
    "test_start_dates": [date(2023, 3, 1), date(2023, 4, 1)]
  }

  result = get_validated_time_period_feature_pipeline(time_period_dict)
  assert result == expected_output, "get_validated_time_period_feature_pipeline function failed with valid input"

  # Test with invalid input: test period before train period
  time_period_dict = {
    "train_start_dates": ["2023-03-01"],
    "test_start_dates": ["2023-02-01"]
  }

  with pytest.raises(ValueError, match="Time periods for testing cannot precede time periods for training."):
    get_validated_time_period_feature_pipeline(time_period_dict)

  # Test with inconsistent cadences
  time_period_dict = {
    "train_start_dates": ["2023-01-01", "2023-01-10"],
    "test_start_dates": ["2023-02-01", "2023-02-20"]
  }

  with pytest.raises(ValueError, match="Irregular cadence detected: the difference between the longest and shortest period lengths must not exceed 5 days."):
    get_validated_time_period_feature_pipeline(time_period_dict)


@pytest.mark.processing_utils
def test_get_validated_time_period_inference_pipeline():
  # Test with valid serving start dates
  time_period_dict = {
    "serving_start_dates": ["2023-01-01", "2023-02-01", "2023-03-01"]
  }
  
  expected_result = {
    "serving_start_dates": [
      date(2023, 1, 1),
      date(2023, 2, 1),
      date(2023, 3, 1)
    ]
  }
  
  result = get_validated_time_period_inference_pipeline(time_period_dict)
  assert result == expected_result, "Test with valid serving start dates failed"
  
  # Test with invalid date format
  time_period_dict = {
    "serving_start_dates": ["2023-01-01", "invalid-date", "2023-03-01"]
  }
  
  with pytest.raises(ValueError):
    get_validated_time_period_inference_pipeline(time_period_dict)
  
  # Test with inconsistent cadences
  time_period_dict = {
    "serving_start_dates": ["2023-01-01", "2023-01-10", "2023-03-01"]
  }
  
  with pytest.raises(ValueError):
    get_validated_time_period_inference_pipeline(time_period_dict)
  
  # Test with empty serving start dates
  time_period_dict = {
    "serving_start_dates": []
  }
  
  expected_result = {
    "serving_start_dates": []
  }
  
  result = get_validated_time_period_inference_pipeline(time_period_dict)
  assert result == expected_result, "Test with empty serving start dates failed"


@pytest.mark.processing_utils
def test_get_closest_date_mapping():
  # Define test cases
  test_cases = [
    # Test case 1: Simple case with distinct dates
    {
      "unique_effective_dates": [date(2023, 1, 1), date(2023, 2, 1), date(2023, 3, 1)],
      "time_periods": [date(2023, 1, 15), date(2023, 2, 15)],
      "expected": {
        "2023-01-01": "2023-01-15",
        "2023-02-01": "2023-02-15"
      }
    },
    # Test case 2: Multiple time periods mapping to the same effective date
    {
      "unique_effective_dates": [date(2023, 1, 1), date(2023, 2, 1)],
      "time_periods": [date(2023, 1, 15), date(2023, 1, 20)],
      "expected": {
        "2023-01-01": "2023-01-15",
        "2023-02-01": "2023-01-20"
      }
    },
    # Test case 3: No effective dates available
    {
      "unique_effective_dates": [],
      "time_periods": [date(2023, 1, 15)],
      "expected": {}
    },
    # Test case 4: No time periods provided
    {
      "unique_effective_dates": [date(2023, 1, 1)],
      "time_periods": [],
      "expected": {}
    },
    # Test case 5: Effective dates with no close match
    {
      "unique_effective_dates": [date(2023, 1, 1), date(2023, 2, 1)],
      "time_periods": [date(2023, 12, 15)],
      "expected": {
        "2023-02-01": "2023-12-15"
      }
    }
  ]

  for i, test_case in enumerate(test_cases):
    result = get_closest_date_mapping(
      unique_effective_dates=test_case["unique_effective_dates"],
      time_periods=test_case["time_periods"]
    )
    assert result == test_case["expected"], f"Test case {i + 1} failed: {result} != {test_case['expected']}"

@pytest.fixture
def input_dataframe_for_create_adjusted_effective_start_date() -> PySparkDataFrame:
  """
  Create the input DataFrame for the test_create_adjusted_effective_start_date function.

  Returns:
  (PySparkDataFrame) - Mock data.
  """
  data = [
    ("2023-01-01",),
    ("2023-02-01",),
    ("2023-03-01",),
    ("2023-04-01",),
    ("2023-05-01",),
  ]

  schema = T.StructType([
    T.StructField("effective_start_date", T.StringType(), True)
  ])

  return spark.createDataFrame(data=data, schema=schema)


def test_filter_after_a_given_date():
    schema = StructType([
        StructField("id", StringType(), True),
        StructField("date_col", DateType(), True)
        ])
    
    data = [
        Row(id="1", date_col=date(2023, 1, 1)),
        Row(id="2", date_col=date(2023, 2, 1)),
        Row(id="3", date_col=date(2023, 3, 1)),
        Row(id="4", date_col=date(2023, 4, 1)),
        ]
            
    df = spark.createDataFrame(data, schema)
            
    # Test filtering with a date that includes some rows
    result_df = filter_after_a_given_date(df, "date_col", "2023-02-01")
    expected_data = [
        Row(id="2", date_col=date(2023, 2, 1)),
        Row(id="3", date_col=date(2023, 3, 1)),
        Row(id="4", date_col=date(2023, 4, 1)),
    ]
    
    expected_df = spark.createDataFrame(expected_data, schema)
    assert result_df.collect() == expected_df.collect()
            
    # Test filtering with a date that includes all rows
    result_df = filter_after_a_given_date(df, "date_col", "2023-01-01")
    expected_data = [
        Row(id="1", date_col=date(2023, 1, 1)),
        Row(id="2", date_col=date(2023, 2, 1)),
        Row(id="3", date_col=date(2023, 3, 1)),
        Row(id="4", date_col=date(2023, 4, 1)),
    ]
    
    expected_df = spark.createDataFrame(expected_data, schema)
    assert result_df.collect() == expected_df.collect()
            
    # Test filtering with a date that excludes all rows
    result_df = filter_after_a_given_date(df, "date_col", "2023-05-01")
    expected_data = []
    expected_df = spark.createDataFrame(expected_data, schema)
    assert result_df.collect() == expected_df.collect()
            
    # Test with invalid date column type
    with pytest.raises(TypeError):
        filter_after_a_given_date(df.withColumn("date_col", F.col("date_col").cast("string")), "date_col", "2023-01-01")


@pytest.fixture
def input_dataframe_for_create_adjusted_effective_start_date() -> PySparkDataFrame:
  """
  Create the input DataFrame for the test_create_adjusted_effective_start_date function.

  Returns:
  (PySparkDataFrame) - Mock data.
  """
  data = [
    ("2023-01-01",),
    ("2023-02-01",),
    ("2023-03-01",),
    ("2023-04-01",),
    ("2023-05-01",),
  ]

  schema = T.StructType([
    T.StructField("effective_date", T.StringType(), True)
  ])

  return spark.createDataFrame(data=data, schema=schema)


@pytest.mark.processing_utils
def test_create_adjusted_effective_start_date(input_dataframe_for_create_adjusted_effective_start_date):
  """
  Test that the create_adjusted_effective_start_date function correctly creates the adjusted effective start date column.
 
  Parameters:
  - input_dataframe_for_create_adjusted_effective_start_date (PySparkDataFrame) : Mock data.
  """

  # Test data type of input
  assert type(input_dataframe_for_create_adjusted_effective_start_date) == PySparkDataFrame, "Input must be PySparkDataFrame"

  # Define time periods
  time_periods = [
    datetime.date(2023, 1, 15),
    datetime.date(2023, 2, 7),
    datetime.date(2023, 3, 15),
    datetime.date(2023, 3, 28),
    datetime.date(2023, 5, 15)
  ]

  # Expected results
  expected_data = [
    ("2023-01-01", "2023-01-15"),
    ("2023-02-01", "2023-02-07"),
    ("2023-03-01", "2023-03-15"),
    ("2023-04-01", "2023-03-28"),
    ("2023-05-01", "2023-05-15"),
  ]

  expected_schema = T.StructType([
    T.StructField("effective_date", T.StringType(), True),
    T.StructField("adjusted_effective_date", T.StringType(), True)
  ])

  expected_df = spark.createDataFrame(data=expected_data, schema=expected_schema)

  # Apply the function
  result_df = input_dataframe_for_create_adjusted_effective_start_date.transform(
    lambda _df: create_adjusted_effective_start_date(_df, "effective_date", time_periods)
  )

  # Collect results
  result_data = result_df.collect()
  expected_data = expected_df.collect()

  # Assert results
  assert result_data == expected_data, "create_adjusted_effective_start_date function failed"


@pytest.mark.processing_utils
def test_get_time_periods_label_mapping_feature_pipeline():
  """
  Test that the _get_time_periods_label_mapping_feature_pipeline function correctly creates a mapping for each user time period to a label.
  """
  time_period_dict = {
    "train_start_dates": [
      datetime.date(2023, 1, 1),
      datetime.date(2023, 2, 1),
      datetime.date(2023, 3, 1)
    ],
    "test_start_dates": [
      datetime.date(2023, 4, 1),
      datetime.date(2023, 5, 1)
    ]
  }

  expected_mapping = {
    "2023-01-01": "train",
    "2023-02-01": "train",
    "2023-03-01": "train",
    "2023-04-01": "test",
    "2023-05-01": "test"
  }

  result_mapping = _get_time_periods_label_mapping_feature_pipeline(time_period_dict)

  assert result_mapping == expected_mapping, "The function _get_time_periods_label_mapping_feature_pipeline did not return the expected mapping."


@pytest.fixture
def input_dataframe_for_create_set_type_for_train_test_data() -> PySparkDataFrame:
  """
  Create the input DataFrame for the test_create_set_type_for_train_test_data function.

  Returns:
  (PySparkDataFrame) - Mock data.
  """
  data = [
    ("2023-01-01",),
    ("2023-02-01",),
    ("2023-03-01",),
    ("2023-04-01",),
  ]

  schema = T.StructType([
    T.StructField("adjusted_effective_date", T.StringType(), True)
  ])

  return spark.createDataFrame(data=data, schema=schema)

@pytest.mark.processing_utils
def test_create_set_type_for_train_test_data(input_dataframe_for_create_set_type_for_train_test_data):
  """
  Test that the create_set_type_for_train_test_data function correctly maps the time periods to their respective labels.
 
  Parameters:
  - input_dataframe_for_create_set_type_for_train_test_data (PySparkDataFrame) : Mock data.
  """

  # Test data type of input
  assert type(input_dataframe_for_create_set_type_for_train_test_data) == PySparkDataFrame, "Input must be PySparkDataFrame"

  # Define the time period dictionary
  time_period_dict = {
    "train_start_dates": [datetime.date(2023, 1, 1), datetime.date(2023, 2, 1)],
    "test_start_dates": [datetime.date(2023, 3, 1), datetime.date(2023, 4, 1)]
  }

  # Expected result
  expected_data = [
    ("2023-01-01", "train"),
    ("2023-02-01", "train"),
    ("2023-03-01", "test"),
    ("2023-04-01", "test"),
  ]

  expected_schema = T.StructType([
    T.StructField("adjusted_effective_date", T.StringType(), True),
    T.StructField("set_type", T.StringType(), True)
  ])

  expected_df = spark.createDataFrame(data=expected_data, schema=expected_schema)

  # Apply the function
  result_df = input_dataframe_for_create_set_type_for_train_test_data.transform(
    lambda _df: create_set_type_for_train_test_data(_df, "adjusted_effective_date", time_period_dict)
  )

  # Assert the result
  assert result_df.collect() == expected_df.collect(), "create_set_type_for_train_test_data function failed"

    

@pytest.mark.processing_utils
def test_combine_rare_levels():
    schema = StructType([
        StructField("id", StringType(), True),
        StructField("category", StringType(), True)
    ])
    
    data = [
        Row(id="1", category="A"),
        Row(id="2", category="A"),
        Row(id="3", category="B"),
        Row(id="4", category="B"),
        Row(id="5", category="B"),
        Row(id="6", category="C"),
        Row(id="7", category="D"),
        Row(id="8", category="D"),
        Row(id="9", category="D"),
        Row(id="10", category="D"),
    ]
    
    # Mock dataframe                    
    df = spark.createDataFrame(data, schema)
    
    # Apply the function with a threshold of 20%
    result_df = combine_rare_levels(df, ["category"], 20.0)
    
    # 'C' is below the threshold
    expected_data = [
        Row(id="1", category="A"),
        Row(id="2", category="A"),
        Row(id="3", category="B"),
        Row(id="4", category="B"),
        Row(id="5", category="B"),
        Row(id="6", category="Others"),  
        Row(id="7", category="D"),
        Row(id="8", category="D"),
        Row(id="9", category="D"),
        Row(id="10", category="D"),
    ]
    
    expected_df = spark.createDataFrame(expected_data, schema)
    assert result_df.collect() == expected_df.collect()


# Test for convert_decimal_to_double function
def test_convert_decimal_to_double():
    
    # Create a sample DataFrame with DecimalType
    data = [
        (Decimal('1.0'), Decimal('1.1'), Decimal('5.0')),
        (Decimal('2.0'), Decimal('2.2'), Decimal('6.0')),
        (Decimal('3.0'), Decimal('3.3'), Decimal('7.0'))
    ]
    schema = StructType([
        StructField("id", DecimalType(10, 1), True),
        StructField("feature_a", DecimalType(10, 2), True),
        StructField("feature_b", DecimalType(10, 1), True) 
    ])
    
    # Mock dataframe
    df = spark.createDataFrame(data, schema=schema)

    # Convert Decimal columns to Double
    processed_df = convert_decimal_to_double(df)

    # Check if the types of the columns are as expected
    for column in processed_df.columns:
        assert processed_df.schema[column].dataType == DoubleType(), f"{column} was not converted to DoubleType"

    # Check if the values are correctly preserved (float conversion)
    result_data = processed_df.collect()
    expected_data = [(1.0, 1.10, 5.0), (2.0, 2.20, 6.0), (3.0, 3.30, 7.0)]
    for row, expected in zip(result_data, expected_data):
        assert (row[0] == expected[0] and row[1] == expected[1]), f"Row {row} does not match expected {expected}"


# Test for convert_integer_to_double function
def test_convert_integer_to_double():
    
    # Create a sample DataFrame with IntegerType
    data = [(1, 2, 3), (4, None, 6), (7, 8, 9)]
    schema = StructType([
        StructField("id", IntegerType(), True),
        StructField("value", IntegerType(), True),
        StructField(config.model.OUTCOME_VARIABLE, IntegerType(), True)  
    ])
    df = spark.createDataFrame(data, schema=schema)

    # Convert integer columns to Double
    processed_df = convert_integer_to_double(df)

    # Check if the types of the columns are as expected
    for column in processed_df.columns:
        if column != config.model.OUTCOME_VARIABLE:  # Exclude the outcome variable
            assert processed_df.schema[column].dataType == DoubleType(), f"{column} was not converted to DoubleType"
        else:
            assert processed_df.schema[column].dataType == IntegerType(), f"{column} should remain as IntegerType"
    
   
    


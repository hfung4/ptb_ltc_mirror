import great_expectations as gx
from ptb_ltc.config.core import config
from pyspark.sql.functions import DataFrame as PySparkDataFrame
import collections
import yaml

class GreatExpectationsContext:
    """
    A class to perform data validation using Great Expectations with a DataFrame.
    Attributes:
        data_source_name (str): The name of the data source. Enter "spark" or "pandas"
        dataframe (Union[DataFrame, pd.DataFrame]): The dataframe to be used.
        context (DataContext): The Great Expectations context.
        data_source (Datasource): The data source added to the context.
        data_asset (DataAsset): The data asset added to the data source.
        batch_definition (BatchDefinition): The batch definition for the data asset.
        batch_parameters (dict): The parameters for the batch.
        batch (Batch): The batch created from the batch definition.
        expectation_suite (ExpectationSuite): The expectation suite created.
        suite (ExpectationSuite): The suite added to the context.
        dataframe_type (str): The type of the dataframe ('pandas' or 'spark').
    """

    def __init__(self, data_source_name, dataframe, dataframe_type="spark"):
        """
        Initializes the GreatExpectation class with the given data source name and dataframe.

        Args:
            data_source_name (str): The name of the data source.
            dataframe (Union[DataFrame, pd.DataFrame]): The dataframe to be used.
            dataframe_type (str): The type of the dataframe ('pandas' or 'spark').
        """
        self.data_source_name = data_source_name
        self.dataframe = dataframe
        self.dataframe_type = dataframe_type

        # Set up context
        self.context = gx.get_context()

        if dataframe_type == "spark":
            # Connect to Spark data source
            self.data_source = self.context.data_sources.add_spark(
                name=f"{self.data_source_name}_data"
            )

        elif dataframe_type == "pandas":
            # Connect to Pandas data source
            self.data_source = self.context.data_sources.add_pandas(
                name=f"{self.data_source_name}_data"
            )

        else:
            raise ValueError("dataframe_type must be either 'pandas' or 'spark'")

        # Add data source asset
        self.data_asset = self.data_source.add_dataframe_asset(
            name=f"{self.data_source_name}_data_asset"
        )
        # Create batch definition
        self.batch_definition = self.data_asset.add_batch_definition_whole_dataframe(
            f"{self.data_source_name}_data_batch"
        )
        
        # Define batch parameters
        self.batch_parameters = {"dataframe": dataframe}

        # Define batch
        self.batch = self.batch_definition.get_batch(
            batch_parameters=self.batch_parameters
        )

        # Create an empty dict of suites
        # name: suite
        self.suites = {}
        
    def add_suite(self, suite_name: str) -> None:
        """
        Creates an expectation suite and adds it to the context.

        Parameters:
        - suite_name (str) : The name of the suite to create.
        """
        # Create expectation suite
        self.expectation_suite = gx.ExpectationSuite(
            name=suite_name
        )

        # Add suite to context
        suite = self.context.suites.add(self.expectation_suite)

        # Add suite to suites
        self.suites[suite_name] = suite

    def validate_suite(
        self,
        suite_name: str
    ) -> gx.core.expectation_validation_result.ExpectationSuiteValidationResult:
        """
        Validates a suite and returns the validation results.

        Parameters:
        - suite_name (str) : The name of the suite to validate.

        Returns:
        - (gx.core.expectation_validation_result.ExpectationSuiteValidationResult)  : The validation results.
        """

        # Create validation definition
        self.validation_definition = gx.ValidationDefinition(
            data=self.batch_definition,
            suite=self.suites[suite_name],
            name=f"{suite_name}_validation",
        )

        # Run validation
        self.validation_results = self.validation_definition.run(
            batch_parameters=self.batch_parameters
        )

        return self.validation_results



# TODO: instantiate class and define expectations
def validate_gold_data(df:PySparkDataFrame, pipeline_type:str) -> collections.namedtuple:
    """Instantiate the GreatExpectationsSuite, define the expectations, and validate

    Args:
        df (PySparkDataFrame): input gold dataframe to be validated

    Returns:
        pipeline_type[str]: pipeline type, can be feature or inference
    """
    # Create namedtuple to store function outputs
    Result = collections.namedtuple('result', ['schema_check', 'missingness_in_response', 'numerical_ranges', 'categorical_options'])

    with open("ptb_ltc/config/gold_expectations.yaml") as file:
        try:
            gold_expectations = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)

    # create new expectation context using gold dataframe
    gx_context = GreatExpectationsContext("gold", df)

    # add suite
    gx_context.add_suite("schema_check")

    # get expected schema
    expected_schema = gold_expectations['TRAIN_SCHEMA'] if pipeline_type == 'feature' else gold_expectations['SERVING_SCHEMA']

    # add a new expecation to the suite for each column, dtype pair
    for column, expected_type in expected_schema.items():
        gx_context.suites['schema_check'].add_expectation(
            gx.expectations.ExpectColumnValuesToBeOfType(
            column=column,
            type_=expected_type
            )
        )

    # validate the expecation suite
    schema_check_validation = gx_context.validate_suite('schema_check')

    # response variable missingness check 
    # Add this validation for gold data processed in the
    # development environment with the feature pipeline
    if pipeline_type=="feature":
        # create new expectation suite using gold dataframe
        gx_context.add_suite("missingness_check")
        # add a new expecation to the suite for response values to not be null
        gx_context.suites['missingness_check'].add_expectation(
            gx.expectations.ExpectColumnValuesToNotBeNull(
                column=config.model.OUTCOME_VARIABLE
            )
        )
        # validate the expecation suite
        missingness_in_response_validation = gx_context.validate_suite('missingness_check')

    # create new expectation suite using gold dataframe
    gx_context.add_suite('numerical_ranges')

    # get expected numerical ranges
    expected_numerical_ranges = gold_expectations['NUMERICAL_RANGES']
    if pipeline_type=="feature":
        # Add the new key-value pair to the dictionary
        expected_numerical_ranges[config.model.OUTCOME_VARIABLE] = [0, 1]

    # add a new expecation to the suite for each expecation in dictionary
    for column, (min_value, max_value) in expected_numerical_ranges.items():
        gx_context.suites['numerical_ranges'].add_expectation(
            gx.expectations.ExpectColumnValuesToBeBetween(
            column=column,
            min_value=min_value,
            max_value=max_value
            )
        )

    # validate the expecation suite
    numerical_range_validation = gx_context.validate_suite('numerical_ranges')

    # create new expectation suite using gold dataframe
    gx_context.add_suite('categorical_options')

    # get expected categorical options
    expected_categorical_options = gold_expectations['TRAIN_CATEGORICAL_OPTIONS'] if pipeline_type == 'feature' else gold_expectations['SERVING_CATEGORICAL_OPTIONS']
    
    for column, expected_type in expected_categorical_options.items():
        gx_context.suites['categorical_options'].add_expectation(
            gx.expectations.ExpectColumnDistinctValuesToBeInSet(
            column=column,
            value_set=expected_type
            )
        )

    # validate the expecation suite
    categorical_options_validation = gx_context.validate_suite('categorical_options')

    result = Result(
        schema_check_validation.success,
        True if pipeline_type != "feature" else missingness_in_response_validation.success,
        numerical_range_validation.success,
        categorical_options_validation.success
    )

    return result






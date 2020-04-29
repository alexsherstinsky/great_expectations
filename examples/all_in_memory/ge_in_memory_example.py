# Basics imports.
import json
import great_expectations as ge
from great_expectations.core import ExpectationSuite
from great_expectations.datasource.types import BatchKwargs


# GreatExpectations Context.
ge.DataContext.create(".")


# Part I: Expectation Authoring.

context = ge.DataContext()
# The next line should return an empty list.
context.list_datasources()

# Add your data source to context.
data_source_name = "my_data_source_name" # The name of the data source corresponding to your use case.
context.add_datasource(
  data_source_name,
  # Next lines are very important the way they appear below.
  class_name="PandasDatasource",
  data_asset_type={
    "module_name": "great_expectations.dataset",
    "class_name": "PandasDataset"
  }
)

# The next line should return your data source, defined above.
context.list_datasources()


# Provide an expectation suite (this can be read from a filesystem or copied and pasted as a literal, as illustrated in the example below).
expectation_suite_name = 'name_of_your_expectation_suite'
# The expections below are included for illustration purposes.  Please create expectations that are relevant to your use case.
expectation_suite_json_string = r"""
{
  "data_asset_type": "Dataset",
  "expectation_suite_name": "name_of_your_expectation_suite",
  "expectations": [
    {
      "expectation_type": "expect_table_row_count_to_be_between",
      "kwargs": {
        "max_value": 100,
        "min_value": 2
      },
      "meta": {}
    },
    {
      "expectation_type": "expect_table_column_count_to_equal",
      "kwargs": {
        "value": 2
      },
      "meta": {}
    }
  ],
  "meta": {
    "great_expectations.__version__": "0.10.1"
  }
}"""

expectation_suite_dict = json.loads(expectation_suite_json_string)
print(expectation_suite_dict)
expectation_suite_obj = ExpectationSuite(**expectation_suite_dict)
print(expectation_suite_obj)

context.save_expectation_suite(expectation_suite_obj, expectation_suite_name)


# Part II: Data Validation.

# Obtain a DataFrame at a relevant point in your data pipeline.
import pandas as pd
data = {'col1': [1, 2], 'col2': [3, 4]}
df = pd.DataFrame(data=data)
print(df)


# Wrap your DataFrame in a batch for the specified expectation suite.
batch = context.get_batch(
  batch_kwargs=BatchKwargs(datasource=data_source_name, dataset=df),
  expectation_suite_name=expectation_suite_name
)


# Run validation operator
results = context.run_validation_operator("action_list_operator", [batch])
print(results)

results_unique_identifier = list(results['details'].keys())[0]
print(results_unique_identifier)

validation_results = results['details'][results_unique_identifier]['validation_result']
print(validation_results)


# Part IV: Generating Data Documentation.

# Add relevant imports.
from great_expectations.profile.basic_dataset_profiler import BasicDatasetProfiler
from great_expectations.render.renderer import ProfilingResultsPageRenderer, ExpectationSuitePageRenderer
from great_expectations.data_context.util import safe_mmkdir
from great_expectations.render.view import DefaultJinjaPageView

# Generate document model.
document_model = ProfilingResultsPageRenderer().render(validation_results)
print(document_model.to_json_dict())

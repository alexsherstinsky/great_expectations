import pandas as pd

import great_expectations as ge

# Data Context is a GE object that represents your project.
# Your project's great_expectations.yml contains all the config
# options for the project's GE Data Context.
context = ge.data_context.DataContext()

datasource_name = "pandas"  # a datasource configured in your great_expectations.yml

# Tell GE how to fetch the batch of data that should be validated...
df = pd.read_json("data/account_20200310_00000.json.gz")
batch_kwargs = {"dataset": df, "datasource": datasource_name}

# ... or from a Pandas or PySpark DataFrame
# batch_kwargs = {"dataset": "your Pandas or PySpark DataFrame", "datasource": datasource_name}

# Get the batch of data you want to validate.
# Specify the name of the expectation suite that holds the expectations.
expectation_suite_name = "validate_salesforce_full_account_v1"
batch = context.get_batch(batch_kwargs, expectation_suite_name)

# Call a validation operator to validate the batch.
# The operator will evaluate the data against the expectations
# and perform a list of actions, such as saving the validation
# result, updating Data Docs, and firing a notification (e.g., Slack).
results = context.run_validation_operator(
    "action_list_operator", assets_to_validate=[batch], run_id=4
)  # e.g., Airflow run id or some run identifier that your pipeline uses.

if not results["success"]:
    # Decide what your pipeline should do in case the data does not
    # meet your expectations.
    print("done")

# Databricks notebook source
import json
import great_expectations as ge
import great_expectations.jupyter_ux
from great_expectations.datasource.types import BatchKwargs
from datetime import datetime

# COMMAND ----------

ge.DataContext.create(".")

# COMMAND ----------

context = ge.DataContext()
context.list_datasources()

# COMMAND ----------

# add a shell data source to context
context.add_datasource(
  "spark", 
  class_name="SparkDFDatasource",
  data_asset_type={
#     "module_name": "CustomSparkDFDataset",
    "class_name": "CustomSparkDFDataset"
  }
)

# COMMAND ----------

context.list_datasources()

# COMMAND ----------

# simulate custom dataset for custom expectation - this module should be loaded into databricks cluster
from great_expectations.dataset import SparkDFDataset, MetaSparkDFDataset

class CustomSparkDFDataset(SparkDFDataset):

    _data_asset_type = "CustomSparkDFDataset"

    @MetaSparkDFDataset.column_map_expectation
    def expect_column_non_percent_count_0(self, column):
        return column.where(column.Unit!=F.lit("Percent")).count() == 0

# COMMAND ----------

"""
1. read the S3 file into spark dataframe
2. validate it against expectation suite json that's passed in
3. save the result to s3
"""

# COMMAND ----------

# mount S3
import urllib.parse
ACCESS_KEY = "ss"
SECRET_KEY = "xx/G+AKR1f+9uONjIi6kTlY"
ENCODED_SECRET_KEY = urllib.parse.quote(SECRET_KEY, "")
AWS_BUCKET_NAME = "serverless-course-b"
MOUNT_NAME = "tests3"
dbutils.fs.mount("s3n://{}:{}@{}".format(ACCESS_KEY, ENCODED_SECRET_KEY, AWS_BUCKET_NAME), "/mnt/{}".format(MOUNT_NAME))


# COMMAND ----------

# test S3 mount
dbutils.fs.ls("/mnt/tests3")

# COMMAND ----------

# read data into spark df
df = spark.read.option("header", True).csv("/mnt/tests3/5602_metadata.csv")

# COMMAND ----------

display(df)

# COMMAND ----------

# create expectation suite from json
from great_expectations.core import ExpectationSuite
import json

suite = ExpectationSuite(**json.loads(r"""
{
  "data_asset_type": "Dataset",
  "expectation_suite_name": "5602_metadata.warning",
  "expectations": [
    {
      "expectation_type": "expect_table_row_count_to_be_between",
      "kwargs": {
        "max_value": 117,
        "min_value": 96
      },
      "meta": {}
    },
    {
      "expectation_type": "expect_table_column_count_to_equal",
      "kwargs": {
        "value": 10
      },
      "meta": {}
    }
  ],
  "meta": {
    "great_expectations.__version__": "0.10.1"
  }
}"""))



# COMMAND ----------

suite

# COMMAND ----------

# create the suite into context
context.save_expectation_suite(suite, "5602_metadata.warning")

# COMMAND ----------

# get a GE batch
batch = context.get_batch(
  batch_kwargs=BatchKwargs(datasource="spark", dataset=df),
#   batch_kwargs= {
#     "path": "/mnt/tests3/5602_metadata.csv",
#     "datasource": "spark"
#   },
  expectation_suite_name="5602_metadata.warning"
)

# COMMAND ----------



# COMMAND ----------

results = context.run_validation_operator("action_list_operator", [batch])

# COMMAND ----------

key0 = list(results['details'].keys())[0]

# COMMAND ----------

validation_results = results['details'][key0]['validation_result']

# COMMAND ----------

from great_expectations.profile.basic_dataset_profiler import BasicDatasetProfiler
from great_expectations.render.renderer import ProfilingResultsPageRenderer, ExpectationSuitePageRenderer
from great_expectations.data_context.util import safe_mmkdir
from great_expectations.render.view import DefaultJinjaPageView

document_model = ProfilingResultsPageRenderer().render(validation_results)


# COMMAND ----------

# write to s3
dbutils.fs.put("/mnt/tests3/output.html", DefaultJinjaPageView().render(document_model))

# COMMAND ----------



# COMMAND ----------

# MAGIC %fs ls .

# COMMAND ----------



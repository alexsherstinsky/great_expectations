import sqlalchemy as sa
from pyspark.sql.functions import countDistinct

from great_expectations.execution_engine import (
    PandasExecutionEngine,
    SparkDFExecutionEngine,
)
from great_expectations.execution_engine.sqlalchemy_execution_engine import (
    SqlAlchemyExecutionEngine,
)
from great_expectations.expectations.metrics.column_aggregate_metric import (
    ColumnAggregateMetric,
)
from great_expectations.expectations.metrics.column_aggregate_metric import F as F
from great_expectations.expectations.metrics.column_aggregate_metric import (
    column_aggregate_metric,
)
from great_expectations.expectations.metrics.column_aggregate_metric import sa as sa


class ColumnDistinctValues(ColumnAggregateMetric):
    metric_name = "column.aggregate.distinct_values"

    @column_aggregate_metric(engine=PandasExecutionEngine)
    def _pandas(cls, column, **kwargs):
        return column.unique()

    @column_aggregate_metric(engine=SqlAlchemyExecutionEngine)
    def _sqlalchemy(cls, column, **kwargs):
        return sa.func.distinct(column)

    @column_aggregate_metric(engine=SparkDFExecutionEngine)
    def _spark(cls, column, **kwargs):
        return F.array_distinct(column)
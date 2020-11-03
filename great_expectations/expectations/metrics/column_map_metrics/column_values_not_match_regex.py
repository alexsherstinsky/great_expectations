import logging

from great_expectations.execution_engine import PandasExecutionEngine
from great_expectations.execution_engine.sqlalchemy_execution_engine import (
    SqlAlchemyExecutionEngine,
)
from great_expectations.expectations.metrics.column_map_metric import (
    ColumnMapMetricProvider,
    column_map_condition,
)
from great_expectations.expectations.metrics.util import get_dialect_regex_expression

logger = logging.getLogger(__name__)


class ColumnValuesNotMatchRegex(ColumnMapMetricProvider):
    condition_metric_name = "column_values.not_match_regex"
    condition_value_keys = ("regex",)

    @column_map_condition(engine=PandasExecutionEngine)
    def _pandas(cls, column, regex, **kwargs):
        return ~column.astype(str).str.contains(regex)

    @column_map_condition(engine=SqlAlchemyExecutionEngine)
    def _sqlalchemy(cls, column, regex, _dialect, **kwargs):
        regex_expression = get_dialect_regex_expression(
            column, regex, _dialect, positive=False
        )
        if regex_expression is None:
            logger.warning("Regex is not supported for dialect %s" % str(_dialect))
            raise NotImplementedError

        return regex_expression
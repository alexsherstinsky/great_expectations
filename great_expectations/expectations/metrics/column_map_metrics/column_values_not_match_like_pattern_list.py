import logging

import sqlalchemy as sa

from great_expectations.execution_engine import PandasExecutionEngine
from great_expectations.execution_engine.sqlalchemy_execution_engine import (
    SqlAlchemyExecutionEngine,
)
from great_expectations.expectations.metrics.column_map_metric import (
    ColumnMapMetricProvider,
    column_map_condition,
)
from great_expectations.expectations.metrics.util import (
    get_dialect_like_pattern_expression,
    get_dialect_regex_expression,
)

logger = logging.getLogger(__name__)


class ColumnValuesNotMatchLikePatternList(ColumnMapMetricProvider):
    condition_metric_name = "column_values.not_match_like_pattern_list"
    condition_value_keys = ("like_pattern_list", "match_on")

    @column_map_condition(engine=SqlAlchemyExecutionEngine)
    def _sqlalchemy(cls, column, like_pattern_list, _dialect, **kwargs):
        if len(like_pattern_list) == 0:
            raise ValueError(
                "At least one like_pattern must be supplied in the like_pattern_list."
            )

        like_pattern_expression = get_dialect_like_pattern_expression(
            column, _dialect, like_pattern_list[0], positive=False
        )
        if like_pattern_expression is None:
            logger.warning(
                "Like patterns are not supported for dialect %s" % str(_dialect.name)
            )
            raise NotImplementedError

        return sa.and_(
            *[
                get_dialect_like_pattern_expression(
                    column, _dialect, like_pattern, positive=False
                )
                for like_pattern in like_pattern_list
            ]
        )
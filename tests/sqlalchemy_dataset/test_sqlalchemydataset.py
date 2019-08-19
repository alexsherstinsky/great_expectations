try:
    from unittest import mock
except ImportError:
    import mock
import pytest

from great_expectations.dataset import MetaSqlAlchemyDataset, SqlAlchemyDataset
import sqlalchemy as sa
import pandas as pd
import pyhive
import sqlalchemy_redshift
import snowflake
from ..test_utils import get_dataset


@pytest.fixture
def custom_dataset():
    class CustomSqlAlchemyDataset(SqlAlchemyDataset):

        @MetaSqlAlchemyDataset.column_map_expectation
        def expect_column_values_to_equal_2(self, column):
            return (sa.column(column) == 2)

        @MetaSqlAlchemyDataset.column_aggregate_expectation
        def expect_column_mode_to_equal_0(self, column):
            mode_query = sa.select([
                sa.column(column).label('value'),
                sa.func.count(sa.column(column)).label('frequency')
            ]).select_from(self._table).group_by(sa.column(column)).order_by(
                sa.desc(sa.column('frequency')))

            mode = self.engine.execute(mode_query).scalar()
            return {
                "success": mode == 0,
                "result": {
                    "observed_value": mode,
                }
            }

        @MetaSqlAlchemyDataset.column_aggregate_expectation
        def broken_aggregate_expectation(self, column):
            return {
                "not_a_success_value": True,
            }

        @MetaSqlAlchemyDataset.column_aggregate_expectation
        def another_broken_aggregate_expectation(self, column):
            return {
                "success": True,
                "result": {
                    "no_observed_value": True
                }
            }

    engine = sa.create_engine('sqlite://')

    data = pd.DataFrame({
        "c1": [2, 2, 2, 2, 0],
        "c2": [4, 4, 5, None, 7],
        "c3": ["cat", "dog", "fish", "tiger", "elephant"]

    })

    data.to_sql(name='test_data', con=engine, index=False)
    custom_dataset = CustomSqlAlchemyDataset('test_data', engine=engine)

    return custom_dataset


def test_custom_sqlalchemydataset(custom_dataset):
    custom_dataset._initialize_expectations()
    custom_dataset.set_default_expectation_argument(
        "result_format", {"result_format": "COMPLETE"})

    result = custom_dataset.expect_column_values_to_equal_2('c1')
    assert result['success'] == False
    assert result['result']['unexpected_list'] == [0]

    result = custom_dataset.expect_column_mode_to_equal_0('c2')
    assert result['success'] == False
    assert result['result']['observed_value'] == 4


def test_broken_decorator_errors(custom_dataset):
    custom_dataset._initialize_expectations()
    custom_dataset.set_default_expectation_argument(
        "result_format", {"result_format": "COMPLETE"})

    with pytest.raises(ValueError) as err:
        custom_dataset.broken_aggregate_expectation('c1')
        assert "Column aggregate expectation failed to return required information: success" in str(
            err)

    with pytest.raises(ValueError) as err:
        custom_dataset.another_broken_aggregate_expectation('c1')
        assert "Column aggregate expectation failed to return required information: observed_value" in str(
            err)


def test_missing_engine_error():
    with pytest.raises(ValueError) as err:
        SqlAlchemyDataset('test_engine', schema='example')
        assert "Engine or connection_string must be provided." in str(err)


def test_only_connection_string():
    SqlAlchemyDataset('test_engine', connection_string='sqlite://')


def test_schema_custom_sql_error():
    engine = sa.create_engine('sqlite://')

    with pytest.raises(ValueError) as err:
        SqlAlchemyDataset('test_schema_custom', schema='example', engine=engine,
                          custom_sql='SELECT * FROM example.fake')
        assert "Cannot specify both schema and custom_sql." in str(err)


def test_sqlalchemydataset_raises_error_on_missing_table_name():
    with pytest.raises(ValueError) as ve:
        SqlAlchemyDataset(table_name=None, engine="foo", connection_string='bar')
    assert str(ve.value) == "No table_name provided."


def test_sqlalchemydataset_builds_guid_for_table_name_on_custom_sql():
    engine = sa.create_engine('sqlite://')

    with mock.patch(
            "great_expectations.dataset.sqlalchemy_dataset.generate_random_temporary_table_name"
    ) as mock_table_name_gen:
        mock_table_name_gen.return_value = "a_guid_with_expected_format"

        dataset = SqlAlchemyDataset(engine=engine, custom_sql="select 1")

        # table name comes from generate_random_temporary_table_name
        assert dataset._table.name == "a_guid_with_expected_format"


def test_sqlalchemydataset_with_custom_sql():
    engine = sa.create_engine('sqlite://')

    data = pd.DataFrame({
        "name": ["Frank", "Steve", "Jane", "Frank", "Michael"],
        "age": [16, 21, 38, 22, 10],
        "pet": ["fish", "python", "cat", "python", "frog"]
    })

    data.to_sql(name='test_sql_data', con=engine, index=False)

    custom_sql = "SELECT name, pet FROM test_sql_data WHERE age > 12"
    custom_sql_dataset = SqlAlchemyDataset(engine=engine, custom_sql=custom_sql)

    custom_sql_dataset._initialize_expectations()
    custom_sql_dataset.set_default_expectation_argument(
        "result_format", {"result_format": "COMPLETE"})

    result = custom_sql_dataset.expect_column_values_to_be_in_set(
        "pet", ["fish", "cat", "python"])
    assert result['success'] == True

    result = custom_sql_dataset.expect_column_to_exist("age")
    assert result['success'] == False


def test_column_fallback():
    engine = sa.create_engine('sqlite://')

    data = pd.DataFrame({
        "name": ["Frank", "Steve", "Jane", "Frank", "Michael"],
        "age": [16, 21, 38, 22, 10],
        "pet": ["fish", "python", "cat", "python", "frog"]
    })

    data.to_sql(name='test_sql_data', con=engine, index=False)
    dataset = SqlAlchemyDataset('test_sql_data', engine=engine)
    fallback_dataset = SqlAlchemyDataset('test_sql_data', engine=engine)
    # override columns attribute to test fallback
    fallback_dataset.columns = fallback_dataset.column_reflection_fallback()

    # check that the results are the same for a few expectations
    assert (dataset.expect_column_to_exist('age') == 
            fallback_dataset.expect_column_to_exist('age'))

    assert (dataset.expect_column_mean_to_be_between('age', min_value=10) == 
            fallback_dataset.expect_column_mean_to_be_between('age', min_value=10))

    # Test a failing expectation
    assert (dataset.expect_table_row_count_to_equal(value=3) == 
            fallback_dataset.expect_table_row_count_to_equal(value=3))


DIALECTS = (
    # dialect name, url to create engine from, loaded dialect
    ("hive", "hive://host:1234", pyhive.sqlalchemy_hive),
    ("postgresql", "postgresql://host:1234", sa.dialects.postgresql),
    ("sqlite", "sqlite://", sa.dialects.sqlite),
    ("snowflake", "snowflake://host:1234", snowflake.sqlalchemy.snowdialect),
    ("redshift", "redshift://host:1234", sqlalchemy_redshift.dialect)
)


@pytest.mark.parametrize("dialect_name,url,dialect_type", DIALECTS)
def test_sqlalchemydataset_dialect_based_on_engine(dialect_name, url, dialect_type):
    """Tests the dialect name and type are correct based on the engine
    provided to SqlAlchemyDataset
    """

    # generate a SQLAlchemy Inspector that does nothing and returns no columns
    class DummyInspector:
        def __init__(self):
            pass

        def get_columns(self, table_name, schema):
            return []

    with mock.patch('great_expectations.dataset.sqlalchemy_dataset.reflection.Inspector.from_engine') as mock_fe:

        # Inspector creation from engine is patched to return a dummy Inspector
        mock_fe.return_value = DummyInspector()

        # we are able to initialize an SqlAlchemyDataset with an engine without needing to connect
        dataset = SqlAlchemyDataset('test_sql_data', engine=sa.create_engine(url))

        # check dialect name is read properly
        assert dataset.get_dialect_name() == dialect_name

        # check our dialect in use is the desired type
        assert isinstance(dataset.dialect, type(dialect_type))


@pytest.fixture
def unexpected_count_df():
    return  get_dataset("sqlite", {"a": [1, 2, 1, 2, 1, 2, 1, 2, 1, 2]})


def test_sqlalchemy_dataset_unexpected_count_calculations(unexpected_count_df):
    # The partial_unexpected_count should not affect overall success calculations, but should limit number of returned rows
    res1 = unexpected_count_df.expect_column_values_to_be_in_set("a", value_set=[1], result_format={"result_format": "BASIC", "partial_unexpected_count": 2})
    res2 = unexpected_count_df.expect_column_values_to_be_in_set("a", value_set=[1], result_format={"result_format": "BASIC", "partial_unexpected_count": 10})

    assert res1["result"]["unexpected_count"] == 5
    assert res2["result"]["unexpected_count"] == 5
    # Note difference here
    assert len(res1["result"]["partial_unexpected_list"]) == 2
    assert len(res2["result"]["partial_unexpected_list"]) == 5


    # However, the "COMPLETE" result format ignores the limit.
    res1 = unexpected_count_df.expect_column_values_to_be_in_set("a", value_set=[1], result_format={"result_format": "COMPLETE", "partial_unexpected_count": 2})
    res2 = unexpected_count_df.expect_column_values_to_be_in_set("a", value_set=[1], result_format={"result_format": "COMPLETE", "partial_unexpected_count": 10})

    assert res1["result"]["unexpected_count"] == 5
    assert res2["result"]["unexpected_count"] == 5


def test_result_format_warning(unexpected_count_df):
    with pytest.warns(UserWarning, match=r'Setting result format to COMPLETE for a SqlAlchemyDataset can be dangerous'):
        unexpected_count_df.expect_column_values_to_be_in_set("a", value_set=[1], result_format={"result_format": "COMPLETE", "partial_unexpected_count": 2})

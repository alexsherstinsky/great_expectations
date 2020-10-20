import pytest
import os
import time
import shutil
import yaml
from typing import Union, List
import pandas as pd


from great_expectations.execution_environment import ExecutionEnvironment
from great_expectations.execution_environment.data_connector.data_connector import DataConnector
from great_expectations.execution_environment.data_connector.partitioner.partition import Partition
from great_expectations.core.batch import (
    Batch,
    BatchRequest,
    BatchDefinition,
    PartitionDefinition,
)
from great_expectations.data_context.util import (
    file_relative_path,
    instantiate_class_from_config,
)
from tests.test_utils import (
    execution_environment_files_data_connector_regex_partitioner_config,
    create_files_for_regex_partitioner,
    create_files_in_directory,
)

@pytest.fixture
def basic_execution_environment(tmp_path_factory):
    base_directory = str(tmp_path_factory.mktemp("basic_execution_environment__filesystem_data_connector"))

    basic_execution_environment = instantiate_class_from_config(
        yaml.load(
            f"""
class_name: ExecutionEnvironment

execution_engine:
    class_name: PandasExecutionEngine

data_connectors:
    my_filesystem_data_connector:
        class_name: FilesDataConnector
        base_directory: {base_directory}
        glob_directive: '*.csv'
            
        default_partitioner: my_regex_partitioner
        partitioners:
            my_regex_partitioner:
                class_name: RegexPartitioner
                regex:
                    group_names:
                        - letter
                        - number
                    pattern: (.+)(\\d+)\\.csv
            """, Loader=yaml.FullLoader),
        runtime_environment={
            "name": "my_execution_environment"
        },
        config_defaults={
            "module_name": "great_expectations.execution_environment"
        }
    )
    return basic_execution_environment


def test_some_very_basic_stuff(basic_execution_environment):
    my_data_connector = basic_execution_environment.get_data_connector("my_filesystem_data_connector")
    create_files_in_directory(
        my_data_connector.base_directory,
        ["A1.csv", "A2.csv", "A3.csv", "B1.csv", "B2.csv", "B3.csv"],
    )

    assert len(basic_execution_environment.get_available_partitions("my_filesystem_data_connector")) == 6

    batch = basic_execution_environment.get_batch_from_batch_definition(BatchDefinition(
        "my_execution_environment",
        "my_filesystem_data_connector",
        "B1",
        partition_definition=PartitionDefinition({
            "letter": "B",
            "number": "1",
        })
    ))
    assert batch.batch_request is None
    assert type(batch.data) == pd.DataFrame
    assert batch.batch_definition == BatchDefinition(
        "my_execution_environment",
        "my_filesystem_data_connector",
        "B1",
        partition_definition=PartitionDefinition({
            "letter": "B",
            "number": "1",
        })
    )

    batch_list = basic_execution_environment.get_batch_list_from_batch_request(BatchRequest(
        execution_environment_name="my_execution_environment",
        data_connector_name="my_filesystem_data_connector",
        data_asset_name="B1",
        partition_request={
            "letter": "B",
            "number": "1",
        }
    ))
    assert len(batch_list) == 1
    assert type(batch_list[0].data) == pd.DataFrame

    my_df = pd.DataFrame({"x": range(10), "y": range(10)})
    batch = basic_execution_environment.get_batch_from_batch_definition(BatchDefinition(
        "my_execution_environment",
        "_pipeline",
        "_pipeline",
        partition_definition=PartitionDefinition({
            "some_random_id": 1
        })
    ), in_memory_dataset=my_df)
    assert batch.batch_request is None


def test_get_batch_list_from_batch_request(basic_execution_environment):
    execution_environment_name: str = "test_execution_environment"
    data_connector_name: str = "test_filesystem_data_connector"
    data_asset_name: str = "Titanic"
    # base_dir_path = str(tmp_path_factory.mktemp("project_dirs"))
    # project_dir_path = os.path.join(base_dir_path, "project_path")
    # titanic_csv_destination_file_path: str = str(os.path.join(project_dir_path, "data/titanic/Titanic.csv"))
    
    batch_request: dict = {
        "execution_environment_name": execution_environment_name,
        "data_connector_name": data_connector_name,
        "data_asset_name": data_asset_name,
        "partition_request": {
            "key": "Titanic.csv"
        },
        # "limit": None,
        # "batch_spec_passthrough": {
        #     "path": titanic_csv_destination_file_path,
        #     "reader_method": "read_csv",
        #     "reader_options": None,
        #     "limit": 2000
        # }
    }
    batch_request: BatchRequest = BatchRequest(**batch_request)
    batch: Batch = basic_execution_environment.get_batch_list_from_batch_request(
        batch_request=batch_request
    )
    # noinspection PyStatementEffect
    batch
    # TODO: <Alex>Add some meaningful assertions to this tet.</Alex>


def test_get_available_partitions_with_opportunistic_partition_caching(tmp_path_factory):
    base_dir_path = str(tmp_path_factory.mktemp("project_dirs"))
    project_dir_path = os.path.join(base_dir_path, "project_path")
    os.mkdir(project_dir_path)

    os.makedirs(os.path.join(project_dir_path, "data"), exist_ok=True)
    os.makedirs(os.path.join(project_dir_path, "data/test_files"), exist_ok=True)

    default_base_directory: str = "data/test_files"
    data_asset_base_directory: Union[str, None] = None

    base_directory_names: list = [default_base_directory, data_asset_base_directory]
    create_files_for_regex_partitioner(root_directory_path=project_dir_path, directory_paths=base_directory_names)

    execution_environment_name: str = "test_execution_environment"
    execution_environment_config: dict = execution_environment_files_data_connector_regex_partitioner_config(
        use_group_names=False,
        use_sorters=False,
        default_base_directory=default_base_directory,
        data_asset_base_directory=data_asset_base_directory
    )[execution_environment_name]
    execution_environment_config.pop("class_name")
    execution_environment: ExecutionEnvironment = ExecutionEnvironment(
        name=execution_environment_name,
        **execution_environment_config,
        data_context_root_directory=project_dir_path
    )

    data_connector_name: str = "test_filesystem_data_connector"

    available_partitions: List[Partition] = execution_environment.get_available_partitions(
        data_connector_name=data_connector_name,
        data_asset_name=None,
        partition_query={
            "custom_filter": None,
            "partition_name": None,
            "partition_definition": None,
            "partition_index": None,
            "limit": None,
        },
        in_memory_dataset=None,
        runtime_parameters=None,
        repartition=False
    )

    assert len(available_partitions) == 10


def test_get_available_partitions_without_partition_caching(tmp_path_factory):
    pass


def test_data_asset_names_as_keys_of_partition_cache(tmp_path_factory):
    base_dir_path = str(tmp_path_factory.mktemp("project_dirs"))
    project_dir_path = os.path.join(base_dir_path, "project_path")
    os.mkdir(project_dir_path)

    os.makedirs(os.path.join(project_dir_path, "data"), exist_ok=True)
    os.makedirs(os.path.join(project_dir_path, "data/test_files"), exist_ok=True)

    default_base_directory: str = "data/test_files"
    data_asset_base_directory: Union[str, None] = None

    base_directory_names: list = [default_base_directory, data_asset_base_directory]
    create_files_for_regex_partitioner(root_directory_path=project_dir_path, directory_paths=base_directory_names)

    execution_environment_name: str = "test_execution_environment"
    execution_environment_config: dict = execution_environment_files_data_connector_regex_partitioner_config(
        use_group_names=False,
        use_sorters=False,
        default_base_directory=default_base_directory,
        data_asset_base_directory=data_asset_base_directory
    )[execution_environment_name]
    execution_environment_config.pop("class_name")

    execution_environment: ExecutionEnvironment = ExecutionEnvironment(
        name=execution_environment_name,
        **execution_environment_config,
        data_context_root_directory=project_dir_path
    )

    data_connector_names: list = [
        data_connector["name"] for data_connector in execution_environment.list_data_connectors()
    ]

    test_df: pd.DataFrame = pd.DataFrame(data={"col1": [1, 2], "col2": [3, 4]})

    for data_connector_name in data_connector_names:
        data_connector: DataConnector = execution_environment.get_data_connector(name=data_connector_name)
        data_asset_names: list = [None]
        if data_connector.assets:
            for data_asset_name in data_connector.assets.keys():
                data_asset_names.append(data_asset_name)
        for data_asset_name in data_asset_names:
            runtime_parameters: dict = {"custom_key_0": int(time.time())}
            # noinspection PyUnusedLocal
            available_partitions: List[Partition] = execution_environment.get_available_partitions(
                data_connector_name=data_connector_name,
                data_asset_name=data_asset_name,
                partition_query={
                    "custom_filter": None,
                    "partition_name": None,
                    "partition_definition": None,
                    "partition_index": None,
                    "limit": None,
                },
                in_memory_dataset= test_df,
                runtime_parameters=runtime_parameters,
                repartition=False
            )

    expected_cached_data_asset_names: dict = {
        "test_pipeline_data_connector": ["test_asset_1"],
        "test_filesystem_data_connector": [
            "test_asset_0",
            "abe_20200809_1040", "james_20200811_1009", "eugene_20201129_1900",
            "will_20200809_1002", "eugene_20200809_1500", "james_20200810_1003",
            "alex_20200819_1300", "james_20200713_1567", "will_20200810_1001", "alex_20200809_1000"
        ]
    }

    for data_connector_name in data_connector_names:
        data_connector: DataConnector = execution_environment.get_data_connector(name=data_connector_name)
        # noinspection PyProtectedMember
        cached_data_asset_names: list = [
            data_asset_name
            for data_asset_name in list(data_connector._partitions_cache.keys())
            if data_asset_name != data_connector.DEFAULT_DATA_ASSET_NAME
        ]
        assert set(cached_data_asset_names) == set(expected_cached_data_asset_names[data_connector_name])


def test_get_available_data_asset_names_with_opportunistic_partition_caching(tmp_path_factory):
    base_dir_path = str(tmp_path_factory.mktemp("project_dirs"))
    project_dir_path = os.path.join(base_dir_path, "project_path")
    os.mkdir(project_dir_path)

    os.makedirs(os.path.join(project_dir_path, "data"), exist_ok=True)
    os.makedirs(os.path.join(project_dir_path, "data/test_files"), exist_ok=True)

    default_base_directory: str = "data/test_files"
    data_asset_base_directory: Union[str, None] = None

    base_directory_names: list = [default_base_directory, data_asset_base_directory]
    create_files_for_regex_partitioner(root_directory_path=project_dir_path, directory_paths=base_directory_names)

    execution_environment_name: str = "test_execution_environment"
    execution_environment_config: dict = execution_environment_files_data_connector_regex_partitioner_config(
        use_group_names=False,
        use_sorters=False,
        default_base_directory=default_base_directory,
        data_asset_base_directory=data_asset_base_directory
    )[execution_environment_name]
    execution_environment_config.pop("class_name")
    execution_environment: ExecutionEnvironment = ExecutionEnvironment(
        name=execution_environment_name,
        **execution_environment_config,
        data_context_root_directory=project_dir_path
    )
    data_connector_names: Union[List, str, None] = None

    expected_data_asset_names_dict: dict = {
        "test_pipeline_data_connector": ["test_asset_1"],
        "test_filesystem_data_connector": [
            "test_asset_0",
            "abe_20200809_1040", "james_20200811_1009", "eugene_20201129_1900",
            "will_20200809_1002", "eugene_20200809_1500", "james_20200810_1003",
            "alex_20200819_1300", "james_20200713_1567", "will_20200810_1001", "alex_20200809_1000"
        ]
    }

    available_data_asset_names_dict: dict = execution_environment.get_available_data_asset_names(
        data_connector_names=data_connector_names,
        clear_cache=False
    )

    assert set(available_data_asset_names_dict.keys()) == set(expected_data_asset_names_dict.keys())
    for connector_name, asset_list in available_data_asset_names_dict.items():
        assert set(asset_list) == set(expected_data_asset_names_dict[connector_name])

    data_connector_names = ["test_filesystem_data_connector", "test_pipeline_data_connector"]

    expected_data_asset_names_dict: dict = {
        "test_pipeline_data_connector": ["test_asset_1"],
        "test_filesystem_data_connector": [
            "test_asset_0",
            "abe_20200809_1040", "james_20200811_1009", "eugene_20201129_1900",
            "will_20200809_1002", "eugene_20200809_1500", "james_20200810_1003",
            "alex_20200819_1300", "james_20200713_1567", "will_20200810_1001", "alex_20200809_1000"
        ]
    }

    available_data_asset_names_dict: dict = execution_environment.get_available_data_asset_names(
        data_connector_names=data_connector_names,
        clear_cache=False
    )

    assert set(available_data_asset_names_dict.keys()) == set(expected_data_asset_names_dict.keys())
    for connector_name, asset_list in available_data_asset_names_dict.items():
        assert set(asset_list) == set(expected_data_asset_names_dict[connector_name])

    data_connector_names = ["test_filesystem_data_connector"]

    expected_data_asset_names_dict: dict = {
        'test_filesystem_data_connector': [
            'test_asset_0',
            'abe_20200809_1040', 'james_20200811_1009', 'eugene_20201129_1900',
            'will_20200809_1002', 'eugene_20200809_1500', 'james_20200810_1003',
            'alex_20200819_1300', 'james_20200713_1567', 'will_20200810_1001', 'alex_20200809_1000'
        ]
    }

    available_data_asset_names_dict: dict = execution_environment.get_available_data_asset_names(
        data_connector_names=data_connector_names,
        clear_cache=False
    )

    assert set(available_data_asset_names_dict.keys()) == set(expected_data_asset_names_dict.keys())
    for connector_name, asset_list in available_data_asset_names_dict.items():
        assert set(asset_list) == set(expected_data_asset_names_dict[connector_name])

    data_connector_names = "test_filesystem_data_connector"

    expected_data_asset_names_dict: dict = {
        'test_filesystem_data_connector': [
            'test_asset_0',
            'abe_20200809_1040', 'james_20200811_1009', 'eugene_20201129_1900',
            'will_20200809_1002', 'eugene_20200809_1500', 'james_20200810_1003',
            'alex_20200819_1300', 'james_20200713_1567', 'will_20200810_1001', 'alex_20200809_1000'
        ]
    }

    available_data_asset_names_dict: dict = execution_environment.get_available_data_asset_names(
        data_connector_names=data_connector_names,
        clear_cache=False
    )

    assert set(available_data_asset_names_dict.keys()) == set(expected_data_asset_names_dict.keys())
    for connector_name, asset_list in available_data_asset_names_dict.items():
        assert set(asset_list) == set(expected_data_asset_names_dict[connector_name])

    data_connector_names = ["test_pipeline_data_connector"]

    expected_data_asset_names_dict: dict = {
        'test_pipeline_data_connector': ['test_asset_1']
    }

    available_data_asset_names_dict: dict = execution_environment.get_available_data_asset_names(
        data_connector_names=data_connector_names,
        clear_cache=False
    )

    assert set(available_data_asset_names_dict.keys()) == set(expected_data_asset_names_dict.keys())
    for connector_name, asset_list in available_data_asset_names_dict.items():
        assert set(asset_list) == set(expected_data_asset_names_dict[connector_name])


def test_get_available_data_asset_names_without_partition_caching(tmp_path_factory):
    base_dir_path = str(tmp_path_factory.mktemp("project_dirs"))
    project_dir_path = os.path.join(base_dir_path, "project_path")
    os.mkdir(project_dir_path)

    os.makedirs(os.path.join(project_dir_path, "data"), exist_ok=True)
    os.makedirs(os.path.join(project_dir_path, "data/test_files"), exist_ok=True)

    default_base_directory: str = "data/test_files"
    data_asset_base_directory: Union[str, None] = None

    base_directory_names: list = [default_base_directory, data_asset_base_directory]
    create_files_for_regex_partitioner(root_directory_path=project_dir_path, directory_paths=base_directory_names)

    execution_environment_name: str = "test_execution_environment"
    execution_environment_config: dict = execution_environment_files_data_connector_regex_partitioner_config(
        use_group_names=False,
        use_sorters=False,
        default_base_directory=default_base_directory,
        data_asset_base_directory=data_asset_base_directory
    )[execution_environment_name]
    execution_environment_config.pop("class_name")
    execution_environment: ExecutionEnvironment = ExecutionEnvironment(
        name=execution_environment_name,
        **execution_environment_config,
        data_context_root_directory=project_dir_path
    )
    data_connector_names: Union[List, str, None] = None

    expected_data_asset_names_dict: dict = {
        "test_pipeline_data_connector": ["test_asset_1"],
        "test_filesystem_data_connector": [
            "test_asset_0",
            "abe_20200809_1040", "james_20200811_1009", "eugene_20201129_1900",
            "will_20200809_1002", "eugene_20200809_1500", "james_20200810_1003",
            "alex_20200819_1300", "james_20200713_1567", "will_20200810_1001", "alex_20200809_1000"
        ]
    }

    available_data_asset_names_dict: dict = execution_environment.get_available_data_asset_names(
        data_connector_names=data_connector_names,
        clear_cache=True
    )

    assert set(available_data_asset_names_dict.keys()) == set(expected_data_asset_names_dict.keys())
    for connector_name, asset_list in available_data_asset_names_dict.items():
        assert set(asset_list) == set(expected_data_asset_names_dict[connector_name])

    data_connector_names = ["test_filesystem_data_connector", "test_pipeline_data_connector"]

    expected_data_asset_names_dict: dict = {
        "test_pipeline_data_connector": ["test_asset_1"],
        "test_filesystem_data_connector": [
            "test_asset_0",
            "abe_20200809_1040", "james_20200811_1009", "eugene_20201129_1900",
            "will_20200809_1002", "eugene_20200809_1500", "james_20200810_1003",
            "alex_20200819_1300", "james_20200713_1567", "will_20200810_1001", "alex_20200809_1000"
        ]
    }

    available_data_asset_names_dict: dict = execution_environment.get_available_data_asset_names(
        data_connector_names=data_connector_names,
        clear_cache=True
    )

    assert set(available_data_asset_names_dict.keys()) == set(expected_data_asset_names_dict.keys())
    for connector_name, asset_list in available_data_asset_names_dict.items():
        assert set(asset_list) == set(expected_data_asset_names_dict[connector_name])

    data_connector_names = ["test_filesystem_data_connector"]

    expected_data_asset_names_dict: dict = {
        'test_filesystem_data_connector': [
            'test_asset_0',
            'abe_20200809_1040', 'james_20200811_1009', 'eugene_20201129_1900',
            'will_20200809_1002', 'eugene_20200809_1500', 'james_20200810_1003',
            'alex_20200819_1300', 'james_20200713_1567', 'will_20200810_1001', 'alex_20200809_1000'
        ]
    }

    available_data_asset_names_dict: dict = execution_environment.get_available_data_asset_names(
        data_connector_names=data_connector_names,
        clear_cache=True
    )

    assert set(available_data_asset_names_dict.keys()) == set(expected_data_asset_names_dict.keys())
    for connector_name, asset_list in available_data_asset_names_dict.items():
        assert set(asset_list) == set(expected_data_asset_names_dict[connector_name])

    data_connector_names = "test_filesystem_data_connector"

    expected_data_asset_names_dict: dict = {
        'test_filesystem_data_connector': [
            'test_asset_0',
            'abe_20200809_1040', 'james_20200811_1009', 'eugene_20201129_1900',
            'will_20200809_1002', 'eugene_20200809_1500', 'james_20200810_1003',
            'alex_20200819_1300', 'james_20200713_1567', 'will_20200810_1001', 'alex_20200809_1000'
        ]
    }

    available_data_asset_names_dict: dict = execution_environment.get_available_data_asset_names(
        data_connector_names=data_connector_names,
        clear_cache=True
    )

    assert set(available_data_asset_names_dict.keys()) == set(expected_data_asset_names_dict.keys())
    for connector_name, asset_list in available_data_asset_names_dict.items():
        assert set(asset_list) == set(expected_data_asset_names_dict[connector_name])

    data_connector_names = ["test_pipeline_data_connector"]

    expected_data_asset_names_dict: dict = {
        'test_pipeline_data_connector': ['test_asset_1']
    }

    available_data_asset_names_dict: dict = execution_environment.get_available_data_asset_names(
        data_connector_names=data_connector_names,
        clear_cache=True
    )

    assert set(available_data_asset_names_dict.keys()) == set(expected_data_asset_names_dict.keys())
    for connector_name, asset_list in available_data_asset_names_dict.items():
        assert set(asset_list) == set(expected_data_asset_names_dict[connector_name])


def test_get_batch_using_batch_spec_passthrough(tmp_path_factory):
    base_dir_path = str(tmp_path_factory.mktemp("project_dirs"))
    project_dir_path = os.path.join(base_dir_path, "project_path")
    os.mkdir(project_dir_path)

    os.makedirs(os.path.join(project_dir_path, "data"), exist_ok=True)
    os.makedirs(os.path.join(project_dir_path, "data/titanic"), exist_ok=True)

    titanic_csv_source_file_path: str = file_relative_path(__file__, "../test_sets/Titanic.csv")
    titanic_csv_destination_file_path: str = str(os.path.join(project_dir_path, "data/titanic/Titanic.csv"))
    shutil.copy(titanic_csv_source_file_path, titanic_csv_destination_file_path)

    default_base_directory: str = "data/titanic"
    data_asset_base_directory: Union[str, None] = None

    execution_environment_name: str = "test_execution_environment"
    execution_environment_config: dict = execution_environment_files_data_connector_regex_partitioner_config(
        use_group_names=False,
        use_sorters=False,
        default_base_directory=default_base_directory,
        data_asset_base_directory=data_asset_base_directory
    )[execution_environment_name]
    execution_environment_config.pop("class_name")
    execution_environment: ExecutionEnvironment = ExecutionEnvironment(
        name=execution_environment_name,
        **execution_environment_config,
        data_context_root_directory=project_dir_path
    )
    data_connector_name: str = "test_filesystem_data_connector"
    data_asset_name: str = "Titanic"

    batch_request: dict = {
        "execution_environment": execution_environment_name,
        "data_connector": data_connector_name,
        "data_asset_name": data_asset_name,
        "in_memory_dataset": None,
        "partition_request": None,
        "limit": None,
        "batch_spec_passthrough": {
            "path": titanic_csv_destination_file_path,
            "reader_method": "read_csv",
            "reader_options": None,
            "limit": 2000
        }
    }
    batch_request: BatchRequest = BatchRequest(**batch_request)
    # TODO: <Alex>get_batch() has been commented out; there is a new replacement method -- use it.</Alex>
    # batch: Batch = execution_environment.get_batch(
    #     batch_request=batch_request
    # )
    #
    # assert batch.batch_spec is not None
    # assert batch.batch_spec["data_asset_name"] == data_asset_name
    # assert isinstance(batch.data, pd.DataFrame)
    # assert batch.data.shape[0] == 1313


def test_get_batch_with_pipeline_style_batch_request():
    execution_environment_name: str = "test_execution_environment"
    execution_environment_config: dict = execution_environment_files_data_connector_regex_partitioner_config(
        use_group_names=False,
        use_sorters=False,
        default_base_directory=None,
        data_asset_base_directory=None
    )[execution_environment_name]
    execution_environment_config.pop("class_name")
    execution_environment: ExecutionEnvironment = ExecutionEnvironment(
        name=execution_environment_name,
        **execution_environment_config,
        data_context_root_directory=None
    )
    data_connector_name: str = "test_pipeline_data_connector"
    data_asset_name: str = "test_asset_1"

    test_df: pd.DataFrame = pd.DataFrame(data={"col1": [1, 2], "col2": [3, 4]})

    batch_request: dict = {
        "execution_environment": execution_environment_name,
        "data_connector": data_connector_name,
        "data_asset_name": data_asset_name,
        "in_memory_dataset": test_df,
        "partition_request": None,
        "limit": None,
    }
    # TODO: <Alex>get_batch() has been commented out; there is a new replacement method -- use it.</Alex>
    # batch_request: BatchRequest = BatchRequest(**batch_request)
    # batch: Batch = execution_environment.get_batch(
    #     batch_request=batch_request
    # )
    # assert batch.batch_spec is not None
    # assert batch.batch_spec["data_asset_name"] == data_asset_name
    # assert isinstance(batch.data, pd.DataFrame)
    # assert batch.data.shape == (2, 2)
    # assert batch.data["col2"].values[1] == 4


def test_get_batch_with_opportunistic_partition_caching():
    pass


def test_get_batch_without_partition_caching():
    pass

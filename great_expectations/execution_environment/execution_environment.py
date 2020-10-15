# -*- coding: utf-8 -*-

import copy
import logging
from typing import Union, List, Dict, Callable, Any

from great_expectations.data_context.types.base import (
    DataConnectorConfig,
    dataConnectorConfigSchema
)
from great_expectations.data_context.util import instantiate_class_from_config
from ruamel.yaml.comments import CommentedMap
from great_expectations.execution_environment.data_connector.data_connector import DataConnector
from great_expectations.execution_environment.data_connector.partitioner.partition import Partition
from great_expectations.core.id_dict import (
    PartitionRequest,
    PartitionDefinition,
    BatchSpec
)
from great_expectations.core.batch import (
    Batch,
    BatchRequestMetadata,
    BatchRequest,
    BatchDefinition
)

import great_expectations.exceptions as ge_exceptions

logger = logging.getLogger(__name__)


class ExecutionEnvironment(object):
    """
    An ExecutionEnvironment is the glue between an ExecutionEngine and a DataConnector.
    """
    recognized_batch_parameters: set = {"limit"}

    def __init__(
        self,
        name: str,
        execution_engine=None,
        data_connectors=None,
        data_context_root_directory: str = None,
    ):
        """
        Build a new ExecutionEnvironment.

        Args:
            name: the name for the datasource
            execution_engine (ClassConfig): the type of compute engine to produce
            data_connectors: DataConnectors to add to the datasource
        """
        self._name = name
        self._execution_engine = instantiate_class_from_config(
            config=execution_engine, runtime_environment={},
        )
        self._execution_environment_config = {
            "execution_engine": execution_engine
        }

        if data_connectors is None:
            data_connectors = {}
        self._execution_environment_config["data_connectors"] = data_connectors

        self._data_connectors_cache = {}

        self._data_context_root_directory = data_context_root_directory

        self._build_data_connectors()

    def get_available_partitions(
        self,
        data_connector_name: str,
        data_asset_name: str = None,
        partition_query: Union[Dict[str, Union[int, list, tuple, slice, str, Dict, Callable, None]], None] = None,
        in_memory_dataset: Any = None,
        runtime_parameters: Union[dict, None] = None,
        repartition: bool = False
    ) -> List[Partition]:
        if not data_connector_name:
            raise ge_exceptions.PartitionerError(message="Finding partitions requires a valid data_connector name.")
        data_connector: DataConnector = self.get_data_connector(
            name=data_connector_name
        )
        available_partitions: List[Partition] = data_connector.get_available_partitions(
            data_asset_name=data_asset_name,
            partition_query=partition_query,
            in_memory_dataset=in_memory_dataset,
            runtime_parameters=runtime_parameters,
            repartition=repartition
        )
        return available_partitions

    def get_batch(
        self,
        batch_request: BatchRequest
    ) -> Batch:
        if not batch_request:
            raise ge_exceptions.BatchDefinitionError(message="Batch request is empty.")

        partition_request: PartitionRequest = batch_request.partition_request
        data_asset_name: str = batch_request.data_asset_name
        partition_query: dict = {
            "custom_filter": None,
            "partition_name": None,
            "partition_definition": copy.deepcopy(partition_request),
            "partition_index": None,
            "limit": None
        }

        in_memory_dataset: Any = batch_request.in_memory_dataset

        data_connector_name: str = batch_request.data_connector_name
        if not data_connector_name:
            raise ge_exceptions.BatchDefinitionError(message="Batch request must specify a data_connector.")
        data_connector: DataConnector = self.get_data_connector(name=data_connector_name)

        partitions: List[Partition] = data_connector.get_available_partitions(
            data_asset_name=data_asset_name,
            partition_query=partition_query,
            in_memory_dataset=in_memory_dataset,
            runtime_parameters=None,
            repartition=False
        )
        if not partitions or len(partitions) == 0:
            raise ge_exceptions.BatchSpecError(
                message=f'''
Unable to build batch_spec for data asset "{data_asset_name}" (found 0 available partitions; must have exactly 1).
                '''
            )
        if len(partitions) > 1:
            raise ge_exceptions.BatchSpecError(
                message=f'''
Unable to build batch_spec for data asset "{data_asset_name}" (found {len(partitions)} partitions; must have exactly 1).
                '''
            )

        partition: Partition = partitions[0]
        # noinspection PyProtectedMember
        batch_spec: BatchSpec = data_connector._build_batch_spec(batch_request=batch_request, partition=partition)
        batch_spec = self.execution_engine.process_batch_request(
            batch_request=batch_request,
            batch_spec=batch_spec
        )

        batch: Batch = self.execution_engine.load_batch(batch_spec=batch_spec)
        partition_definition: PartitionDefinition = partition.definition
        batch_definition: BatchDefinition = BatchDefinition(
            execution_environment_name=self.name,
            data_connector_name=data_connector_name,
            data_asset_name=batch_request.data_asset_name,
            partition_definition=partition_definition
        )
        batch_request_metadata: BatchRequestMetadata = batch_request.batch_request_metadata
        batch.batch_request = batch_request_metadata
        batch.batch_definition = batch_definition

        return batch

    @property
    def name(self):
        """
        Property for datasource name
        """
        return self._name

    @property
    def execution_engine(self):
        return self._execution_engine

    @property
    def config(self):
        return copy.deepcopy(self._execution_environment_config)

    def _build_data_connectors(self):
        """
        Build DataConnector objects from the ExecutionEnvironment configuration.

        Returns:
            None
        """
        if "data_connectors" in self._execution_environment_config:
            for data_connector in self._execution_environment_config["data_connectors"].keys():
                self.get_data_connector(name=data_connector)

    # TODO Abe 10/6/2020: Should this be an internal method?
    def get_data_connector(self, name: str) -> DataConnector:
        """Get the (named) DataConnector from an ExecutionEnvironment)

        Args:
            name (str): name of DataConnector
            runtime_environment (dict):

        Returns:
            DataConnector (DataConnector)
        """
        data_connector: DataConnector
        if name in self._data_connectors_cache:
            return self._data_connectors_cache[name]
        elif (
            "data_connectors" in self._execution_environment_config
            and name in self._execution_environment_config["data_connectors"]
        ):
            data_connector_config: dict = copy.deepcopy(
                self._execution_environment_config["data_connectors"][name]
            )
        else:
            raise ge_exceptions.DataConnectorError(
                f'Unable to load data connector "{name}" -- no configuration found or invalid configuration.'
            )
        data_connector_config: CommentedMap = dataConnectorConfigSchema.load(
            data_connector_config
        )
        data_connector: DataConnector = self._build_data_connector_from_config(
            name=name, config=data_connector_config
        )
        self._data_connectors_cache[name] = data_connector
        return data_connector

    def _build_data_connector_from_config(
        self,
        name: str,
        config: CommentedMap,
    ) -> DataConnector:
        """Build a DataConnector using the provided configuration and return the newly-built DataConnector."""
        # We convert from the type back to a dictionary for purposes of instantiation
        if isinstance(config, DataConnectorConfig):
            config: dict = dataConnectorConfigSchema.dump(config)
        module_name: str = "great_expectations.execution_environment.data_connector.data_connector"
        runtime_environment: dict = {
            "name": name,
            "data_context_root_directory": self._data_context_root_directory
        }
        if self._execution_engine is not None:
            runtime_environment.update({"execution_engine": self._execution_engine})
        data_connector: DataConnector = instantiate_class_from_config(
            config=config,
            runtime_environment=runtime_environment,
            config_defaults={"module_name": module_name},
        )
        if not data_connector:
            raise ge_exceptions.ClassInstantiationError(
                module_name="great_expectations.execution_environment.data_connector.data_connector",
                package_name=None,
                class_name=config["class_name"],
            )
        return data_connector

    # TODO Abe 10/6/2020: Should this be an internal method?<Alex>Pros/cons for either choice exist; happy to discuss.</Alex>
    def list_data_connectors(self) -> List[dict]:
        """List currently-configured DataConnector for this ExecutionEnvironment.

        Returns:
            List(dict): each dictionary includes "name" and "type" keys
        """
        data_connectors: List[dict] = []

        if "data_connectors" in self._execution_environment_config:
            for key, value in self._execution_environment_config["data_connectors"].items():
                data_connectors.append({"name": key, "class_name": value["class_name"]})

        return data_connectors

    def get_available_data_asset_names(self, data_connector_names: list = None, clear_cache: bool = False) -> dict:
        """
        Returns a dictionary of data_asset_names that the specified data
        connector can provide. Note that some data_connectors may not be
        capable of describing specific named data assets, and some (such as
        files_data_connectors) require the user to configure
        data asset names.

        Args:
            data_connector_names: the DataConnector for which to get available data asset names.
            clear_cache: if True, clears the cache in the underlying implementation (False by default for efficiency)

        Returns:
            dictionary consisting of sets of data assets available for the specified data connectors:
            ::

                {
                  data_connector_name: {
                    names: [ (data_asset_1, data_asset_1_type), (data_asset_2, data_asset_2_type) ... ]
                  }
                  ...
                }
        """
        available_data_asset_names: dict = {}
        if data_connector_names is None:
            data_connector_names = [
                data_connector["name"] for data_connector in self.list_data_connectors()
            ]
        elif isinstance(data_connector_names, str):
            data_connector_names = [data_connector_names]

        for data_connector_name in data_connector_names:
            data_connector = self.get_data_connector(name=data_connector_name)
            available_data_asset_names[data_connector_name] = data_connector.get_available_data_asset_names(
                repartition=clear_cache
            )
        return available_data_asset_names

# -*- coding: utf-8 -*-

import copy
import logging
from typing import Union, List, Dict, Callable, Any

from great_expectations.data_context.util import instantiate_class_from_config
from great_expectations.execution_environment.data_connector.data_connector import DataConnector
from great_expectations.execution_environment.data_connector.partitioner.partition import Partition
from great_expectations.core.id_dict import BatchSpec
from great_expectations.core.batch import (
    Batch,
    BatchRequest,
    BatchDefinition,
    BatchMarkers
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
            config=execution_engine,
            runtime_environment={},
            config_defaults={
                "module_name": "great_expectations.execution_engine"
            }
        )
        self._execution_environment_config = {
            "execution_engine": execution_engine
        }

        if data_connectors is None:
            data_connectors = {}
        self._execution_environment_config["data_connectors"] = data_connectors

        self._data_connectors_cache = {}

        self._data_context_root_directory = data_context_root_directory

        # if data_connectors is None:
        #     data_connectors = {}
        self._data_connectors_cache = {}
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

# TODO: <Alex>We need to determine the proper replacement method.  Right now, we do not have a method that returns the batch with all metadata and batch_request, including "in_memory_dataset", included.</Alex>
#     def get_batch(
#         self,
#         batch_request: BatchRequest
#     ) -> Batch:
#         if not batch_request:
#             raise ge_exceptions.BatchDefinitionError(message="Batch request is empty.")

#         partition_request: PartitionRequest = batch_request.partition_request
#         data_asset_name: str = batch_request.data_asset_name
#         partition_query: dict = {
#             "custom_filter": None,
#             "partition_name": None,
#             "partition_definition": copy.deepcopy(partition_request),
#             "partition_index": None,
#             "limit": None
#         }

#         in_memory_dataset: Any = batch_request.in_memory_dataset

#         data_connector_name: str = batch_request.data_connector_name
#         if not data_connector_name:
#             raise ge_exceptions.BatchDefinitionError(message="Batch request must specify a data_connector.")
#         data_connector: DataConnector = self.get_data_connector(name=data_connector_name)

#         partitions: List[Partition] = data_connector.get_available_partitions(
#             data_asset_name=data_asset_name,
#             partition_query=partition_query,
#             in_memory_dataset=in_memory_dataset,
#             runtime_parameters=None,
#             repartition=False
#         )
#         if not partitions or len(partitions) == 0:
#             raise ge_exceptions.BatchSpecError(
#                 message=f'''
# Unable to build batch_spec for data asset "{data_asset_name}" (found 0 available partitions; must have exactly 1).
#                 '''
#             )
#         if len(partitions) > 1:
#             raise ge_exceptions.BatchSpecError(
#                 message=f'''
# Unable to build batch_spec for data asset "{data_asset_name}" (found {len(partitions)} partitions; must have exactly 1).
#                 '''
#             )

#         partition: Partition = partitions[0]
#         # noinspection PyProtectedMember
#         batch_spec: BatchSpec = data_connector._build_batch_spec(batch_request=batch_request, partition=partition)
#         batch_spec = self.execution_engine.process_batch_request(
#             batch_request=batch_request,
#             batch_spec=batch_spec
#         )

#         batch: Batch = self.execution_engine.load_batch(batch_spec=batch_spec)
#         partition_definition: PartitionDefinition = partition.definition
#         batch_definition: BatchDefinition = BatchDefinition(
#             execution_environment_name=self.name,
#             data_connector_name=data_connector_name,
#             data_asset_name=batch_request.data_asset_name,
#             partition_definition=partition_definition
#         )
#         batch_request_metadata: BatchRequestMetadata = batch_request.batch_request_metadata
#         batch.batch_request = batch_request_metadata
#         batch.batch_definition = batch_definition

#         return batch

    # TODO: <Alex>This method can become the main "get_batch" method (mentioned above) as long as it can represent the "in_memory_dataset" option.</Alex>
    # TODO: <Alex>If so, then this method should either use "get_batch_from_batch_definition()", or they should be combined into one method, using a boolean option to distinguish between the two usecases.</Alex>
    def get_batch_list_from_batch_request(
        self,
        batch_request: BatchRequest
    ) -> List[Batch]:
        data_connector = self.get_data_connector(
            name=batch_request.data_connector_name
        )
        batch_definition_list: List[BatchDefinition] = data_connector.get_batch_definition_list_from_batch_request(
            batch_request=batch_request
        )
        batches: List[Batch] = []
        for batch_definition in batch_definition_list:
            batch_data, batch_spec, batch_markers = data_connector.get_batch_data_and_metadata_from_batch_definition(
                batch_definition=batch_definition
            )
            batch: Batch = Batch(
                data=batch_data,
                batch_request=batch_request,
                batch_definition=batch_definition,
                batch_spec=batch_spec,
                batch_markers=batch_markers,
            )
            batches.append(batch)
        return batches

    # TODO: <Alex>What is the purpose of this method, given the comment below about how it not should be used.</Alex>
    # TODO: <Alex>This method can be combined with "get_batch_list_from_batch_request()" into one method, using a boolean option to distinguish between the two usecases.</Alex>
    def get_batch_from_batch_definition(
        self,
        batch_definition: BatchDefinition,
        in_memory_dataset: Any = None,
    ) -> Batch:
        """
        Note: this method should *not* be used when getting a Batch from a BatchRequest, since it does not capture BatchRequest metadata.
        """

        batch_data: Any
        batch_spec: Union[BatchSpec, None] = None
        batch_markers: Union[BatchMarkers, None] = None
        # TODO: <Alex>More work is needed to figure out the appropriate safeguards for the "in_memory_dataset".</Alex>
        if not isinstance(in_memory_dataset, type(None)):
            # TODO: <Alex>Good idea from Abe below.</Alex>
            # NOTE Abe 20201014: Maybe do more careful type checking here?
            # Seems like we should verify that in_memory_dataset is compatible with the execution_engine...?
            batch_data = in_memory_dataset

            # TODO: <Alex>
            # execution_environment_name in batch_definition must match the name of the present execution_environment object.
            #  </Alex>
            # NOTE Abe 20201014: We should also verify that the keys in batch_definition.partition_definition are compatible with the DataConnector?

        else:
            data_connector: DataConnector = self.get_data_connector(
                name=batch_definition.data_connector_name
            )
            batch_data, batch_spec, batch_markers = data_connector.get_batch_data_and_metadata_from_batch_definition(
                batch_definition=batch_definition
            )
        batch: Batch = Batch(
            data=batch_data,
            batch_request=None,
            batch_definition=batch_definition,
            batch_spec=batch_spec,
            batch_markers=batch_markers,
        )
        return batch

    def _build_data_connectors(self):
        """
        Build DataConnector objects from the ExecutionEnvironment configuration.

        Returns:
            None
        """
        if "data_connectors" in self._execution_environment_config:
            for data_connector_name in self._execution_environment_config["data_connectors"].keys():
                self.get_data_connector(name=data_connector_name)

    # TODO Abe 10/6/2020: Should this be an internal method?
    def get_data_connector(self, name: str) -> DataConnector:
        """Get the (named) DataConnector from an ExecutionEnvironment)

        Args:
            name (str): name of DataConnector

        Returns:
            DataConnector (DataConnector)
        """
        data_connector_config: dict
        data_connector: DataConnector
        if name in self._data_connectors_cache:
            return self._data_connectors_cache[name]
        elif (
            "data_connectors" in self._execution_environment_config
            and name in self._execution_environment_config["data_connectors"]
        ):
            data_connector_config = copy.deepcopy(
                self._execution_environment_config["data_connectors"][name]
            )
        else:
            raise ge_exceptions.DataConnectorError(
                f'Unable to load data connector "{name}" -- no configuration found or invalid configuration.'
            )
        data_connector: DataConnector = self._build_data_connector_from_config(
            name=name, config=data_connector_config
        )
        self._data_connectors_cache[name] = data_connector
        return data_connector

    def _build_data_connector_from_config(
        self,
        name: str,
        config: dict
    ) -> DataConnector:
        """Build a DataConnector using the provided configuration and return the newly-built DataConnector."""
        module_name: str = "great_expectations.execution_environment.data_connector.data_connector"
        runtime_environment: dict = {
            "name": name,
            "data_context_root_directory": self._data_context_root_directory
        }
        if self._execution_engine is not None:
            runtime_environment.update({"execution_engine": self._execution_engine})
        data_connector: DataConnector = instantiate_class_from_config(
            config=config,
            runtime_environment={
                "name": name,
                "data_context_root_directory": self._data_context_root_directory,
                "execution_engine": self._execution_engine
            },
            config_defaults={
                "module_name": "great_expectations.execution_environment.data_connector"
            },
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

    def self_check(self, pretty_print=True, max_examples=3):
        return_object = {
            "execution_engine": {
                "class_name": self._execution_engine.__class__.__name__,
            }
        }

        if pretty_print:
            print(f"Execution engine: {self._execution_engine.__class__.__name__}")

        if pretty_print:
            print(f"Data connectors:")

        data_connector_list = self.list_data_connectors()
        data_connector_list.sort()
        return_object["data_connectors"] = {
            "count" : len(data_connector_list)
        }

        for data_connector in data_connector_list:
            if pretty_print:
                print("\t"+data_connector["name"], ":", data_connector["class_name"])
                print()

            asset_names = self.get_available_data_asset_names(data_connector["name"])[data_connector["name"]]
            asset_names.sort()
            len_asset_names = len(asset_names)

            data_connector_obj = {
                "class_name" : data_connector["class_name"],
                "data_asset_count" : len_asset_names,
                "example_data_asset_names": asset_names[:max_examples],
                "data_assets" : {}
            }

            if pretty_print:
                print(f"\tAvailable data_asset_names ({min(len_asset_names, max_examples)} of {len_asset_names}):")
            
            for asset_name in asset_names[:max_examples]:
                partitions = self.get_available_partitions(data_connector["name"], asset_name)
                len_partitions = len(partitions)
                example_partition_names = [partition.data_reference for partition in partitions][:max_examples]
                if pretty_print:
                    print(f"\t\t{asset_name} ({min(len_partitions, max_examples)} of {len_partitions}):", example_partition_names)

                data_connector_obj["data_assets"][asset_name] = {
                    "partition_count": len_partitions,
                    "example_partition_names": example_partition_names
                }

            instantiated_data_connector = self.get_data_connector(data_connector["name"])
            instantiated_data_connector.refresh_data_object_cache()
            unmatched_data_references = instantiated_data_connector.get_unmatched_data_objects()
            len_unmatched_data_references = len(unmatched_data_references)
            if pretty_print:
                print(f"\n\tUnmatched data_references ({min(len_unmatched_data_references, max_examples)} of {len_unmatched_data_references}):", unmatched_data_references[:max_examples])

            return_object["data_connectors"][data_connector["name"]] = data_connector_obj

        return return_object

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

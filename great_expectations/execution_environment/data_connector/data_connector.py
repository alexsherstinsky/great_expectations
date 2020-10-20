# -*- coding: utf-8 -*-

import copy
import itertools
from typing import List, Dict, Union, Callable, Any, Tuple

import logging

from great_expectations.execution_engine import ExecutionEngine
from great_expectations.execution_environment.data_connector.asset.asset import Asset
from great_expectations.execution_environment.data_connector.partitioner.partitioner import Partitioner
from great_expectations.execution_environment.data_connector.partitioner.partition import Partition
from great_expectations.execution_environment.data_connector.partitioner.partition_query import (
    PartitionQuery,
    build_partition_query
)
from great_expectations.core.batch import BatchRequest
from great_expectations.core.id_dict import (
    PartitionDefinitionSubset,
    PartitionDefinition,
    BatchSpec,
)
from great_expectations.core.batch import (
    BatchMarkers,
    BatchDefinition,
)
from great_expectations.data_context.util import instantiate_class_from_config
import great_expectations.exceptions as ge_exceptions

logger = logging.getLogger(__name__)


class DataConnector(object):
    """
    DataConnectors produce identifying information, called "batch_spec" that ExecutionEngines
    can use to get individual batches of data. They add flexibility in how to obtain data
    such as with time-based partitioning, downsampling, or other techniques appropriate
    for the ExecutionEnvironment.

    For example, a DataConnector could produce a SQL query that logically represents "rows in
    the Events table with a timestamp on February 7, 2012," which a SqlAlchemyExecutionEnvironment
    could use to materialize a SqlAlchemyDataset corresponding to that batch of data and
    ready for validation.

    A batch is a sample from a data asset, sliced according to a particular rule. For
    example, an hourly slide of the Events table or “most recent `users` records.”

    A Batch is the primary unit of validation in the Great Expectations DataContext.
    Batches include metadata that identifies how they were constructed--the same “batch_spec”
    assembled by the data connector, While not every ExecutionEnvironment will enable re-fetching a
    specific batch of data, GE can store snapshots of batches or store metadata from an
    external data version control system.
    """
    DEFAULT_DATA_ASSET_NAME: str = "DEFAULT_DATA_ASSET"

    _default_reader_options: dict = {}

    def __init__(
        self,
        name: str,
        assets: dict = None,
        partitioners: dict = None,
        default_partitioner: str = None,
        execution_engine: ExecutionEngine = None,
        data_context_root_directory: str = None,
    ):
        self._name = name
        if assets is None:
            assets = {}
        self._assets = assets
        if partitioners is None:
            partitioners = {}
        self._partitioners = partitioners
        self._default_partitioner = default_partitioner
        self._execution_engine = execution_engine
        self._data_context_root_directory = data_context_root_directory

        self._assets_cache: dict = {}

        self._partitioners_cache: dict = {}

        self._build_partitioners_from_config(config=partitioners)

        # The partitions cache is a dictionary, which maintains lists of partitions for a data_asset_name as the key.
        self._partitions_cache: dict = {}

        # This is a dictionary which maps data_references onto batch_requests
        self._cached_data_reference_to_batch_definition_map = {}

    @property
    def name(self) -> str:
        return self._name

    @property
    def assets(self) -> dict:
        return self._assets

    # TODO: <Alex></Alex>
    @property
    def partitioners(self) -> dict:
        return self._partitioners

    @property
    def default_partitioner(self) -> Partitioner:
        return self.get_partitioner(name=self._default_partitioner)
        # TODO: <Alex>Cleanup</Alex>
        # try:
        #     return self.partitioners[self._default_partitioner]
        # except KeyError:
        #     raise ValueError("No default partitioner has been set")

    def get_asset(self, name) -> Asset:
        """Get the (named) Asset from a DataConnector)

        Args:
            name (str): name of Asset

        Returns:
            Asset (Asset)
        """
        asset_config: dict
        if name in self._assets_cache:
            return self._assets_cache[name]
        else:
            if self._assets:
                if name in self._assets.keys():
                    asset_config = copy.deepcopy(
                        self._assets[name]
                    )
                else:
                    raise ge_exceptions.AssetError(
                        f'''Unable to load asset with the name "{name}" -- no configuration found or invalid
configuration.
                        '''
                    )
            else:
                raise ge_exceptions.AssetError(
                    f'Unable to load asset with the name "{name}" -- no configuration found or invalid configuration.'
                )
        asset: Asset = self._build_asset_from_config(
            name=name, config=asset_config
        )
        self._assets_cache[name] = asset
        return asset

    @staticmethod
    def _build_asset_from_config(name: str, config: dict) -> Asset:
        """Build an Asset using the provided configuration and return the newly-built Asset."""
        runtime_environment: dict = {
            "name": name
        }
        asset: Asset = instantiate_class_from_config(
            config=config,
            runtime_environment=runtime_environment,
            config_defaults={
                "module_name": "great_expectations.execution_environment.data_connector.asset.asset"
            },
        )
        if not asset:
            raise ge_exceptions.ClassInstantiationError(
                module_name="great_expectations.execution_environment.data_connector.asset.asset",
                package_name=None,
                class_name=config["class_name"],
            )
        return asset

    def _build_partitioners_from_config(self, config: dict):
        for name in config.keys():
            # noinspection PyUnusedLocal
            new_partitioner: Partitioner = self.get_partitioner(name=name)

    def get_partitioner(self, name: str) -> Partitioner:
        """Get the (named) Partitioner from a DataConnector)

        Args:
            name (str): name of Partitioner

        Returns:
            Partitioner (Partitioner)
        """
        partitioner_config: dict
        if name in self._partitioners_cache:
            return self._partitioners_cache[name]
        elif name in self.partitioners:
            partitioner_config = copy.deepcopy(
                self.partitioners[name]
            )
        else:
            raise ge_exceptions.PartitionerError(
                f'Unable to load partitioner "{name}" -- no configuration found or invalid configuration.'
            )
        partitioner: Partitioner = self._build_partitioner_from_config(
            name=name, config=partitioner_config
        )
        self._partitioners_cache[name] = partitioner
        return partitioner

    def _build_partitioner_from_config(self, name: str, config: dict):
        """Build a Partitioner using the provided configuration and return the newly-built Partitioner."""
        runtime_environment: dict = {
            "name": name,
            "data_connector": self
        }
        partitioner: Partitioner = instantiate_class_from_config(
            config=config,
            runtime_environment=runtime_environment,
            config_defaults={
                "module_name": "great_expectations.execution_environment.data_connector.partitioner"
            },
        )
        if not partitioner:
            raise ge_exceptions.ClassInstantiationError(
                module_name="great_expectations.execution_environment.data_connector.partitioner",
                package_name=None,
                class_name=config["class_name"],
            )
        return partitioner

    def _get_cached_partitions(
        self,
        data_asset_name: str = None,
        runtime_parameters: Union[PartitionDefinitionSubset, None] = None
    ) -> List[Partition]:
        cached_partitions: List[Partition]
        if data_asset_name is None:
            cached_partitions = list(
                itertools.chain.from_iterable(
                    [
                        partitions for name, partitions in self._partitions_cache.items()
                    ]
                )
            )
        else:
            cached_partitions = self._partitions_cache.get(data_asset_name)
        if runtime_parameters is None:
            return cached_partitions
        else:
            if not cached_partitions:
                return []
            return list(
                filter(
                    lambda partition: self._cache_partition_runtime_parameters_filter(
                        partition=partition,
                        parameters=runtime_parameters
                    ),
                    cached_partitions
                )
            )

    @staticmethod
    def _cache_partition_runtime_parameters_filter(partition: Partition, parameters: PartitionDefinitionSubset) -> bool:
        partition_definition: PartitionDefinition = partition.definition
        for key, value in parameters.items():
            if not (key in partition_definition and partition_definition[key] == value):
                return False
        return True

    def update_partitions_cache(
        self,
        partitions: List[Partition],
        partitioner_name: str,
        runtime_parameters: PartitionDefinition,
        allow_multipart_partitions: bool = False
    ):
        """
        The cache of partitions (keyed by a data_asset_name) is identified by the combination of the following entities:
        -- name of the partition (string)
        -- data_asset_name (string)
        -- partition definition (dictionary)
        Configurably, these entities are either supplied by the user function or computed by the specific partitioner.

        In order to serve as the identity of the Partition object, the above fields are hashed.  Hence, Partition
        objects can be utilized in set operations, tested for presence in lists, containing multiple Partition objects,
        and participate in any other logic operations, where equality checks play a role.  This is particularly
        important, because one-to-many mappings (multiple "same" partition objects, differing only by the data_reference
        property) especially if referencing different data objects, are illegal (unless configured otherwise).

        In addition, it is considered illegal to have the same data_reference property in multiple Partition objects.

        Note that the data_reference field of the Partition object does not participate in identifying a partition.
        The reason for this is that data references can vary in type (files on a filesystem, S3 objects, Pandas or Spark
        DataFrame objects, etc.) and maintaining references to them in general can result in large memory consumption.
        Moreover, in the case of Pandas and Spark DataFrame objects (and other in-memory datasets), the metadata aspects
        of the data object (as captured by the partition information) may remain the same, while the actual dataset
        changes (e.g., the output of a processing stage of a data pipeline).  In such situations, if a new partition
        is found to be identical to an existing partition, the data_reference property of the new partition is accepted.
        """

        # Prevent non-unique partitions in submitted list of partitions.
        if not allow_multipart_partitions and partitions and len(partitions) > len(set(partitions)):
            raise ge_exceptions.PartitionerError(
                f'''Partitioner "{partitioner_name}" detected multiple data references in one or more partitions for the
given data asset; however, allow_multipart_partitions is set to False.  Please consider modifying the directives, used
to partition your dataset, or set allow_multipart_partitions to True, but be aware that unless you have a specific use
case for multipart partitions, there is most likely a mismatch between the partitioning directives and the actual
structure of data under consideration.
                '''
            )
        for partition in partitions:
            data_asset_name: str = partition.data_asset_name
            cached_partitions: List[Partition] = self._get_cached_partitions(
                data_asset_name=data_asset_name,
                runtime_parameters=runtime_parameters
            )
            if cached_partitions is None or len(cached_partitions) == 0:
                cached_partitions = []
            if partition in cached_partitions:
                # Prevent non-unique partitions in anticipated list of partitions.
                non_unique_partitions: List[Partition] = [
                    temp_partition
                    for temp_partition in cached_partitions
                    if temp_partition == partition
                ]
                if not allow_multipart_partitions and len(non_unique_partitions) > 1:
                    raise ge_exceptions.PartitionerError(
                        f'''Partitioner "{partitioner_name}" detected multiple data references for partition
"{partition}" of data asset "{partition.data_asset_name}"; however, allow_multipart_partitions is set to
False.  Please consider modifying the directives, used to partition your dataset, or set allow_multipart_partitions to
True, but be aware that unless you have a specific use case for multipart partitions, there is most likely a mismatch
between the partitioning directives and the actual structure of data under consideration.
                        '''
                    )
                # Attempt to update the data_reference property with that provided as part of the submitted partition.
                specific_partition_idx: int = cached_partitions.index(partition)
                specific_partition: Partition = cached_partitions[specific_partition_idx]
                if specific_partition.data_reference != partition.data_reference:
                    specific_partition.data_reference = partition.data_reference
            else:
                # Prevent the same data_reference property value to be represented by multiple partitions.
                partitions_with_given_data_reference: List[Partition] = [
                    temp_partition
                    for temp_partition in cached_partitions
                    if temp_partition.data_reference == partition.data_reference
                ]
                if len(partitions_with_given_data_reference) > 0:
                    raise ge_exceptions.PartitionerError(
                        f'''Partitioner "{partitioner_name}" for data asset "{partition.data_asset_name}" detected
multiple partitions, including "{partition}", for the same data reference -- this is illegal.
                        '''
                    )
                cached_partitions.append(partition)
            self._partitions_cache[data_asset_name] = cached_partitions

    def reset_partitions_cache(self, data_asset_name: str = None):
        if data_asset_name is None:
            self._partitions_cache = {}
        else:
            if data_asset_name in self._partitions_cache:
                self._partitions_cache[data_asset_name] = []

    def add_partitioner(self, partitioner_name: str, partitioner_config: dict) -> Partitioner:
        """Add a new Partitioner to the DataConnector and (for convenience) return the instantiated Partitioner object.

        Args:
            partitioner_name (str): a key for the new Store in in self._stores
            partitioner_config (dict): a config for the Store to add

        Returns:
            partitioner (Partitioner)
        """
        new_partitioner: Partitioner = self._build_partitioner_from_config(
            name=partitioner_name,
            config=partitioner_config
        )
        self.partitioners[partitioner_name] = new_partitioner

        return new_partitioner

    def get_partitioner_for_data_asset(self, data_asset_name: str = None) -> Partitioner:
        partitioner_name: str
        data_asset_config_exists: bool = data_asset_name and self.assets and self.assets.get(data_asset_name)
        if data_asset_config_exists and self.assets[data_asset_name].get("partitioner"):
            partitioner_name = self.assets[data_asset_name]["partitioner"]
        else:
            partitioner_name = self.default_partitioner.name
        partitioner: Partitioner
        if partitioner_name is None:
            raise ge_exceptions.BatchSpecError(
                message=f'''
No partitioners found for data connector "{self.name}" -- at least one partitioner must be configured for a data
connector and the default_partitioner set to one of the configured partitioners.
                '''
            )
        else:
            # TODO: <Alex>Cleanup</Alex>
            # partitioner = self.partitioners[partitioner_name]
            partitioner = self.get_partitioner(name=partitioner_name)
        return partitioner

    def get_available_data_asset_names(self, repartition: bool = False) -> List[str]:
        """Return the list of asset names known by this data connector.

        Returns:
            A list of available names
        """
        available_data_asset_names: List[str] = []

        if self.assets:
            available_data_asset_names = list(self.assets.keys())

        available_partitions: List[Partition] = self.get_available_partitions(
            data_asset_name=None,
            partition_query={
                "custom_filter": None,
                "partition_name": None,
                "partition_definition": None,
                "partition_index": None,
                "limit": None,
            },
            runtime_parameters=None,
            repartition=repartition
        )
        if available_partitions and len(available_partitions) > 0:
            for partition in available_partitions:
                available_data_asset_names.append(partition.data_asset_name)

        return list(set(available_data_asset_names))

    def get_available_partitions(
        self,
        data_asset_name: str = None,
        partition_query: Union[
            Dict[str, Union[int, list, tuple, slice, str, Union[Dict, PartitionDefinitionSubset], Callable, None]], None
        ] = None,
        in_memory_dataset: Any = None,
        runtime_parameters: Union[dict, None] = None,
        repartition: bool = False
    ) -> List[Partition]:
        partitioner: Partitioner = self.get_partitioner_for_data_asset(data_asset_name=data_asset_name)
        partition_query_obj: PartitionQuery = build_partition_query(partition_query_dict=partition_query)
        if runtime_parameters is not None:
            runtime_parameters: PartitionDefinitionSubset = PartitionDefinitionSubset(runtime_parameters)
        return self._get_available_partitions(
            partitioner=partitioner,
            data_asset_name=data_asset_name,
            partition_query=partition_query_obj,
            in_memory_dataset=in_memory_dataset,
            runtime_parameters=runtime_parameters,
            repartition=repartition
        )

    def _get_available_partitions(
        self,
        partitioner: Partitioner,
        data_asset_name: str = None,
        partition_query: Union[PartitionQuery, None] = None,
        in_memory_dataset: Any = None,
        runtime_parameters: Union[PartitionDefinitionSubset, None] = None,
        repartition: bool = False
    ) -> List[Partition]:
        raise NotImplementedError

    # TODO: <Alex>
    #  Approach:
    #  1) Get the list of batch definitions; then
    #  2) Get the list of batch data tuples, corresponding to the batch definitions.
    #  Steps:
    #  Generate partition definition list from batch request (one to many).
    #  With the help of the other attributes (execution_environment_name, data_connector_name, and data_asset_name), turn each partition definition into a batch definition.
    #  With the help of specific data connector, turn each batch definition into a batch spec.
    #  Using the batch spec, get the batch data and batch markers from the execution engine.
    #  </Alex>
    def get_batch_definition_list_from_batch_request(
        self,
        batch_request: BatchRequest,
    ) -> List[BatchDefinition]:
        if batch_request.data_connector_name != self.name:
            raise ge_exceptions.DataConnectorError(
                f'''Unexpected Error: The data connector name in the batch request is
"{batch_request.data_connector_name}"; this does not match the actual data connector name, "{self.name}".
                '''
            )

        partition_definition_list: List[PartitionDefinition] = \
            self._generate_partition_definition_list_from_batch_request(
                batch_request=batch_request
            )
        batch_definition_list: List[BatchDefinition] = []
        for partition_definition in partition_definition_list:
            batch_definition_list.append(BatchDefinition(
                execution_environment_name=batch_request.execution_environment_name,
                data_connector_name=self.name,
                data_asset_name=batch_request.data_asset_name,
                partition_definition=partition_definition,
            ))
        return batch_definition_list

    # TODO: <Alex>We could create a one to many method that turns a batch request into list of batch data and metadata Tuples (as in below).</Alex>
    def get_batch_data_and_metadata_from_batch_definition(
        self,
        batch_definition: BatchDefinition,
    ) -> Tuple[
        Any,  # batch_data
        BatchSpec,
        BatchMarkers,
    ]:
        batch_spec: BatchSpec = self._build_batch_spec_from_batch_definition(
            batch_definition=batch_definition
        )
        batch_data: Any
        batch_markers: BatchMarkers
        batch_data, batch_markers = self._execution_engine.get_batch_data_and_markers(**batch_spec)
        return (
            batch_data,
            batch_spec,
            batch_markers,
        )

    # TODO: <Alex>This method does not take into account the fact that batch_request may carry a non-null partition_request -- should we incorporate it?</Alex>
    # TODO: <Alex>We also need to discuss how the "in_memory_dataset" is handled.</Alex>
    def _generate_partition_definition_list_from_batch_request(
        self,
        batch_request: BatchRequest
    ) -> List[PartitionDefinition]:
        # TODO: <Alex>We need to discuss whether or not it is advantageous to use the data_reference_cache, given that a partition_request may be contained as part the batch_request?</Alex>
        # FIXME: switch this to use the data_reference cache instead of the partition cache.
        available_partitions = self.get_available_partitions(
            data_asset_name=batch_request.data_asset_name,
            # TODO: <Alex>The following comment needs discussion.</Alex>
            ### Need to pass data connector
        )
        return [partition.definition for partition in available_partitions]

    def _build_batch_spec_from_batch_definition(
        self,
        batch_definition: BatchDefinition
    ) -> BatchSpec:
        batch_spec_params: dict = self._generate_batch_spec_parameters_from_batch_definition(
            batch_definition=batch_definition
        )

        # TODO Abe 20201018: Reincorporate info from the execution_engine.
        # Note: This might not be necessary now that we're encoding that info as default params on get_batch_data_and_markers
        # batch_spec = self._execution_engine.process_batch_request(
        #     batch_request=batch_request,
        #     batch_spec=batch_spec
        # )
        # TODO: <Alex></Alex>

        # TODO Abe 20201018: Decide if we want to allow batch_spec_passthrough parameters anywhere.
        # TODO: <Alex>Hope not. :)</Alex>

        batch_spec: BatchSpec = BatchSpec(**batch_spec_params)

        return batch_spec

    def _generate_batch_spec_parameters_from_batch_definition(
        self,
        batch_definition: BatchDefinition
    ) -> dict:
        raise NotImplementedError

    # TODO: <Alex>This was commented out, because, apparently, it is not used any more.</Alex>
    # def _build_batch_spec(self, batch_request: BatchRequest, partition: Partition) -> BatchSpec:
    #     if not batch_request.data_asset_name:
    #         raise ge_exceptions.BatchSpecError("Batch request must have a data_asset_name.")

    #     batch_spec_scaffold: BatchSpec
    #     batch_spec_passthrough: BatchSpec = batch_request.batch_spec_passthrough
    #     if batch_spec_passthrough is None:
    #         batch_spec_scaffold = BatchSpec()
    #     else:
    #         batch_spec_scaffold = copy.deepcopy(batch_spec_passthrough)

    #     data_asset_name: str = batch_request.data_asset_name
    #     batch_spec_scaffold["data_asset_name"] = data_asset_name

    #     batch_spec: BatchSpec = self._build_batch_spec_from_partition(
    #         partition=partition, batch_request=batch_request, batch_spec=batch_spec_scaffold
    #     )

    #     return batch_spec

    # def _build_batch_spec_from_partition(
    #     self,
    #     partition: Partition,
    #     batch_request: BatchRequest,
    #     batch_spec: BatchSpec
    # ) -> BatchSpec:
    #     raise NotImplementedError

    # TODO: <Alex>Commenting out for now (DO NOT DELETE) -- implementation and bug fixes are pending.</Alex>
    # def refresh_data_reference_cache(self):
    #     #Map data_references to batch_definitions
    #     self._cached_data_reference_to_batch_definition_map = {}
    #
    #     for data_reference in self._get_data_reference_list():
    #         mapped_batch_definition_list = self._map_data_reference_to_batch_request_list(data_reference)
    #         self._cached_data_reference_to_batch_definition_map[data_reference] = mapped_batch_definition_list
    #
    # def get_unmatched_data_references(self):
    #     if self._cached_data_reference_to_batch_definition_map == None:
    #         raise ValueError("_cached_data_reference_to_batch_definition_map is None. Have you called refresh_data_reference_cache yet?")
    #
    #     return [k for k,v in self._cached_data_reference_to_batch_definition_map.items() if v == None]
    #
    # def get_data_reference_list_count(self):
    #     return len(self._cached_data_reference_to_batch_definition_map)
    #
    # #TODO Abe 20201015: This method is extremely janky. Needs better supporting methods, plus more thought and hardening.
    # def _map_data_reference_to_batch_request_list(self, data_reference) -> List[BatchDefinition]:
    #     #FIXME: Make this smarter about choosing the right partitioner
    #
    #     # Verify that a default_partitioner has been chosen
    #     try:
    #         self.default_partitioner
    #     except ValueError:
    #         #If not, return None
    #         return
    #
    #     #TODO Abe 20201016: Instead of _find_partitions_for_path, this should be convert_data_reference_to_batch_request
    #     partition = self.default_partitioner._find_partitions_for_path(data_reference)
    #     if partition == None:
    #         return None
    #
    #     return BatchRequest(
    #         execution_environment_name="FAKE_EXECUTION_ENVIRONMENT_NAME",
    #         data_connector_name=self.name,
    #         data_asset_name="FAKE_DATA_ASSET_NAME",
    #         partition_request=partition.definition,
    #     )

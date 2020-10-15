# -*- coding: utf-8 -*-

import copy
import itertools
from typing import List, Dict, Union, Callable, Any
from ruamel.yaml.comments import CommentedMap

import logging

from great_expectations.data_context.types.base import (
    PartitionerConfig,
    partitionerConfigSchema
)
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
    BatchSpec
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
        partitioners: dict = None,
        default_partitioner: str = None,
        assets: dict = None,
        config_params: dict = None,
        data_context_root_directory: str = None,
        **kwargs
    ):
        self._name = name

        self._partitioners = partitioners or {}
        self._default_partitioner = default_partitioner
        self._assets = assets
        self._config_params = config_params

        self._partitioners_cache: dict = {}

        # The partitions cache is a dictionary, which maintains lists of partitions for a data_asset_name as the key.
        self._partitions_cache: dict = {}

        self._data_context_root_directory = data_context_root_directory

    @property
    def name(self) -> str:
        return self._name

    @property
    def partitioners(self) -> dict:
        return self._partitioners

    @property
    def default_partitioner(self) -> str:
        return self._default_partitioner

    @property
    def assets(self) -> dict:
        return self._assets

    @property
    def config_params(self) -> dict:
        return self._config_params

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

    def get_partitioner(self, name: str):
        """Get the (named) Partitioner from a DataConnector)

        Args:
            name (str): name of Partitioner

        Returns:
            Partitioner (Partitioner)
        """
        if name in self._partitioners_cache:
            return self._partitioners_cache[name]
        elif name in self.partitioners:
            partitioner_config: dict = copy.deepcopy(
                self.partitioners[name]
            )
        else:
            raise ge_exceptions.PartitionerError(
                f'Unable to load partitioner "{name}" -- no configuration found or invalid configuration.'
            )
        partitioner_config: CommentedMap = partitionerConfigSchema.load(
            partitioner_config
        )
        partitioner: Partitioner = self._build_partitioner_from_config(
            name=name, config=partitioner_config
        )
        self._partitioners_cache[name] = partitioner
        return partitioner

    def _build_partitioner_from_config(self, name: str, config: CommentedMap):
        """Build a Partitioner using the provided configuration and return the newly-built Partitioner."""
        # We convert from the type back to a dictionary for purposes of instantiation
        if isinstance(config, PartitionerConfig):
            config: dict = partitionerConfigSchema.dump(config)
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

    def get_partitioner_for_data_asset(self, data_asset_name: str = None) -> Partitioner:
        partitioner_name: str
        data_asset_config_exists: bool = data_asset_name and self.assets and self.assets.get(data_asset_name)
        if data_asset_config_exists and self.assets[data_asset_name].get("partitioner"):
            partitioner_name = self.assets[data_asset_name]["partitioner"]
        else:
            partitioner_name = self.default_partitioner
        partitioner: Partitioner
        if partitioner_name is None:
            raise ge_exceptions.BatchSpecError(
                message=f'''
No partitioners found for data connector "{self.name}" -- at least one partitioner must be configured for a data
connector and the default_partitioner set to one of the configured partitioners.
                '''
            )
        else:
            partitioner = self.get_partitioner(name=partitioner_name)
        return partitioner

    def _build_batch_spec(self, batch_request: BatchRequest, partition: Partition) -> BatchSpec:
        if not batch_request.data_asset_name:
            raise ge_exceptions.BatchSpecError("Batch request must have a data_asset_name.")

        batch_spec_scaffold: BatchSpec
        batch_spec_passthrough: BatchSpec = batch_request.batch_spec_passthrough
        if batch_spec_passthrough is None:
            batch_spec_scaffold = BatchSpec()
        else:
            batch_spec_scaffold = copy.deepcopy(batch_spec_passthrough)

        data_asset_name: str = batch_request.data_asset_name
        batch_spec_scaffold["data_asset_name"] = data_asset_name

        batch_spec: BatchSpec = self._build_batch_spec_from_partition(
            partition=partition, batch_request=batch_request, batch_spec=batch_spec_scaffold
        )

        return batch_spec

    def _build_batch_spec_from_partition(
        self,
        partition: Partition,
        batch_request: BatchRequest,
        batch_spec: BatchSpec
    ) -> BatchSpec:
        raise NotImplementedError

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

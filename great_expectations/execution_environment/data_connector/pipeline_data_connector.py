from typing import Union, List, Any

import logging

from great_expectations.execution_environment.data_connector.partitioner.partitioner import Partitioner
from great_expectations.execution_environment.data_connector.partitioner.partition_query import PartitionQuery
from great_expectations.execution_environment.data_connector.partitioner.partition import Partition
from great_expectations.execution_environment.data_connector.data_connector import DataConnector
from great_expectations.core.batch import BatchRequest
from great_expectations.core.id_dict import (
    PartitionDefinitionSubset,
    BatchSpec
)
from great_expectations.execution_environment.types.batch_spec import InMemoryBatchSpec

logger = logging.getLogger(__name__)


class PipelineDataConnector(DataConnector):
    DEFAULT_DATA_ASSET_NAME: str = "IN_MEMORY_DATA_ASSET"

    def __init__(
        self,
        name: str,
        partitioners: dict = None,
        default_partitioner: str = None,
        assets: dict = None,
        data_context_root_directory: str = None,
        **kwargs
    ):
        logger.debug(f'Constructing PipelineDataConnector "{name}".')
        super().__init__(
            name=name,
            partitioners=partitioners,
            default_partitioner=default_partitioner,
            assets=assets,
            data_context_root_directory=data_context_root_directory,
            **kwargs
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
        # TODO: <Alex>TODO: Each specific data_connector should verify the given partitioner against the list of supported partitioners.</Alex>
        pipeline_data_asset_name: str = self.DEFAULT_DATA_ASSET_NAME
        if data_asset_name and self.assets and data_asset_name in self.assets:
            pipeline_data_asset_name = data_asset_name
        partition_name: Union[str, None] = None
        if partition_query:
            partition_name = partition_query.partition_name
        # TODO: <Alex>For the future multi-batch support, this can become a list of partition configurations.</Alex>
        partition_config: dict = {
            "name": partition_name,
            "data_asset_name": pipeline_data_asset_name,
            "definition": runtime_parameters,
            "data_reference": in_memory_dataset
        }
        return partitioner.find_or_create_partitions(
            data_asset_name=data_asset_name,
            partition_query=partition_query,
            runtime_parameters=runtime_parameters,
            repartition=repartition,
            # The partition_config parameter is for the specific partitioners, working under the present data connector.
            partition_config=partition_config
        )

    def _build_batch_spec_from_partition(
        self,
        partition: Partition,
        batch_request: BatchRequest,
        batch_spec: BatchSpec
    ) -> InMemoryBatchSpec:
        """
        Args:
            partition:
            batch_request:
            batch_spec:
        Returns:
            batch_spec
        """
        if not batch_spec.get("dataset"):
            in_memory_dataset: Any = partition.data_reference
            batch_spec["dataset"] = in_memory_dataset
        return InMemoryBatchSpec(batch_spec)

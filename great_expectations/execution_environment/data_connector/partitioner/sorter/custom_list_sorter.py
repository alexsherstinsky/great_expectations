# -*- coding: utf-8 -*-

from typing import Any, List

import logging

from great_expectations.execution_environment.data_connector.partitioner.partition import Partition
from great_expectations.execution_environment.data_connector.partitioner.sorter.sorter import Sorter
import great_expectations.exceptions as ge_exceptions

logger = logging.getLogger(__name__)


class CustomListSorter(Sorter):
    """
    CustomListSorter
        - The CustomListSorter is able to sort partitions values according to a user-provided custom list.
    """
    def __init__(self, name: str, orderby: str = "asc", reference_list: List[str] = None):
        super().__init__(name=name, orderby=orderby)

        self._reference_list = self._validate_reference_list(reference_list=reference_list)

    @staticmethod
    def _validate_reference_list(reference_list: List[str] = None) -> List[str]:
        if not (reference_list and isinstance(reference_list, list)):
            raise ge_exceptions.SorterError(
                "CustomListSorter requires reference_list which was not provided."
            )
        for item in reference_list:
            if not isinstance(item, str):
                raise ge_exceptions.SorterError(
                    f"Items in reference list for CustomListSorter must have string type (actual type is `{str(type(item))}`)."
                )
        return reference_list

    def get_partition_key(self, partition: Partition) -> Any:
        partition_definition: dict = partition.definition
        partition_value: Any = partition_definition[self.name]
        if partition_value in self._reference_list:
            return self._reference_list.index(partition_value)
        else:
            raise ge_exceptions.SorterError(f'Source {partition_value} was not found in Reference list.  Try again...')

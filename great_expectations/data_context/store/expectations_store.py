from great_expectations.common import (
    substitute_none_for_missing,
    instantiate_class_from_config,
    convert_to_string_and_escape,
    load_class,
    safe_mmkdir,
    format_dict_for_error_message,
    substitute_config_variable,
    substitute_all_config_variables,
    file_relative_path,
)
from great_expectations.core import ExpectationSuiteSchema
from great_expectations.data_context.store.database_store_backend import DatabaseStoreBackend
from great_expectations.data_context.store.tuple_store_backend import TupleStoreBackend
from great_expectations.data_context.store.store import Store
from great_expectations.data_context.types.resource_identifiers import ExpectationSuiteIdentifier
import copy
import importlib
import six
from six import string_types
import inspect


class ExpectationsStore(Store):
    _key_class = ExpectationSuiteIdentifier

    def __init__(self, store_backend=None, runtime_environment=None):
        self._expectationSuiteSchema = ExpectationSuiteSchema(strict=True)

        if store_backend is not None:
            store_backend_module_name = store_backend.get("module_name", "great_expectations.data_context.store")
            store_backend_class_name = store_backend.get("class_name", "InMemoryStoreBackend")
            store_backend_class = load_class(store_backend_class_name, store_backend_module_name)

            if issubclass(store_backend_class, TupleStoreBackend):
                # Provide defaults for this common case
                store_backend["filepath_suffix"] = store_backend.get("filepath_suffix", ".json")
            elif issubclass(store_backend_class, DatabaseStoreBackend):
                # Provide defaults for this common case
                store_backend["table_name"] = store_backend.get("table_name", "ge_expectations_store")
                store_backend["key_columns"] = store_backend.get(
                    "key_columns", ["expectation_suite_name"]
                )

        super(ExpectationsStore, self).__init__(store_backend=store_backend, runtime_environment=runtime_environment)

    def serialize(self, key, value):
        return self._expectationSuiteSchema.dumps(value).data

    def deserialize(self, key, value):
        return self._expectationSuiteSchema.loads(value).data

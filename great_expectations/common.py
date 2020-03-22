# -*- coding: utf-8 -*-
from __future__ import division

# PYTHON 2 - py2 - update to ABC direct use rather than __metaclass__ once we drop py2 support
from abc import ABCMeta, abstractmethod

import sys
import logging
import os
import errno
import six
from six import (
    string_types,
    integer_types,
    PY3,
)
from marshmallow import (
    Schema,
    fields,
    post_load,
    ValidationError,
)

import importlib
from pydoc import locate
import copy
import re
import random
import inspect
import warnings
from collections import (
    OrderedDict,
    Hashable,
    Counter,
    defaultdict,
    namedtuple,
)

import json
import enum
from string import Template as pTemplate
import datetime
import uuid
from uuid import uuid4

#from builtins import str  # PY2 compatibility

from functools import wraps

import traceback

import altair as alt
import pandas as pd

from jinja2 import (
    DictLoader,
    ChoiceLoader,
    Environment,
    PackageLoader,
    FileSystemLoader,
    select_autoescape,
    contextfilter,
)
import mistune

from great_expectations import __version__ as ge_version
from great_expectations.render.types import (
    RenderedDocumentContent,
    RenderedSectionContent,
    RenderedComponentContent,
    RenderedContent,
    RenderedHeaderContent,
    RenderedTableContent,
    TextContent,
    RenderedStringTemplateContent,
    RenderedBulletListContent,
    RenderedMarkdownContent,
    CollapseContent,
    RenderedContentBlockContainer,
    RenderedGraphContent,
)

from great_expectations.render.util import ordinal, num_to_str

from great_expectations.data_context.types.base import DataContextConfig

import great_expectations.exceptions as ge_exceptions
from great_expectations.exceptions import (
    GreatExpectationsError,
    PluginModuleNotFoundError,
    PluginClassNotFoundError,
    InvalidConfigError,
    InvalidDataContextKeyError,
    DataContextError,
    StoreBackendError,
)

from great_expectations.core import (
    ExpectationConfiguration,
    ExpectationSuiteSchema,
    expectationSuiteSchema,
    ExpectationSuite,
    ExpectationSuiteValidationResultSchema,
    ExpectationSuiteValidationResult,
    ExpectationValidationResult,
    ensure_json_serializable,
    IDDict,
)
from great_expectations.core.id_dict import BatchKwargs
from great_expectations.core.data_context_key import DataContextKey
from great_expectations.core.metric import ValidationMetricIdentifier
from great_expectations.core.id_dict import BatchKwargs

# Gross legacy python 2 hacks
try:
    ModuleNotFoundError
except NameError:
    ModuleNotFoundError = ImportError

try:
    import sqlalchemy
    from sqlalchemy import (
        create_engine,
        Column,
        String,
        MetaData,
        Table,
        select,
        and_,
        column,
    )
    from sqlalchemy.engine.url import URL
except ImportError:
    sqlalchemy = None
    create_engine = None

try:
    from sqlalchemy.exc import OperationalError
except ModuleNotFoundError:
    OperationalError = RuntimeError

from great_expectations.data_asset.util import (
    recursively_convert_to_json_serializable,
    parse_result_format,
)

from great_expectations.validation_operators.util import send_slack_notification

# NOTE: Abe 2019/08/24 : This is first implementation of all these classes. Consider them UNSTABLE for now. 
# Updated: Aalex 2020/03/18 : This is a rearranged implementation of all these classes in order to support AWS Glue.  Consider them UNSTABLE for now. 

logger = logging.getLogger(__name__)
logging.captureWarnings(True)


def allocate_module_to_class_mappings():
    return {
        'great_expectations.render.renderer.site_builder': {
            'SiteBuilder': SiteBuilder,
            'DefaultSiteSectionBuilder': DefaultSiteSectionBuilder,
            'DefaultSiteIndexBuilder': DefaultSiteIndexBuilder,
            'CallToActionButton': CallToActionButton,
        },
        'great_expectations.render.renderer': {
            'SiteIndexPageRenderer': SiteIndexPageRenderer,
            'ValidationResultsPageRenderer': ValidationResultsPageRenderer,
            'ExpectationSuitePageRenderer': ExpectationSuitePageRenderer,
            'ProfilingResultsPageRenderer': ProfilingResultsPageRenderer,
        },
        'great_expectations.render.view': {
            'SiteIndexPageRenderer': SiteIndexPageRenderer,
            'NoOpTemplate': NoOpTemplate,
            'PrettyPrintTemplate': PrettyPrintTemplate,
            'DefaultJinjaView': DefaultJinjaView,
            'DefaultJinjaPageView': DefaultJinjaPageView,
            'DefaultJinjaIndexPageView': DefaultJinjaIndexPageView,
            'DefaultJinjaSectionView': DefaultJinjaSectionView,
            'DefaultJinjaComponentView': DefaultJinjaComponentView,
        },
        'great_expectations.render.renderer.column_section_renderer': {
            'ColumnSectionRenderer': ColumnSectionRenderer,
            'ProfilingResultsColumnSectionRenderer': ProfilingResultsColumnSectionRenderer,
            'ValidationResultsColumnSectionRenderer': ValidationResultsColumnSectionRenderer,
            'ExpectationSuiteColumnSectionRenderer': ExpectationSuiteColumnSectionRenderer,
        },
        'great_expectations.render.renderer.other_section_renderer': {
            'ProfilingResultsOverviewSectionRenderer': ProfilingResultsOverviewSectionRenderer,
        },
        'great_expectations.render.renderer.content_block': {
            'ProfilingOverviewTableContentBlockRenderer': ProfilingOverviewTableContentBlockRenderer,
            'ExpectationStringRenderer': ExpectationStringRenderer,
            'ExpectationSuiteBulletListContentBlockRenderer': ExpectationSuiteBulletListContentBlockRenderer,
            'ValidationResultsTableContentBlockRenderer': ValidationResultsTableContentBlockRenderer,
        },
        'great_expectations.render.renderer.slack_renderer': {
            'SlackRenderer': SlackRenderer,
        },
        'great_expectations.data_context.store': {
            'EvaluationParameterStore': EvaluationParameterStore,
            'MetricStore': MetricStore,
            'EvaluationParameterStore': EvaluationParameterStore,
            'InMemoryStoreBackend': InMemoryStoreBackend,
            'DatabaseStoreBackend': DatabaseStoreBackend,
            'ValidationsStore': ValidationsStore,
            'TupleS3StoreBackend': TupleS3StoreBackend,
            'ExpectationsStore': ExpectationsStore,
        },
        'great_expectations.validation_operators': {
            'ValidationOperator': ValidationOperator,
            'ActionListValidationOperator': ActionListValidationOperator,
            'WarningAndFailureExpectationSuitesValidationOperator': WarningAndFailureExpectationSuitesValidationOperator,
            'ValidationAction': ValidationAction,
            'NoOpAction': NoOpAction,
            'SlackNotificationAction': SlackNotificationAction,
            'StoreValidationResultAction': StoreValidationResultAction,
            'StoreEvaluationParametersAction': StoreEvaluationParametersAction,
            'StoreEvaluationParametersAction': StoreEvaluationParametersAction,
            'UpdateDataDocsAction': UpdateDataDocsAction,
        },
    }


def substitute_none_for_missing(kwargs, kwarg_list):
    """Utility function to plug Nones in when optional parameters are not specified in expectation kwargs.

    Example:
        Input:
            kwargs={"a":1, "b":2},
            kwarg_list=["c", "d"]

        Output: {"a":1, "b":2, "c": None, "d": None}

    This is helpful for standardizing the input objects for rendering functions.
    The alternative is lots of awkward `if "some_param" not in kwargs or kwargs["some_param"] == None:` clauses in renderers.
    """

    new_kwargs = copy.deepcopy(kwargs)
    for kwarg in kwarg_list:
        if kwarg not in new_kwargs:
            new_kwargs[kwarg] = None
    return new_kwargs


# TODO: Rename config to constructor_kwargs and config_defaults -> constructor_kwarg_default
# TODO: Improve error messages in this method. Since so much of our workflow is config-driven, this will be a *super* important part of DX.
def instantiate_class_from_config(config, runtime_environment, config_defaults=None):
    """Build a GE class from configuration dictionaries."""

    if config_defaults is None:
        config_defaults = {}

    config = copy.deepcopy(config)

    module_name = config.pop("module_name", None)
    if module_name is None:
        try:
            module_name = config_defaults.pop("module_name")
        except KeyError as e:
            raise KeyError("Neither config : {} nor config_defaults : {} contains a module_name key.".format(
                config, config_defaults,
            ))
    else:
        # Pop the value without using it, to avoid sending an unwanted value to the config_class
        config_defaults.pop("module_name", None)

    class_name = config.pop("class_name", None)
    if class_name is None:
        logger.warning("Instantiating class from config without an explicit class_name is dangerous. Consider adding "
                       "an explicit class_name for %s" % config.get("name"))
        try:
            class_name = config_defaults.pop("class_name")
        except KeyError as e:
            raise KeyError("Neither config : {} nor config_defaults : {} contains a class_name key.".format(
                config, config_defaults,
            ))
    else:
        # Pop the value without using it, to avoid sending an unwanted value to the config_class
        config_defaults.pop("class_name", None)

    class_ = load_class(class_name=class_name, module_name=module_name)
    if class_ is None:
        raise Exception('NONE CLASS FOR MODULE_NAME: {} ; CLASS_NAME: {}'.format(module_name, class_name))

    config_with_defaults = copy.deepcopy(config_defaults)
    config_with_defaults.update(config)
    if runtime_environment is not None:
        # If there are additional kwargs available in the runtime_environment requested by a
        # class to be instantiated, provide them
        if six.PY3:
            argspec = inspect.getfullargspec(class_.__init__)[0][1:]
        else:
            argspec = inspect.getargspec(class_.__init__)[0][1:]
        missing_args = set(argspec) - set(config_with_defaults.keys())
        config_with_defaults.update(
            {missing_arg: runtime_environment[missing_arg] for missing_arg in missing_args
             if missing_arg in runtime_environment}
        )
        # Add the entire runtime_environment as well if it's requested
        if "runtime_environment" in missing_args:
            config_with_defaults.update({"runtime_environment": runtime_environment})

    try:
        class_instance = class_(**config_with_defaults)
    except TypeError as e:
        raise TypeError("Couldn't instantiate class : {} with config : \n\t{}\n \n".format(
            class_name,
            format_dict_for_error_message(config_with_defaults)
        ) + str(e))

    return class_instance


def convert_to_string_and_escape(var):
    return re.sub(r"\$", r"$$", str(var))


def load_class(class_name, module_name):
    """Dynamically load a class from strings or raise a helpful error."""

    # TODO remove this nasty python 2 hack
    try:
        ModuleNotFoundError
    except NameError:
        ModuleNotFoundError = ImportError

    if module_name in LOCAL_MODULE_NAMES_LIST:
        return LOCAL_MODULE_TO_CLASS_MAPPING_DICT[module_name].get(class_name)

    try:
        loaded_module = importlib.import_module(module_name)
        class_ = getattr(loaded_module, class_name)
    except ModuleNotFoundError as e0:
        try:
            loaded_module = __import__(module_name, fromlist=[module_name])
            class_ = getattr(loaded_module, class_name)
        except ModuleNotFoundError as e1:
            try:
                fully_qualified_class_name = '.'.join([module_name, class_name])
                # https://stackoverflow.com/questions/547829/how-to-dynamically-load-a-python-class
                class_ = locate(fully_qualified_class_name)
            except Exception as e2:
                raise PluginModuleNotFoundError(module_name=module_name)
    except AttributeError as e:
        raise PluginClassNotFoundError(
            module_name=module_name,
            class_name=class_name
        )
    return class_


def safe_mmkdir(directory, exist_ok=True):
    """Simple wrapper since exist_ok is not available in python 2"""
    if not isinstance(directory, six.string_types):
        raise TypeError("directory must be of type str, not {0}".format({
            "directory_type": str(type(directory))
        }))

    if not exist_ok:
        raise ValueError(
            "This wrapper should only be used for exist_ok=True; it is designed to make porting easier later")
    try:
        os.makedirs(directory)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise


def format_dict_for_error_message(dict_):
    # TODO : Tidy this up a bit. Indentation isn't fully consistent.

    return '\n\t'.join('\t\t'.join((str(key), str(dict_[key]))) for key in dict_)


def substitute_config_variable(template_str, config_variables_dict):
    """
    This method takes a string, and if it contains a pattern ${SOME_VARIABLE} or $SOME_VARIABLE,
    returns a string where the pattern is replaced with the value of SOME_VARIABLE,
    otherwise returns the string unchanged.

    If the environment variable SOME_VARIABLE is set, the method uses its value for substitution.
    If it is not set, the value of SOME_VARIABLE is looked up in the config variables store (file).
    If it is not found there, the input string is returned as is.

    :param template_str: a string that might or might not be of the form ${SOME_VARIABLE}
            or $SOME_VARIABLE
    :param config_variables_dict: a dictionary of config variables. It is loaded from the
            config variables store (by default, "uncommitted/config_variables.yml file)
    :return:
    """
    if template_str is None:
        return template_str

    try:
        match = re.search(r'\$\{(.*?)\}', template_str) or re.search(r'\$([_a-z][_a-z0-9]*)', template_str)
    except TypeError:
        # If the value is not a string (e.g., a boolean), we should return it as is
        return template_str

    if match:
        config_variable_value = os.getenv(match.group(1))
        if not config_variable_value:
            config_variable_value = config_variables_dict.get(match.group(1))

        if config_variable_value:
            if match.start() == 0 and match.end() == len(template_str):
                return config_variable_value
            else:
                return template_str[:match.start()] + config_variable_value + template_str[match.end():]

        raise InvalidConfigError("Unable to find match for config variable {:s}. See https://great-expectations.readthedocs.io/en/latest/reference/data_context_reference.html#managing-environment-and-secrets".format(match.group(1)))

    return template_str


def substitute_all_config_variables(data, replace_variables_dict):
    """
    Substitute all config variables of the form ${SOME_VARIABLE} in a dictionary-like
    config object for their values.

    The method traverses the dictionary recursively.

    :param data:
    :param replace_variables_dict:
    :return: a dictionary with all the variables replaced with their values
    """
    if isinstance(data, DataContextConfig):
        data = data.as_dict()

    if isinstance(data, dict) or isinstance(data, OrderedDict):
        return {k: substitute_all_config_variables(v, replace_variables_dict) for
                k, v in data.items()}
    elif isinstance(data, list):
        return [substitute_all_config_variables(v, replace_variables_dict) for v in data]
    return substitute_config_variable(data, replace_variables_dict)


def file_relative_path(dunderfile, relative_path):
    """
    This function is useful when one needs to load a file that is
    relative to the position of the current file. (Such as when
    you encode a configuration file path in source file and want
    in runnable in any current working directory)

    It is meant to be used like the following:
    file_relative_path(__file__, 'path/relative/to/file')

    H/T https://github.com/dagster-io/dagster/blob/8a250e9619a49e8bff8e9aa7435df89c2d2ea039/python_modules/dagster/dagster/utils/__init__.py#L34
    """
    return os.path.join(os.path.dirname(dunderfile), relative_path)


class DataAsset(object):

    # This should in general only be changed when a subclass *adds expectations* or *changes expectation semantics*
    # That way, multiple backends can implement the same data_asset_type
    _data_asset_type = "DataAsset"

    def __init__(self, *args, **kwargs):
        """
        Initialize the DataAsset.

        :param profiler (profiler class) = None: The profiler that should be run on the data_asset to
            build a baseline expectation suite.

        Note: DataAsset is designed to support multiple inheritance (e.g. PandasDataset inherits from both a
        Pandas DataFrame and Dataset which inherits from DataAsset), so it accepts generic *args and **kwargs arguments
        so that they can also be passed to other parent classes. In python 2, there isn't a clean way to include all of
        *args, **kwargs, and a named kwarg...so we use the inelegant solution of popping from kwargs, leaving the
        support for the profiler parameter not obvious from the signature.

        """
        interactive_evaluation = kwargs.pop("interactive_evaluation", True)
        profiler = kwargs.pop("profiler", None)
        expectation_suite = kwargs.pop("expectation_suite", None)
        expectation_suite_name = kwargs.pop("expectation_suite_name", None)
        data_context = kwargs.pop("data_context", None)
        batch_kwargs = kwargs.pop("batch_kwargs", BatchKwargs(ge_batch_id=str(uuid.uuid1())))
        batch_parameters = kwargs.pop("batch_parameters", {})
        batch_markers = kwargs.pop("batch_markers", {})

        if "autoinspect_func" in kwargs:
            warnings.warn("Autoinspect_func is no longer supported; use a profiler instead (migration is easy!).",
                          category=DeprecationWarning)
        super(DataAsset, self).__init__(*args, **kwargs)
        self._config = {
            "interactive_evaluation": interactive_evaluation
        }
        self._initialize_expectations(
            expectation_suite=expectation_suite,
            expectation_suite_name=expectation_suite_name
        )
        self._data_context = data_context
        self._batch_kwargs = BatchKwargs(batch_kwargs)
        self._batch_markers = batch_markers
        self._batch_parameters = batch_parameters

        # This special state variable tracks whether a validation run is going on, which will disable
        # saving expectation config objects
        self._active_validation = False
        if profiler is not None:
            profiler.profile(self)
        if data_context and hasattr(data_context, '_expectation_explorer_manager'):
            self.set_default_expectation_argument("include_config", True)

    def autoinspect(self, profiler):
        """Deprecated: use profile instead.

        Use the provided profiler to evaluate this data_asset and assign the resulting expectation suite as its own.

        Args:
            profiler: The profiler to use

        Returns:
            tuple(expectation_suite, validation_results)
        """
        warnings.warn("The term autoinspect is deprecated and will be removed in a future release. Please use 'profile'\
        instead.")
        expectation_suite, validation_results = profiler.profile(self)
        return expectation_suite, validation_results

    def profile(self, profiler):
        """Use the provided profiler to evaluate this data_asset and assign the resulting expectation suite as its own.

        Args:
            profiler: The profiler to use

        Returns:
            tuple(expectation_suite, validation_results)

        """
        expectation_suite, validation_results = profiler.profile(self)
        return expectation_suite, validation_results

    #TODO: add warning if no expectation_explorer_manager and how to turn on
    def edit_expectation_suite(self):
        return self._data_context._expectation_explorer_manager.edit_expectation_suite(self)

    @classmethod
    def expectation(cls, method_arg_names):
        """Manages configuration and running of expectation objects.

        Expectation builds and saves a new expectation configuration to the DataAsset object. It is the core decorator \
        used by great expectations to manage expectation configurations.

        Args:
            method_arg_names (List) : An ordered list of the arguments used by the method implementing the expectation \
                (typically the result of inspection). Positional arguments are explicitly mapped to \
                keyword arguments when the expectation is run.

        Notes:
            Intermediate decorators that call the core @expectation decorator will most likely need to pass their \
            decorated methods' signature up to the expectation decorator. For example, the MetaPandasDataset \
            column_map_expectation decorator relies on the DataAsset expectation decorator, but will pass through the \
            signature from the implementing method.

            @expectation intercepts and takes action based on the following parameters:
                * include_config (boolean or None) : \
                    If True, then include the generated expectation config as part of the result object. \
                    For more detail, see :ref:`include_config`.
                * catch_exceptions (boolean or None) : \
                    If True, then catch exceptions and include them as part of the result object. \
                    For more detail, see :ref:`catch_exceptions`.
                * result_format (str or None) : \
                    Which output mode to use: `BOOLEAN_ONLY`, `BASIC`, `COMPLETE`, or `SUMMARY`.
                    For more detail, see :ref:`result_format <result_format>`.
                * meta (dict or None): \
                    A JSON-serializable dictionary (nesting allowed) that will be included in the output without \
                    modification. For more detail, see :ref:`meta`.
        """
        def outer_wrapper(func):
            @wraps(func)
            def wrapper(self, *args, **kwargs):

                # Get the name of the method
                method_name = func.__name__

                # Combine all arguments into a single new "all_args" dictionary to name positional parameters
                all_args = dict(zip(method_arg_names, args))
                all_args.update(kwargs)

                # Unpack display parameters; remove them from all_args if appropriate
                if "include_config" in kwargs:
                    include_config = kwargs["include_config"]
                    del all_args["include_config"]
                else:
                    include_config = self.default_expectation_args["include_config"]

                if "catch_exceptions" in kwargs:
                    catch_exceptions = kwargs["catch_exceptions"]
                    del all_args["catch_exceptions"]
                else:
                    catch_exceptions = self.default_expectation_args["catch_exceptions"]

                if "result_format" in kwargs:
                    result_format = kwargs["result_format"]
                else:
                    result_format = self.default_expectation_args["result_format"]

                # Extract the meta object for use as a top-level expectation_config holder
                if "meta" in kwargs:
                    meta = kwargs["meta"]
                    del all_args["meta"]
                else:
                    meta = None

                # Get the signature of the inner wrapper:
                if PY3:
                    argspec = inspect.getfullargspec(func)[0][1:]
                else:
                    argspec = inspect.getargspec(func)[0][1:]

                if "result_format" in argspec:
                    all_args["result_format"] = result_format
                else:
                    if "result_format" in all_args:
                        del all_args["result_format"]

                all_args = recursively_convert_to_json_serializable(all_args)

                # Patch in PARAMETER args, and remove locally-supplied arguments
                # This will become the stored config
                expectation_args = copy.deepcopy(all_args)

                if self._expectation_suite.evaluation_parameters:
                    evaluation_args = self._build_evaluation_parameters(
                        expectation_args,
                        self._expectation_suite.evaluation_parameters
                    )
                else:
                    evaluation_args = self._build_evaluation_parameters(
                        expectation_args, None)

                # Construct the expectation_config object
                expectation_config = ExpectationConfiguration(
                    expectation_type=method_name,
                    kwargs=expectation_args,
                    meta=meta
                )

                raised_exception = False
                exception_traceback = None
                exception_message = None

                # Finally, execute the expectation method itself
                if self._config.get("interactive_evaluation", True) or self._active_validation:
                    try:
                        return_obj = func(self, **evaluation_args)
                        if isinstance(return_obj, dict):
                            return_obj = ExpectationValidationResult(**return_obj)

                    except Exception as err:
                        if catch_exceptions:
                            raised_exception = True
                            exception_traceback = traceback.format_exc()
                            exception_message = "{}: {}".format(type(err).__name__, str(err))

                            return_obj = ExpectationValidationResult(success=False)

                        else:
                            raise err

                else:
                    return_obj = ExpectationValidationResult(expectation_config=copy.deepcopy(
                        expectation_config))

                # If validate has set active_validation to true, then we do not save the config to avoid
                # saving updating expectation configs to the same suite during validation runs
                if self._active_validation is True:
                    pass
                else:
                    # Append the expectation to the config.
                    self._append_expectation(expectation_config)

                if include_config:
                    return_obj.expectation_config = copy.deepcopy(expectation_config)

                # If there was no interactive evaluation, success will not have been computed.
                if return_obj.success is not None:
                    # Add a "success" object to the config
                    expectation_config.success_on_last_run = return_obj.success

                if catch_exceptions:
                    return_obj.exception_info = {
                        "raised_exception": raised_exception,
                        "exception_message": exception_message,
                        "exception_traceback": exception_traceback
                    }

                # Add meta to return object
                if meta is not None:
                    return_obj.meta = meta

                return_obj = recursively_convert_to_json_serializable(
                    return_obj)

                if self._data_context is not None:
                    return_obj = self._data_context.update_return_obj(self, return_obj)

                return return_obj

            return wrapper

        return outer_wrapper

    def _initialize_expectations(self, expectation_suite=None, expectation_suite_name=None):
        """Instantiates `_expectation_suite` as empty by default or with a specified expectation `config`.
        In addition, this always sets the `default_expectation_args` to:
            `include_config`: False,
            `catch_exceptions`: False,
            `output_format`: 'BASIC'

        By default, initializes data_asset_type to the name of the implementing class, but subclasses
        that have interoperable semantics (e.g. Dataset) may override that parameter to clarify their
        interoperability.

        Args:
            expectation_suite (json): \
                A json-serializable expectation config. \
                If None, creates default `_expectation_suite` with an empty list of expectations and \
                key value `data_asset_name` as `data_asset_name`.

            expectation_suite_name (string): \
                The name to assign to the `expectation_suite.expectation_suite_name`

        Returns:
            None
        """
        if expectation_suite is not None:
            if isinstance(expectation_suite, dict):
                expectation_suite = expectationSuiteSchema.load(expectation_suite).data
            else:
                expectation_suite = copy.deepcopy(expectation_suite)
            self._expectation_suite = expectation_suite

            if expectation_suite_name is not None:
                if self._expectation_suite.expectation_suite_name != expectation_suite_name:
                    logger.warning(
                        "Overriding existing expectation_suite_name {n1} with new name {n2}"
                        .format(n1=self._expectation_suite.expectation_suite_name, n2=expectation_suite_name)
                    )
                self._expectation_suite.expectation_suite_name = expectation_suite_name

        else:
            if expectation_suite_name is None:
                expectation_suite_name = "default"
            self._expectation_suite = ExpectationSuite(expectation_suite_name=expectation_suite_name)

        self._expectation_suite.data_asset_type = self._data_asset_type
        self.default_expectation_args = {
            "include_config": True,
            "catch_exceptions": False,
            "result_format": 'BASIC',
        }

    def _append_expectation(self, expectation_config):
        """Appends an expectation to `DataAsset._expectation_suite` and drops existing expectations of the same type.

           If `expectation_config` is a column expectation, this drops existing expectations that are specific to \
           that column and only if it is the same expectation type as `expectation_config`. Otherwise, if it's not a \
           column expectation, this drops existing expectations of the same type as `expectation config`. \
           After expectations of the same type are dropped, `expectation_config` is appended to \
           `DataAsset._expectation_suite`.

           Args:
               expectation_config (json): \
                   The JSON-serializable expectation to be added to the DataAsset expectations in `_expectation_suite`.

           Notes:
               May raise future errors once json-serializable tests are implemented to check for correct arg formatting

        """
        expectation_type = expectation_config.expectation_type

        # Test to ensure the new expectation is serializable.
        # FIXME: If it's not, are we sure we want to raise an error?
        # FIXME: Should we allow users to override the error?
        # FIXME: Should we try to convert the object using something like recursively_convert_to_json_serializable?
        # json.dumps(expectation_config)

        # Drop existing expectations with the same expectation_type.
        # For column_expectations, _append_expectation should only replace expectations
        # where the expectation_type AND the column match
        # !!! This is good default behavior, but
        # !!!    it needs to be documented, and
        # !!!    we need to provide syntax to override it.

        if 'column' in expectation_config.kwargs:
            column = expectation_config.kwargs['column']

            self._expectation_suite.expectations = [f for f in filter(
                lambda exp: (exp.expectation_type != expectation_type) or (
                    'column' in exp.kwargs and exp.kwargs['column'] != column),
                self._expectation_suite.expectations
            )]
        else:
            self._expectation_suite.expectations = [f for f in filter(
                lambda exp: exp.expectation_type != expectation_type,
                self._expectation_suite.expectations
            )]

        self._expectation_suite.expectations.append(expectation_config)

    def _copy_and_clean_up_expectation(self,
                                       expectation,
                                       discard_result_format_kwargs=True,
                                       discard_include_config_kwargs=True,
                                       discard_catch_exceptions_kwargs=True,
                                       ):
        """Returns copy of `expectation` without `success_on_last_run` and other specified key-value pairs removed

          Returns a copy of specified expectation will not have `success_on_last_run` key-value. The other key-value \
          pairs will be removed by default but will remain in the copy if specified.

          Args:
              expectation (json): \
                  The expectation to copy and clean.
              discard_result_format_kwargs (boolean): \
                  if True, will remove the kwarg `output_format` key-value pair from the copied expectation.
              discard_include_config_kwargs (boolean):
                  if True, will remove the kwarg `include_config` key-value pair from the copied expectation.
              discard_catch_exceptions_kwargs (boolean):
                  if True, will remove the kwarg `catch_exceptions` key-value pair from the copied expectation.

          Returns:
              A copy of the provided expectation with `success_on_last_run` and other specified key-value pairs removed
        """
        new_expectation = copy.deepcopy(expectation)

        if "success_on_last_run" in new_expectation:
            del new_expectation["success_on_last_run"]

        if discard_result_format_kwargs:
            if "result_format" in new_expectation.kwargs:
                del new_expectation.kwargs["result_format"]
                # discards["result_format"] += 1

        if discard_include_config_kwargs:
            if "include_config" in new_expectation.kwargs:
                del new_expectation.kwargs["include_config"]
                # discards["include_config"] += 1

        if discard_catch_exceptions_kwargs:
            if "catch_exceptions" in new_expectation.kwargs:
                del new_expectation.kwargs["catch_exceptions"]
                # discards["catch_exceptions"] += 1

        return new_expectation

    def _copy_and_clean_up_expectations_from_indexes(
        self,
        match_indexes,
        discard_result_format_kwargs=True,
        discard_include_config_kwargs=True,
        discard_catch_exceptions_kwargs=True,
    ):
        """Copies and cleans all expectations provided by their index in DataAsset._expectation_suite.expectations.

           Applies the _copy_and_clean_up_expectation method to multiple expectations, provided by their index in \
           `DataAsset,_expectation_suite.expectations`. Returns a list of the copied and cleaned expectations.

           Args:
               match_indexes (List): \
                   Index numbers of the expectations from `expectation_config.expectations` to be copied and cleaned.
               discard_result_format_kwargs (boolean): \
                   if True, will remove the kwarg `output_format` key-value pair from the copied expectation.
               discard_include_config_kwargs (boolean):
                   if True, will remove the kwarg `include_config` key-value pair from the copied expectation.
               discard_catch_exceptions_kwargs (boolean):
                   if True, will remove the kwarg `catch_exceptions` key-value pair from the copied expectation.

           Returns:
               A list of the copied expectations with `success_on_last_run` and other specified \
               key-value pairs removed.

           See also:
               _copy_and_clean_expectation
        """
        rval = []
        for i in match_indexes:
            rval.append(
                self._copy_and_clean_up_expectation(
                    self._expectation_suite.expectations[i],
                    discard_result_format_kwargs,
                    discard_include_config_kwargs,
                    discard_catch_exceptions_kwargs,
                )
            )

        return rval

    def find_expectation_indexes(self,
                                 expectation_type=None,
                                 column=None,
                                 expectation_kwargs=None
                                 ):
        """Find matching expectations within _expectation_config.
        Args:
            expectation_type=None                : The name of the expectation type to be matched.
            column=None                          : The name of the column to be matched.
            expectation_kwargs=None              : A dictionary of kwargs to match against.

        Returns:
            A list of indexes for matching expectation objects.
            If there are no matches, the list will be empty.
        """
        if expectation_kwargs is None:
            expectation_kwargs = {}

        if "column" in expectation_kwargs and column is not None and column is not expectation_kwargs["column"]:
            raise ValueError("Conflicting column names in remove_expectation: %s and %s" % (
                column, expectation_kwargs["column"]))

        if column is not None:
            expectation_kwargs["column"] = column

        match_indexes = []
        for i, exp in enumerate(self._expectation_suite.expectations):
            if expectation_type is None or (expectation_type == exp.expectation_type):
                # if column == None or ('column' not in exp['kwargs']) or
                # (exp['kwargs']['column'] == column) or (exp['kwargs']['column']==:
                match = True

                for k, v in expectation_kwargs.items():
                    if k in exp['kwargs'] and exp['kwargs'][k] == v:
                        continue
                    else:
                        match = False

                if match:
                    match_indexes.append(i)

        return match_indexes

    def find_expectations(self,
                          expectation_type=None,
                          column=None,
                          expectation_kwargs=None,
                          discard_result_format_kwargs=True,
                          discard_include_config_kwargs=True,
                          discard_catch_exceptions_kwargs=True,
                          ):
        """Find matching expectations within _expectation_config.
        Args:
            expectation_type=None                : The name of the expectation type to be matched.
            column=None                          : The name of the column to be matched.
            expectation_kwargs=None              : A dictionary of kwargs to match against.
            discard_result_format_kwargs=True    : In returned expectation object(s), \
            suppress the `result_format` parameter.
            discard_include_config_kwargs=True  : In returned expectation object(s), \
            suppress the `include_config` parameter.
            discard_catch_exceptions_kwargs=True : In returned expectation object(s), \
            suppress the `catch_exceptions` parameter.

        Returns:
            A list of matching expectation objects.
            If there are no matches, the list will be empty.
        """

        match_indexes = self.find_expectation_indexes(
            expectation_type,
            column,
            expectation_kwargs,
        )

        return self._copy_and_clean_up_expectations_from_indexes(
            match_indexes,
            discard_result_format_kwargs,
            discard_include_config_kwargs,
            discard_catch_exceptions_kwargs,
        )

    def remove_expectation(self,
                           expectation_type=None,
                           column=None,
                           expectation_kwargs=None,
                           remove_multiple_matches=False,
                           dry_run=False,
                           ):
        """Remove matching expectation(s) from _expectation_config.
        Args:
            expectation_type=None                : The name of the expectation type to be matched.
            column=None                          : The name of the column to be matched.
            expectation_kwargs=None              : A dictionary of kwargs to match against.
            remove_multiple_matches=False        : Match multiple expectations
            dry_run=False                        : Return a list of matching expectations without removing

        Returns:
            None, unless dry_run=True.
            If dry_run=True and remove_multiple_matches=False then return the expectation that *would be* removed.
            If dry_run=True and remove_multiple_matches=True then return a list of expectations that *would be* removed.

        Note:
            If remove_expectation doesn't find any matches, it raises a ValueError.
            If remove_expectation finds more than one matches and remove_multiple_matches!=True, it raises a ValueError.
            If dry_run=True, then `remove_expectation` acts as a thin layer to find_expectations, with the default \
            values for discard_result_format_kwargs, discard_include_config_kwargs, and discard_catch_exceptions_kwargs
        """

        match_indexes = self.find_expectation_indexes(
            expectation_type,
            column,
            expectation_kwargs,
        )

        if len(match_indexes) == 0:
            raise ValueError('No matching expectation found.')

        elif len(match_indexes) > 1:
            if not remove_multiple_matches:
                raise ValueError(
                    'Multiple expectations matched arguments. No expectations removed.')
            else:

                if not dry_run:
                    self._expectation_suite.expectations = [i for j, i in enumerate(
                        self._expectation_suite.expectations) if j not in match_indexes]
                else:
                    return self._copy_and_clean_up_expectations_from_indexes(match_indexes)

        else:  # Exactly one match
            expectation = self._copy_and_clean_up_expectation(
                self._expectation_suite.expectations[match_indexes[0]]
            )

            if not dry_run:
                del self._expectation_suite.expectations[match_indexes[0]]

            else:
                if remove_multiple_matches:
                    return [expectation]
                else:
                    return expectation

    def set_config_value(self, key, value):
        self._config[key] = value

    def get_config_value(self, key):
        return self._config[key]

    @property
    def batch_kwargs(self):
        return self._batch_kwargs

    @property
    def batch_id(self):
        return self.batch_kwargs.to_id()

    @property
    def batch_markers(self):
        return self._batch_markers

    @property
    def batch_parameters(self):
        return self._batch_parameters

    def discard_failing_expectations(self):
        res = self.validate(only_return_failures=True).results
        if any(res):
            for item in res:
                self.remove_expectation(expectation_type=item.expectation_config.expectation_type,
                                        expectation_kwargs=item.expectation_config['kwargs'])
            warnings.warn(
                "Removed %s expectations that were 'False'" % len(res))

    def get_default_expectation_arguments(self):
        """Fetch default expectation arguments for this data_asset

        Returns:
            A dictionary containing all the current default expectation arguments for a data_asset

            Ex::

                {
                    "include_config" : True,
                    "catch_exceptions" : False,
                    "result_format" : 'BASIC'
                }

        See also:
            set_default_expectation_arguments
        """
        return self.default_expectation_args

    def set_default_expectation_argument(self, argument, value):
        """Set a default expectation argument for this data_asset

        Args:
            argument (string): The argument to be replaced
            value : The New argument to use for replacement

        Returns:
            None

        See also:
            get_default_expectation_arguments
        """
        # !!! Maybe add a validation check here?

        self.default_expectation_args[argument] = value

    def get_expectations_config(self,
                                discard_failed_expectations=True,
                                discard_result_format_kwargs=True,
                                discard_include_config_kwargs=True,
                                discard_catch_exceptions_kwargs=True,
                                suppress_warnings=False
                                ):
        warnings.warn("get_expectations_config is deprecated, and will be removed in a future release. " +
                      "Please use get_expectation_suite instead.", DeprecationWarning)
        return self.get_expectation_suite(
            discard_failed_expectations,
            discard_result_format_kwargs,
            discard_include_config_kwargs,
            discard_catch_exceptions_kwargs,
            suppress_warnings
            )

    def get_expectation_suite(self,
                              discard_failed_expectations=True,
                              discard_result_format_kwargs=True,
                              discard_include_config_kwargs=True,
                              discard_catch_exceptions_kwargs=True,
                              suppress_warnings=False
                              ):
        """Returns _expectation_config as a JSON object, and perform some cleaning along the way.

        Args:
            discard_failed_expectations (boolean): \
                Only include expectations with success_on_last_run=True in the exported config.  Defaults to `True`.
            discard_result_format_kwargs (boolean): \
                In returned expectation objects, suppress the `result_format` parameter. Defaults to `True`.
            discard_include_config_kwargs (boolean): \
                In returned expectation objects, suppress the `include_config` parameter. Defaults to `True`.
            discard_catch_exceptions_kwargs (boolean): \
                In returned expectation objects, suppress the `catch_exceptions` parameter.  Defaults to `True`.

        Returns:
            An expectation suite.

        Note:
            get_expectation_suite does not affect the underlying expectation suite at all. The returned suite is a \
             copy of _expectation_suite, not the original object.
        """

        expectation_suite = copy.deepcopy(self._expectation_suite)
        expectations = expectation_suite.expectations

        discards = defaultdict(int)

        if discard_failed_expectations:
            new_expectations = []

            for expectation in expectations:
                # Note: This is conservative logic.
                # Instead of retaining expectations IFF success==True, it discard expectations IFF success==False.
                # In cases where expectation.success is missing or None, expectations are *retained*.
                # Such a case could occur if expectations were loaded from a config file and never run.
                if expectation.success_on_last_run is False:
                    discards["failed_expectations"] += 1
                else:
                    new_expectations.append(expectation)

            expectations = new_expectations

        message = "\t%d expectation(s) included in expectation_suite." % len(expectations)

        if discards["failed_expectations"] > 0 and not suppress_warnings:
            message += " Omitting %d expectation(s) that failed when last run; set " \
                       "discard_failed_expectations=False to include them." \
                        % discards["failed_expectations"]

        for expectation in expectations:
            # FIXME: Factor this out into a new function. The logic is duplicated in remove_expectation,
            #  which calls _copy_and_clean_up_expectation
            expectation.success_on_last_run = None

            if discard_result_format_kwargs:
                if "result_format" in expectation.kwargs:
                    del expectation.kwargs["result_format"]
                    discards["result_format"] += 1

            if discard_include_config_kwargs:
                if "include_config" in expectation.kwargs:
                    del expectation.kwargs["include_config"]
                    discards["include_config"] += 1

            if discard_catch_exceptions_kwargs:
                if "catch_exceptions" in expectation.kwargs:
                    del expectation.kwargs["catch_exceptions"]
                    discards["catch_exceptions"] += 1

        settings_message = ""

        if discards["result_format"] > 0 and not suppress_warnings:
            settings_message += " result_format"

        if discards["include_config"] > 0 and not suppress_warnings:
            settings_message += " include_config"

        if discards["catch_exceptions"] > 0 and not suppress_warnings:
            settings_message += " catch_exceptions"

        if len(settings_message) > 1:  # Only add this if we added one of the settings above.
            settings_message += " settings filtered."

        expectation_suite.expectations = expectations
        logger.info(message + settings_message)
        return expectation_suite

    def save_expectation_suite(
        self,
        filepath=None,
        discard_failed_expectations=True,
        discard_result_format_kwargs=True,
        discard_include_config_kwargs=True,
        discard_catch_exceptions_kwargs=True,
        suppress_warnings=False
    ):
        """Writes ``_expectation_config`` to a JSON file.

           Writes the DataAsset's expectation config to the specified JSON ``filepath``. Failing expectations \
           can be excluded from the JSON expectations config with ``discard_failed_expectations``. The kwarg key-value \
           pairs :ref:`result_format`, :ref:`include_config`, and :ref:`catch_exceptions` are optionally excluded from \
           the JSON expectations config.

           Args:
               filepath (string): \
                   The location and name to write the JSON config file to.
               discard_failed_expectations (boolean): \
                   If True, excludes expectations that do not return ``success = True``. \
                   If False, all expectations are written to the JSON config file.
               discard_result_format_kwargs (boolean): \
                   If True, the :ref:`result_format` attribute for each expectation is not written to the JSON config \
                   file.
               discard_include_config_kwargs (boolean): \
                   If True, the :ref:`include_config` attribute for each expectation is not written to the JSON config \
                   file.
               discard_catch_exceptions_kwargs (boolean): \
                   If True, the :ref:`catch_exceptions` attribute for each expectation is not written to the JSON \
                   config file.
               suppress_warnings (boolean): \
                  It True, all warnings raised by Great Expectations, as a result of dropped expectations, are \
                  suppressed.

        """
        expectation_suite = self.get_expectation_suite(
            discard_failed_expectations,
            discard_result_format_kwargs,
            discard_include_config_kwargs,
            discard_catch_exceptions_kwargs,
            suppress_warnings
        )
        if filepath is None and self._data_context is not None:
            self._data_context.save_expectation_suite(expectation_suite)
        elif filepath is not None:
            with open(filepath, 'w') as outfile:
                json.dump(expectationSuiteSchema.dump(expectation_suite).data, outfile, indent=2)
        else:
            raise ValueError("Unable to save config: filepath or data_context must be available.")

    def validate(self,
                 expectation_suite=None,
                 run_id=None,
                 data_context=None,
                 evaluation_parameters=None,
                 catch_exceptions=True,
                 result_format=None,
                 only_return_failures=False):
        """Generates a JSON-formatted report describing the outcome of all expectations.

        Use the default expectation_suite=None to validate the expectations config associated with the DataAsset.

        Args:
            expectation_suite (json or None): \
                If None, uses the expectations config generated with the DataAsset during the current session. \
                If a JSON file, validates those expectations.
            run_id (str): \
                A string used to identify this validation result as part of a collection of validations. See \
                DataContext for more information.
            data_context (DataContext): \
                A datacontext object to use as part of validation for binding evaluation parameters and \
                registering validation results.
            evaluation_parameters (dict or None): \
                If None, uses the evaluation_paramters from the expectation_suite provided or as part of the \
                data_asset. If a dict, uses the evaluation parameters in the dictionary.
            catch_exceptions (boolean): \
                If True, exceptions raised by tests will not end validation and will be described in the returned \
                report.
            result_format (string or None): \
                If None, uses the default value ('BASIC' or as specified). \
                If string, the returned expectation output follows the specified format ('BOOLEAN_ONLY','BASIC', \
                etc.).
            only_return_failures (boolean): \
                If True, expectation results are only returned when ``success = False`` \

        Returns:
            A JSON-formatted dictionary containing a list of the validation results. \
            An example of the returned format::

            {
              "results": [
                {
                  "unexpected_list": [unexpected_value_1, unexpected_value_2],
                  "expectation_type": "expect_*",
                  "kwargs": {
                    "column": "Column_Name",
                    "output_format": "SUMMARY"
                  },
                  "success": true,
                  "raised_exception: false.
                  "exception_traceback": null
                },
                {
                  ... (Second expectation results)
                },
                ... (More expectations results)
              ],
              "success": true,
              "statistics": {
                "evaluated_expectations": n,
                "successful_expectations": m,
                "unsuccessful_expectations": n - m,
                "success_percent": m / n
              }
            }

        Notes:
           If the configuration object was built with a different version of great expectations then the \
           current environment. If no version was found in the configuration file.

        Raises:
           AttributeError - if 'catch_exceptions'=None and an expectation throws an AttributeError
        """
        try:
            self._active_validation = True

            # If a different validation data context was provided, override
            validate__data_context = self._data_context
            if data_context is None and self._data_context is not None:
                data_context = self._data_context
            elif data_context is not None:
                # temporarily set self._data_context so it is used inside the expectation decorator
                self._data_context = data_context

            results = []

            if expectation_suite is None:
                expectation_suite = self.get_expectation_suite(
                    discard_failed_expectations=False,
                    discard_result_format_kwargs=False,
                    discard_include_config_kwargs=False,
                    discard_catch_exceptions_kwargs=False,
                )
            elif isinstance(expectation_suite, string_types):
                try:
                    with open(expectation_suite, 'r') as infile:
                        expectation_suite = expectationSuiteSchema.loads(infile.read()).data
                except ValidationError:
                    raise
                except IOError:
                    raise GreatExpectationsError(
                        "Unable to load expectation suite: IO error while reading %s" % expectation_suite)
            elif not isinstance(expectation_suite, ExpectationSuite):
                logger.error("Unable to validate using the provided value for expectation suite; does it need to be "
                             "loaded from a dictionary?")
                return ExpectationValidationResult(success=False)
            # Evaluation parameter priority is
            # 1. from provided parameters
            # 2. from expectation configuration
            # 3. from data context
            # So, we load them in reverse order

            if data_context is not None:
                runtime_evaluation_parameters = \
                    data_context.evaluation_parameter_store.get_bind_params(run_id)
            else:
                runtime_evaluation_parameters = {}

            if expectation_suite.evaluation_parameters:
                runtime_evaluation_parameters.update(expectation_suite.evaluation_parameters)

            if evaluation_parameters is not None:
                runtime_evaluation_parameters.update(evaluation_parameters)

            # Convert evaluation parameters to be json-serializable
            runtime_evaluation_parameters = recursively_convert_to_json_serializable(runtime_evaluation_parameters)

            # Warn if our version is different from the version in the configuration
            try:
                if expectation_suite.meta['great_expectations.__version__'] != ge_version:
                    warnings.warn(
                        "WARNING: This configuration object was built using version %s of great_expectations, but "
                        "is currently being validated by version %s."
                        % (expectation_suite.meta['great_expectations.__version__'], ge_version))
            except KeyError:
                warnings.warn(
                    "WARNING: No great_expectations version found in configuration object.")

            ###
            # This is an early example of what will become part of the ValidationOperator
            # This operator would be dataset-semantic aware
            # Adding now to simply ensure we can be slightly better at ordering our expectation evaluation
            ###

            # Group expectations by column
            columns = {}

            for expectation in expectation_suite.expectations:
                if "column" in expectation.kwargs and isinstance(expectation.kwargs["column"], Hashable):
                    column = expectation.kwargs["column"]
                else:
                    column = "_nocolumn"
                if column not in columns:
                    columns[column] = []
                columns[column].append(expectation)

            expectations_to_evaluate = []
            for col in columns:
                expectations_to_evaluate.extend(columns[col])

            for expectation in expectations_to_evaluate:

                try:
                    # copy the config so we can modify it below if needed
                    expectation = copy.deepcopy(expectation)

                    expectation_method = getattr(self, expectation.expectation_type)

                    if result_format is not None:
                        expectation.kwargs.update({'result_format': result_format})

                    # A missing parameter should raise a KeyError
                    evaluation_args = self._build_evaluation_parameters(
                        expectation.kwargs, runtime_evaluation_parameters)

                    result = expectation_method(
                        catch_exceptions=catch_exceptions,
                        include_config=True,
                        **evaluation_args
                    )

                except Exception as err:
                    if catch_exceptions:
                        raised_exception = True
                        exception_traceback = traceback.format_exc()

                        result = ExpectationValidationResult(
                            success=False,
                            exception_info={
                                "raised_exception": raised_exception,
                                "exception_traceback": exception_traceback,
                                "exception_message": str(err)
                            }
                        )

                    else:
                        raise err

                # if include_config:
                result.expectation_config = expectation

                # Add an empty exception_info object if no exception was caught
                if catch_exceptions and result.exception_info is None:
                    result.exception_info = {
                        "raised_exception": False,
                        "exception_traceback": None,
                        "exception_message": None
                    }

                results.append(result)

            statistics = _calc_validation_statistics(results)

            if only_return_failures:
                abbrev_results = []
                for exp in results:
                    if not exp.success:
                        abbrev_results.append(exp)
                results = abbrev_results

            expectation_suite_name = expectation_suite.expectation_suite_name

            if run_id is None:
                run_id = datetime.datetime.utcnow().strftime("%Y%m%dT%H%M%S.%fZ")

            result = ExpectationSuiteValidationResult(
                results=results,
                success=statistics.success,
                statistics={
                    "evaluated_expectations": statistics.evaluated_expectations,
                    "successful_expectations": statistics.successful_expectations,
                    "unsuccessful_expectations": statistics.unsuccessful_expectations,
                    "success_percent": statistics.success_percent,
                },
                evaluation_parameters=runtime_evaluation_parameters,
                meta={
                    "great_expectations.__version__": ge_version,
                    "expectation_suite_name": expectation_suite_name,
                    "run_id": run_id,
                    "batch_kwargs": self.batch_kwargs,
                    "batch_markers": self.batch_markers,
                    "batch_parameters": self.batch_parameters
                }
            )

            self._data_context = validate__data_context
        except Exception:
            raise
        finally:
            self._active_validation = False

        return result

    def get_evaluation_parameter(self, parameter_name, default_value=None):
        """Get an evaluation parameter value that has been stored in meta.

        Args:
            parameter_name (string): The name of the parameter to store.
            default_value (any): The default value to be returned if the parameter is not found.

        Returns:
            The current value of the evaluation parameter.
        """
        if parameter_name in self._expectation_suite.evaluation_parameters:
            return self._expectation_suite.evaluation_parameters[parameter_name]
        else:
            return default_value

    def set_evaluation_parameter(self, parameter_name, parameter_value):
        """Provide a value to be stored in the data_asset evaluation_parameters object and used to evaluate
        parameterized expectations.

        Args:
            parameter_name (string): The name of the kwarg to be replaced at evaluation time
            parameter_value (any): The value to be used
        """
        self._expectation_suite.evaluation_parameters.update(
            {parameter_name: parameter_value})

    def add_citation(self, comment, batch_kwargs=None, batch_markers=None, batch_parameters=None, citation_date=None):
        if batch_kwargs is None:
            batch_kwargs = self.batch_kwargs
        if batch_markers is None:
            batch_markers = self.batch_markers
        if batch_parameters is None:
            batch_parameters = self.batch_parameters
        self._expectation_suite.add_citation(comment, batch_kwargs=batch_kwargs, batch_markers=batch_markers,
                                             batch_parameters=batch_parameters,
                                             citation_date=citation_date)

    # PENDING DELETION: 20200130 - JPC - Ready for deletion upon release of 0.9.0 with no data_asset_name
    #
    # @property
    # def data_asset_name(self):
    #     """Gets the current name of this data_asset as stored in the expectations configuration."""
    #     return self._expectation_suite.data_asset_name
    #
    # @data_asset_name.setter
    # def data_asset_name(self, data_asset_name):
    #     """Sets the name of this data_asset as stored in the expectations configuration."""
    #     self._expectation_suite.data_asset_name = data_asset_name

    @property
    def expectation_suite_name(self):
        """Gets the current expectation_suite name of this data_asset as stored in the expectations configuration."""
        return self._expectation_suite.expectation_suite_name

    @expectation_suite_name.setter
    def expectation_suite_name(self, expectation_suite_name):
        """Sets the expectation_suite name of this data_asset as stored in the expectations configuration."""
        self._expectation_suite.expectation_suite_name = expectation_suite_name

    def _build_evaluation_parameters(self, expectation_args, evaluation_parameters):
        """Build a dictionary of parameters to evaluate, using the provided evaluation_parameters,
        AND mutate expectation_args by removing any parameter values passed in as temporary values during
        exploratory work.
        """

        evaluation_args = copy.deepcopy(expectation_args)

        # Iterate over arguments, and replace $PARAMETER-defined args with their
        # specified parameters.
        for key, value in evaluation_args.items():
            if isinstance(value, dict) and '$PARAMETER' in value:
                # First, check to see whether an argument was supplied at runtime
                # If it was, use that one, but remove it from the stored config
                if "$PARAMETER." + value["$PARAMETER"] in value:
                    evaluation_args[key] = evaluation_args[key]["$PARAMETER." +
                                                                value["$PARAMETER"]]
                    del expectation_args[key]["$PARAMETER." +
                                              value["$PARAMETER"]]
                elif evaluation_parameters is not None and value["$PARAMETER"] in evaluation_parameters:
                    evaluation_args[key] = evaluation_parameters[value['$PARAMETER']]
                elif not self._config.get("interactive_evaluation", True):
                    pass
                else:
                    raise KeyError(
                        "No value found for $PARAMETER " + value["$PARAMETER"])

        return evaluation_args

    ###
    #
    # Output generation
    #
    ###

    def _format_map_output(
        self,
        result_format,
        success,
        element_count,
        nonnull_count,
        unexpected_count,
        unexpected_list,
        unexpected_index_list,
    ):
        """Helper function to construct expectation result objects for map_expectations (such as column_map_expectation
        and file_lines_map_expectation).

        Expectations support four result_formats: BOOLEAN_ONLY, BASIC, SUMMARY, and COMPLETE.
        In each case, the object returned has a different set of populated fields.
        See :ref:`result_format` for more information.

        This function handles the logic for mapping those fields for column_map_expectations.
        """
        # NB: unexpected_count parameter is explicit some implementing classes may limit the length of unexpected_list

        # Retain support for string-only output formats:
        result_format = parse_result_format(result_format)

        # Incrementally add to result and return when all values for the specified level are present
        return_obj = {
            'success': success
        }

        if result_format['result_format'] == 'BOOLEAN_ONLY':
            return return_obj

        missing_count = element_count - nonnull_count

        if element_count > 0:
            unexpected_percent = unexpected_count / element_count * 100
            missing_percent = missing_count / element_count * 100

            if nonnull_count > 0:
                unexpected_percent_nonmissing = unexpected_count / nonnull_count * 100
            else:
                unexpected_percent_nonmissing = None

        else:
            missing_percent = None
            unexpected_percent = None
            unexpected_percent_nonmissing = None

        return_obj['result'] = {
            'element_count': element_count,
            'missing_count': missing_count,
            'missing_percent': missing_percent,
            'unexpected_count': unexpected_count,
            'unexpected_percent': unexpected_percent,
            'unexpected_percent_nonmissing': unexpected_percent_nonmissing,
            'partial_unexpected_list': unexpected_list[:result_format['partial_unexpected_count']]
        }

        if result_format['result_format'] == 'BASIC':
            return return_obj

        # Try to return the most common values, if possible.
        if 0 < result_format.get('partial_unexpected_count'):
            try:
                partial_unexpected_counts = [
                    {'value': key, 'count': value}
                    for key, value
                    in sorted(
                        Counter(unexpected_list).most_common(result_format['partial_unexpected_count']),
                        key=lambda x: (-x[1], x[0]))
                ]
            except TypeError:
                partial_unexpected_counts = [
                    'partial_exception_counts requires a hashable type']
            finally:
                return_obj['result'].update(
                    {
                        'partial_unexpected_index_list': unexpected_index_list[:result_format[
                            'partial_unexpected_count']] if unexpected_index_list is not None else None,
                        'partial_unexpected_counts': partial_unexpected_counts
                    }
                )

        if result_format['result_format'] == 'SUMMARY':
            return return_obj

        return_obj['result'].update(
            {
                'unexpected_list': unexpected_list,
                'unexpected_index_list': unexpected_index_list
            }
        )

        if result_format['result_format'] == 'COMPLETE':
            return return_obj

        raise ValueError("Unknown result_format %s." %
                         (result_format['result_format'],))

    def _calc_map_expectation_success(self, success_count, nonnull_count, mostly):
        """Calculate success and percent_success for column_map_expectations

        Args:
            success_count (int): \
                The number of successful values in the column
            nonnull_count (int): \
                The number of nonnull values in the column
            mostly (float or None): \
                A value between 0 and 1 (or None), indicating the fraction of successes required to pass the \
                expectation as a whole. If mostly=None, then all values must succeed in order for the expectation as \
                a whole to succeed.

        Returns:
            success (boolean), percent_success (float)
        """

        if nonnull_count > 0:
            # percent_success = float(success_count)/nonnull_count
            percent_success = success_count / nonnull_count

            if mostly is not None:
                success = bool(percent_success >= mostly)

            else:
                success = bool(nonnull_count-success_count == 0)

        else:
            success = True
            percent_success = None

        return success, percent_success

    ###
    #
    # Iterative testing for custom expectations
    #
    ###

    def test_expectation_function(self, function, *args, **kwargs):
        """Test a generic expectation function

        Args:
            function (func): The function to be tested. (Must be a valid expectation function.)
            *args          : Positional arguments to be passed the the function
            **kwargs       : Keyword arguments to be passed the the function

        Returns:
            A JSON-serializable expectation result object.

        Notes:
            This function is a thin layer to allow quick testing of new expectation functions, without having to \
            define custom classes, etc. To use developed expectations from the command-line tool, you will still need \
            to define custom classes, etc.

            Check out :ref:`custom_expectations_reference` for more information.
        """

        if PY3:
            argspec = inspect.getfullargspec(function)[0][1:]
        else:
            # noinspection PyDeprecation
            argspec = inspect.getargspec(function)[0][1:]

        new_function = self.expectation(argspec)(function)
        return new_function(self, *args, **kwargs)


ValidationStatistics = namedtuple("ValidationStatistics", [
    "evaluated_expectations",
    "successful_expectations",
    "unsuccessful_expectations",
    "success_percent",
    "success",
])


def _calc_validation_statistics(validation_results):
    """
    Calculate summary statistics for the validation results and
    return ``ExpectationStatistics``.
    """
    # calc stats
    successful_expectations = sum(exp.success for exp in validation_results)
    evaluated_expectations = len(validation_results)
    unsuccessful_expectations = evaluated_expectations - successful_expectations
    success = successful_expectations == evaluated_expectations
    try:
        success_percent = successful_expectations / evaluated_expectations * 100
    except ZeroDivisionError:
        # success_percent = float("nan")
        success_percent = None

    return ValidationStatistics(
        successful_expectations=successful_expectations,
        evaluated_expectations=evaluated_expectations,
        unsuccessful_expectations=unsuccessful_expectations,
        success=success,
        success_percent=success_percent
    )


class SiteSectionIdentifier(DataContextKey):
    def __init__(self, site_section_name, resource_identifier):
        self._site_section_name = site_section_name
        if site_section_name in ["validations", "profiling"]:
            if isinstance(resource_identifier, ValidationResultIdentifier):
                self._resource_identifier = resource_identifier
            elif isinstance(resource_identifier, (tuple, list)):
                self._resource_identifier = ValidationResultIdentifier(*resource_identifier)
            else:
                self._resource_identifier = ValidationResultIdentifier(**resource_identifier)
        elif site_section_name == "expectations":
            if isinstance(resource_identifier, ExpectationSuiteIdentifier):
                self._resource_identifier = resource_identifier
            elif isinstance(resource_identifier, (tuple, list)):
                self._resource_identifier = ExpectationSuiteIdentifier(*resource_identifier)
            else:
                self._resource_identifier = ExpectationSuiteIdentifier(**resource_identifier)
        else:
            raise InvalidDataContextKeyError(
                "SiteSectionIdentifier only supports 'validations' and 'expectations' as site section names"
            )

    @property
    def site_section_name(self):
        return self._site_section_name

    @property
    def resource_identifier(self):
        return self._resource_identifier

    def to_tuple(self):
        # if PY3:
        #     return (self.site_section_name, *self.resource_identifier.to_tuple())
        # else:
        site_section_identifier_tuple_list = [self.site_section_name] + list(self.resource_identifier.to_tuple())
        return tuple(site_section_identifier_tuple_list)

    @classmethod
    def from_tuple(cls, tuple_):
        if tuple_[0] == "validations":
            return cls(
                site_section_name=tuple_[0],
                resource_identifier=ValidationResultIdentifier.from_tuple(tuple_[1:])
            )
        elif tuple_[0] == "expectations":
            return cls(
                site_section_name=tuple_[0],
                resource_identifier=ExpectationSuiteIdentifier.from_tuple(tuple_[1:])
            )
        else:
            raise InvalidDataContextKeyError(
                "SiteSectionIdentifier only supports 'validations' and 'expectations' as site section names"
            )


class ExpectationSuiteIdentifierSchema(Schema):
    expectation_suite_name = fields.Str()

    # noinspection PyUnusedLocal
    @post_load
    def make_expectation_suite_identifier(self, data, **kwargs):
        return ExpectationSuiteIdentifier(**data)


class BatchIdentifier(DataContextKey):

    def __init__(self, batch_identifier):
        super(BatchIdentifier, self).__init__()
        # batch_kwargs
        # if isinstance(batch_identifier, (BatchKwargs, dict)):
        #     self._batch_identifier = batch_identifier.batch_fingerprint
        # else:
        self._batch_identifier = batch_identifier

    @property
    def batch_identifier(self):
        return self._batch_identifier

    def to_tuple(self):
        return self.batch_identifier,

    @classmethod
    def from_tuple(cls, tuple_):
        return cls(batch_identifier=tuple_[0])


class BatchIdentifierSchema(Schema):
    batch_identifier = fields.Str()

    # noinspection PyUnusedLocal
    @post_load
    def make_batch_identifier(self, data, **kwargs):
        return BatchIdentifier(**data)


class ValidationResultIdentifier(DataContextKey):
    """A ValidationResultIdentifier identifies a validation result by the fully qualified expectation_suite_identifer
    and run_id.
    """

    def __init__(self, expectation_suite_identifier, run_id, batch_identifier):
        """Constructs a ValidationResultIdentifier

        Args:
            expectation_suite_identifier (ExpectationSuiteIdentifier, list, tuple, or dict):
                identifying information for the fully qualified expectation suite used to validate
            run_id (str): The run_id for which validation occurred
        """
        super(ValidationResultIdentifier, self).__init__()
        self._expectation_suite_identifier = expectation_suite_identifier
        self._run_id = run_id
        self._batch_identifier = batch_identifier

    @property
    def expectation_suite_identifier(self):
        return self._expectation_suite_identifier

    @property
    def run_id(self):
        return self._run_id

    @property
    def batch_identifier(self):
        return self._batch_identifier

    def to_tuple(self):
        return tuple(
            list(self.expectation_suite_identifier.to_tuple()) + [
                self.run_id or "__none__",
                self.batch_identifier or "__none__"
            ]
        )

    def to_fixed_length_tuple(self):
        return self.expectation_suite_identifier.expectation_suite_name, self.run_id or "__none__", \
               self.batch_identifier or "__none__"

    @classmethod
    def from_tuple(cls, tuple_):
        return cls(ExpectationSuiteIdentifier.from_tuple(tuple_[0:-2]), tuple_[-2], tuple_[-1])

    @classmethod
    def from_fixed_length_tuple(cls, tuple_):
        return cls(ExpectationSuiteIdentifier(tuple_[0]), tuple_[1], tuple_[2])

    @classmethod
    def from_object(cls, validation_result):
        batch_kwargs = validation_result.meta.get("batch_kwargs", {})
        if isinstance(batch_kwargs, IDDict):
            batch_identifier = batch_kwargs.to_id()
        elif isinstance(batch_kwargs, dict):
            batch_identifier = IDDict(batch_kwargs).to_id()
        else:
            raise DataContextError("Unable to construct ValidationResultIdentifier from provided object.")
        return cls(
            expectation_suite_identifier=ExpectationSuiteIdentifier(validation_result.meta["expectation_suite_name"]),
            run_id=validation_result.meta.get("run_id"),
            batch_identifier=batch_identifier
        )


class ValidationResultIdentifierSchema(Schema):
    expectation_suite_identifier = fields.Nested(ExpectationSuiteIdentifierSchema, required=True, error_messages={
        'required': 'expectation_suite_identifier is required for a ValidationResultIdentifier'})
    run_id = fields.Str(required=True, error_messages={'required': "run_id is required for a "
                                                                   "ValidationResultIdentifier"})
    batch_identifier = fields.Nested(BatchIdentifierSchema, required=True)

    # noinspection PyUnusedLocal
    @post_load
    def make_validation_result_identifier(self, data, **kwargs):
        return ValidationResultIdentifier(**data)


expectationSuiteIdentifierSchema = ExpectationSuiteIdentifierSchema(strict=True)
validationResultIdentifierSchema = ValidationResultIdentifierSchema(strict=True)


class HtmlSiteStore(object):
    _key_class = SiteSectionIdentifier

    def __init__(self, store_backend=None, runtime_environment=None):
        store_backend_module_name = store_backend.get("module_name", "great_expectations.data_context.store")
        store_backend_class_name = store_backend.get("class_name", "TupleFilesystemStoreBackend")
        store_class = load_class(store_backend_class_name, store_backend_module_name)

        if not issubclass(store_class, TupleStoreBackend):
            raise DataContextError("Invalid configuration: HtmlSiteStore needs a TupleStoreBackend")
        if "filepath_template" in store_backend or ("fixed_length_key" in store_backend and
                                                    store_backend["fixed_length_key"] is True):
            logger.warning("Configuring a filepath_template or using fixed_length_key is not supported in SiteBuilder: "
                           "filepaths will be selected based on the type of asset rendered.")

        # One thing to watch for is reversibility of keys.
        # If several types are being written to overlapping directories, we could get collisions.
        self.store_backends = {
            ExpectationSuiteIdentifier: instantiate_class_from_config(
                config=store_backend,
                runtime_environment=runtime_environment,
                config_defaults={
                    #"module_name": "great_expectations.data_context.store",
                    "module_name": "great_expectations.common",
                    "filepath_prefix": "expectations",
                    "filepath_suffix": ".html"
                }
            ),
            ValidationResultIdentifier: instantiate_class_from_config(
                config=store_backend,
                runtime_environment=runtime_environment,
                config_defaults={
                    #"module_name": "great_expectations.data_context.store",
                    "module_name": "great_expectations.common",
                    "filepath_prefix": "validations",
                    "filepath_suffix": ".html"
                }
            ),
            "index_page": instantiate_class_from_config(
                config=store_backend,
                runtime_environment=runtime_environment,
                config_defaults={
                    #"module_name": "great_expectations.data_context.store",
                    "module_name": "great_expectations.common",
                    "filepath_template": 'index.html',
                }
            ),
            "static_assets": instantiate_class_from_config(
                config=store_backend,
                runtime_environment=runtime_environment,
                config_defaults={
                    #"module_name": "great_expectations.data_context.store",
                    "module_name": "great_expectations.common",
                    "filepath_template": None,
                }
            ),
        }

        # NOTE: Instead of using the filesystem as the source of record for keys,
        # this class tracks keys separately in an internal set.
        # This means that keys are stored for a specific session, but can't be fetched after the original
        # HtmlSiteStore instance leaves scope.
        # Doing it this way allows us to prevent namespace collisions among keys while still having multiple
        # backends that write to the same directory structure.
        # It's a pretty reasonable way for HtmlSiteStore to do its job---you just ahve to remember that it
        # can't necessarily set and list_keys like most other Stores.
        self.keys = set()

    def get(self, key):
        self._validate_key(key)
        return self.store_backends[
            type(key.resource_identifier)
        ].get(key.to_tuple())

    def set(self, key, serialized_value):
        self._validate_key(key)
        self.keys.add(key)

        return self.store_backends[
            type(key.resource_identifier)
        ].set(key.resource_identifier.to_tuple(), serialized_value,
              content_encoding='utf-8', content_type='text/html; charset=utf-8')

    def get_url_for_resource(self, resource_identifier=None):
        """
        Return the URL of the HTML document that renders a resource
        (e.g., an expectation suite or a validation result).

        :param resource_identifier: ExpectationSuiteIdentifier, ValidationResultIdentifier
                or any other type's identifier. The argument is optional - when
                not supplied, the method returns the URL of the index page.
        :return: URL (string)
        """
        if resource_identifier is None:
            store_backend = self.store_backends["index_page"]
            key = ()
        elif isinstance(resource_identifier, ExpectationSuiteIdentifier):
            store_backend = self.store_backends[ExpectationSuiteIdentifier]
            key = resource_identifier.to_tuple()
        elif isinstance(resource_identifier, ValidationResultIdentifier):
            store_backend = self.store_backends[ValidationResultIdentifier]
            key = resource_identifier.to_tuple()
        else:
            # this method does not support getting the URL of static assets
            raise ValueError("Cannot get URL for resource {0:s}".format(str(resource_identifier)))

        return store_backend.get_url_for_key(key)

    def _validate_key(self, key):
        if not isinstance(key, SiteSectionIdentifier):
            raise TypeError("key: {!r} must a SiteSectionIdentifier, not {!r}".format(
                key,
                type(key),
            ))

        for key_class in self.store_backends.keys():
            try:
                if isinstance(key.resource_identifier, key_class):
                    return
            except TypeError:
                # it's ok to have a key that is not a type (e.g. the string "index_page")
                continue

        # The key's resource_identifier didn't match any known key_class
        raise TypeError("resource_identifier in key: {!r} must one of {}, not {!r}".format(
            key,
            set(self.store_backends.keys()),
            type(key),
        ))

    def list_keys(self):
        keys = []
        for type_, backend in self.store_backends.items():
            try:
                # If the store_backend does not support list_keys...
                key_tuples = backend.list_keys()
            except NotImplementedError:
                pass
            try:
                if issubclass(type_, DataContextKey):
                    keys += [type_.from_tuple(tuple_) for tuple_ in key_tuples]
            except TypeError:
                # If the key in store_backends is not itself a type...
                pass
        return keys

    def write_index_page(self, page):
        """This third param_store has a special method, which uses a zero-length tuple as a key."""
        return self.store_backends["index_page"].set((), page, content_encoding='utf-8', content_type='text/html; '
                                                                                                      'charset=utf-8')

    def copy_static_assets(self, static_assets_source_dir=None):
        """
        Copies static assets, using a special "static_assets" backend store that accepts variable-length tuples as
        keys, with no filepath_template.
        """
        file_exclusions = [".DS_Store"]
        dir_exclusions = []

        if not static_assets_source_dir:
            static_assets_source_dir = file_relative_path(__file__, os.path.join("..", "..", "render", "view", "static"))

        for item in os.listdir(static_assets_source_dir):
            # Directory
            if os.path.isdir(os.path.join(static_assets_source_dir, item)):
                if item in dir_exclusions:
                    continue
                # Recurse
                new_source_dir = os.path.join(static_assets_source_dir, item)
                self.copy_static_assets(new_source_dir)
            # File
            else:
                # Copy file over using static assets store backend
                if item in file_exclusions:
                    continue
                source_name = os.path.join(static_assets_source_dir, item)
                with open(source_name, 'rb') as f:
                    # Only use path elements starting from static/ for key
                    store_key = tuple(os.path.normpath(source_name).split(os.sep))
                    store_key = store_key[store_key.index('static'):]
                    content_type, content_encoding = guess_type(item, strict=False)

                    if content_type is None:
                        # Use GE-known content-type if possible
                        if source_name.endswith(".otf"):
                            content_type = "font/opentype"
                        else:
                            # fallback
                            logger.warning("Unable to automatically determine content_type for {}".format(source_name))
                            content_type = "text/html; charset=utf8"

                    self.store_backends["static_assets"].set(
                        store_key,
                        f.read(),
                        content_encoding=content_encoding,
                        content_type=content_type
                    )


class SiteBuilder(object):
    """SiteBuilder builds data documentation for the project defined by a DataContext.

    A data documentation site consists of HTML pages for expectation suites, profiling and validation results, and
    an index.html page that links to all the pages.

    The exact behavior of SiteBuilder is controlled by configuration in the DataContext's great_expectations.yml file.

    Users can specify:

        * which datasources to document (by default, all)
        * whether to include expectations, validations and profiling results sections (by default, all)
        * where the expectations and validations should be read from (filesystem or S3)
        * where the HTML files should be written (filesystem or S3)
        * which renderer and view class should be used to render each section

    Here is an example of a minimal configuration for a site::

        local_site:
            class_name: SiteBuilder
            store_backend:
                class_name: TupleS3StoreBackend
                bucket: data_docs.my_company.com
                prefix: /data_docs/


    A more verbose configuration can also control individual sections and override renderers, views, and stores::

        local_site:
            class_name: SiteBuilder
            store_backend:
                class_name: TupleS3StoreBackend
                bucket: data_docs.my_company.com
                prefix: /data_docs/
            site_index_builder:
                class_name: DefaultSiteIndexBuilder

            # Verbose version:
            # index_builder:
            #     module_name: great_expectations.render.builder
            #     class_name: DefaultSiteIndexBuilder
            #     renderer:
            #         module_name: great_expectations.render.renderer
            #         class_name: SiteIndexPageRenderer
            #     view:
            #         module_name: great_expectations.render.view
            #         class_name: DefaultJinjaIndexPageView

            site_section_builders:
                # Minimal specification
                expectations:
                    class_name: DefaultSiteSectionBuilder
                    source_store_name: expectation_store
                renderer:
                    module_name: great_expectations.render.renderer
                    class_name: ExpectationSuitePageRenderer

                # More verbose specification with optional arguments
                validations:
                    module_name: great_expectations.data_context.render
                    class_name: DefaultSiteSectionBuilder
                    source_store_name: local_validation_store
                    renderer:
                        module_name: great_expectations.render.renderer
                        class_name: SiteIndexPageRenderer
                    view:
                        module_name: great_expectations.render.view
                        class_name: DefaultJinjaIndexPageView
    """

    def __init__(self,
                 data_context,
                 store_backend,
                 site_name=None,
                 site_index_builder=None,
                 site_section_builders=None,
                 runtime_environment=None
                 ):
        self.site_name = site_name
        self.data_context = data_context
        self.store_backend = store_backend

        # set custom_styles_directory if present
        custom_styles_directory = None
        plugins_directory = data_context.plugins_directory
        if plugins_directory and os.path.isdir(os.path.join(plugins_directory, "custom_data_docs", "styles")):
            custom_styles_directory = os.path.join(plugins_directory, "custom_data_docs", "styles")

        # The site builder is essentially a frontend store. We'll open up three types of backends using the base
        # type of the configuration defined in the store_backend section

        self.target_store = HtmlSiteStore(
            store_backend=store_backend,
            runtime_environment=runtime_environment
        )

        if site_index_builder is None:
            site_index_builder = {
                "class_name": "DefaultSiteIndexBuilder"
            }
        self.site_index_builder = instantiate_class_from_config(
            config=site_index_builder,
            runtime_environment={
                "data_context": data_context,
                "custom_styles_directory": custom_styles_directory,
                "target_store": self.target_store,
                "site_name": self.site_name
            },
            config_defaults={
                "name": "site_index_builder",
                #"module_name": "great_expectations.render.renderer.site_builder",
                "module_name": "great_expectations.common",
                "class_name": "DefaultSiteIndexBuilder"
            }
        )

        if site_section_builders is None:
            site_section_builders = {
                "expectations": {
                    "class_name": "DefaultSiteSectionBuilder",
                    "source_store_name":  data_context.expectations_store_name,
                    "renderer": {
                        "class_name": "ExpectationSuitePageRenderer"
                    }
                },
                "validations": {
                    "class_name": "DefaultSiteSectionBuilder",
                    "source_store_name": data_context.validations_store_name,
                    "run_id_filter": {
                        "ne": "profiling"
                    },
                    "renderer": {
                        "class_name": "ValidationResultsPageRenderer"
                    },
                    "validation_results_limit": site_index_builder.get("validation_results_limit")
                },
                "profiling": {
                    "class_name": "DefaultSiteSectionBuilder",
                    "source_store_name":  data_context.validations_store_name,
                    "run_id_filter": {
                        "eq": "profiling"
                    },
                    "renderer": {
                        "class_name": "ProfilingResultsPageRenderer"
                    }
                }
            }
        self.site_section_builders = {}
        for site_section_name, site_section_config in site_section_builders.items():
            self.site_section_builders[site_section_name] = instantiate_class_from_config(
                config=site_section_config,
                runtime_environment={
                    "data_context": data_context,
                    "target_store": self.target_store,
                    "custom_styles_directory": custom_styles_directory
                },
                config_defaults={
                    "name": site_section_name,
                    #"module_name": "great_expectations.render.renderer.site_builder"
                    "module_name": "great_expectations.common"
                }
            )

    def build(self, resource_identifiers=None):
        """

        :param resource_identifiers: a list of resource identifiers (ExpectationSuiteIdentifier,
                            ValidationResultIdentifier). If specified, rebuild HTML
                            (or other views the data docs site renders) only for
                            the resources in this list. This supports incremental build
                            of data docs sites (e.g., when a new validation result is created)
                            and avoids full rebuild.
        :return:
        """

        # copy static assets
        # TODO: <Alex>Dynamic loading in AWS Glue (due to Spark batch execution mode) is not supported.
        # TODO: Copying static assets to S3 by other means and commenting out this call.</Alex>
        # self.target_store.copy_static_assets()

        for site_section, site_section_builder in self.site_section_builders.items():
            site_section_builder.build(resource_identifiers=resource_identifiers)

        return self.site_index_builder.build()

    def get_resource_url(self, resource_identifier=None):
        """
        Return the URL of the HTML document that renders a resource
        (e.g., an expectation suite or a validation result).

        :param resource_identifier: ExpectationSuiteIdentifier, ValidationResultIdentifier
                or any other type's identifier. The argument is optional - when
                not supplied, the method returns the URL of the index page.
        :return: URL (string)
        """

        return self.target_store.get_url_for_resource(resource_identifier=resource_identifier)


class DefaultSiteSectionBuilder(object):

    def __init__(
            self,
            name,
            data_context,
            target_store,
            source_store_name,
            custom_styles_directory=None,
            run_id_filter=None,
            validation_results_limit=None,
            renderer=None,
            view=None,
    ):
        self.name = name
        self.source_store = data_context.stores[source_store_name]
        self.target_store = target_store
        self.run_id_filter = run_id_filter
        self.validation_results_limit = validation_results_limit

        if renderer is None:
            raise ge_exceptions.InvalidConfigError(
                "SiteSectionBuilder requires a renderer configuration with a class_name key."
            )
        self.renderer_class = instantiate_class_from_config(
            config=renderer,
            runtime_environment={
                "data_context": data_context
            },
            config_defaults={
                #"module_name": "great_expectations.render.renderer"
                "module_name": "great_expectations.common"
            }
        )
        if view is None:
            view = {
                #"module_name": "great_expectations.render.view",
                #"module_name": "great_expectations.data_context.util",
                "module_name": "great_expectations.common",
                "class_name": "DefaultJinjaPageView",
            }

        self.view_class = instantiate_class_from_config(
            config=view,
            runtime_environment={
                "custom_styles_directory": custom_styles_directory
            },
            config_defaults={
                #"module_name": "great_expectations.render.view"
                "module_name": "great_expectations.common"
            }
        )

    def build(self, resource_identifiers=None):
        source_store_keys = self.source_store.list_keys()
        if self.name == "validations" and self.validation_results_limit:
            source_store_keys = sorted(source_store_keys, key=lambda x: x.run_id, reverse=True)[:self.validation_results_limit]

        for resource_key in source_store_keys:

            # if no resource_identifiers are passed, the section builder will build
            # a page for every keys in its source store.
            # if the caller did pass resource_identifiers, the section builder
            # will build pages only for the specified resources
            if resource_identifiers and resource_key not in resource_identifiers:
                continue

            if self.run_id_filter:
                if not self._resource_key_passes_run_id_filter(resource_key):
                    continue

            resource = self.source_store.get(resource_key)

            if isinstance(resource_key, ExpectationSuiteIdentifier):
                expectation_suite_name = resource_key.expectation_suite_name
                logger.debug("        Rendering expectation suite {}".format(expectation_suite_name))
            elif isinstance(resource_key, ValidationResultIdentifier):
                run_id = resource_key.run_id
                expectation_suite_name = resource_key.expectation_suite_identifier.expectation_suite_name
                if run_id == "profiling":
                    logger.debug("        Rendering profiling for batch {}".format(resource_key.batch_identifier))
                else:

                    logger.debug("        Rendering validation: run id: {}, suite {} for batch {}".format(run_id,
                                                                                                              expectation_suite_name,
                                                                                                              resource_key.batch_identifier))

            try:
                rendered_content = self.renderer_class.render(resource)
                viewable_content = self.view_class.render(rendered_content)
            except Exception as e:
                logger.error("Exception occurred during data docs rendering: ", e, exc_info=True)
                exception_traceback = traceback.format_exc()
                exception_message = "[DefaultSiteSectionBuilder:build] {}".format(type(e).__name__, str(e)) + " [TRACEBACK] " + exception_traceback
                raise Exception(exception_message)
                #continue

            self.target_store.set(
                SiteSectionIdentifier(
                    site_section_name=self.name,
                    resource_identifier=resource_key,
                ),
                viewable_content
            )

    def _resource_key_passes_run_id_filter(self, resource_key):
        if type(resource_key) == ValidationResultIdentifier:
            run_id = resource_key.run_id
        else:
            raise TypeError("run_id_filter filtering is only implemented for ValidationResultResources.")

        if self.run_id_filter.get("eq"):
            return self.run_id_filter.get("eq") == run_id

        elif self.run_id_filter.get("ne"):
            return self.run_id_filter.get("ne") != run_id


class DatasourceTypes(enum.Enum):
    PANDAS = "pandas"
    SQL = "sql"
    SPARK = "spark"
    # TODO DBT = "dbt"


DATASOURCE_TYPE_BY_DATASOURCE_CLASS = {
    "PandasDatasource": DatasourceTypes.PANDAS,
    "SparkDFDatasource": DatasourceTypes.SPARK,
    "SqlAlchemyDatasource": DatasourceTypes.SQL,
}


class DefaultSiteIndexBuilder(object):

    def __init__(
            self,
            name,
            site_name,
            data_context,
            target_store,
            custom_styles_directory=None,
            show_cta_footer=True,
            validation_results_limit=None,
            renderer=None,
            view=None,
    ):
        # NOTE: This method is almost identical to DefaultSiteSectionBuilder
        self.name = name
        self.site_name = site_name
        self.data_context = data_context
        self.target_store = target_store
        self.show_cta_footer = show_cta_footer
        self.validation_results_limit = validation_results_limit

        if renderer is None:
            renderer = {
                #"module_name": "great_expectations.render.renderer",
                #"module_name": "great_expectations.data_context.util",
                "module_name": "great_expectations.common",
                "class_name": "SiteIndexPageRenderer",
            }
        self.renderer_class = instantiate_class_from_config(
            config=renderer,
            runtime_environment={
                "data_context": data_context
            },
            config_defaults={
                #"module_name": "great_expectations.render.renderer"
                "module_name": "great_expectations.common"
            }
        )

        if view is None:
            view = {
                #"module_name": "great_expectations.render.view",
                #"module_name": "great_expectations.data_context.util",
                "module_name": "great_expectations.common",
                "class_name": "DefaultJinjaIndexPageView",
            }
        self.view_class = instantiate_class_from_config(
            config=view,
            runtime_environment={
                "custom_styles_directory": custom_styles_directory
            },
            config_defaults={
                #"module_name": "great_expectations.render.view"
                "module_name": "great_expectations.common"
            }
        )

    def add_resource_info_to_index_links_dict(self,
                                              index_links_dict,
                                              expectation_suite_name,
                                              section_name,
                                              batch_identifier=None,
                                              run_id=None,
                                              validation_success=None
                                              ):
        import os

        if section_name + "_links" not in index_links_dict:
            index_links_dict[section_name + "_links"] = []

        if run_id:
            path_components = ["validations"] + expectation_suite_name.split(".") + [run_id] + [batch_identifier]
            # py2 doesn't support
            # filepath = os.path.join("validations", batch_identifier, *expectation_suite_name.split("."), run_id)
            filepath = os.path.join(*path_components)
            filepath += ".html"
        else:
            filepath = os.path.join("expectations", *expectation_suite_name.split("."))
            filepath += ".html"

        index_links_dict[section_name + "_links"].append(
            {
                "expectation_suite_name": expectation_suite_name,
                "filepath": filepath,
                "run_id": run_id,
                "batch_identifier": batch_identifier,
                "validation_success": validation_success
            }
        )

        return index_links_dict

    def get_calls_to_action(self):
        telemetry = None
        db_driver = None
        datasource_classes_by_name = self.data_context.list_datasources()

        if datasource_classes_by_name:
            last_datasource_class_by_name = datasource_classes_by_name[-1]
            last_datasource_class_name = last_datasource_class_by_name["class_name"]
            last_datasource_name = last_datasource_class_by_name["name"]
            last_datasource = self.data_context.datasources[last_datasource_name]

            if last_datasource_class_name == "SqlAlchemyDatasource":
                try:
                    db_driver = last_datasource.drivername
                except AttributeError:
                    pass

            datasource_type = DATASOURCE_TYPE_BY_DATASOURCE_CLASS[last_datasource_class_name].value
            telemetry = "?utm_source={}&utm_medium={}&utm_campaign={}".format(
                "ge-init-datadocs-v2",
                datasource_type,
                db_driver,
            )

        return {
            "header": "To continue exploring Great Expectations check out one of these tutorials...",
            "buttons": self._get_call_to_action_buttons(telemetry)
        }

    def _get_call_to_action_buttons(self, telemetry):
        """
        Build project and user specific calls to action buttons.

        This can become progressively smarter about project and user specific
        calls to action.
        """
        create_expectations = CallToActionButton(
            "How To Create Expectations",
            # TODO update this link to a proper tutorial
            "https://docs.greatexpectations.io/en/latest/tutorials/create_expectations.html"
        )
        see_glossary = CallToActionButton(
            "See more kinds of Expectations",
            "http://docs.greatexpectations.io/en/latest/reference/expectation_glossary.html"
        )
        validation_playground = CallToActionButton(
            "How To Validate data",
            # TODO update this link to a proper tutorial
            "https://docs.greatexpectations.io/en/latest/tutorials/validate_data.html"
        )
        customize_data_docs = CallToActionButton(
            "How To Customize Data Docs",
            "https://docs.greatexpectations.io/en/latest/reference/data_docs_reference.html#customizing-data-docs"
        )
        s3_team_site = CallToActionButton(
            "How To Set up a team site on AWS S3",
            "https://docs.greatexpectations.io/en/latest/tutorials/publishing_data_docs_to_s3.html"
        )
        # TODO gallery does not yet exist
        # gallery = CallToActionButton(
        #     "Great Expectations Gallery",
        #     "https://greatexpectations.io/gallery"
        # )

        results = []
        results.append(create_expectations)

        # Show these no matter what
        results.append(validation_playground)
        results.append(customize_data_docs)
        results.append(s3_team_site)

        if telemetry:
            for button in results:
                button.link = button.link + telemetry

        return results

    def build(self):
        # Loop over sections in the HtmlStore
        logger.debug("DefaultSiteIndexBuilder.build")

        expectation_suite_keys = [
            ExpectationSuiteIdentifier.from_tuple(expectation_suite_tuple) for expectation_suite_tuple in
            self.target_store.store_backends[ExpectationSuiteIdentifier].list_keys()
        ]
        validation_and_profiling_result_keys = [
            ValidationResultIdentifier.from_tuple(validation_result_tuple) for validation_result_tuple in
            self.target_store.store_backends[ValidationResultIdentifier].list_keys()
        ]
        profiling_result_keys = [
            validation_result_key for validation_result_key in validation_and_profiling_result_keys
            if validation_result_key.run_id == "profiling"
        ]
        validation_result_keys = [
            validation_result_key for validation_result_key in validation_and_profiling_result_keys
            if validation_result_key.run_id != "profiling"
        ]
        validation_result_keys = sorted(validation_result_keys, key=lambda x: x.run_id, reverse=True)
        if self.validation_results_limit:
            validation_result_keys = validation_result_keys[:self.validation_results_limit]

        index_links_dict = OrderedDict()
        index_links_dict["site_name"] = self.site_name

        if self.show_cta_footer:
            index_links_dict["cta_object"] = self.get_calls_to_action()

        for expectation_suite_key in expectation_suite_keys:
            self.add_resource_info_to_index_links_dict(
                index_links_dict=index_links_dict,
                expectation_suite_name=expectation_suite_key.expectation_suite_name,
                section_name="expectations"
            )

        for profiling_result_key in profiling_result_keys:
            try:
                validation = self.data_context.get_validation_result(
                    batch_identifier=profiling_result_key.batch_identifier,
                    expectation_suite_name=profiling_result_key.expectation_suite_identifier.expectation_suite_name,
                    run_id=profiling_result_key.run_id
                )

                validation_success = validation.success

                self.add_resource_info_to_index_links_dict(
                    index_links_dict=index_links_dict,
                    expectation_suite_name=profiling_result_key.expectation_suite_identifier.expectation_suite_name,
                    section_name="profiling",
                    batch_identifier=profiling_result_key.batch_identifier,
                    run_id=profiling_result_key.run_id,
                    validation_success=validation_success
                )
            except Exception as e:
                error_msg = "Profiling result not found: {0:s} - skipping".format(str(profiling_result_key.to_tuple()))
                logger.warning(error_msg)

        for validation_result_key in validation_result_keys:
            try:
                validation = self.data_context.get_validation_result(
                    batch_identifier=validation_result_key.batch_identifier,
                    expectation_suite_name=validation_result_key.expectation_suite_identifier.expectation_suite_name,
                    run_id=validation_result_key.run_id
                )

                validation_success = validation.success

                self.add_resource_info_to_index_links_dict(
                    index_links_dict=index_links_dict,
                    expectation_suite_name=validation_result_key.expectation_suite_identifier.expectation_suite_name,
                    section_name="validations",
                    batch_identifier=validation_result_key.batch_identifier,
                    run_id=validation_result_key.run_id,
                    validation_success=validation_success
                )
            except Exception as e:
                error_msg = "Validation result not found: {0:s} - skipping".format(str(validation_result_key.to_tuple()))
                logger.warning(error_msg)

        try:
            rendered_content = self.renderer_class.render(index_links_dict)
            viewable_content = self.view_class.render(rendered_content)
        except Exception as e:
            logger.error("Exception occurred during data docs rendering: ", e, exc_info=True)
            exception_traceback = traceback.format_exc()
            exception_message = "[DefaultSiteIndexBuilder:build] {}".format(type(e).__name__, str(e)) + " [TRACEBACK] " + exception_traceback
            raise Exception(exception_message)
            #return None

        return (
            self.target_store.write_index_page(viewable_content),
            index_links_dict
        )


class CallToActionButton(object):
    def __init__(self, title, link):
        self.title = title
        self.link = link


class Renderer(object):

    def __init__(self):
        # This is purely a convenience to provide an explicit mechanism to instantiate any Renderer, even ones that
        # used to be composed exclusively of classmethods
        pass

    @classmethod
    def render(cls, ge_object):
        return ge_object

    @classmethod
    def _get_expectation_type(cls, ge_object):
        if isinstance(ge_object, ExpectationConfiguration):
            return ge_object.expectation_type

        elif isinstance(ge_object, ExpectationValidationResult):
            # This is a validation
            return ge_object.expectation_config.expectation_type

    #TODO: When we implement a ValidationResultSuite class, this method will move there.
    @classmethod
    def _find_evr_by_type(cls, evrs, type_):
        for evr in evrs:
            if evr.expectation_config.expectation_type == type_:
                return evr

    #TODO: When we implement a ValidationResultSuite class, this method will move there.
    @classmethod
    def _find_all_evrs_by_type(cls, evrs, type_, column_=None):
        ret = []
        for evr in evrs:
            if evr.expectation_config.expectation_type == type_\
                    and (not column_ or column_ == evr.expectation_config.kwargs.get("column")):
                ret.append(evr)

        return ret

    #TODO: When we implement a ValidationResultSuite class, this method will move there.
    @classmethod
    def _get_column_list_from_evrs(cls, evrs):
        """
        Get list of column names.

        If expect_table_columns_to_match_ordered_list EVR is present, use it as the list, including the order.

        Otherwise, get the list of all columns mentioned in the expectations and order it alphabetically.

        :param evrs:
        :return: list of columns with best effort sorting
        """
        evrs_ = evrs if isinstance(evrs, list) else evrs.results

        expect_table_columns_to_match_ordered_list_evr = cls._find_evr_by_type(evrs_, "expect_table_columns_to_match_ordered_list")
        # Group EVRs by column
        sorted_columns = sorted(list(set([evr.expectation_config.kwargs["column"] for evr in evrs_ if "column" in
                                          evr.expectation_config.kwargs])))

        if expect_table_columns_to_match_ordered_list_evr:
            ordered_columns = expect_table_columns_to_match_ordered_list_evr.result["observed_value"]
        else:
            ordered_columns = []

        # only return ordered columns from expect_table_columns_to_match_ordered_list evr if they match set of column
        # names from entire evr
        if set(sorted_columns) == set(ordered_columns):
            return ordered_columns
        else:
            return sorted_columns

    #TODO: When we implement a ValidationResultSuite class, this method will move there.
    @classmethod
    def _group_evrs_by_column(cls, validation_results):
        columns = {}
        for evr in validation_results.results:
            if "column" in evr.expectation_config.kwargs:
                column = evr.expectation_config.kwargs["column"]
            else:
                column = "Table-level Expectations"

            if column not in columns:
                columns[column] = []
            columns[column].append(evr)

        return columns

    #TODO: When we implement an ExpectationSuite class, this method will move there.
    @classmethod
    def _group_and_order_expectations_by_column(cls, expectations):
        """Group expectations by column."""
        expectations_by_column = {}
        ordered_columns = []

        for expectation in expectations.expectations:
            if "column" in expectation.kwargs:
                column = expectation.kwargs["column"]
            else:
                column = "_nocolumn"
            if column not in expectations_by_column:
                expectations_by_column[column] = []
            expectations_by_column[column].append(expectation)

            # if possible, get the order of columns from expect_table_columns_to_match_ordered_list
            if expectation.expectation_type == "expect_table_columns_to_match_ordered_list":
                exp_column_list = expectation.kwargs["column_list"]
                if exp_column_list and len(exp_column_list) > 0:
                    ordered_columns = exp_column_list

        # Group items by column
        sorted_columns = sorted(list(expectations_by_column.keys()))

        # only return ordered columns from expect_table_columns_to_match_ordered_list evr if they match set of column
        # names from entire evr, else use alphabetic sort
        if set(sorted_columns) == set(ordered_columns):
            return expectations_by_column, ordered_columns
        else:
            return expectations_by_column, sorted_columns


class SlackRenderer(Renderer):
    
    def __init__(self):
        pass
    
    def render(self, validation_result=None):
        # Defaults
        timestamp = datetime.datetime.strftime(datetime.datetime.now(), "%x %X")
        default_text = "No validation occurred. Please ensure you passed a validation_result."
        status = "Failed :x:"

        title_block = {
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": default_text,
            },
        }

        query = {
            "blocks": [title_block],
            # this abbreviated root level "text" will show up in the notification and not the message
            "text": default_text
        }

        # TODO improve this nested logic
        if validation_result:
            expectation_suite_name = validation_result.meta.get("expectation_suite_name",
                                                                "__no_expectation_suite_name__")
        
            n_checks_succeeded = validation_result.statistics["successful_expectations"]
            n_checks = validation_result.statistics["evaluated_expectations"]
            run_id = validation_result.meta.get("run_id", "__no_run_id__")
            batch_id = BatchKwargs(validation_result.meta.get("batch_kwargs", {})).to_id()
            check_details_text = "*{}* of *{}* expectations were met".format(
                n_checks_succeeded, n_checks)
        
            if validation_result.success:
                status = "Success :tada:"

            summary_text = """*Batch Validation Status*: {}
*Expectation suite name*: `{}`
*Run ID*: `{}`
*Batch ID*: `{}`
*Timestamp*: `{}`
*Summary*: {}""".format(
                status,
                expectation_suite_name,
                run_id,
                batch_id,
                timestamp,
                check_details_text
            )
            query["blocks"][0]["text"]["text"] = summary_text
            # this abbreviated root level "text" will show up in the notification and not the message
            query["text"] = "{}: {}".format(expectation_suite_name, status)

            if "result_reference" in validation_result.meta:
                report_element = {
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": "- *Validation Report*: {}".format(validation_result.meta["result_reference"])},
                }
                query["blocks"].append(report_element)
        
            if "dataset_reference" in validation_result.meta:
                dataset_element = {
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": "- *Validation data asset*: {}".format(validation_result.meta["dataset_reference"])
                    },
                }
                query["blocks"].append(dataset_element)

        custom_blocks = self._custom_blocks(evr=validation_result)
        if custom_blocks:
            query["blocks"].append(custom_blocks)

        documentation_url = "https://docs.greatexpectations.io/en/latest/features/validation.html#reviewing-validation-results"
        footer_section = {
            "type": "context",
            "elements": [
                {
                    "type": "mrkdwn",
                    "text": "Learn how to review validation results: {}".format(documentation_url),
                }
            ],
        }

        divider_block = {"type": "divider"}
        query["blocks"].append(divider_block)
        query["blocks"].append(footer_section)
        return query

    def _custom_blocks(self, evr):
        return None
# FIXME : This class needs to be rebuilt to accept SiteSectionIdentifiers as input.
# FIXME : This class needs tests.


class CallToActionRenderer(object):
    _document_defaults = {
        "header": "What would you like to do next?",
        "styling": {
            "classes": [
                "border",
                "border-info",
                "alert",
                "alert-info",
                "fixed-bottom",
                "alert-dismissible",
                "fade",
                "show",
                "m-0",
                "rounded-0",
                "invisible"
            ],
            "attributes": {
                "id": "ge-cta-footer",
                "role": "alert"
            }
        }
    }
    
    @classmethod
    def render(cls, cta_object):
        """
        :param cta_object: dict
            {
                "header": # optional, can be a string or string template
                "buttons": # list of CallToActionButtons
            }
        :return: dict
            {
                "header": # optional, can be a string or string template
                "buttons": # list of CallToActionButtons
            }
        """
        
        if not cta_object.get("header"):
            cta_object["header"] = cls._document_defaults.get("header")
        
        cta_object["styling"] = cls._document_defaults.get("styling")
        cta_object["tooltip_icon"] = {
            "template": "$icon",
            "params": {
                "icon": ""
            },
            "tooltip": {
                "content": "To disable this footer, set the show_cta_footer flag in your project config to false."
            },
            "styling": {
                "params": {
                    "icon": {
                        "tag": "i",
                        "classes": ["m-1", "fas", "fa-question-circle"],
                    }
                }
            }
        }
        
        return cta_object


class SiteIndexPageRenderer(Renderer):

    @classmethod
    def _generate_links_table_rows(cls, index_links_dict, link_list_keys_to_render):
        section_rows = []

        column_count = len(link_list_keys_to_render)
        validations_links = index_links_dict.get("validations_links")
        expectations_links = index_links_dict.get("expectations_links")

        if column_count:
            cell_width_pct = 100.0/column_count

        if "expectations_links" in link_list_keys_to_render:
            for expectation_suite_link_dict in expectations_links:
                expectation_suite_row = []
                expectation_suite_name = expectation_suite_link_dict["expectation_suite_name"]

                expectation_suite_link = RenderedStringTemplateContent(**{
                    "content_block_type": "string_template",
                    "string_template": {
                        "template": "$link_text",
                        "params": {
                            "link_text": expectation_suite_name
                        },
                        "tag": "a",
                        "styling": {
                            "attributes": {
                                "href": expectation_suite_link_dict["filepath"]
                            },
                            "classes": ["ge-index-page-table-expectation-suite-link"]
                        }
                    },
                    "styling": {
                        "parent": {
                            "styles": {
                                "width": "{}%".format(cell_width_pct),
                            }
                        }
                    }
                })
                expectation_suite_row.append(expectation_suite_link)

                if "validations_links" in link_list_keys_to_render:
                    sorted_validations_links = [
                        link_dict for link_dict in sorted(validations_links, key=lambda x: x["run_id"], reverse=True)
                        if link_dict["expectation_suite_name"] == expectation_suite_name
                    ]
                    validation_link_bullets = [
                        RenderedStringTemplateContent(**{
                            "content_block_type": "string_template",
                            "string_template": {
                                "template": "${validation_success} $link_text",
                                "params": {
                                    "link_text": link_dict["run_id"],
                                    "validation_success": ""
                                },
                                "tag": "a",
                                "styling": {
                                    "attributes": {
                                        "href": link_dict["filepath"]
                                    },
                                    "params": {
                                        "validation_success": {
                                            "tag": "i",
                                            "classes": ["fas", "fa-check-circle",  "text-success"] if link_dict["validation_success"] else ["fas", "fa-times", "text-danger"]
                                        }
                                    },
                                    "classes": ["ge-index-page-table-validation-links-item"]
                                }
                            },
                            "styling": {
                                "parent": {
                                    "classes": ["hide-succeeded-validation-target"] if link_dict[
                                        "validation_success"] else []
                                }
                            }
                        }) for link_dict in sorted_validations_links if
                        link_dict["expectation_suite_name"] == expectation_suite_name
                    ]
                    validation_link_bullet_list = RenderedBulletListContent(**{
                        "content_block_type": "bullet_list",
                        "bullet_list": validation_link_bullets,
                        "styling": {
                            "parent": {
                                "styles": {
                                    "width": "{}%".format(cell_width_pct)
                                }
                            },
                            "body": {
                                "classes": ["ge-index-page-table-validation-links-list"]
                            }
                        }
                    })
                    expectation_suite_row.append(validation_link_bullet_list)

                section_rows.append(expectation_suite_row)

        if not expectations_links and "validations_links" in link_list_keys_to_render:
            sorted_validations_links = [
                link_dict for link_dict in sorted(validations_links, key=lambda x: x["run_id"], reverse=True)
            ]
            validation_link_bullets = [
                RenderedStringTemplateContent(**{
                    "content_block_type": "string_template",
                    "string_template": {
                        "template": "${validation_success} $link_text",
                        "params": {
                            "link_text": link_dict["run_id"],
                            "validation_success": ""
                        },
                        "tag": "a",
                        "styling": {
                            "attributes": {
                                "href": link_dict["filepath"]
                            },
                            "params": {
                                "validation_success": {
                                    "tag": "i",
                                    "classes": ["fas", "fa-check-circle", "text-success"] if link_dict[
                                        "validation_success"] else ["fas", "fa-times", "text-danger"]
                                }
                            },
                            "classes": ["ge-index-page-table-validation-links-item"]
                        }
                    },
                    "styling": {
                        "parent": {
                            "classes": ["hide-succeeded-validation-target"] if link_dict[
                                "validation_success"] else []
                        }
                    }
                }) for link_dict in sorted_validations_links
            ]
            validation_link_bullet_list = RenderedBulletListContent(**{
                "content_block_type": "bullet_list",
                "bullet_list": validation_link_bullets,
                "styling": {
                    "parent": {
                        "styles": {
                            "width": "{}%".format(cell_width_pct)
                        }
                    },
                    "body": {
                        "classes": ["ge-index-page-table-validation-links-list"]
                    }
                }
            })
            section_rows.append([validation_link_bullet_list])

        return section_rows

    @classmethod
    def render(cls, index_links_dict):

        sections = []
        cta_object = index_links_dict.pop("cta_object", None)

        try:
            content_blocks = []
            # site name header
            site_name_header_block = RenderedHeaderContent(**{
                "content_block_type": "header",
                "header": RenderedStringTemplateContent(**{
                    "content_block_type": "string_template",
                    "string_template": {
                        "template": "$title_prefix | $site_name",
                        "params": {
                            "site_name": index_links_dict.get("site_name"),
                            "title_prefix": "Data Docs"
                        },
                        "styling": {
                            "params": {
                                "title_prefix": {
                                    "tag": "strong"
                                }
                            }
                        },
                    }
                }),
                "styling": {
                    "classes": ["col-12", "ge-index-page-site-name-title"],
                    "header": {
                        "classes": ["alert", "alert-secondary"]
                    }
                }
            })
            content_blocks.append(site_name_header_block)

            table_rows = []
            table_header_row = []
            link_list_keys_to_render = []

            header_dict = OrderedDict([
                ["expectations_links", "Expectation Suite"],
                ["validations_links", "Validation Results (run_id)"]
            ])

            for link_lists_key, header in header_dict.items():
                if index_links_dict.get(link_lists_key):
                    class_header_str = link_lists_key.replace("_", "-")
                    class_str = "ge-index-page-table-{}-header".format(class_header_str)
                    header = RenderedStringTemplateContent(**{
                        "content_block_type": "string_template",
                        "string_template": {
                            "template": header,
                            "params": {},
                            "styling": {
                                "classes": [class_str],
                            }
                        }
                    })
                    table_header_row.append(header)
                    link_list_keys_to_render.append(link_lists_key)

            generator_table = RenderedTableContent(**{
                "content_block_type": "table",
                "header_row": table_header_row,
                "table": table_rows,
                "styling": {
                    "classes": ["col-12", "ge-index-page-table-container"],
                    "styles": {
                        "margin-top": "10px"
                    },
                    "body": {
                        "classes": ["table", "table-sm", "ge-index-page-generator-table"]
                    }
                }
            })

            table_rows += cls._generate_links_table_rows(
                index_links_dict, link_list_keys_to_render=link_list_keys_to_render
            )

            content_blocks.append(generator_table)

            if index_links_dict.get("profiling_links"):
                profiling_table_rows = []
                for profiling_link_dict in index_links_dict.get("profiling_links"):
                    profiling_table_rows.append(
                        [
                            RenderedStringTemplateContent(**{
                                "content_block_type": "string_template",
                                "string_template": {
                                    "template": "$link_text",
                                    "params": {
                                        "link_text": profiling_link_dict["expectation_suite_name"] + "." + profiling_link_dict["batch_identifier"]
                                    },
                                    "tag": "a",
                                    "styling": {
                                        "attributes": {
                                            "href": profiling_link_dict["filepath"]
                                        },
                                        "classes": ["ge-index-page-table-expectation-suite-link"]
                                    }
                                },
                            })
                        ]
                    )
                content_blocks.append(
                    RenderedTableContent(**{
                        "content_block_type": "table",
                        "header_row": ["Profiling Results"],
                        "table": profiling_table_rows,
                        "styling": {
                            "classes": ["col-12", "ge-index-page-table-container"],
                            "styles": {
                                "margin-top": "10px"
                            },
                            "body": {
                                "classes": ["table", "table-sm", "ge-index-page-generator-table"]
                            }
                        }
                    })

                )

            section = RenderedSectionContent(**{
                "section_name": index_links_dict.get("site_name"),
                "content_blocks": content_blocks
            })
            sections.append(section)

            index_page_document = RenderedDocumentContent(**{
                "renderer_type": "SiteIndexPageRenderer",
                "utm_medium": "index-page",
                "sections": sections
            })

            if cta_object:
                index_page_document.cta_footer = CallToActionRenderer.render(cta_object)

            return index_page_document

        except Exception as e:
            logger.error("Exception occurred during data docs rendering: ", e, exc_info=True)
            exception_traceback = traceback.format_exc()
            exception_message = "[SiteIndexPageRenderer:render] {}".format(type(e).__name__, str(e)) + " [TRACEBACK] " + exception_traceback
            raise Exception(exception_message)


class NoOpTemplate(object):
    def render(self, document):
        return document


class PrettyPrintTemplate(object):
    def render(self, document, indent=2):
        print(json.dumps(document, indent=indent))


# Abe 2019/06/26: This View should probably actually be called JinjaView or something similar.
# Down the road, I expect to wind up with class hierarchy along the lines of:
#   View > JinjaView > GEContentBlockJinjaView
class DefaultJinjaView(object):
    """
    Defines a method for converting a document to human-consumable form

    Dependencies
    ~~~~~~~~~~~~
    * Font Awesome 5.10.1
    * Bootstrap 4.3.1
    * jQuery 3.2.1
    * Vega 5.3.5
    * Vega-Lite 3.2.1
    * Vega-Embed 4.0.0

    """
    _template = NoOpTemplate

    bullet_list_j2 = '''
{% include 'content_block_header.j2' %}

{% if "styling" in content_block and "body" in content_block["styling"] -%}
    {% set content_block_body_styling = content_block["styling"]["body"] | render_styling -%}
{% else -%}
    {% set content_block_body_styling = "" -%}
{% endif -%}

<ul id="{{content_block_id}}-body" {{ content_block_body_styling | replace("{{section_id}}", section_id) | replace("{{content_block_id}}", content_block_id) }}>
    {% for bullet_point in content_block["bullet_list"] -%}
        {% if bullet_point is mapping and "styling" in bullet_point %}
            {% set bullet_point_styling = bullet_point["styling"].get("parent", {}) | render_styling %}
        {% else %}
            {% set bullet_point_styling = "" %}
        {% endif %}
        <li {{ bullet_point_styling }}>{{ bullet_point | render_content_block }}</li>
    {% endfor %}
</ul>
    '''

    carousel_modal_j2 = '''
{% if expectation_suite_name %}
  {% set expectation_suite_name_dot_count = expectation_suite_name.count(".") -%}
{% endif %}

{% set glossary_url = "http://docs.greatexpectations.io/en/latest/expectation_glossary.html?utm_source=walkthrough&utm_medium=glossary" %}

{% if utm_medium == "validation-results-page" or utm_medium == "profiling-results-page" %}
  {% set static_images_dir = ((expectation_suite_name_dot_count + 3) * "../") + "static/images/" -%}
{% elif utm_medium == "expectation-suite-page" %}
  {% set static_images_dir = ((expectation_suite_name_dot_count + 1) * "../") + "static/images/" -%}
{% elif utm_medium == "index-page" %}
  {% set static_images_dir = "./static/images/" -%}
{% endif %}
<style type="text/css">
div.card-footer .container {
    padding-top: 0;
}
div.walkthrough-card {
    height: 100%;
 }
div.modal-body {
    height: 700px
}
div.image-sizer {
    max-height: 400px;
    width: 100%;
    padding: 0 40px;
    overflow: hidden;
    margin-bottom: 1em;
}
.code-snippet {
    margin: 0 40px 1em 40px;
    padding: 1em 40px;
}
code.inline-code {
    padding: 2px 5px;
}
img.dev-loop {
    max-height: 250px;
}
.json-key {
    color: #333333;
    font-weight: bold;
}
.json-str {
    color: darkgreen;
}
.json-bool {
    color: darkorange;
}
.json-number {
    color: darkblue;
}
</style>

<div class="modal fade ge-walkthrough-modal" tabindex="-1" role="dialog" aria-labelledby="ge-walkthrough-modal-title"
     aria-hidden="true">
  <div class="modal-dialog modal-dialog-centered modal-lg">
    <div class="modal-content">
      <div class="modal-header">
        <h6 class="modal-title" id="ge-walkthrough-modal-title">Great Expectations Walkthrough</h6>
        <button type="button" class="close" data-dismiss="modal" aria-label="Close">
          <span aria-hidden="true">&times;</span>
        </button>
      </div>
      <div class="modal-body">
        <div class="card walkthrough-card bg-dark text-white walkthrough-1">
        <div class="card-header"><h4>How to work with Great Expectations</h4></div>
            <div class="card-body">
              <div class="image-sizer text-center">
                  <img src="{{ static_images_dir + "iterative-dev-loop.png" }}" class="img-fluid rounded-sm mx-auto dev-loop">
              </div>
                  <p>
                      Welcome! Now that you have initialized your project, the best way to work with Great Expectations is in this iterative dev loop:
                  </p>
                <ol>
                    <li>Let Great Expectations create a (terrible) first draft suite, by running <code class="inline-code bg-light">great_expectations suite new</code>.</li>
                    <li>View the suite here in Data Docs.</li>
                    <li>Edit the suite in a Jupyter notebook by running <code class="inline-code bg-light">great_expectations suite edit</code></li>
                    <li>Repeat Steps 2-3 until you are happy with your suite.</li>
                    <li>Commit this suite to your source control repository.</li>
                </ol>
            </div>
            <div class="card-footer walkthrough-links">
              <div class="container">
                <div class="row">
                  <div class="col-sm">
                    &nbsp;
                  </div>
                  <div class="col-6 text-center text-secondary">
                    1 of 7
                    <div class="progress" style="height: 2px">
                      <div class="progress-bar bg-info" role="progressbar" style="width: 16%" aria-valuenow="16" aria-valuemin="0" aria-valuemax="16"></div>
                    </div>
                  </div>
                  <div class="col-sm">
                    <a href="#" class="next-link btn btn-primary float-right" onclick="go_to_slide(2)">Next</a>
                  </div>
                </div>
              </div>
            </div>
        </div>


                <div class="card walkthrough-card bg-dark text-white walkthrough-2">
                <div class="card-header"><h4>What are Expectations?</h4></div>
                <div class="card-body">
                    <ul class="code-snippet bg-light text-muted rounded-sm">
                        <li>expect_column_to_exist</li>
                        <li>expect_table_row_count_to_be_between</li>
                        <li>expect_column_values_to_be_unique</li>
                        <li>expect_column_values_to_not_be_null</li>
                        <li>expect_column_values_to_be_between</li>
                        <li>expect_column_values_to_match_regex</li>
                        <li>expect_column_mean_to_be_between</li>
                        <li>expect_column_kl_divergence_to_be_less_than</li>
                        <li>... <a href="{{ glossary_url }}">and many more</a></li>
                    </ul>
                  <p>An expectation is a falsifiable, verifiable statement about data.</p>
                  <p>Expectations provide a language to talk about data characteristics and data quality - humans to humans, humans to machines and machines to machines.</p>
                  <p>Expectations are both data tests and docs!</p>
                </div>
                <div class="card-footer walkthrough-links">
                  <div class="container">
                    <div class="row">
                      <div class="col-sm">
                        <a href="#" class="next-link btn btn-secondary float-left" onclick="go_to_slide(1)">Back</a>
                      </div>
                      <div class="col-6 text-center text-secondary">
                        2 of 7
                        <div class="progress" style="height: 2px">
                          <div class="progress-bar bg-info" role="progressbar" style="width: 32%" aria-valuenow="32" aria-valuemin="0" aria-valuemax="32"></div>
                        </div>
                      </div>
                      <div class="col-sm">
                        <a href="#" class="next-link btn btn-primary float-right" onclick="go_to_slide(3)">Next</a>
                      </div>
                    </div>
                  </div>
                </div>
                </div>


                <div class="card walkthrough-card bg-dark text-white walkthrough-3">
                <div class="card-header"><h4>Expectations can be presented in a machine-friendly JSON</h4></div>
                <div class="card-body">
                    <p class="code-snippet bg-light text-muted rounded-sm">
{<br />
&nbsp;&nbsp;&nbsp;&nbsp;<span class="json-key">"expectation_type"</span>: <span class="json-str">"expect_column_values_to_not_be_null",</span><br />
&nbsp;&nbsp;&nbsp;&nbsp;<span class="json-key">"kwargs"</span>: {<br />
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<span class="json-key">"column"</span>: <span class="json-str">"user_id"</span><br />
&nbsp;&nbsp;&nbsp;&nbsp;}<br />
}<br />
                    </p>
                  <p>A machine can test if a dataset conforms to the expectation.</p>
                </div>
                <div class="card-footer walkthrough-links">
                  <div class="container">
                    <div class="row">
                      <div class="col-sm">
                        <a href="#" class="next-link btn btn-secondary float-left" onclick="go_to_slide(2)">Back</a>
                      </div>
                      <div class="col-6 text-center text-secondary">
                        3 of 7
                        <div class="progress" style="height: 2px">
                          <div class="progress-bar bg-info" role="progressbar" style="width: 50%" aria-valuenow="50" aria-valuemin="0" aria-valuemax="50"></div>
                        </div>
                      </div>
                      <div class="col-sm">
                        <a href="#" class="next-link btn btn-primary float-right" onclick="go_to_slide(4)">Next</a>
                      </div>
                    </div>
                  </div>
                </div>
                </div>

                <div class="card walkthrough-card bg-dark text-white walkthrough-4">
                <div class="card-header"><h4>Validation produces a validation result object</h4></div>
                <div class="card-body">
                     <p class="code-snippet bg-light text-muted rounded-sm">
{<br />
&nbsp;&nbsp;&nbsp;&nbsp;<span class="json-key">"success"</span>: <span class="json-bool">false</span>,<br />
&nbsp;&nbsp;&nbsp;&nbsp;<span class="json-key">"result":</span> {<br />
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<span class="json-key">"element_count"</span>: <span class="json-number">253405,</span><br />
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<span class="json-key">"unexpected_count"</span>: <span class="json-number">7602,</span><br />
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<span class="json-key">"unexpected_percent"</span>: <span class="json-number">2.999</span><br />
&nbsp;&nbsp;&nbsp;&nbsp;},<br />
&nbsp;&nbsp;&nbsp;&nbsp;<span class="json-key">"expectation_config"</span>: {<br />
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<span class="json-key">"expectation_type"</span>: <span class="json-str">"expect_column_values_to_not_be_null"</span>,<br />
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<span class="json-key">"kwargs"</span>: {<br />
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<span class="json-key">"column"</span>: <span class="json-str">"user_id"</span><br />
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;}<br />
}<br />
                    </p>
                  <p>Here's an example Validation Result (not from your data) in JSON format. This object has rich context about the test failure.</p>
                </div>
                <div class="card-footer walkthrough-links">
                  <div class="container">
                    <div class="row">
                      <div class="col-sm">
                        <a href="#" class="next-link btn btn-secondary float-left" onclick="go_to_slide(3)">Back</a>
                      </div>
                      <div class="col-6 text-center text-secondary">
                        4 of 7
                        <div class="progress" style="height: 2px">
                          <div class="progress-bar bg-info" role="progressbar" style="width: 68%" aria-valuenow="68" aria-valuemin="0" aria-valuemax="68"></div>
                        </div>
                      </div>
                      <div class="col-sm">
                        <a href="#" class="next-link btn btn-primary float-right" onclick="go_to_slide(5)">Next</a>
                      </div>
                    </div>
                  </div>
                </div>
                </div>

                <div class="card walkthrough-card bg-dark text-white walkthrough-5">
                <div class="card-header"><h4>Validation results save you time.</h4></div>
                <div class="card-body">
                  <div class="image-sizer text-center">
                      <img src="{{ static_images_dir + "validation_failed_unexpected_values.gif" }}" class="img-fluid rounded-sm mx-auto">
                  </div>
                  <p>This is an example of what a single failed Expectation looks like in Data Docs. Note the failure includes unexpected values from your data. This helps you debug pipeines faster.</p>
                </div>
                <div class="card-footer walkthrough-links">
                  <div class="container">
                    <div class="row">
                      <div class="col-sm">
                        <a href="#" class="next-link btn btn-secondary float-left" onclick="go_to_slide(4)">Back</a>
                      </div>
                      <div class="col-6 text-center text-secondary">
                        5 of 7
                        <div class="progress" style="height: 2px">
                          <div class="progress-bar bg-info" role="progressbar" style="width: 84%" aria-valuenow="84" aria-valuemin="0" aria-valuemax="84"></div>
                        </div>
                      </div>
                      <div class="col-sm">
                        <a href="#" class="next-link btn btn-primary float-right" onclick="go_to_slide(6)">Next</a>
                      </div>
                    </div>
                  </div>
                </div>
                </div>

                <div class="card walkthrough-card bg-dark text-white walkthrough-6">
                <div class="card-header"><h4>Great Expectations provides a large library of expectations.</h4></div>
                <div class="card-body">
                  <div class="image-sizer text-center">
                      <img src="{{ static_images_dir + "glossary_scroller.gif" }}" class="img-fluid rounded-sm mx-auto">
                  </div>
                  <p><a href="{{ glossary_url }}">Nearly 50 built in expectations</a> allow you to express how you understand your data, and you can add custom
                  expectations if you need a new one.
                  </p>
                </div>
                <div class="card-footer walkthrough-links">
                  <div class="container">
                    <div class="row">
                      <div class="col-sm">
                        <a href="#" class="next-link btn btn-secondary float-left" onclick="go_to_slide(5)">Back</a>
                      </div>
                      <div class="col-6 text-center text-secondary">
                        6 of 7
                        <div class="progress" style="height: 2px">
                          <div class="progress-bar bg-info" role="progressbar" style="width: 84%" aria-valuenow="84" aria-valuemin="0" aria-valuemax="84"></div>
                        </div>
                      </div>
                      <div class="col-sm">
                        <a href="#" class="next-link btn btn-primary float-right" onclick="go_to_slide(7)">Next</a>
                      </div>
                    </div>
                  </div>
                </div>
                </div>

                <div class="card walkthrough-card bg-dark text-white walkthrough-7">
                <div class="card-header"><h4>Now explore and edit the sample suite!</h4></div>
                <div class="card-body">
                  <p>This sample suite shows you a few examples of expectations.</p>
                  <p>Note this is <strong>not a production suite</strong> and was generated using only a small sample of your data.</p>
                  <p>When you are ready, press the <strong>How to Edit</strong> button to kick off the iterative dev loop.</p>
                </div>
                <div class="card-footer walkthrough-links">
                  <div class="container">
                    <div class="row">
                      <div class="col-sm">
                        <a href="#" class="next-link btn btn-secondary float-left" onclick="go_to_slide(6)">Back</a>
                      </div>
                      <div class="col-6 text-center text-secondary">
                        7 of 7
                        <div class="progress" style="height: 2px">
                          <div class="progress-bar bg-info" role="progressbar" style="width: 100%" aria-valuenow="100" aria-valuemin="0" aria-valuemax="100"></div>
                        </div>
                      </div>
                      <div class="col-sm">
                        <button type="button" class="btn btn-primary float-right" data-dismiss="modal" aria-label="Close">
                          <span aria-hidden="true">Done</span>
                        </button>
                      </div>
                    </div>
                  </div>
                </div>
                </div>
            </div>
        </div>
    </div>
</div>

<script type="text/JavaScript">
  $(".ge-walkthrough-modal").on('hide.bs.modal', function (e) {
    try {
      localStorage.setItem('ge-walkthrough-modal-dismissed', 'true');
    }
    catch (e) {
      console.log(e);
    }
    go_to_slide(1);
  })
</script>


<script type="text/JavaScript">
  function go_to_slide(slide_number) {
    hide_cards();
    $('.walkthrough-' + slide_number).show();
  }
  function hide_cards() {
    $('.walkthrough-card').hide();
  }
  hide_cards();
  go_to_slide(1);
</script>
    '''

    collapse_j2 = '''
{% set collapse_toggle_link = content_block.get("collapse_toggle_link", "Show more...") %}

{% if "styling" in content_block and "parent" in content_block["styling"] -%}
    {% set content_block_parent_styling = content_block["styling"]["parent"] | render_styling -%}
{% else -%}
    {% set content_block_parent_styling = "" -%}
{% endif -%}

{% if "styling" in content_block and content_block["styling"].get("body") -%}
  {% set content_block_body_styling_dict = content_block["styling"]["body"] %}
  {% if content_block_body_styling_dict.get("classes") %}
    {% do content_block_body_styling_dict["classes"].append("collapse") %}
    {% do content_block_body_styling_dict["classes"].append("m-2") %}
  {% endif %}
    {% set content_block_body_styling = content_block_body_styling_dict | render_styling -%}
{% else -%}
  {% set content_block_body_styling_dict = {"classes": ["collapse", "m-2"]} %}
  {% set content_block_body_styling = content_block_body_styling_dict | render_styling -%}
{% endif -%}

{% if "styling" in content_block and content_block["styling"].get("collapse_link") %}
  {% set collapse_link_styling = content_block["styling"].get("collapse_link") %}
{% else %}
  {% set collapse_link_styling = "" %}
{% endif %}

{% set collapse_id = content_block_id ~ "-collapse-body-" | generate_html_element_uuid %}

{% if content_block["inline_link"] %}
  <span class="m-0">
    <a data-toggle="collapse" href="#{{collapse_id}}" aria-expanded="false" aria-controls="{{content_block_id}}-collapse-body" {{ collapse_link_styling | replace("{{section_id}}", section_id) | replace("{{content_block_id}}", content_block_id) }}>
      {{ collapse_toggle_link | render_content_block }}
    </a>
  </span>
{% endif %}

<div id="{{content_block_id}}-parent" {{ content_block_parent_styling | replace("{{section_id}}", section_id) | replace("{{content_block_id}}", content_block_id) }}>
  {% if not content_block["inline_link"] %}
    <p class="m-0">
      <a data-toggle="collapse" href="#{{collapse_id}}" aria-expanded="false" aria-controls="{{content_block_id}}-collapse-body" {{ collapse_link_styling | replace("{{section_id}}", section_id) | replace("{{content_block_id}}", content_block_id) }}>
        {{ collapse_toggle_link | render_content_block }}
      </a>
    </p>
  {% endif %}

  <div id={{collapse_id}} {{ content_block_body_styling | replace("{{section_id}}", section_id) | replace("{{content_block_id}}", content_block_id) }}>
    {% set content_block_body_styling = "" %}
    {% include 'content_block_header.j2' %}
    {% for block in content_block["collapse"] %}
      {% set collapse_block_loop = loop %}
      {% set content_block_id = content_block_id ~ "-collapse-item-" ~ collapse_block_loop.index %}
      {{ block | render_content_block(content_block_id=content_block_id) }}
    {% endfor %}
  </div>
</div>
    '''

    component_j2 = '''
{%- if section_loop is defined -%}
    {%- set section_id = "section-"~section_loop.index -%}
{%- else -%}
    {%- set section_id = None -%}
{%- endif -%}
{% if content_block_loop is defined -%}
    {%- if section_loop is defined -%}
        {%- set content_block_id = "section-"~section_loop.index~"-content-block-"~content_block_loop.index -%}
    {%- else -%}
        {%- set content_block_id = "content-block-"~content_block_loop.index -%}
    {%- endif -%}
{%- else -%}
    {%- set content_block_id = "content-block" -%}
{% endif %}

{% if "styling" in content_block and "body" in content_block["styling"] -%}
    {% set content_block_body_styling = content_block["styling"]["body"] | render_styling -%}
{% else -%}
    {% set content_block_body_styling = "" -%}
{% endif -%}

<div id="{{content_block_id}}" {{ content_block | render_styling_from_string_template }}>

    {{ content_block | render_content_block }}

</div>
    '''

    content_block_container_j2 = '''
{% include 'content_block_header.j2' %}

{% if "styling" in content_block -%}
    {% set content_block_styling = content_block["styling"] | render_styling -%}
{% else -%}
    {% set content_block_styling = "" -%}
{% endif -%}

<div id="{{ content_block_id }}-container" {{ content_block_body_styling | replace("{{section_id}}", section_id) | replace("{{content_block_id}}", content_block_id) }}>
  {% for el in content_block["content_blocks"] %}
    {{ el | render_content_block }}
  {% endfor %}
</div>
    '''

    content_block_header_j2 = '''
{% if "styling" in content_block and "header" in content_block["styling"] -%}
    {% set content_block_header_styling = content_block["styling"]["header"] | render_styling -%}
{% else -%}
    {% set content_block_header_styling = "" -%}
{% endif -%}

{% if "styling" in content_block and "subheader" in content_block["styling"] -%}
    {% set content_block_subheader_styling = content_block["styling"]["subheader"] | render_styling -%}
{% else -%}
    {% set content_block_subheader_styling = "" -%}
{% endif -%}

{% if (content_block.get("header") or content_block.get("subheader")) and content_block["content_block_type"] != "header" -%}
    <div id="{{content_block_id}}-header" {{ content_block_header_styling | replace("{{section_id}}", section_id) | replace("{{content_block_id}}", content_block_id) }}>
        {% if content_block.get("header") and content_block["content_block_type"] != "header" %}
          {% if content_block["header"] is mapping %}
            <div>
              {{ content_block["header"] | render_content_block }}
            </div>
          {% else %}
            <h5>
                {{ content_block["header"] | render_content_block }}
            </h5>
          {% endif %}
        {% endif %}
        {%- if content_block.get("subheader") and content_block["content_block_type"] != "header" -%}
          {% if content_block["subheader"] is mapping %}
            <div id="{{content_block_id}}-subheader" {{ content_block_subheader_styling | replace("{{section_id}}", section_id) | replace("{{content_block_id}}", content_block_id) }}>
              {{ content_block["subheader"] | render_content_block }}
            </div>
          {% else %}
            <h6 id="{{content_block_id}}-subheader" {{ content_block_subheader_styling | replace("{{section_id}}", section_id) | replace("{{content_block_id}}", content_block_id) }}>
                {{ content_block["subheader"] | render_content_block }}
            </h6>
          {% endif %}
        {% endif -%}
    </div>
{% endif -%}
    '''

    cta_footer_j2 = '''
{% if "styling" in cta_footer  -%}
    {% set cta_footer_styling = cta_footer["styling"] | render_styling -%}
{% else -%}
    {% set cta_footer_styling = "" -%}
{% endif -%}

<footer {{ cta_footer_styling }}>
  <h5 class="alert-heading text-center">
    {{ cta_footer["tooltip_icon"] | render_string_template }}
    {{ cta_footer["header"] | render_string_template }}
  </h5>
  <div class="d-flex justify-content-center flex-sm-row flex-column">
    {% for cta_button in cta_footer["buttons"] %}
      <a href="{{ cta_button.link }}" class="btn btn-primary m-2" rel="noopener noreferrer" target="_blank">{{ cta_button.title }}</a>
    {% endfor %}
  </div>
  <button type="button" class="close" data-dismiss="alert" aria-label="Close">
    <span aria-hidden="true">&times;</span>
  </button>
</footer>
<script>
  if (sessionStorage.getItem("showCta") !== "false") {
    $('#ge-cta-footer').removeClass("invisible")
  }
  $('#ge-cta-footer').on('closed.bs.alert', function () {
    sessionStorage.setItem("showCta", false)
  })
</script>
    '''

    edit_expectations_instructions_modal_j2 = '''
{% if expectation_suite_name %}
  {% set expectation_suite_name_dot_count = expectation_suite_name.count(".") -%}
{% endif %}

{% set edit_suite_command = "great_expectations suite edit " + expectation_suite_name  %}

{# TODO move this logic into a testable method on DefaultJinjaView #}
{% if utm_medium == "validation-results-page" or utm_medium == "profiling-results-page" %}
  {% set static_images_dir = ((expectation_suite_name_dot_count + 3) * "../") + "static/images/" -%}
{% elif utm_medium == "expectation-suite-page" %}
  {% set static_images_dir = ((expectation_suite_name_dot_count + 1) * "../") + "static/images/" -%}
{% elif utm_medium == "index-page" %}
  {% set static_images_dir = "./static/images/" -%}
{% endif %}

<script type="text/javascript">
$(function() {
    $('button.copy-edit-command').click(function() {
        $('.edit-command').focus();
        $('.edit-command').select();
        document.execCommand('copy');
    });
});
</script>

<div class="modal fade ge-expectation-editing-instructions-modal" tabindex="-1" role="dialog" aria-labelledby="ge-expectation-editing-instructions-modal-title"
     aria-hidden="true">
  <div class="modal-dialog modal-dialog-centered modal-lg">
    <div class="modal-content">
      <div class="modal-header">
        <h5 class="modal-title" id="ge-expectation-editing-instructions-modal-title">How to Edit This Expectation Suite</h5>
        <button type="button" class="close" data-dismiss="modal" aria-label="Close">
          <span aria-hidden="true">&times;</span>
        </button>
      </div>
      <div class="modal-body" style="height: 350px">
        <p>Expectations are best <strong>edited interactively in Jupyter notebooks</strong>.</p>
        <p>To automatically generate a notebook that does this run:</p>
        <div class="input-group mb-3">
          {% if batch_kwargs %}
            <textarea class="form-control edit-command" readonly>{{ edit_suite_command | safe }}</textarea>
          {% else %}
            <input type="text" class="form-control edit-command" readonly value="{{ edit_suite_command | safe }}">
          {% endif %}
            <div class="input-group-append">
                <button class="btn btn-primary copy-edit-command" type="button"><i class="far fa-clipboard"></i> Copy</button>
            </div>
        </div>
      <p>Once you have made your changes and <strong>run the entire notebook</strong> you can kill the notebook by pressing <strong>Ctr-C</strong> in your terminal.</p>
      <p>Because these notebooks are generated from an Expectation Suite, these notebooks are <strong>entirely disposable</strong>.</p>
      </div>
    </div>
  </div>
</div>
    '''

    favicon_j2 = '''
{% if expectation_suite_name %}
  {% set expectation_suite_name_dot_count = expectation_suite_name.count(".") -%}
{% endif %}

{% if utm_medium == "validation-results-page" or utm_medium == "profiling-results-page" %}
  {% set static_images_dir = ((expectation_suite_name_dot_count + 3) * "../") + "static/images/" -%}
{% elif utm_medium == "expectation-suite-page" %}
  {% set static_images_dir = ((expectation_suite_name_dot_count + 1) * "../") + "static/images/" -%}
{% elif utm_medium == "index-page" %}
  {% set static_images_dir = "./static/images/" -%}
{% endif %}

<link rel="shortcut icon" href="{{ static_images_dir + "favicon.ico" }}" type="image/x-icon"/>
    '''

    ge_info_j2 = '''
<div class="mb-4">
  {% if renderer_type == "ValidationResultsPageRenderer" %}
    <div class="col-12 p-0">
      <h4>Expectation Validation Result</h4>
      <p class="lead">Evaluates whether a batch of data matches expectations.</p>
    </div>
  {% elif renderer_type == "ExpectationSuitePageRenderer" %}
    <div class="col-12 p-0">
      <h4>Expectation Suite</h4>
      <p class="lead">A collection of Expectations defined for batches of data.</p>
    </div>
  {% elif renderer_type == "ProfilingResultsPageRenderer" %}
    <div class="col-12 p-0">
      <h4>Data Asset Profile</h4>
      <p class="lead">Overview of data values, statistics, and distributions.</p>
    </div>
  {% else %}
    <div class="col-12 p-0">
      <p>
        Data Docs autogenerated using
        <a href="https://greatexpectations.io">Great Expectations</a>.
      </p>
    </div>
  {% endif %}
</div>
    '''

    graph_j2 = '''
{% include 'content_block_header.j2' %}

{%- if index -%}
    {%- set child_id = "-child-" ~ index -%}
{%- else -%}
    {%- set child_id = "" -%}
{%- endif -%}

<div class="show-scrollbars">
  <div id="{{content_block_id}}-graph{{child_id}}" {{content_block_body_styling | replace("{{section_id}}", section_id) | replace("{{content_block_id}}", content_block_id)}}></div>
</div>
<script>
    // Assign the specification to a local variable vlSpec.
    vlSpec = {{content_block["graph"]}};
    // Embed the visualization in the container with id `vis`
    vegaEmbed('#{{content_block_id}}-graph{{child_id}}', vlSpec, {
        actions: false
    }).then(result=>console.log(result)).catch(console.warn);
</script>
    '''

    header_j2 = '''
{% if "styling" in content_block and "header" in content_block["styling"] -%}
    {% set content_block_header_styling = content_block["styling"]["header"] | render_styling -%}
{% else -%}
    {% set content_block_header_styling = "" -%}
{% endif -%}

{% if "styling" in content_block and "subheader" in content_block["styling"] -%}
    {% set content_block_subheader_styling = content_block["styling"]["subheader"] | render_styling -%}
{% else -%}
    {% set content_block_subheader_styling = "" -%}
{% endif -%}

<div id="{{content_block_id}}-header" {{ content_block_header_styling | replace("{{section_id}}", section_id) | replace("{{content_block_id}}", content_block_id) }}>
    {%- if "header" in content_block -%}
      {% if content_block["header"] is mapping %}
        <div>
          {{ content_block["header"] | render_content_block }}
        </div>
      {% else %}
        <h5>
            {{ content_block["header"] | render_content_block }}
        </h5>
      {% endif %}
    {% endif -%}
    {%- if "subheader" in content_block -%}
      {% if content_block["subheader"] is mapping %}
        <div id="{{content_block_id}}-subheader" {{ content_block_subheader_styling | replace("{{section_id}}", section_id) | replace("{{content_block_id}}", content_block_id) }}>
          {{ content_block["subheader"] | render_content_block }}
        </div>
      {% else %}
        <h6 id="{{content_block_id}}-subheader" {{ content_block_subheader_styling | replace("{{section_id}}", section_id) | replace("{{content_block_id}}", content_block_id) }}>
            {{ content_block["subheader"] | render_content_block }}
        </h6>
      {% endif %}
    {% endif -%}
</div>
    '''

    index_page_j2 = '''
<!DOCTYPE html>
<html>
  <head>
    <title>Data Docs created by Great Expectations</title>
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <meta charset="UTF-8">
    <title>{% block title %}{% endblock %}</title>

    {# {# Remove this when not debugging: #}
    {# <meta http-equiv="refresh" content="1"/> #}
    <link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/4.3.1/css/bootstrap.min.css"/>

    <style>{% include 'data_docs_default_styles.css' %}</style>

    <style>{% include 'data_docs_custom_styles.css' ignore missing %}</style>

    {% include 'js_script_imports.j2' %}
    {% include 'favicon.j2' %}
  </head>

  <body
          data-spy="scroll"
          data-target="#navigation"
          data-offset="50"
  >
    {% include 'carousel_modal.j2' %}

    <script>
      try {
        if (localStorage.getItem('ge-walkthrough-modal-dismissed') !== 'true') {
          $(".ge-walkthrough-modal").modal();
        }
      }
      catch(error) {
        $(".ge-walkthrough-modal").modal();
        console.log(error);
      }
    </script>

    {% include 'top_navbar.j2' %}
    <div class="container-fluid pt-4 pb-4 pl-5 pr-5">
      <div class="row">
        {% include 'sidebar.j2' %}
        <div class="col-md-10 col-lg-10 col-xs-12 pl-md-4 pr-md-3">
          {% for section in sections %}
            {% set section_loop = loop -%}
            {% include 'section.j2' %}
          {% endfor %}
        </div>
      </div>
    </div>

    {% if cta_footer %}
      {% include 'cta_footer.j2' %}
    {% endif %}

  </body>
</html>
    '''

    js_script_imports_j2 = '''
<script src="https://cdn.jsdelivr.net/npm/vega@5.3.5/build/vega.js"></script>
<script src="https://cdn.jsdelivr.net/npm/vega-lite@3.2.1/build/vega-lite.js"></script>
<script src="https://cdn.jsdelivr.net/npm/vega-embed@4.0.0/build/vega-embed.js"></script>
<script src="https://kit.fontawesome.com/8217dffd95.js"></script>

<script src="https://code.jquery.com/jquery-3.4.1.min.js"></script>
<script src="https://stackpath.bootstrapcdn.com/bootstrap/4.3.1/js/bootstrap.min.js" integrity="sha384-JjSmVgyd0p3pXB1rRibZUAYoIIy6OrQ6VrjIEaFf/nJGzIxFDsf4x0xIM+B07jRM" crossorigin="anonymous"></script>
    '''

    markdown_j2 = '''
{{ content_block["markdown"] | render_markdown }}
    '''

    page_j2 = '''
<!DOCTYPE html>
<html>
  <head>
    <title>Data documentation compiled by Great Expectations</title>
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <meta charset="UTF-8">
    <title>{% block title %}{% endblock %}</title>

    {# {# Remove this when not debugging: #}
    {# <meta http-equiv="refresh" content="1"/> #}
    <link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/4.3.1/css/bootstrap.min.css"/>

    <style>{% include 'data_docs_default_styles.css' %}</style>
    <style>{% include 'data_docs_custom_styles.css' ignore missing %}</style>

    {% include 'js_script_imports.j2' %}
    {% include 'favicon.j2' %}
  </head>

  <body
        data-spy="scroll"
        data-target="#navigation"
        data-offset="50"
    >
    {% include 'carousel_modal.j2' %}
    {% include 'edit_expectations_instructions_modal.j2' %}
    {% include 'top_navbar.j2' %}
    {% if renderer_type == "ValidationResultsPageRenderer" %}
      <script>
        try {
          if (localStorage.getItem('ge-walkthrough-modal-dismissed') !== 'true') {
            $(".ge-walkthrough-modal").modal();
          }
        }
        catch(error) {
          $(".ge-walkthrough-modal").modal();
          console.log(error);
        }
      </script>
    {% endif %}

    <div class="container-fluid pt-4 pb-4 pl-5 pr-5">
      <div class="row">
        {% include 'sidebar.j2' %}
        <div class="col-md-10 col-lg-10 col-xs-12 pl-md-4 pr-md-3">
        {% for section in sections %}
          {% set section_loop = loop -%}
          {% include 'section.j2' %}
        {% endfor %}
        </div>
      </div>
    </div>
  </body>
</html>
    '''

    page_action_card_j2 = '''
<script>
    function showAllValidations() {
    $(".hide-succeeded-validation-target-child").parent().fadeIn();
    $(".hide-succeeded-validation-target").fadeIn();
    $(".hide-succeeded-validations-column-section-target-child").parent().parent().each((idx, el) => {
      $(el).fadeIn();
      const elId = el.id;
      $(`a[href$=${elId}]`).fadeIn();
    })
  }

    function hideSucceededValidations() {
    $(".hide-succeeded-validation-target-child").parent().fadeOut();
    $(".hide-succeeded-validation-target").fadeOut();
    $(".hide-succeeded-validations-column-section-target-child").parent().parent().each((idx, el) => {
      $(el).fadeOut();
      const elId = el.id;
      $(`a[href$=${elId}]`).fadeOut();
    })
  }
</script>

<div class="card bg-light mb-3">
  <div class="card-header p-2">
    <strong>Actions</strong>
  </div>
  <div class="card-body p-3">
    {% if renderer_type in ["ValidationResultsPageRenderer", "SiteIndexPageRenderer"] %}
      <div class="mb-2">
        <p class="card-text col-12 p-0 mb-1">
          Validation Filter:
        </p>
        <div class="d-flex justify-content-center">
          <div class="btn-group btn-group-toggle" data-toggle="buttons">
            <label class="btn btn-primary active" onclick="showAllValidations()">
              <input type="radio" name="options" id="option1" autocomplete="off" checked> Show All
            </label>
            <label class="btn btn-primary" onclick="hideSucceededValidations()">
              <input type="radio" name="options" id="option2" autocomplete="off"> Failed Only
            </label>
          </div>
        </div>
      </div>
    {% endif %}

    {% if renderer_type in ["ValidationResultsPageRenderer", "ExpectationSuitePageRenderer"] %}
      <div class="mb-2">
        <div class="d-flex justify-content-center">
          <button type="button" class="btn btn-warning" data-toggle="modal" data-target=".ge-expectation-editing-instructions-modal">
            <i class="fas fa-edit"></i> How to Edit This Suite
          </button>
        </div>
      </div>
    {% endif %}

    <div class="mb-2">
      <div class="d-flex justify-content-center">
        <button type="button" class="btn btn-info" data-toggle="modal" data-target=".ge-walkthrough-modal">
          Show Walkthrough
        </button>
      </div>
    </div>
  </div>
</div>
    '''

    section_j2 = '''
<div id="section-{{section_loop.index|default(1)}}" class="ge-section container-fluid mb-1 pb-1 pl-sm-3 px-0">
    <div class="row" {{ section | render_styling_from_string_template }}>
        {% set section_loop = loop -%}
        {% for content_block in section["content_blocks"] -%}
            {% set content_block_loop = loop -%}
            {% include 'component.j2' %}
        {% endfor %}
    </div>
</div>
    '''

    sidebar_j2 = '''
<div class="col-lg-2 col-md-2 col-sm-12 d-sm-block px-0">
  {% include 'ge_info.j2' %}
  <div class="sticky">
    {% include 'page_action_card.j2' %}
    {% if renderer_type != "SiteIndexPageRenderer" %}
      {% include 'table_of_contents.j2' %}
    {% endif %}
  </div>
</div>
    '''

    string_template_j2 = '''
{{ content_block["string_template"] | render_string_template }}
    '''

    table_j2 = '''
{% include 'content_block_header.j2' %}

{% if "styling" in content_block and "body" in content_block["styling"] -%}
    {% set content_block_body_styling = content_block["styling"]["body"] | render_styling -%}
{% else -%}
    {% set content_block_body_styling = "" -%}
{% endif -%}

<table id="{{content_block_id}}-body" {{ content_block_body_styling | replace("{{section_id}}", section_id) | replace("{{content_block_id}}", content_block_id) }}>
    {% if content_block.get("header_row") %}
        <tr>
            {% for header_cell in content_block.get("header_row") %}
                <th>{{ header_cell | render_content_block }}</th>
            {% endfor %}
        </tr>
    {% endif %}

    {% for row in content_block["table"] -%}
        <tr>
        {% set rowloop = loop -%}
        {% for cell in row -%}
            {%- set content_block_id = content_block_id ~ "-cell-" ~ rowloop.index ~ "-" ~ loop.index -%}
            {% if cell is mapping and "styling" in cell -%}
                {% set cell_styling = cell["styling"].get("parent", {}) | render_styling | replace("{{section_id}}", section_id) | replace("{{content_block_id}}", content_block_id) -%}
            {% else -%}
                {% set cell_styling = "" -%}
            {% endif -%}
            <td id="{{content_block_id}}" {{ cell_styling }}><div class="show-scrollbars">{{ cell | render_content_block }}</div></td>
        {%- endfor -%}
        </tr>
    {%- endfor -%}
</table>
    '''

    table_of_contents_j2 = '''
<script>
  $(window).on('activate.bs.scrollspy', function () {
    document.querySelector(".nav-link.active").scrollIntoViewIfNeeded();
  });
</script>

<div class="card bg-light d-md-block d-none" style="max-height: 75vh">
  <div class="card-header p-2">
    <strong>Table of Contents</strong>
  </div>
  <div class="card-body p-0" style="overflow: auto; height: 100%">
    <nav id="navigation" class="rounded navbar navbar-light bg-light ge-navigation-sidebar-container p-1">
      <ul class="nav nav-pills flex-column ge-navigation-sidebar-content col-12 p-0">
        {% for section in sections %}
          {% if section['section_name'] == "Overview" %}
            <li class="nav-item">
              <a class="nav-link ge-navigation-sidebar-link" href="#section-{{ loop.index }}"
               style="white-space: normal; word-break: break-all;overflow-wrap: normal;">
                <strong>{{ section['section_name'] }}</strong>
              </a>
            </li>
          {% else %}
            <li class="nav-item">
              <a class="nav-link ge-navigation-sidebar-link ml-1" href="#section-{{ loop.index }}"
                 style="white-space: normal; word-break: break-all;overflow-wrap: normal;">
                {{ section['section_name'] }}
              </a>
            </li>
          {% endif %}
        {% endfor %}
      </ul>
    </nav>
  </div>
</div>
    '''

    text_j2 = '''
{% if not content_block_body_styling and content_block.get("styling") %}
    {% set content_block_body_styling = content_block.get("styling") | render_styling %}
{% endif %}

<div id="{{content_block_id}}-body" {{ content_block_body_styling | replace("{{section_id}}", section_id) | replace("{{content_block_id}}", content_block_id) }}>
  {% include 'content_block_header.j2' %}
  {% for text in content_block["text"] -%}
      {% if text is mapping and "styling" in text -%}
          {% set paragraph_styling = text["styling"].get("parent", {}) | render_styling -%}
      {% else -%}
          {% set paragraph_styling = "" -%}
      {% endif -%}

      {% if text is mapping and text.get("content_block_type") == "markdown" %}
        <div {{ paragraph_styling }}>{{ text | render_content_block }}</div>
      {% else %}
        <p {{ paragraph_styling }}>{{ text | render_content_block }}</p>
      {% endif %}
    {% endfor -%}
</div>
    '''

    top_navbar_j2 = '''
{% if expectation_suite_name %}
  {% set expectation_suite_name_dot_count = expectation_suite_name.count(".") -%}
{% endif %}

{% if utm_medium == "validation-results-page" or utm_medium == "profiling-results-page" %}
  {% set home_url =  ((expectation_suite_name_dot_count + 3) * "../") + "index.html" -%}
  {% set static_images_dir = ((expectation_suite_name_dot_count + 3) * "../") + "static/images/" -%}
{% elif utm_medium == "expectation-suite-page" %}
  {% set home_url = ((expectation_suite_name_dot_count + 1) * "../") + "index.html" -%}
  {% set static_images_dir = ((expectation_suite_name_dot_count + 1) * "../") + "static/images/" -%}
{% elif utm_medium == "index-page" %}
  {% set home_url = "#" -%}
{% endif %}

{% set static_images_dir = "https://great-expectations-web-assets.s3.us-east-2.amazonaws.com/" -%}

<nav class="navbar bg-light navbar-expand-md sticky-top border-bottom" style="height: 70px">
  <div class="mr-auto">
    <nav class="d-flex align-items-center">
      <div class="float-left navbar-brand m-0 h-100">
        <a href="{{ home_url }}">
          <img
            class="NO-CACHE"
            src="{{ static_images_dir + "logo-long.png" }}"
            alt="Great Expectations"
            style="width: auto; height: 50px"
          />
        </a>
      </div>
      {% if page_title %}
        <ol class="ge-breadcrumbs breadcrumb d-md-inline-flex bg-light ml-2 mr-0 mt-0 mb-0 pt-0 pb-0 d-none">
            <li class="ge-breadcrumbs-item breadcrumb-item"><a href="{{ home_url }}">Home</a></li>
            <li class="ge-breadcrumbs-item breadcrumb-item active" aria-current="page">{{page_title}}</li>
        </ol>
      {% endif %}
    </nav>
  </div>
</nav>

<script>
  var d = new Date()
  var els = document.getElementsByClassName('NO-CACHE');
  for (var i = 0; i < els.length; i++)
  {
      els[i].attributes['src'].value += "?d=" + d.toISOString();
  }
</script>
    '''

    value_list_j2 = '''
{% include 'content_block_header.j2' %}

<p id="{{content_block_id}}-body" {{ content_block_body_styling | replace("{{section_id}}", section_id) | replace("{{content_block_id}}", content_block_id) }}>
    {% for value in content_block["value_list"] -%}
        {{ value | render_content_block }}
    {% endfor -%}
</p>
    '''

    data_docs_custom_styles_template_css = '''
/*index page*/
.ge-index-page-site-name-title {}
.ge-index-page-table-container {}
.ge-index-page-table {}
.ge-index-page-table-profiling-links-header {}
.ge-index-page-table-expectations-links-header {}
.ge-index-page-table-validations-links-header {}
.ge-index-page-table-profiling-links-list {}
.ge-index-page-table-profiling-links-item {}
.ge-index-page-table-expectation-suite-link {}
.ge-index-page-table-validation-links-list {}
.ge-index-page-table-validation-links-item {}

/*breadcrumbs*/
.ge-breadcrumbs {}
.ge-breadcrumbs-item {}

/*navigation sidebar*/
.ge-navigation-sidebar-container {}
.ge-navigation-sidebar-content {}
.ge-navigation-sidebar-title {}
.ge-navigation-sidebar-link {}
    '''

    data_docs_default_styles_css = '''
{% if utm_medium == "validation-results-page" or utm_medium == "profiling-results-page" %}
  {% set static_fonts_dir = "../../../../../static/fonts/" -%}
{% elif utm_medium == "expectation-suite-page" %}
  {% set static_fonts_dir = "../../../../static/fonts/" -%}
{% elif utm_medium == "index-page" %}
  {% set static_fonts_dir = "./static/fonts/" -%}
{% endif %}

@font-face {
  font-family: HKGroteskPro;
  src: url("{{ static_fonts_dir +  'HKGrotesk/HKGrotesk-Regular.otf'}}");
  font-weight: normal;
}

@font-face {
  font-family: HKGrotesk-Light;
  src: url("{{ static_fonts_dir +  'HKGrotesk/HKGrotesk-Light.otf'}}");
  font-weight: 300;
}

body {
  position: relative;
}

.container {
  padding-top: 50px;
}

.sticky {
  position: -webkit-sticky;
  position: sticky;
  top: 90px;
  z-index: 1;
}

.ge-section {
  clear: both;
  margin-bottom: 30px;
  padding-bottom: 20px;
}

.popover {
  max-width: 100%;
}

.cooltip {
  display: inline-block;
  position: relative;
  text-align: left;
  cursor: pointer;
}

.cooltip .top {
  min-width: 200px;
  top: -6px;
  left: 50%;
  transform: translate(-50%, -100%);
  padding: 10px 20px;
  color: #FFFFFF;
  background-color: #222222;
  font-weight: normal;
  font-size: 13px;
  border-radius: 8px;
  position: absolute;
  z-index: 99999999 !important;
  box-sizing: border-box;
  box-shadow: 0 1px 8px rgba(0, 0, 0, 0.5);
  display: none;
}

.cooltip:hover .top {
  display: block;
  z-index: 99999999 !important;
}

.cooltip .top i {
  position: absolute;
  top: 100%;
  left: 50%;
  margin-left: -12px;
  width: 24px;
  height: 12px;
  overflow: hidden;
}

.cooltip .top i::after {
  content: '';
  position: absolute;
  width: 12px;
  height: 12px;
  left: 50%;
  transform: translate(-50%, -50%) rotate(45deg);
  background-color: #222222;
  box-shadow: 0 1px 8px rgba(0, 0, 0, 0.5);
}

ul {
  padding-inline-start: 20px;
}

.show-scrollbars {
  overflow: auto;
}

td .show-scrollbars {
  max-height: 80vh;
}

/*.show-scrollbars ul {*/
/*  padding-bottom: 20px*/
/*}*/

.show-scrollbars::-webkit-scrollbar {
  -webkit-appearance: none;
}

.show-scrollbars::-webkit-scrollbar:vertical {
  width: 11px;
}

.show-scrollbars::-webkit-scrollbar:horizontal {
  height: 11px;
}

.show-scrollbars::-webkit-scrollbar-thumb {
  border-radius: 8px;
  border: 2px solid white; /* should match background, can't be transparent */
  background-color: rgba(0, 0, 0, .5);
}

#ge-cta-footer {
  opacity: 0.9;
  border-left-width: 4px
}

.carousel-caption {
    position: relative;
    left: 0;
    top: 0;
}
    '''

    TEMPLATES_LOAD_DICT = {
        'bullet_list.j2': bullet_list_j2,
        'carousel_modal.j2': carousel_modal_j2,
        'collapse.j2': collapse_j2,
        'component.j2': component_j2,
        'content_block_container.j2': content_block_container_j2,
        'content_block_header.j2': content_block_header_j2,
        'cta_footer.j2': cta_footer_j2,
        'edit_expectations_instructions_modal.j2': edit_expectations_instructions_modal_j2,
        'favicon.j2': favicon_j2,
        'ge_info.j2': ge_info_j2,
        'graph.j2': graph_j2,
        'header.j2': header_j2,
        'index_page.j2': index_page_j2,
        'js_script_imports.j2': js_script_imports_j2,
        'markdown.j2': markdown_j2,
        'page.j2': page_j2,
        'page_action_card.j2': page_action_card_j2,
        'section.j2': section_j2,
        'sidebar.j2': sidebar_j2,
        'string_template.j2': string_template_j2,
        'table.j2': table_j2,
        'table_of_contents.j2': table_of_contents_j2,
        'text.j2': text_j2,
        'top_navbar.j2': top_navbar_j2,
        'value_list.j2': value_list_j2
    }

    STYLES_LOAD_DICT = {
        'data_docs_custom_styles_template.css': data_docs_custom_styles_template_css,
        'data_docs_default_styles.css': data_docs_default_styles_css
    }

    def __init__(self, custom_styles_directory=None):
        self.custom_styles_directory = custom_styles_directory

    def render(self, document, template=None, **kwargs):
        self._validate_document(document)

        if template is None:
            template = self._template

        t = self._get_template(template)
        if isinstance(document, RenderedContent):
            document = document.to_json_dict()
        return t.render(document, **kwargs)

    def _get_template(self, template):
        if template is None:
            return NoOpTemplate

        #templates_loader = PackageLoader(
        #    'great_expectations',
        #    'render/view/templates'
        #)
        #styles_loader = PackageLoader(
        #    'great_expectations',
        #    'render/view/static/styles'
        #)
        templates_loader = DictLoader(DefaultJinjaView.TEMPLATES_LOAD_DICT)
        styles_loader = DictLoader(DefaultJinjaView.STYLES_LOAD_DICT)

        loaders = [
            templates_loader,
            styles_loader
        ]

        if self.custom_styles_directory:
            loaders.append(FileSystemLoader(self.custom_styles_directory))

        env = Environment(
            loader=ChoiceLoader(loaders),
            autoescape=select_autoescape(['html', 'xml']),
            extensions=["jinja2.ext.do"]
        )
        env.filters['render_string_template'] = self.render_string_template
        env.filters['render_styling_from_string_template'] = self.render_styling_from_string_template
        env.filters['render_styling'] = self.render_styling
        env.filters['render_content_block'] = self.render_content_block
        env.filters['render_markdown'] = self.render_markdown
        env.filters['get_html_escaped_json_string_from_dict'] = self.get_html_escaped_json_string_from_dict
        env.filters['generate_html_element_uuid'] = self.generate_html_element_uuid
        env.globals['ge_version'] = ge_version

        template = env.get_template(template)
        template.globals['now'] = datetime.datetime.utcnow

        return template

    @contextfilter
    def render_content_block(self, context, content_block, index=None, content_block_id=None):
        if type(content_block) is str:
            return "<span>{content_block}</span>".format(content_block=content_block)
        elif content_block is None:
            return ""
        elif type(content_block) is list:
            # If the content_block item here is actually a list of content blocks then we want to recursively render
            rendered_block = ""
            for idx, content_block_el in enumerate(content_block):
                if (isinstance(content_block_el, RenderedComponentContent) or
                        isinstance(content_block_el, dict) and "content_block_type" in content_block_el):
                    rendered_block += self.render_content_block(context, content_block_el, idx)
                else:
                    rendered_block += "<span>" + str(content_block_el) + "</span>"
            return rendered_block
        elif not isinstance(content_block, dict):
            return content_block
        content_block_type = content_block.get("content_block_type")
        template = self._get_template(template="{content_block_type}.j2".format(content_block_type=content_block_type))
        if content_block_id:
            return template.render(context, content_block=content_block, index=index, content_block_id=content_block_id)
        else:
            return template.render(context, content_block=content_block, index=index)

    def get_html_escaped_json_string_from_dict(self, source_dict):
        return json.dumps(source_dict).replace('"', '\\"').replace('"', '&quot;')

    def render_styling(self, styling):

        """Adds styling information suitable for an html tag.

        Example styling block::

            styling = {
                "classes": ["alert", "alert-warning"],
                "attributes": {
                    "role": "alert",
                    "data-toggle": "popover",
                },
                "styles" : {
                    "padding" : "10px",
                    "border-radius" : "2px",
                }
            }

        The above block returns a string similar to::

            'class="alert alert-warning" role="alert" data-toggle="popover" style="padding: 10px; border-radius: 2px"'

        "classes", "attributes" and "styles" are all optional parameters.
        If they aren't present, they simply won't be rendered.

        Other dictionary keys are also allowed and ignored.
        """

        class_list = styling.get("classes", None)
        if class_list is None:
            class_str = ""
        else:
            if type(class_list) == str:
                raise TypeError("classes must be a list, not a string.")
            class_str = 'class="' + ' '.join(class_list) + '" '

        attribute_dict = styling.get("attributes", None)
        if attribute_dict is None:
            attribute_str = ""
        else:
            attribute_str = ""
            for k, v in attribute_dict.items():
                attribute_str += k + '="' + v + '" '

        style_dict = styling.get("styles", None)
        if style_dict is None:
            style_str = ""
        else:
            style_str = 'style="'
            style_str += " ".join([k + ':' + v + ';' for k, v in style_dict.items()])
            style_str += '" '

        styling_string = pTemplate('$classes$attributes$style').substitute({
            "classes": class_str,
            "attributes": attribute_str,
            "style": style_str,
        })

        return styling_string

    def render_styling_from_string_template(self, template):
        # NOTE: We should add some kind of type-checking to template
        """This method is a thin wrapper use to call `render_styling` from within jinja templates.
        """
        if not isinstance(template, (dict, OrderedDict)):
            return template

        if "styling" in template:
            return self.render_styling(template["styling"])

        else:
            return ""

    def generate_html_element_uuid(self, prefix=None):
        if prefix:
            return prefix + str(uuid4())
        else:
            return str(uuid4())

    def render_markdown(self, markdown):
        try:
            return mistune.markdown(markdown)
        except OSError:
            return markdown

    def render_string_template(self, template):
        # NOTE: Using this line for debugging. This should probably be logged...?
        # print(template)

        # NOTE: We should add some kind of type-checking to template
        if not isinstance(template, (dict, OrderedDict)):
            return template

        tag = template.get("tag", "span")
        template["template"] = template.get("template", "").replace("\n", "<br>")

        if "tooltip" in template:
            if template.get("styling", {}).get("classes"):
                classes = template.get("styling", {}).get("classes")
                classes.append("cooltip")
                template["styling"]["classes"] = classes
            elif template.get("styling"):
                template["styling"]["classes"] = ["cooltip"]
            else:
                template["styling"] = {
                    "classes": ["cooltip"]
                }

            tooltip_content = template["tooltip"]["content"]
            tooltip_content.replace("\n", "<br>")
            placement = template["tooltip"].get("placement", "top")
            base_template_string = """
                <{tag} $styling>
                    $template
                    <span class={placement}>
                        {tooltip_content}
                    </span>
                </{tag}>
            """.format(placement=placement, tooltip_content=tooltip_content, tag=tag)
        else:
            base_template_string = """
                <{tag} $styling>
                    $template
                </{tag}>
            """.format(tag=tag)

        if "styling" in template:
            params = template.get("params", {})

            # Apply default styling
            if "default" in template["styling"]:
                default_parameter_styling = template["styling"]["default"]
                default_param_tag = default_parameter_styling.get("tag", "span")
                base_param_template_string = "<{param_tag} $styling>$content</{param_tag}>".format(
                    param_tag=default_param_tag)

                for parameter in template["params"].keys():

                    # If this param has styling that over-rides the default, skip it here and get it in the next loop.
                    if "params" in template["styling"]:
                        if parameter in template["styling"]["params"]:
                            continue

                    params[parameter] = pTemplate(base_param_template_string).substitute({
                        "styling": self.render_styling(default_parameter_styling),
                        "content": params[parameter],
                    })

            # Apply param-specific styling
            if "params" in template["styling"]:
                # params = template["params"]
                for parameter, parameter_styling in template["styling"]["params"].items():
                    if parameter not in params:
                        continue
                    param_tag = parameter_styling.get("tag", "span")
                    param_template_string = "<{param_tag} $styling>$content</{param_tag}>".format(param_tag=param_tag)
                    params[parameter] = pTemplate(param_template_string).substitute({
                        "styling": self.render_styling(parameter_styling),
                        "content": params[parameter],
                    })

            string = pTemplate(
                pTemplate(base_template_string).substitute(
                    {"template": template["template"], "styling": self.render_styling(template.get("styling", {}))})
            ).substitute(params)
            return string

        return pTemplate(
            pTemplate(base_template_string).substitute(
                {"template": template.get("template", ""), "styling": self.render_styling(template.get("styling", {}))})
        ).substitute(template.get("params", {}))

    def _validate_document(self, document):
        raise NotImplementedError


class DefaultJinjaPageView(DefaultJinjaView):
    _template = "page.j2"

    def _validate_document(self, document):
        assert isinstance(document, RenderedDocumentContent)


class DefaultJinjaIndexPageView(DefaultJinjaPageView):
    _template = "index_page.j2"


class DefaultJinjaSectionView(DefaultJinjaView):
    _template = "section.j2"

    def _validate_document(self, document):
        assert isinstance(document["section"], dict)  # For now low-level views take dicts


class DefaultJinjaComponentView(DefaultJinjaView):
    _template = "component.j2"

    def _validate_document(self, document):
        assert isinstance(document["content_block"], dict)  # For now low-level views take dicts


class ValidationResultsPageRenderer(Renderer):

    def __init__(self, column_section_renderer=None):
        if column_section_renderer is None:
            column_section_renderer = {
                "class_name": "ValidationResultsColumnSectionRenderer"
            }
        self._column_section_renderer = instantiate_class_from_config(
            config=column_section_renderer,
            runtime_environment={},
            config_defaults={
                #"module_name": column_section_renderer.get("module_name", "great_expectations.render.renderer.column_section_renderer")
                "module_name": column_section_renderer.get("module_name", "great_expectations.common")
            }
        )

    def render(self, validation_results):
        run_id = validation_results.meta['run_id']
        batch_id = BatchKwargs(validation_results.meta['batch_kwargs']).to_id()
        expectation_suite_name = validation_results.meta['expectation_suite_name']
        batch_kwargs = validation_results.meta.get("batch_kwargs")

        # add datasource key to batch_kwargs if missing
        if 'datasource' not in validation_results.meta.get("batch_kwargs", {}):
            # check if expectation_suite_name follows datasource.generator.data_asset_name.suite_name pattern
            if len(expectation_suite_name.split('.')) == 4:
                batch_kwargs['datasource'] = expectation_suite_name.split('.')[0]

        # Group EVRs by column
        columns = {}
        for evr in validation_results.results:
            if "column" in evr.expectation_config.kwargs:
                column = evr.expectation_config.kwargs["column"]
            else:
                column = "Table-Level Expectations"

            if column not in columns:
                columns[column] = []
            columns[column].append(evr)

        ordered_columns = Renderer._get_column_list_from_evrs(validation_results)

        overview_content_blocks = [
            self._render_validation_header(validation_results),
            self._render_validation_statistics(validation_results=validation_results),
        ]

        collapse_content_blocks = [self._render_validation_info(validation_results=validation_results)]

        if validation_results["meta"].get("batch_markers"):
            collapse_content_blocks.append(
                self._render_nested_table_from_dict(
                    input_dict=validation_results["meta"].get("batch_markers"),
                    header="Batch Markers"
                )
            )

        if validation_results["meta"].get("batch_kwargs"):
            collapse_content_blocks.append(
                self._render_nested_table_from_dict(
                    input_dict=validation_results["meta"].get("batch_kwargs"),
                    header="Batch Kwargs"
                )
            )

        if validation_results["meta"].get("batch_parameters"):
            collapse_content_blocks.append(
                self._render_nested_table_from_dict(
                    input_dict=validation_results["meta"].get("batch_parameters"),
                    header="Batch Parameters"
                )
            )

        collapse_content_block = CollapseContent(**{
            "collapse_toggle_link": "Show more info...",
            "collapse": collapse_content_blocks,
            "styling": {
                "body": {
                    "classes": ["card", "card-body"]
                },
                "classes": ["col-12", "p-1"]
            }
        })

        overview_content_blocks.append(collapse_content_block)

        sections = [
            RenderedSectionContent(**{
                "section_name": "Overview",
                "content_blocks": overview_content_blocks
            })
        ]

        if "Table-Level Expectations" in columns:
            sections += [
                self._column_section_renderer.render(
                    validation_results=columns["Table-Level Expectations"]
                )
            ]

        sections += [
            self._column_section_renderer.render(
                validation_results=columns[column],
            ) for column in ordered_columns
        ]

        return RenderedDocumentContent(**{
            "renderer_type": "ValidationResultsPageRenderer",
            "page_title": expectation_suite_name + " / " + run_id + " / " + batch_id,
            "batch_kwargs": batch_kwargs,
            "expectation_suite_name": expectation_suite_name,
            "sections": sections,
            "utm_medium": "validation-results-page",
        })

    @classmethod
    def _render_validation_header(cls, validation_results):
        success = validation_results.success
        expectation_suite_name = validation_results.meta['expectation_suite_name']
        if success:
            success = '<i class="fas fa-check-circle text-success" aria-hidden="true"></i> Succeeded'
        else:
            success = '<i class="fas fa-times text-danger" aria-hidden="true"></i> Failed'
        return RenderedHeaderContent(**{
            "content_block_type": "header",
            "header": RenderedStringTemplateContent(**{
                "content_block_type": "string_template",
                "string_template": {
                    "template": 'Overview',
                    "tag": "h5",
                    "styling": {
                        "classes": ["m-0"]
                    }
                }
            }),
            "subheader": RenderedStringTemplateContent(**{
                "content_block_type": "string_template",
                "string_template": {
                    "template": "${suite_title} ${expectation_suite_name}\n${status_title} ${success}",
                    "params": {
                        "suite_title": "Expectation Suite:",
                        "status_title": "Status:",
                        "expectation_suite_name": expectation_suite_name,
                        "success": success
                    },
                    "styling": {
                        "params": {
                            "suite_title": {
                                "classes": ["h6"]
                            },
                            "status_title": {
                                "classes": ["h6"]
                            }
                        },
                        "classes": ["mb-0", "mt-1"]
                    }
                }
            }),
            "styling": {
                "classes": ["col-12", "p-0"],
                "header": {
                    "classes": ["alert", "alert-secondary"]
                }
            }
        })

    @classmethod
    def _render_validation_info(cls, validation_results):
        run_id = validation_results.meta['run_id']
        ge_version = validation_results.meta["great_expectations.__version__"]

        return RenderedTableContent(**{
            "content_block_type": "table",
            "header": RenderedStringTemplateContent(**{
                "content_block_type": "string_template",
                "string_template": {
                    "template": 'Info',
                    "tag": "h6",
                    "styling": {
                        "classes": ["m-0"]
                    }
                }
            }),
            "table": [
                ["Great Expectations Version", ge_version],
                ["Run ID", run_id]
            ],
            "styling": {
                "classes": ["col-12", "table-responsive", "mt-1"],
                "body": {
                    "classes": ["table", "table-sm"]
                }
            },
        })

    @classmethod
    def _render_nested_table_from_dict(cls, input_dict, header=None, sub_table=False):
        table_rows = []

        for kwarg, value in input_dict.items():
            if not isinstance(value, (dict, OrderedDict)):
                table_row = [
                    RenderedStringTemplateContent(**{
                        "content_block_type": "string_template",
                        "string_template": {
                            "template": "$value",
                            "params": {
                                "value": str(kwarg)
                            },
                            "styling": {
                                "default": {
                                    "styles": {
                                        "word-break": "break-all"
                                    }
                                },
                            }
                        },
                        "styling": {
                            "parent": {
                                "classes": ["pr-3"],
                            }
                        }
                    }),
                    RenderedStringTemplateContent(**{
                        "content_block_type": "string_template",
                        "string_template": {
                            "template": "$value",
                            "params": {
                                "value": str(value)
                            },
                            "styling": {
                                "default": {
                                    "styles": {
                                        "word-break": "break-all"
                                    }
                                },
                            }
                        },
                        "styling": {
                            "parent": {
                                "classes": [],
                            }
                        }
                    })
                ]
            else:
                table_row = [
                    RenderedStringTemplateContent(**{
                        "content_block_type": "string_template",
                        "string_template": {
                            "template": "$value",
                            "params": {
                                "value": str(kwarg)
                            },
                            "styling": {
                                "default": {
                                    "styles": {
                                        "word-break": "break-all"
                                    }
                                },
                            }
                        },
                        "styling": {
                            "parent": {
                                "classes": ["pr-3"],
                            }
                        }
                    }),
                    cls._render_nested_table_from_dict(value, sub_table=True)
                ]
            table_rows.append(table_row)

        table_rows.sort(key=lambda row: row[0].string_template["params"]["value"])

        if sub_table:
            return RenderedTableContent(**{
                "content_block_type": "table",
                "table": table_rows,
                "styling": {
                    "classes": ["col-6", "table-responsive"],
                    "body": {
                        "classes": ["table", "table-sm", "m-0"]
                    },
                    "parent": {
                        "classes": ["pt-0", "pl-0", "border-top-0"]
                    }
                },
            })
        else:
            return RenderedTableContent(**{
                "content_block_type": "table",
                "header": RenderedStringTemplateContent(**{
                "content_block_type": "string_template",
                "string_template": {
                    "template": header,
                    "tag": "h6",
                    "styling": {
                        "classes": ["m-0"]
                    }
                }
            }),
                "table": table_rows,
                "styling": {
                    "classes": ["col-6", "table-responsive", "mt-1"],
                    "body": {
                        "classes": ["table", "table-sm"]
                    }
                },
            })

    @classmethod
    def _render_validation_statistics(cls, validation_results):
        statistics = validation_results["statistics"]
        statistics_dict = OrderedDict([
            ("evaluated_expectations", "Evaluated Expectations"),
            ("successful_expectations", "Successful Expectations"),
            ("unsuccessful_expectations", "Unsuccessful Expectations"),
            ("success_percent", "Success Percent")
        ])
        table_rows = []
        for key, value in statistics_dict.items():
            if statistics.get(key) is not None:
                if key == "success_percent":
                    # table_rows.append([value, "{0:.2f}%".format(statistics[key])])
                    table_rows.append([value, num_to_str(statistics[key], precision=4) + "%"])
                else:
                    table_rows.append([value, statistics[key]])

        return RenderedTableContent(**{
            "content_block_type": "table",
            "header": RenderedStringTemplateContent(**{
                "content_block_type": "string_template",
                "string_template": {
                    "template": 'Statistics',
                    "tag": "h6",
                    "styling": {
                        "classes": ["m-0"]
                    }
                }
            }),
            "table": table_rows,
            "styling": {
                "classes": ["col-6", "table-responsive", "mt-1", "p-1"],
                "body": {
                    "classes": ["table", "table-sm"]
                }
            },
        })


class ExpectationSuitePageRenderer(Renderer):

    def __init__(self, column_section_renderer=None):
        if column_section_renderer is None:
            column_section_renderer = {
                "class_name": "ExpectationSuiteColumnSectionRenderer"
            }
        self._column_section_renderer = instantiate_class_from_config(
            config=column_section_renderer,
            runtime_environment={},
            config_defaults={
                #"module_name": column_section_renderer.get("module_name", "great_expectations.render.renderer.column_section_renderer")
                "module_name": column_section_renderer.get("module_name", "great_expectations.common")
            }
        )

    def render(self, expectations):
        columns, ordered_columns = self._group_and_order_expectations_by_column(expectations)
        expectation_suite_name = expectations.expectation_suite_name

        overview_content_blocks = [
            self._render_expectation_suite_header(),
            self._render_expectation_suite_info(expectations)
        ]

        table_level_expectations_content_block = self._render_table_level_expectations(columns)
        if table_level_expectations_content_block is not None:
            overview_content_blocks.append(table_level_expectations_content_block)

        asset_notes_content_block = self._render_expectation_suite_notes(expectations)
        if asset_notes_content_block is not None:
            overview_content_blocks.append(asset_notes_content_block)

        sections = [
            RenderedSectionContent(**{
                "section_name": "Overview",
                "content_blocks": overview_content_blocks,
            })
        ]

        sections += [
            self._column_section_renderer.render(expectations=columns[column]) for column in ordered_columns if column != "_nocolumn"
        ]
        return RenderedDocumentContent(**{
            "renderer_type": "ExpectationSuitePageRenderer",
            "page_title": expectation_suite_name,
            "expectation_suite_name": expectation_suite_name,
            "utm_medium": "expectation-suite-page",
            "sections": sections
        })

    def _render_table_level_expectations(self, columns):
        table_level_expectations = columns.get("_nocolumn")
        if not table_level_expectations:
            return None
        else:
            expectation_bullet_list = self._column_section_renderer.render(
                expectations=table_level_expectations).content_blocks[1]
            expectation_bullet_list.header = RenderedStringTemplateContent(**{
                "content_block_type": "string_template",
                "string_template": {
                    "template": 'Table-Level Expectations',
                    "tag": "h6",
                    "styling": {
                        "classes": ["m-0"]
                    }
                }
            })
            return expectation_bullet_list

    @classmethod
    def _render_expectation_suite_header(cls):
        return RenderedHeaderContent(**{
            "content_block_type": "header",
            "header": RenderedStringTemplateContent(**{
                "content_block_type": "string_template",
                "string_template": {
                    "template": 'Overview',
                    "tag": "h5",
                    "styling": {
                        "classes": ["m-0"]
                    }
                }
            }),
            "styling": {
                "classes": ["col-12"],
                "header": {
                    "classes": ["alert", "alert-secondary"]
                }
            }
        })

    @classmethod
    def _render_expectation_suite_info(cls, expectations):
        expectation_suite_name = expectations.expectation_suite_name
        ge_version = expectations.meta["great_expectations.__version__"]

        return RenderedTableContent(**{
            "content_block_type": "table",
            "header": RenderedStringTemplateContent(**{
                "content_block_type": "string_template",
                "string_template": {
                    "template": 'Info',
                    "tag": "h6",
                    "styling": {
                        "classes": ["m-0"]
                    }
                }
            }),
            "table": [
                ["Expectation Suite Name", expectation_suite_name],
                ["Great Expectations Version", ge_version]
            ],
            "styling": {
                "classes": ["col-12", "table-responsive", "mt-1"],
                "body": {
                    "classes": ["table", "table-sm"]
                }
            },
        })

    # TODO: Update tests
    @classmethod
    def _render_expectation_suite_notes(cls, expectations):

        content = []

        total_expectations = len(expectations.expectations)
        columns = []
        for exp in expectations.expectations:
            if "column" in exp.kwargs:
                columns.append(exp.kwargs["column"])
        total_columns = len(set(columns))

        content = content + [
            # TODO: Leaving these two paragraphs as placeholders for later development.
            # "This Expectation suite was first generated by {BasicDatasetProfiler} on {date}, using version {xxx} of Great Expectations.",
            # "{name}, {name}, and {name} have also contributed additions and revisions.",
            "This Expectation suite currently contains %d total Expectations across %d columns." % (
                total_expectations,
                total_columns,
            ),
        ]

        if "notes" in expectations.meta:
            notes = expectations.meta["notes"]
            note_content = None

            if isinstance(notes, string_types):
                note_content = [notes]

            elif isinstance(notes, list):
                note_content = notes

            elif isinstance(notes, dict):
                if "format" in notes:
                    if notes["format"] == "string":
                        if isinstance(notes["content"], string_types):
                            note_content = [notes["content"]]
                        elif isinstance(notes["content"], list):
                            note_content = notes["content"]
                        else:
                            logger.warning("Unrecognized Expectation suite notes format. Skipping rendering.")

                    elif notes["format"] == "markdown":
                        if isinstance(notes["content"], string_types):
                            note_content = [
                                RenderedMarkdownContent(**{
                                    "content_block_type": "markdown",
                                    "markdown": notes["content"],
                                    "styling": {
                                        "parent": {
                                        }
                                    }
                                })
                            ]
                        elif isinstance(notes["content"], list):
                            note_content = [
                                RenderedMarkdownContent(**{
                                    "content_block_type": "markdown",
                                    "markdown": note,
                                    "styling": {
                                        "parent": {
                                        }
                                    }
                                }) for note in notes["content"]
                            ]
                        else:
                            logger.warning("Unrecognized Expectation suite notes format. Skipping rendering.")
                else:
                    logger.warning("Unrecognized Expectation suite notes format. Skipping rendering.")

            if note_content is not None:
                content = content + note_content

        return TextContent(**{
            "content_block_type": "text",
            "header": RenderedStringTemplateContent(**{
                "content_block_type": "string_template",
                "string_template": {
                    "template": 'Notes',
                    "tag": "h6",
                    "styling": {
                        "classes": ["m-0"]
                    }
                }
            }),
            "text": content,
            "styling": {
                "classes": ["col-12", "table-responsive", "mt-1"],
                "body": {
                    "classes": ["table", "table-sm"]
                }
            },
        })


class ProfilingResultsPageRenderer(Renderer):

    def __init__(self, overview_section_renderer=None, column_section_renderer=None):
        if overview_section_renderer is None:
            overview_section_renderer = {
                "class_name": "ProfilingResultsOverviewSectionRenderer"
            }
        if column_section_renderer is None:
            column_section_renderer = {
                "class_name": "ProfilingResultsColumnSectionRenderer"
            }
        self._overview_section_renderer = instantiate_class_from_config(
            config=overview_section_renderer,
            runtime_environment={},
            config_defaults={
                #"module_name": overview_section_renderer.get("module_name", "great_expectations.render.renderer.other_section_renderer")
                "module_name": overview_section_renderer.get("module_name", "great_expectations.common")
            }
        )
        self._column_section_renderer = instantiate_class_from_config(
            config=column_section_renderer,
            runtime_environment={},
            config_defaults={
                #"module_name": column_section_renderer.get("module_name", "great_expectations.render.renderer.other_section_renderer")
                "module_name": column_section_renderer.get("module_name", "great_expectations.common")
            }
        )

    def render(self, validation_results):
        run_id = validation_results.meta['run_id']
        expectation_suite_name = validation_results.meta['expectation_suite_name']
        batch_kwargs = validation_results.meta.get("batch_kwargs")

        # add datasource key to batch_kwargs if missing
        if 'datasource' not in validation_results.meta.get("batch_kwargs", {}):
            # check if expectation_suite_name follows datasource.generator.data_asset_name.suite_name pattern
            if len(expectation_suite_name.split('.')) == 4:
                batch_kwargs['datasource'] = expectation_suite_name.split('.')[0]

        # Group EVRs by column
        #TODO: When we implement a ValidationResultSuite class, this method will move there.
        columns = self._group_evrs_by_column(validation_results)

        ordered_columns = Renderer._get_column_list_from_evrs(validation_results)
        column_types = self._overview_section_renderer._get_column_types(validation_results)

        return RenderedDocumentContent(**{
            "renderer_type": "ProfilingResultsPageRenderer",
            "page_title": run_id + "-" + expectation_suite_name + "-ProfilingResults",
            "expectation_suite_name": expectation_suite_name,
            "utm_medium": "profiling-results-page",
            "batch_kwargs": batch_kwargs,
            "sections":
                [
                    self._overview_section_renderer.render(
                        validation_results,
                        section_name="Overview"
                    )
                ] +
                [
                    self._column_section_renderer.render(
                        columns[column],
                        section_name=column,
                        column_type=column_types.get(column),
                    ) for column in ordered_columns
                ]
        })


class ContentBlockRenderer(Renderer):

    _rendered_component_type = TextContent
    _default_header = ""

    _default_content_block_styling = {
        "classes": ["col-12"]
    }

    _default_element_styling = {}

    @classmethod
    def validate_input(cls, render_object):
        pass

    @classmethod
    def render(cls, render_object, **kwargs):
        cls.validate_input(render_object)

        if isinstance(render_object, list):
            blocks = []
            has_failed_evr = False if isinstance(render_object[0], ExpectationValidationResult) else None
            for obj_ in render_object:
                expectation_type = cls._get_expectation_type(obj_)

                content_block_fn = cls._get_content_block_fn(expectation_type)

                if isinstance(obj_, ExpectationValidationResult) and not obj_.success:
                    has_failed_evr = True

                if content_block_fn is not None:
                    try:
                        result = content_block_fn(
                            obj_,
                            styling=cls._get_element_styling(),
                            **kwargs
                        )
                    except Exception as e:
                        logger.error("Exception occurred during data docs rendering: ", e, exc_info=True)

                        if isinstance(obj_, ExpectationValidationResult):
                            content_block_fn = cls._get_content_block_fn("_missing_content_block_fn")
                        else:
                            content_block_fn = cls._missing_content_block_fn
                        result = content_block_fn(
                            obj_,
                            cls._get_element_styling(),
                            **kwargs
                        )
                else:
                    result = cls._missing_content_block_fn(
                        obj_,
                        cls._get_element_styling(),
                        **kwargs
                    )

                if result is not None:
                    if isinstance(obj_, ExpectationConfiguration):
                        expectation_meta_notes = cls._render_expectation_meta_notes(obj_)
                        if expectation_meta_notes:
                            # this adds collapse content block to expectation string
                            result[0] = [result[0], expectation_meta_notes]

                        horizontal_rule = RenderedStringTemplateContent(**{
                            "content_block_type": "string_template",
                            "string_template": {
                                "template": "",
                                "tag": "hr",
                                "styling": {
                                    "classes": ["mt-1", "mb-1"],
                                }
                            },
                            "styling": {
                                "parent": {
                                    "styles": {
                                        "list-style-type": "none"
                                    }
                                }
                            }
                        })
                        result.append(horizontal_rule)

                    blocks += result

            if len(blocks) > 0:
                content_block = cls._rendered_component_type(**{
                    cls._content_block_type: blocks,
                    "styling": cls._get_content_block_styling(),
                })
                cls._process_content_block(content_block, has_failed_evr=has_failed_evr)

                return content_block
            else:
                return None
        else:
            expectation_type = cls._get_expectation_type(render_object)

            content_block_fn = getattr(cls, expectation_type, None)
            if content_block_fn is not None:
                try:
                    result = content_block_fn(
                        render_object,
                        styling=cls._get_element_styling(),
                        **kwargs
                    )
                except Exception as e:
                    logger.error("Exception occurred during data docs rendering: ", e, exc_info=True)

                    if isinstance(render_object, ExpectationValidationResult):
                        content_block_fn = cls._get_content_block_fn("_missing_content_block_fn")
                    else:
                        content_block_fn = cls._missing_content_block_fn
                    result = content_block_fn(
                        render_object,
                        cls._get_element_styling(),
                        **kwargs
                    )
            else:
                result = cls._missing_content_block_fn(
                            render_object,
                            cls._get_element_styling(),
                            **kwargs
                        )
            if result is not None:
                if isinstance(render_object, ExpectationConfiguration):
                    expectation_meta_notes = cls._render_expectation_meta_notes(render_object)
                    if expectation_meta_notes:
                        result.append(expectation_meta_notes)
            return result

    @classmethod
    def _render_expectation_meta_notes(cls, expectation):
        if not expectation.meta.get("notes"):
            return None
        else:
            collapse_link = RenderedStringTemplateContent(**{
                "content_block_type": "string_template",
                "string_template": {
                    "template": "$icon",
                    "params": {"icon": ""},
                    "styling": {
                        "params": {
                            "icon": {
                                "classes": ["fas", "fa-comment", "text-info"],
                                "tag": "i"
                            }
                        }
                    }
                }
            })
            notes = expectation.meta["notes"]
            note_content = None

            if isinstance(notes, string_types):
                note_content = [notes]

            elif isinstance(notes, list):
                note_content = notes

            elif isinstance(notes, dict):
                if "format" in notes:
                    if notes["format"] == "string":
                        if isinstance(notes["content"], string_types):
                            note_content = [notes["content"]]
                        elif isinstance(notes["content"], list):
                            note_content = notes["content"]
                        else:
                            logger.warning("Unrecognized Expectation suite notes format. Skipping rendering.")

                    elif notes["format"] == "markdown":
                        if isinstance(notes["content"], string_types):
                            note_content = [
                                RenderedMarkdownContent(**{
                                    "content_block_type": "markdown",
                                    "markdown": notes["content"],
                                    "styling": {
                                        "parent": {
                                            "styles": {
                                                "color": "red"
                                            }
                                        }
                                    }
                                })
                            ]
                        elif isinstance(notes["content"], list):
                            note_content = [
                                RenderedMarkdownContent(**{
                                    "content_block_type": "markdown",
                                    "markdown": note,
                                    "styling": {
                                        "parent": {
                                        }
                                    }
                                }) for note in notes["content"]
                            ]
                        else:
                            logger.warning("Unrecognized Expectation suite notes format. Skipping rendering.")
                else:
                    logger.warning("Unrecognized Expectation suite notes format. Skipping rendering.")

            notes_block = TextContent(**{
                "content_block_type": "text",
                "subheader": "Notes:",
                "text": note_content,
                "styling": {
                    "classes": ["col-12", "mt-2", "mb-2"],
                    "parent": {
                        "styles": {
                            "list-style-type": "none"
                        }
                    }
                },
            })

            return CollapseContent(**{
                "collapse_toggle_link": collapse_link,
                "collapse": [notes_block],
                "inline_link": True,
                "styling": {
                    "body": {
                        "classes": ["card", "card-body", "p-1"]
                    },
                    "parent": {
                        "styles": {
                            "list-style-type": "none"
                        }
                    },
                }
            })

    @classmethod
    def _process_content_block(cls, content_block, has_failed_evr):
        header = cls._get_header()
        if header != "":
            content_block.header = header

    @classmethod
    def _get_content_block_fn(cls, expectation_type):
        return getattr(cls, expectation_type, None)

    @classmethod
    def list_available_expectations(cls):
        expectations = [attr for attr in dir(cls) if attr[:7] == "expect_"]
        return expectations

    @classmethod
    def _missing_content_block_fn(cls, obj, styling, **kwargs):
        return []

    @classmethod
    def _get_content_block_styling(cls):
        return cls._default_content_block_styling

    @classmethod
    def _get_element_styling(cls):
        return cls._default_element_styling

    @classmethod
    def _get_header(cls):
        return cls._default_header


class ExceptionListContentBlockRenderer(ContentBlockRenderer):
    """Render a bullet list of exception messages raised for provided EVRs"""

    _rendered_component_type = RenderedBulletListContent
    _content_block_type = "bullet_list"

    _default_header = 'Failed expectations <span class="mr-3 triangle"></span>'

    _default_content_block_styling = {
        "classes": ["col-12"],
        "styles": {
            "margin-top": "20px"
        },
        "header": {
            "classes": ["collapsed"],
            "attributes": {
                "data-toggle": "collapse",
                "href": "#{{content_block_id}}-body",
                "role": "button",
                "aria-expanded": "true",
                "aria-controls": "collapseExample",
            },
            "styles": {
                "cursor": "pointer",
            }
        },
        "body": {
            "classes": ["list-group", "collapse"],
        }
    }

    _default_element_styling = {
        "classes": ["list-group-item"],  # "d-flex", "justify-content-between", "align-items-center"],
        "params": {
            "column": {
                "classes": ["badge", "badge-primary"]
            },
            "expectation_type": {
                "classes": ["text-monospace"]
            },
            "exception_message": {
                "classes": ["text-monospace"]
            }
        }
    }

    @classmethod
    def _missing_content_block_fn(cls, evr, styling=None, include_column_name=True):
        # Only render EVR objects for which an exception was raised
        if evr.exception_info["raised_exception"] is True:
            template_str = "$expectation_type raised an exception: $exception_message"
            if include_column_name:
                template_str = "$column: " + template_str

            try:
                column = evr.expectation_config.kwargs["column"]
            except KeyError:
                column = None
            return [RenderedStringTemplateContent(**{
                "content_block_type": "string_template",
                "string_template": {
                    "template": template_str,
                    "params": {
                        "column": column,
                        "expectation_type": evr.expectation_config.expectation_type,
                        "exception_message": evr.exception_info["exception_message"]
                    },
                    "styling": styling,
                }
            })]


class ProfilingOverviewTableContentBlockRenderer(ContentBlockRenderer):

    @classmethod
    def render(cls, ge_object, header_row=[]):
        """Each expectation method should return a list of rows"""
        if isinstance(ge_object, list):
            table_entries = []
            for sub_object in ge_object:
                expectation_type = cls._get_expectation_type(sub_object)
                extra_rows_fn = getattr(cls, expectation_type, None)
                if extra_rows_fn is not None:
                    rows = extra_rows_fn(sub_object)
                    table_entries.extend(rows)
        else:
            table_entries = []
            expectation_type = cls._get_expectation_type(ge_object)
            extra_rows_fn = getattr(cls, expectation_type, None)
            if extra_rows_fn is not None:
                rows = extra_rows_fn(ge_object)
                table_entries.extend(rows)

        return RenderedTableContent(**{
            "content_block_type": "table",
            "header_row": header_row,
            "table": table_entries
        })

    @classmethod
    def expect_column_values_to_not_match_regex(cls, ge_object):
        regex = ge_object.expectation_config.kwargs["regex"]
        unexpected_count = ge_object.result["unexpected_count"]
        if regex == '^\\s+|\\s+$':
            return [["Leading or trailing whitespace (n)", unexpected_count]]
        else:
            return [["Regex: %s" % regex, unexpected_count]]

    @classmethod
    def expect_column_unique_value_count_to_be_between(cls, ge_object):
        observed_value = ge_object.result["observed_value"]
        return [
            [
                RenderedStringTemplateContent(**{
                    "content_block_type": "string_template",
                    "string_template": {
                        "template": "Distinct (n)",
                        "tooltip": {
                            "content": "expect_column_unique_value_count_to_be_between"
                        }
                    }
                }),
                observed_value,
            ]
        ]

    @classmethod
    def expect_column_proportion_of_unique_values_to_be_between(cls, ge_object):
        observed_value = ge_object.result["observed_value"]
        template_string_object = RenderedStringTemplateContent(**{
            "content_block_type": "string_template",
            "string_template": {
                "template": "Distinct (%)",
                "tooltip": {
                    "content": "expect_column_proportion_of_unique_values_to_be_between"
                }
            }
        })
        if not observed_value:
            return [[template_string_object, "--"]]
        else:
            return [[template_string_object, "%.1f%%" % (100*observed_value)]]

    @classmethod
    def expect_column_max_to_be_between(cls, ge_object):
        observed_value = ge_object.result["observed_value"]
        return [["Max", observed_value]]

    @classmethod
    def expect_column_mean_to_be_between(cls, ge_object):
        observed_value = ge_object.result["observed_value"]
        return [["Mean", observed_value]]

    @classmethod
    def expect_column_values_to_not_be_null(cls, ge_object):
        return [
            [
                RenderedStringTemplateContent(**{
                    "content_block_type": "string_template",
                    "string_template": {
                        "template": "Missing (n)",
                        "tooltip": {
                            "content": "expect_column_values_to_not_be_null"
                        }
                    }
                }),
                ge_object.result["unexpected_count"] if "unexpected_count" in ge_object.result and ge_object.result["unexpected_count"] is not None else "--",
            ],
            [
                RenderedStringTemplateContent(**{
                    "content_block_type": "string_template",
                    "string_template": {
                        "template": "Missing (%)",
                        "tooltip": {
                            "content": "expect_column_values_to_not_be_null"
                        }
                    }
                }),
                "%.1f%%" % ge_object.result["unexpected_percent"] if "unexpected_percent" in ge_object.result and ge_object.result["unexpected_percent"] is not None else "--",
            ]
        ]

    @classmethod
    def expect_column_values_to_be_null(cls, ge_object):
        return [
            ["Populated (n)", ge_object.result["unexpected_count"]],
            ["Populated (%)", "%.1f%%" %
             ge_object.result["unexpected_percent"]]
        ]


class ExpectationStringRenderer(ContentBlockRenderer):

    @classmethod
    def _missing_content_block_fn(cls, expectation, styling=None, include_column_name=True):
        return [RenderedStringTemplateContent(**{
            "content_block_type": "string_template",
            "styling": {
              "parent": {
                  "classes": ["alert", "alert-warning"]
              }
            },
            "string_template": {
                "template": "$expectation_type(**$kwargs)",
                "params": {
                    "expectation_type": expectation.expectation_type,
                    "kwargs": expectation.kwargs
                },
                "styling": {
                    "params": {
                        "expectation_type": {
                            "classes": ["badge", "badge-warning"],
                        }
                    }
                },
            }
        })]

    @classmethod
    def expect_column_to_exist(cls, expectation, styling=None, include_column_name=True):
        params = substitute_none_for_missing(
            expectation.kwargs,
            ["column", "column_index"],
        )

        if params["column_index"] is None:
            if include_column_name:
                template_str = "$column is a required field."
            else:
                template_str = "is a required field."
        else:
            params["column_indexth"] = ordinal(params["column_index"])
            if include_column_name:
                template_str = "$column must be the $column_indexth field"
            else:
                template_str = "must be the $column_indexth field"

        return [RenderedStringTemplateContent(**{
            "content_block_type": "string_template",
            "string_template": {
                "template": template_str,
                "params": params,
                "styling": styling,
            }
        })]

    @classmethod
    def expect_column_unique_value_count_to_be_between(cls, expectation, styling=None, include_column_name=True):
        params = substitute_none_for_missing(
            expectation.kwargs,
            ["column", "min_value", "max_value", "mostly"],
        )

        if (params["min_value"] is None) and (params["max_value"] is None):
            template_str = "may have any number of unique values."
        else:
            if params["mostly"] is not None:
                params["mostly_pct"] = num_to_str(params["mostly"] * 100, precision=15, no_scientific=True)
                # params["mostly_pct"] = "{:.14f}".format(params["mostly"]*100).rstrip("0").rstrip(".")
                if params["min_value"] is None:
                    template_str = "must have fewer than $max_value unique values, at least $mostly_pct % of the time."
                elif params["max_value"] is None:
                    template_str = "must have more than $min_value unique values, at least $mostly_pct % of the time."
                else:
                    template_str = "must have between $min_value and $max_value unique values, at least $mostly_pct % of the time."
            else:
                if params["min_value"] is None:
                    template_str = "must have fewer than $max_value unique values."
                elif params["max_value"] is None:
                    template_str = "must have more than $min_value unique values."
                else:
                    template_str = "must have between $min_value and $max_value unique values."

        if include_column_name:
            template_str = "$column " + template_str

        return [RenderedStringTemplateContent(**{
            "content_block_type": "string_template",
            "string_template": {
                "template": template_str,
                "params": params,
                "styling": styling,
            }
        })]

    # NOTE: This method is a pretty good example of good usage of `params`.
    @classmethod
    def expect_column_values_to_be_between(cls, expectation, styling=None, include_column_name=True):
        params = substitute_none_for_missing(
            expectation.kwargs,
            ["column", "min_value", "max_value", "mostly"]
        )

        if (params["min_value"] is None) and (params["max_value"] is None):
            template_str = "may have any numerical value."
        else:
            if params["mostly"] is not None:
                params["mostly_pct"] = num_to_str(params["mostly"] * 100, precision=15, no_scientific=True)
                # params["mostly_pct"] = "{:.14f}".format(params["mostly"]*100).rstrip("0").rstrip(".")
                if params["min_value"] is not None and params["max_value"] is not None:
                    template_str = "values must be between $min_value and $max_value, at least $mostly_pct % of the time."

                elif params["min_value"] is None:
                    template_str = "values must be less than $max_value, at least $mostly_pct % of the time."

                elif params["max_value"] is None:
                    template_str = "values must be less than $max_value, at least $mostly_pct % of the time."
            else:
                if params["min_value"] is not None and params["max_value"] is not None:
                    template_str = "values must always be between $min_value and $max_value."

                elif params["min_value"] is None:
                    template_str = "values must always be less than $max_value."

                elif params["max_value"] is None:
                    template_str = "values must always be more than $min_value."

        if include_column_name:
            template_str = "$column " + template_str

        return [RenderedStringTemplateContent(**{
            "content_block_type": "string_template",
            "string_template": {
                "template": template_str,
                "params": params,
                "styling": styling,
            }
        })]

    @classmethod
    def expect_column_pair_values_A_to_be_greater_than_B(cls, expectation, styling=None, include_column_name=True):
        params = substitute_none_for_missing(
            expectation.kwargs,
            ["column_A", "column_B", "parse_strings_as_datetimes",
             "ignore_row_if", "mostly", "or_equal"]
        )

        if (params["column_A"] is None) or (params["column_B"] is None):
            template_str = "$column has a bogus `expect_column_pair_values_A_to_be_greater_than_B` expectation."

        if params["mostly"] is None:
            if params["or_equal"] in [None, False]:
                template_str = "Values in $column_A must always be greater than those in $column_B."
            else:
                template_str = "Values in $column_A must always be greater than or equal to those in $column_B."
        else:
            params["mostly_pct"] = num_to_str(params["mostly"] * 100, precision=15, no_scientific=True)
            # params["mostly_pct"] = "{:.14f}".format(params["mostly"]*100).rstrip("0").rstrip(".")
            if params["or_equal"] in [None, False]:
                template_str = "Values in $column_A must be greater than those in $column_B, at least $mostly_pct % of the time."
            else:
                template_str = "Values in $column_A must be greater than or equal to those in $column_B, at least $mostly_pct % of the time."

        if params.get("parse_strings_as_datetimes"):
            template_str += " Values should be parsed as datetimes."

        return [RenderedStringTemplateContent(**{
            "content_block_type": "string_template",
            "string_template": {
                "template": template_str,
                "params": params,
                "styling": styling,
            }
        })]

    @classmethod
    def expect_column_pair_values_to_be_equal(cls, expectation, styling=None, include_column_name=True):
        params = substitute_none_for_missing(
            expectation.kwargs,
            ["column_A", "column_B",
             "ignore_row_if", "mostly", ]
        )

        # NOTE: This renderer doesn't do anything with "ignore_row_if"

        if (params["column_A"] is None) or (params["column_B"] is None):
            template_str = " unrecognized kwargs for expect_column_pair_values_to_be_equal: missing column."

        if params["mostly"] is None:
            template_str = "Values in $column_A and $column_B must always be equal."
        else:
            params["mostly_pct"] = num_to_str(params["mostly"] * 100, precision=15, no_scientific=True)
            # params["mostly_pct"] = "{:.14f}".format(params["mostly"]*100).rstrip("0").rstrip(".")
            template_str = "Values in $column_A and $column_B must be equal, at least $mostly_pct % of the time."

        return [RenderedStringTemplateContent(**{
            "content_block_type": "string_template",
            "string_template": {
                "template": template_str,
                "params": params,
                "styling": styling,
            }
        })]

    @classmethod
    def expect_table_columns_to_match_ordered_list(cls, expectation, styling=None, include_column_name=True):
        params = substitute_none_for_missing(
            expectation.kwargs,
            ["column_list"]
        )

        if params["column_list"] is None:
            template_str = "Must have a list of columns in a specific order, but that order is not specified."

        else:
            template_str = "Must have these columns in this order: "
            for idx in range(len(params["column_list"]) - 1):
                template_str += "$column_list_" + str(idx) + ", "
                params["column_list_" + str(idx)] = params["column_list"][idx]

            last_idx = len(params["column_list"]) - 1
            template_str += "$column_list_" + str(last_idx)
            params["column_list_" + str(last_idx)] = params["column_list"][last_idx]

        return [RenderedStringTemplateContent(**{
            "content_block_type": "string_template",
            "string_template": {
                "template": template_str,
                "params": params,
                "styling": styling,
            }
        })]

    @classmethod
    def expect_multicolumn_values_to_be_unique(cls, expectation, styling=None, include_column_name=True):
        params = substitute_none_for_missing(
            expectation.kwargs,
            ["column_list", "ignore_row_if"]
        )

        template_str = "Values must always be unique across columns: "
        for idx in range(len(params["column_list"]) - 1):
            template_str += "$column_list_" + str(idx) + ", "
            params["column_list_" + str(idx)] = params["column_list"][idx]

        last_idx = len(params["column_list"]) - 1
        template_str += "$column_list_" + str(last_idx)
        params["column_list_" + str(last_idx)] = params["column_list"][last_idx]

        return [RenderedStringTemplateContent(**{
            "content_block_type": "string_template",
            "string_template": {
                "template": template_str,
                "params": params,
                "styling": styling,
            }
        })]

    @classmethod
    def expect_table_column_count_to_equal(cls, expectation, styling=None, include_column_name=True):
        params = substitute_none_for_missing(
            expectation.kwargs,
            ["value"]
        )
        template_str = "Must have exactly $value columns."
        return [RenderedStringTemplateContent(**{
            "content_block_type": "string_template",
            "string_template": {
                "template": template_str,
                "params": params,
                "styling": styling,
            }
        })]

    @classmethod
    def expect_table_column_count_to_be_between(cls, expectation, styling=None, include_column_name=True):
        params = substitute_none_for_missing(
            expectation.kwargs,
            ["min_value", "max_value"]
        )
        if params["min_value"] is None and params["max_value"] is None:
            template_str = "May have any number of columns."
        else:
            if params["min_value"] is not None and params["max_value"] is not None:
                template_str = "Must have between $min_value and $max_value columns."
            elif params["min_value"] is None:
                template_str = "Must have less than than $max_value columns."
            elif params["max_value"] is None:
                template_str = "Must have more than $min_value columns."
        return [RenderedStringTemplateContent(**{
            "content_block_type": "string_template",
            "string_template": {
                "template": template_str,
                "params": params,
                "styling": styling,
            }
        })]

    @classmethod
    def expect_table_row_count_to_be_between(cls, expectation, styling=None, include_column_name=True):
        params = substitute_none_for_missing(
            expectation.kwargs,
            ["min_value", "max_value"]
        )

        if params["min_value"] is None and params["max_value"] is None:
            template_str = "May have any number of rows."
        else:
            if params["min_value"] is not None and params["max_value"] is not None:
                template_str = "Must have between $min_value and $max_value rows."
            elif params["min_value"] is None:
                template_str = "Must have less than than $max_value rows."
            elif params["max_value"] is None:
                template_str = "Must have more than $min_value rows."

        return [RenderedStringTemplateContent(**{
            "content_block_type": "string_template",
            "string_template": {
                "template": template_str,
                "params": params,
                "styling": styling,
            }
        })]

    @classmethod
    def expect_table_row_count_to_equal(cls, expectation, styling=None, include_column_name=True):
        params = substitute_none_for_missing(
            expectation.kwargs,
            ["value"]
        )
        template_str = "Must have exactly $value rows."

        return [RenderedStringTemplateContent(**{
            "content_block_type": "string_template",
            "string_template": {
                "template": template_str,
                "params": params,
                "styling": styling,
            }
        })]

    @classmethod
    def expect_column_distinct_values_to_be_in_set(cls, expectation, styling=None, include_column_name=True):
        params = substitute_none_for_missing(
            expectation.kwargs,
            ["column", "value_set"],
        )

        if params["value_set"] is None or len(params["value_set"]) == 0:

            if include_column_name:
                template_str = "$column distinct values must belong to this set: [ ]"
            else:
                template_str = "distinct values must belong to a set, but that set is not specified."

        else:

            for i, v in enumerate(params["value_set"]):
                params["v__" + str(i)] = v
            values_string = " ".join(
                ["$v__" + str(i) for i, v in enumerate(params["value_set"])]
            )

            if include_column_name:
                template_str = "$column distinct values must belong to this set: " + values_string + "."
            else:
                template_str = "distinct values must belong to this set: " + values_string + "."

        return [RenderedStringTemplateContent(**{
            "content_block_type": "string_template",
            "string_template": {
                "template": template_str,
                "params": params,
                "styling": styling,
            }
        })]

    @classmethod
    def expect_column_values_to_not_be_null(cls, expectation, styling=None, include_column_name=True):
        params = substitute_none_for_missing(
            expectation.kwargs,
            ["column", "mostly"],
        )

        if params["mostly"] is not None:
            params["mostly_pct"] = num_to_str(params["mostly"] * 100, precision=15, no_scientific=True)
            # params["mostly_pct"] = "{:.14f}".format(params["mostly"]*100).rstrip("0").rstrip(".")
            if include_column_name:
                template_str = "$column values must not be null, at least $mostly_pct % of the time."
            else:
                template_str = "values must not be null, at least $mostly_pct % of the time."
        else:
            if include_column_name:
                template_str = "$column values must never be null."
            else:
                template_str = "values must never be null."

        return [RenderedStringTemplateContent(**{
            "content_block_type": "string_template",
            "string_template": {
                "template": template_str,
                "params": params,
                "styling": styling,
            }
        })]

    @classmethod
    def expect_column_values_to_be_null(cls, expectation, styling=None, include_column_name=True):
        params = substitute_none_for_missing(
            expectation.kwargs,
            ["column", "mostly"]
        )

        if params["mostly"] is not None:
            params["mostly_pct"] = num_to_str(params["mostly"] * 100, precision=15, no_scientific=True)
            # params["mostly_pct"] = "{:.14f}".format(params["mostly"]*100).rstrip("0").rstrip(".")
            template_str = "values must be null, at least $mostly_pct % of the time."
        else:
            template_str = "values must be null."

        if include_column_name:
            template_str = "$column " + template_str

        return [RenderedStringTemplateContent(**{
            "content_block_type": "string_template",
            "string_template": {
                "template": template_str,
                "params": params,
                "styling": styling,
            }
        })]

    @classmethod
    def expect_column_values_to_be_of_type(cls, expectation, styling=None, include_column_name=True):
        params = substitute_none_for_missing(
            expectation.kwargs,
            ["column", "type_", "mostly"]
        )

        if params["mostly"] is not None:
            params["mostly_pct"] = num_to_str(params["mostly"] * 100, precision=15, no_scientific=True)
            # params["mostly_pct"] = "{:.14f}".format(params["mostly"]*100).rstrip("0").rstrip(".")
            template_str = "values must be of type $type_, at least $mostly_pct % of the time."
        else:
            template_str = "values must be of type $type_."

        if include_column_name:
            template_str = "$column " + template_str

        return [RenderedStringTemplateContent(**{
            "content_block_type": "string_template",
            "string_template": {
                "template": template_str,
                "params": params,
                "styling": styling,
            }
        })]

    @classmethod
    def expect_column_values_to_be_in_type_list(cls, expectation, styling=None, include_column_name=True):
        params = substitute_none_for_missing(
            expectation.kwargs,
            ["column", "type_list", "mostly"],
        )

        if params["type_list"] is not None:
            for i, v in enumerate(params["type_list"]):
                params["v__"+str(i)] = v
            values_string = " ".join(
                ["$v__"+str(i) for i, v in enumerate(params["type_list"])]
            )

            if params["mostly"] is not None:
                params["mostly_pct"] = num_to_str(params["mostly"] * 100, precision=15, no_scientific=True)
                # params["mostly_pct"] = "{:.14f}".format(params["mostly"]*100).rstrip("0").rstrip(".")
                if include_column_name:
                    template_str = "$column value types must belong to this set: " + values_string + ", at least $mostly_pct % of the time."
                else:
                    template_str = "value types must belong to this set: " + values_string + ", at least $mostly_pct % of the time."
            else:
                if include_column_name:
                    template_str = "$column value types must belong to this set: "+values_string+"."
                else:
                    template_str = "value types must belong to this set: "+values_string+"."
        else:
            if include_column_name:
                template_str = "$column value types may be any value, but observed value will be reported"
            else:
                template_str = "value types may be any value, but observed value will be reported"

        return [RenderedStringTemplateContent(**{
            "content_block_type": "string_template",
            "string_template": {
                "template": template_str,
                "params": params,
                "styling": styling,
            }
        })]

    @classmethod
    def expect_column_values_to_be_in_set(cls, expectation, styling=None, include_column_name=True):
        params = substitute_none_for_missing(
            expectation.kwargs,
            ["column", "value_set", "mostly", "parse_strings_as_datetimes"]
        )

        if params["value_set"] is None or len(params["value_set"]) == 0:
            values_string = "[ ]"
        else:
            for i, v in enumerate(params["value_set"]):
                params["v__" + str(i)] = v

            values_string = " ".join(
                ["$v__" + str(i) for i, v in enumerate(params["value_set"])]
            )

        template_str = "values must belong to this set: " + values_string

        if params["mostly"] is not None:
            params["mostly_pct"] = num_to_str(params["mostly"] * 100, precision=15, no_scientific=True)
            # params["mostly_pct"] = "{:.14f}".format(params["mostly"]*100).rstrip("0").rstrip(".")
            template_str += ", at least $mostly_pct % of the time."
        else:
            template_str += "."

        if params.get("parse_strings_as_datetimes"):
            template_str += " Values should be parsed as datetimes."

        if include_column_name:
            template_str = "$column " + template_str

        return [RenderedStringTemplateContent(**{
            "content_block_type": "string_template",
            "string_template": {
                "template": template_str,
                "params": params,
                "styling": styling,
            }
        })]

    @classmethod
    def expect_column_values_to_not_be_in_set(cls, expectation, styling=None, include_column_name=True):
        params = substitute_none_for_missing(
            expectation.kwargs,
            ["column", "value_set", "mostly", "parse_strings_as_datetimes"]
        )

        if params["value_set"] is None or len(params["value_set"]) == 0:
            values_string = "[ ]"
        else:
            for i, v in enumerate(params["value_set"]):
                params["v__" + str(i)] = v

            values_string = " ".join(
                ["$v__" + str(i) for i, v in enumerate(params["value_set"])]
            )

        template_str = "values must not belong to this set: " + values_string

        if params["mostly"] is not None:
            params["mostly_pct"] = num_to_str(params["mostly"] * 100, precision=15, no_scientific=True)
            # params["mostly_pct"] = "{:.14f}".format(params["mostly"]*100).rstrip("0").rstrip(".")
            template_str += ", at least $mostly_pct % of the time."
        else:
            template_str += "."

        if params.get("parse_strings_as_datetimes"):
            template_str += " Values should be parsed as datetimes."

        if include_column_name:
            template_str = "$column"

        return [RenderedStringTemplateContent(**{
            "content_block_type": "string_template",
            "string_template": {
                "template": template_str,
                "params": params,
                "styling": styling,
            }
        })]

    @classmethod
    def expect_column_proportion_of_unique_values_to_be_between(cls, expectation, styling=None,
                                                                include_column_name=True):
        params = substitute_none_for_missing(
            expectation.kwargs,
            ["column", "min_value", "max_value"],
        )

        if params["min_value"] is None and params["max_value"] is None:
            template_str = "may have any fraction of unique values."
        else:
            if params["min_value"] is None:
                template_str = "fraction of unique values must be less than $max_value."
            elif params["max_value"] is None:
                template_str = "fraction of unique values must be at least $min_value."
            else:
                template_str = "fraction of unique values must be between $min_value and $max_value."

        if include_column_name:
            template_str = "$column " + template_str

        return [RenderedStringTemplateContent(**{
            "content_block_type": "string_template",
            "string_template": {
                "template": template_str,
                "params": params,
                "styling": styling,
            }
        })]

    # TODO: test parse_strings_as_datetimes
    @classmethod
    def expect_column_values_to_be_increasing(cls, expectation, styling=None, include_column_name=True):
        params = substitute_none_for_missing(
            expectation.kwargs,
            ["column", "strictly", "mostly", "parse_strings_as_datetimes"]
        )

        if params.get("strictly"):
            template_str = "values must be strictly greater than previous values"
        else:
            template_str = "values must be greater than or equal to previous values"

        if params["mostly"] is not None:
            params["mostly_pct"] = num_to_str(params["mostly"] * 100, precision=15, no_scientific=True)
            # params["mostly_pct"] = "{:.14f}".format(params["mostly"]*100).rstrip("0").rstrip(".")
            template_str += ", at least $mostly_pct % of the time."
        else:
            template_str += "."

        if params.get("parse_strings_as_datetimes"):
            template_str += " Values should be parsed as datetimes."

        if include_column_name:
            template_str = "$column " + template_str

        return [RenderedStringTemplateContent(**{
            "content_block_type": "string_template",
            "string_template": {
                "template": template_str,
                "params": params,
                "styling": styling,
            }
        })]

    # TODO: test parse_strings_as_datetimes
    @classmethod
    def expect_column_values_to_be_decreasing(cls, expectation, styling=None, include_column_name=True):
        params = substitute_none_for_missing(
            expectation.kwargs,
            ["column", "strictly", "mostly", "parse_strings_as_datetimes"]
        )

        if params.get("strictly"):
            template_str = "values must be strictly less than previous values"
        else:
            template_str = "values must be less than or equal to previous values"

        if params["mostly"] is not None:
            params["mostly_pct"] = num_to_str(params["mostly"] * 100, precision=15, no_scientific=True)
            # params["mostly_pct"] = "{:.14f}".format(params["mostly"]*100).rstrip("0").rstrip(".")
            template_str += ", at least $mostly_pct % of the time."
        else:
            template_str += "."

        if params.get("parse_strings_as_datetimes"):
            template_str += " Values should be parsed as datetimes."

        if include_column_name:
            template_str = "$column " + template_str

        return [RenderedStringTemplateContent(**{
            "content_block_type": "string_template",
            "string_template": {
                "template": template_str,
                "params": params,
                "styling": styling,
            }
        })]

    @classmethod
    def expect_column_value_lengths_to_be_between(cls, expectation, styling=None, include_column_name=True):
        params = substitute_none_for_missing(
            expectation.kwargs,
            ["column", "min_value", "max_value", "mostly"],
        )

        if (params["min_value"] is None) and (params["max_value"] is None):
            template_str = "values may have any length."
        else:
            if params["mostly"] is not None:
                params["mostly_pct"] = num_to_str(params["mostly"] * 100, precision=15, no_scientific=True)
                # params["mostly_pct"] = "{:.14f}".format(params["mostly"]*100).rstrip("0").rstrip(".")
                if params["min_value"] is not None and params["max_value"] is not None:
                    template_str = "values must be between $min_value and $max_value characters long, at least $mostly_pct % of the time."

                elif params["min_value"] is None:
                    template_str = "values must be less than $max_value characters long, at least $mostly_pct % of the time."

                elif params["max_value"] is None:
                    template_str = "values must be more than $min_value characters long, at least $mostly_pct % of the time."
            else:
                if params["min_value"] is not None and params["max_value"] is not None:
                    template_str = "values must always be between $min_value and $max_value characters long."

                elif params["min_value"] is None:
                    template_str = "values must always be less than $max_value characters long."

                elif params["max_value"] is None:
                    template_str = "values must always be more than $min_value characters long."

        if include_column_name:
            template_str = "$column " + template_str

        return [RenderedStringTemplateContent(**{
            "content_block_type": "string_template",
            "string_template": {
                "template": template_str,
                "params": params,
                "styling": styling,
            }
        })]

    @classmethod
    def expect_column_value_lengths_to_equal(cls, expectation, styling=None, include_column_name=True):
        params = substitute_none_for_missing(
            expectation.kwargs,
            ["column", "value", "mostly"]
        )

        if params.get("value") is None:
            template_str = "values may have any length."
        else:
            template_str = "values must be $value characters long"
            if params["mostly"] is not None:
                params["mostly_pct"] = num_to_str(params["mostly"] * 100, precision=15, no_scientific=True)
                # params["mostly_pct"] = "{:.14f}".format(params["mostly"]*100).rstrip("0").rstrip(".")
                template_str += ", at least $mostly_pct % of the time."
            else:
                template_str += "."

        if include_column_name:
            template_str = "$column " + template_str

        return [RenderedStringTemplateContent(**{
            "content_block_type": "string_template",
            "string_template": {
                "template": template_str,
                "params": params,
                "styling": styling,
            }
        })]

    @classmethod
    def expect_column_values_to_match_regex(cls, expectation, styling=None, include_column_name=True):
        params = substitute_none_for_missing(
            expectation.kwargs,
            ["column", "regex", "mostly"]
        )

        if not params.get("regex"):
            template_str = "values must match a regular expression but none was specified."
        else:
            template_str = "values must match this regular expression: $regex"
            if params["mostly"] is not None:
                params["mostly_pct"] = num_to_str(params["mostly"] * 100, precision=15, no_scientific=True)
                # params["mostly_pct"] = "{:.14f}".format(params["mostly"]*100).rstrip("0").rstrip(".")
                template_str += ", at least $mostly_pct % of the time."
            else:
                template_str += "."

        if include_column_name:
            template_str = "$column " + template_str

        return [RenderedStringTemplateContent(**{
            "content_block_type": "string_template",
            "string_template": {
                "template": template_str,
                "params": params,
                "styling": styling,
            }
        })]

    @classmethod
    def expect_column_values_to_not_match_regex(cls, expectation, styling=None, include_column_name=True):
        params = substitute_none_for_missing(
            expectation.kwargs,
            ["column", "regex", "mostly"],
        )

        if not params.get("regex"):
            template_str = "values must not match a regular expression but none was specified."
        else:
            if params["mostly"] is not None:
                params["mostly_pct"] = num_to_str(params["mostly"] * 100, precision=15, no_scientific=True)
                # params["mostly_pct"] = "{:.14f}".format(params["mostly"]*100).rstrip("0").rstrip(".")
                if include_column_name:
                    template_str = "$column values must not match this regular expression: $regex, at least $mostly_pct % of the time."
                else:
                    template_str = "values must not match this regular expression: $regex, at least $mostly_pct % of the time."
            else:
                if include_column_name:
                    template_str = "$column values must not match this regular expression: $regex."
                else:
                    template_str = "values must not match this regular expression: $regex."

        return [RenderedStringTemplateContent(**{
            "content_block_type": "string_template",
            "string_template": {
                "template": template_str,
                "params": params,
                "styling": styling,
            }
        })]

    @classmethod
    def expect_column_values_to_match_regex_list(cls, expectation, styling=None, include_column_name=True):
        params = substitute_none_for_missing(
            expectation.kwargs,
            ["column", "regex_list", "mostly", "match_on"],
        )

        if not params.get("regex_list") or len(params.get("regex_list")) == 0:
            values_string = "[ ]"
        else:
            for i, v in enumerate(params["regex_list"]):
                params["v__" + str(i)] = v
            values_string = " ".join(
                ["$v__" + str(i) for i, v in enumerate(params["regex_list"])]
            )

        if params.get("match_on") == "all":
            template_str = "values must match all of the following regular expressions: " + values_string
        else:
            template_str = "values must match any of the following regular expressions: " + values_string

        if params["mostly"] is not None:
            params["mostly_pct"] = num_to_str(params["mostly"] * 100, precision=15, no_scientific=True)
            # params["mostly_pct"] = "{:.14f}".format(params["mostly"]*100).rstrip("0").rstrip(".")
            template_str += ", at least $mostly_pct % of the time."
        else:
            template_str += "."

        if include_column_name:
            template_str = "$column " + template_str

        return [RenderedStringTemplateContent(**{
            "content_block_type": "string_template",
            "string_template": {
                "template": template_str,
                "params": params,
                "styling": styling,
            }
        })]

    @classmethod
    def expect_column_values_to_not_match_regex_list(cls, expectation, styling=None, include_column_name=True):
        params = substitute_none_for_missing(
            expectation.kwargs,
            ["column", "regex_list", "mostly"],
        )

        if not params.get("regex_list") or len(params.get("regex_list")) == 0:
            values_string = "[ ]"
        else:
            for i, v in enumerate(params["regex_list"]):
                params["v__" + str(i)] = v
            values_string = " ".join(
                ["$v__" + str(i) for i, v in enumerate(params["regex_list"])]
            )

        template_str = "values must not match any of the following regular expressions: " + values_string

        if params["mostly"] is not None:
            params["mostly_pct"] = num_to_str(params["mostly"] * 100, precision=15, no_scientific=True)
            # params["mostly_pct"] = "{:.14f}".format(params["mostly"]*100).rstrip("0").rstrip(".")
            template_str += ", at least $mostly_pct % of the time."
        else:
            template_str += "."

        if include_column_name:
            template_str = "$column " + template_str

        return [RenderedStringTemplateContent(**{
            "content_block_type": "string_template",
            "string_template": {
                "template": template_str,
                "params": params,
                "styling": styling,
            }
        })]

    @classmethod
    def expect_column_values_to_match_strftime_format(cls, expectation, styling=None, include_column_name=True):
        params = substitute_none_for_missing(
            expectation.kwargs,
            ["column", "strftime_format", "mostly"],
        )

        if not params.get("strftime_format"):
            template_str = "values must match a strftime format but none was specified."
        else:
            template_str = "values must match the following strftime format: $strftime_format"
            if params["mostly"] is not None:
                params["mostly_pct"] = num_to_str(params["mostly"] * 100, precision=15, no_scientific=True)
                # params["mostly_pct"] = "{:.14f}".format(params["mostly"]*100).rstrip("0").rstrip(".")
                template_str += ", at least $mostly_pct % of the time."
            else:
                template_str += "."

        if include_column_name:
            template_str = "$column " + template_str

        return [RenderedStringTemplateContent(**{
            "content_block_type": "string_template",
            "string_template": {
                "template": template_str,
                "params": params,
                "styling": styling,
            }
        })]

    @classmethod
    def expect_column_values_to_be_dateutil_parseable(cls, expectation, styling=None, include_column_name=True):
        params = substitute_none_for_missing(
            expectation.kwargs,
            ["column", "mostly"],
        )

        template_str = "values must be parseable by dateutil"

        if params["mostly"] is not None:
            params["mostly_pct"] = num_to_str(params["mostly"] * 100, precision=15, no_scientific=True)
            # params["mostly_pct"] = "{:.14f}".format(params["mostly"]*100).rstrip("0").rstrip(".")
            template_str += ", at least $mostly_pct % of the time."
        else:
            template_str += "."

        if include_column_name:
            template_str = "$column " + template_str

        return [RenderedStringTemplateContent(**{
            "content_block_type": "string_template",
            "string_template": {
                "template": template_str,
                "params": params,
                "styling": styling,
            }
        })]

    @classmethod
    def expect_column_values_to_be_json_parseable(cls, expectation, styling=None, include_column_name=True):
        params = substitute_none_for_missing(
            expectation.kwargs,
            ["column", "mostly"],
        )

        template_str = "values must be parseable as JSON"

        if params["mostly"] is not None:
            params["mostly_pct"] = num_to_str(params["mostly"] * 100, precision=15, no_scientific=True)
            # params["mostly_pct"] = "{:.14f}".format(params["mostly"]*100).rstrip("0").rstrip(".")
            template_str += ", at least $mostly_pct % of the time."
        else:
            template_str += "."

        if include_column_name:
            template_str = "$column " + template_str

        return [RenderedStringTemplateContent(**{
            "content_block_type": "string_template",
            "string_template": {
                "template": template_str,
                "params": params,
                "styling": styling,
            }
        })]

    @classmethod
    def expect_column_values_to_match_json_schema(cls, expectation, styling=None, include_column_name=True):
        params = substitute_none_for_missing(
            expectation.kwargs,
            ["column", "mostly", "json_schema"],
        )

        if not params.get("json_schema"):
            template_str = "values must match a JSON Schema but none was specified."
        else:
            params["formatted_json"] = "<pre>" + json.dumps(params.get("json_schema"), indent=4) + "</pre>"
            if params["mostly"] is not None:
                params["mostly_pct"] = num_to_str(params["mostly"] * 100, precision=15, no_scientific=True)
                # params["mostly_pct"] = "{:.14f}".format(params["mostly"]*100).rstrip("0").rstrip(".")
                template_str = "values must match the following JSON Schema, at least $mostly_pct % of the time: $formatted_json"
            else:
                template_str = "values must match the following JSON Schema: $formatted_json"

        if include_column_name:
            template_str = "$column " + template_str

        return [RenderedStringTemplateContent(**{
            "content_block_type": "string_template",
            "string_template": {
                "template": template_str,
                "params": params,
                "styling": {
                    "params":
                        {
                            "formatted_json": {
                                "classes": []
                            }
                        }
                },
            }
        })]

    @classmethod
    def expect_column_distinct_values_to_contain_set(cls, expectation, styling=None, include_column_name=True):
        params = substitute_none_for_missing(
            expectation.kwargs,
            ["column", "value_set", "parse_strings_as_datetimes"]
        )

        if params["value_set"] is None or len(params["value_set"]) == 0:
            values_string = "[ ]"
        else:
            for i, v in enumerate(params["value_set"]):
                params["v__" + str(i)] = v

            values_string = " ".join(
                ["$v__" + str(i) for i, v in enumerate(params["value_set"])]
            )

        template_str = "distinct values must contain this set: " + values_string + "."

        if params.get("parse_strings_as_datetimes"):
            template_str += " Values should be parsed as datetimes."

        if include_column_name:
            template_str = "$column " + template_str

        return [RenderedStringTemplateContent(**{
            "content_block_type": "string_template",
            "string_template": {
                "template": template_str,
                "params": params,
                "styling": styling,
            }
        })]

    @classmethod
    def expect_column_distinct_values_to_equal_set(cls, expectation, styling=None, include_column_name=True):
        params = substitute_none_for_missing(
            expectation.kwargs,
            ["column", "value_set", "parse_strings_as_datetimes"]
        )

        if params["value_set"] is None or len(params["value_set"]) == 0:
            values_string = "[ ]"
        else:
            for i, v in enumerate(params["value_set"]):
                params["v__" + str(i)] = v

            values_string = " ".join(
                ["$v__" + str(i) for i, v in enumerate(params["value_set"])]
            )

        template_str = "distinct values must match this set: " + values_string + "."

        if params.get("parse_strings_as_datetimes"):
            template_str += " Values should be parsed as datetimes."

        if include_column_name:
            template_str = "$column " + template_str

        return [RenderedStringTemplateContent(**{
            "content_block_type": "string_template",
            "string_template": {
                "template": template_str,
                "params": params,
                "styling": styling,
            }
        })]

    @classmethod
    def expect_column_mean_to_be_between(cls, expectation, styling=None, include_column_name=True):
        params = substitute_none_for_missing(
            expectation.kwargs,
            ["column", "min_value", "max_value"]
        )

        if (params["min_value"] is None) and (params["max_value"] is None):
            template_str = "mean may have any numerical value."
        else:
            if params["min_value"] is not None and params["max_value"] is not None:
                template_str = "mean must be between $min_value and $max_value."
            elif params["min_value"] is None:
                template_str = "mean must be less than $max_value."
            elif params["max_value"] is None:
                template_str = "mean must be more than $min_value."

        if include_column_name:
            template_str = "$column " + template_str

        return [RenderedStringTemplateContent(**{
            "content_block_type": "string_template",
            "string_template": {
                "template": template_str,
                "params": params,
                "styling": styling,
            }
        })]

    @classmethod
    def expect_column_median_to_be_between(cls, expectation, styling=None, include_column_name=True):
        params = substitute_none_for_missing(
            expectation.kwargs,
            ["column", "min_value", "max_value"]
        )

        if (params["min_value"] is None) and (params["max_value"] is None):
            template_str = "median may have any numerical value."
        else:
            if params["min_value"] is not None and params["max_value"] is not None:
                template_str = "median must be between $min_value and $max_value."
            elif params["min_value"] is None:
                template_str = "median must be less than $max_value."
            elif params["max_value"] is None:
                template_str = "median must be more than $min_value."

        if include_column_name:
            template_str = "$column " + template_str

        return [RenderedStringTemplateContent(**{
            "content_block_type": "string_template",
            "string_template": {
                "template": template_str,
                "params": params,
                "styling": styling,
            }
        })]

    @classmethod
    def expect_column_stdev_to_be_between(cls, expectation, styling=None, include_column_name=True):
        params = substitute_none_for_missing(
            expectation.kwargs,
            ["column", "min_value", "max_value"]
        )

        if (params["min_value"] is None) and (params["max_value"] is None):
            template_str = "standard deviation may have any numerical value."
        else:
            if params["min_value"] is not None and params["max_value"] is not None:
                template_str = "standard deviation must be between $min_value and $max_value."
            elif params["min_value"] is None:
                template_str = "standard deviation must be less than $max_value."
            elif params["max_value"] is None:
                template_str = "standard deviation must be more than $min_value."

        if include_column_name:
            template_str = "$column " + template_str

        return [RenderedStringTemplateContent(**{
            "content_block_type": "string_template",
            "string_template": {
                "template": template_str,
                "params": params,
                "styling": styling,
            }
        })]

    @classmethod
    def expect_column_max_to_be_between(cls, expectation, styling=None, include_column_name=True):
        params = substitute_none_for_missing(
            expectation.kwargs,
            ["column", "min_value", "max_value", "parse_strings_as_datetimes"]
        )

        if (params["min_value"] is None) and (params["max_value"] is None):
            template_str = "maximum value may have any numerical value."
        else:
            if params["min_value"] is not None and params["max_value"] is not None:
                template_str = "maximum value must be between $min_value and $max_value."
            elif params["min_value"] is None:
                template_str = "maximum value must be less than $max_value."
            elif params["max_value"] is None:
                template_str = "maximum value must be more than $min_value."

        if params.get("parse_strings_as_datetimes"):
            template_str += " Values should be parsed as datetimes."

        if include_column_name:
            template_str = "$column " + template_str

        return [RenderedStringTemplateContent(**{
            "content_block_type": "string_template",
            "string_template": {
                "template": template_str,
                "params": params,
                "styling": styling,
            }
        })]

    @classmethod
    def expect_column_min_to_be_between(cls, expectation, styling=None, include_column_name=True):
        params = substitute_none_for_missing(
            expectation.kwargs,
            ["column", "min_value", "max_value", "parse_strings_as_datetimes"]
        )

        if (params["min_value"] is None) and (params["max_value"] is None):
            template_str = "minimum value may have any numerical value."
        else:
            if params["min_value"] is not None and params["max_value"] is not None:
                template_str = "minimum value must be between $min_value and $max_value."
            elif params["min_value"] is None:
                template_str = "minimum value must be less than $max_value."
            elif params["max_value"] is None:
                template_str = "minimum value must be more than $min_value."

        if params.get("parse_strings_as_datetimes"):
            template_str += " Values should be parsed as datetimes."

        if include_column_name:
            template_str = "$column " + template_str

        return [RenderedStringTemplateContent(**{
            "content_block_type": "string_template",
            "string_template": {
                "template": template_str,
                "params": params,
                "styling": styling,
            }
        })]

    @classmethod
    def expect_column_sum_to_be_between(cls, expectation, styling=None, include_column_name=True):
        params = substitute_none_for_missing(
            expectation.kwargs,
            ["column", "min_value", "max_value"]
        )

        if (params["min_value"] is None) and (params["max_value"] is None):
            template_str = "sum may have any numerical value."
        else:
            if params["min_value"] is not None and params["max_value"] is not None:
                template_str = "sum must be between $min_value and $max_value."
            elif params["min_value"] is None:
                template_str = "sum must be less than $max_value."
            elif params["max_value"] is None:
                template_str = "sum must be more than $min_value."

        if include_column_name:
            template_str = "$column " + template_str

        return [RenderedStringTemplateContent(**{
            "content_block_type": "string_template",
            "string_template": {
                "template": template_str,
                "params": params,
                "styling": styling,
            }
        })]

    @classmethod
    def expect_column_most_common_value_to_be_in_set(cls, expectation, styling=None, include_column_name=True):
        params = substitute_none_for_missing(
            expectation.kwargs,
            ["column", "value_set", "ties_okay"]
        )

        if params["value_set"] is None or len(params["value_set"]) == 0:
            values_string = "[ ]"
        else:
            for i, v in enumerate(params["value_set"]):
                params["v__" + str(i)] = v

            values_string = " ".join(
                ["$v__" + str(i) for i, v in enumerate(params["value_set"])]
            )

        template_str = "most common value must belong to this set: " + values_string + "."

        if params.get("ties_okay"):
            template_str += " Values outside this set that are as common (but not more common) are allowed."

        if include_column_name:
            template_str = "$column " + template_str

        return [RenderedStringTemplateContent(**{
            "content_block_type": "string_template",
            "string_template": {
                "template": template_str,
                "params": params,
                "styling": styling,
            }
        })]

    @classmethod
    def _get_kl_divergence_partition_object_table(cls, partition_object, header=None):
        table_rows = []
        fractions = partition_object["weights"]

        if partition_object.get("bins"):
            bins = partition_object["bins"]

            for idx, fraction in enumerate(fractions):
                if idx == len(fractions) - 1:
                    table_rows.append([
                        "[{} - {}]".format(num_to_str(bins[idx]), num_to_str(bins[idx + 1])),
                        num_to_str(fraction)
                    ])
                else:
                    table_rows.append([
                        "[{} - {})".format(num_to_str(bins[idx]), num_to_str(bins[idx + 1])),
                        num_to_str(fraction)
                    ])
        else:
            values = partition_object["values"]
            table_rows = [[value, num_to_str(fractions[idx])] for idx, value in enumerate(values)]

        if header:
            return {
                "content_block_type": "table",
                "header": header,
                "header_row": ["Interval", "Fraction"] if partition_object.get("bins") else ["Value", "Fraction"],
                "table": table_rows,
                "styling": {
                    "classes": ["table-responsive"],
                    "body": {
                        "classes": [
                            "table",
                            "table-sm",
                            "table-bordered",
                            "mt-2",
                            "mb-2"
                        ],
                    },
                    "parent": {
                        "classes": [
                            "show-scrollbars",
                            "p-2"
                        ],
                        "styles": {
                            "list-style-type": "none",
                            "overflow": "auto",
                            "max-height": "80vh"
                        }
                    }
                }
            }
        else:
            return {
                "content_block_type": "table",
                "header_row": ["Interval", "Fraction"] if partition_object.get("bins") else ["Value", "Fraction"],
                "table": table_rows,
                "styling": {
                    "classes": ["table-responsive"],
                    "body": {
                        "classes": [
                            "table",
                            "table-sm",
                            "table-bordered",
                            "mt-2",
                            "mb-2"
                        ],
                    },
                    "parent": {
                        "classes": [
                            "show-scrollbars",
                            "p-2"
                        ],
                        "styles": {
                            "list-style-type": "none",
                            "overflow": "auto",
                            "max-height": "80vh"
                        }
                    }
                }
            }

    @classmethod
    def expect_column_quantile_values_to_be_between(cls, expectation, styling=None, include_column_name=True):
        params = substitute_none_for_missing(
            expectation["kwargs"],
            ["column", "quantile_ranges"]
        )
        template_str = "quantiles must be within the following value ranges."

        if include_column_name:
            template_str = "$column " + template_str

        expectation_string_obj = {
            "content_block_type": "string_template",
            "string_template": {
                "template": template_str,
                "params": params
            }
        }

        quantiles = params["quantile_ranges"]["quantiles"]
        value_ranges = params["quantile_ranges"]["value_ranges"]

        table_header_row = ["Quantile", "Min Value", "Max Value"]
        table_rows = []

        quantile_strings = {
            .25: "Q1",
            .75: "Q3",
            .50: "Median"
        }

        for idx, quantile in enumerate(quantiles):
            quantile_string = quantile_strings.get(quantile)
            table_rows.append([
                quantile_string if quantile_string else "{:3.2f}".format(quantile),
                str(value_ranges[idx][0]) if value_ranges[idx][0] else "Any",
                str(value_ranges[idx][1]) if value_ranges[idx][1] else "Any",
            ])

        quantile_range_table = {
            "content_block_type": "table",
            "header_row": table_header_row,
            "table": table_rows,
            "styling": {
                "body": {
                    "classes": ["table", "table-sm", "table-unbordered", "col-4", "mt-2"],
                },
                "parent": {
                    "styles": {
                        "list-style-type": "none"
                    }
                }
            }
        }

        return [
            expectation_string_obj,
            quantile_range_table
        ]

    @classmethod
    def _get_kl_divergence_chart(cls, partition_object, header=None):
        weights = partition_object["weights"]

        if len(weights) > 60:
            expected_distribution = cls._get_kl_divergence_partition_object_table(partition_object, header=header)
        else:
            chart_pixel_width = (len(weights) / 60.0) * 500
            if chart_pixel_width < 250:
                chart_pixel_width = 250
            chart_container_col_width = round((len(weights) / 60.0) * 6)
            if chart_container_col_width < 4:
                chart_container_col_width = 4
            elif chart_container_col_width >= 5:
                chart_container_col_width = 6
            elif chart_container_col_width >= 4:
                chart_container_col_width = 5

            mark_bar_args = {}
            if len(weights) == 1:
                mark_bar_args["size"] = 20

            if partition_object.get("bins"):
                bins = partition_object["bins"]
                bins_x1 = [round(value, 1) for value in bins[:-1]]
                bins_x2 = [round(value, 1) for value in bins[1:]]

                df = pd.DataFrame({
                    "bin_min": bins_x1,
                    "bin_max": bins_x2,
                    "fraction": weights,
                })

                bars = alt.Chart(df).mark_bar().encode(
                    x='bin_min:O',
                    x2='bin_max:O',
                    y="fraction:Q",
                    tooltip=["bin_min", "bin_max", "fraction"]
                ).properties(width=chart_pixel_width, height=400, autosize="fit")

                chart = bars.to_json()
            elif partition_object.get("values"):
                values = partition_object["values"]

                df = pd.DataFrame({
                    "values": values,
                    "fraction": weights
                })

                bars = alt.Chart(df).mark_bar().encode(
                    x='values:N',
                    y="fraction:Q",
                    tooltip=["values", "fraction"]
                ).properties(width=chart_pixel_width, height=400, autosize="fit")
                chart = bars.to_json()

            if header:
                expected_distribution = RenderedGraphContent(**{
                    "content_block_type": "graph",
                    "graph": chart,
                    "header": header,
                    "styling": {
                        "classes": ["col-" + str(chart_container_col_width), "mt-2", "pl-1", "pr-1"],
                        "parent": {
                            "styles": {
                                "list-style-type": "none"
                            }
                        }
                    }
                })
            else:
                expected_distribution = RenderedGraphContent(**{
                    "content_block_type": "graph",
                    "graph": chart,
                    "styling": {
                        "classes": ["col-" + str(chart_container_col_width), "mt-2", "pl-1", "pr-1"],
                        "parent": {
                            "styles": {
                                "list-style-type": "none"
                            }
                        }
                    }
                })
        return expected_distribution

    @classmethod
    def expect_column_kl_divergence_to_be_less_than(cls, expectation, styling=None, include_column_name=True):
        params = substitute_none_for_missing(
            expectation.kwargs,
            ["column", "partition_object", "threshold"]
        )

        expected_distribution = None
        if not params.get("partition_object"):
            template_str = "can match any distribution."
        else:
            template_str = "Kullback-Leibler (KL) divergence with respect to the following distribution must be " \
                           "lower than $threshold."
            expected_distribution = cls._get_kl_divergence_chart(params.get("partition_object"))

        if include_column_name:
            template_str = "$column " + template_str

        expectation_string_obj = {
            "content_block_type": "string_template",
            "string_template": {
                "template": template_str,
                "params": params
            }
        }

        if expected_distribution:
            return [
                expectation_string_obj,
                expected_distribution
            ]
        else:
            return [expectation_string_obj]

    @classmethod
    def expect_column_values_to_be_unique(cls, expectation, styling=None, include_column_name=True):
        params = substitute_none_for_missing(
            expectation.kwargs,
            ["column", "mostly"],
        )

        if include_column_name:
            template_str = "$column values must be unique"
        else:
            template_str = "values must be unique"

        if params["mostly"] is not None:
            params["mostly_pct"] = num_to_str(params["mostly"] * 100, precision=15, no_scientific=True)
            # params["mostly_pct"] = "{:.14f}".format(params["mostly"]*100).rstrip("0").rstrip(".")
            template_str += ", at least $mostly_pct % of the time."
        else:
            template_str += "."

        return [RenderedStringTemplateContent(**{
            "content_block_type": "string_template",
            "string_template": {
                "template": template_str,
                "params": params,
                "styling": styling,
            }
        })]


class ExpectationSuiteBulletListContentBlockRenderer(ExpectationStringRenderer):
    _rendered_component_type = RenderedBulletListContent
    _content_block_type = "bullet_list"

    _default_element_styling = {
        "default": {
            "classes": ["badge", "badge-secondary"]
        },
        "params": {
            "column": {
                "classes": ["badge", "badge-primary"]
            }
        }
    }


class ValidationResultsTableContentBlockRenderer(ExpectationStringRenderer):
    _content_block_type = "table"
    _rendered_component_type = RenderedTableContent

    _default_element_styling = {
        "default": {
            "classes": ["badge", "badge-secondary"]
        },
        "params": {
            "column": {
                "classes": ["badge", "badge-primary"]
            }
        }
    }

    _default_content_block_styling = {
        "body": {
            "classes": ["table"],
        },
        "classes": ["ml-2", "mr-2", "mt-0", "mb-0", "table-responsive"],
    }

    @classmethod
    def _get_status_icon(cls, evr):
        if evr.exception_info["raised_exception"]:
            return RenderedStringTemplateContent(**{
                "content_block_type": "string_template",
                "string_template": {
                    "template": "$icon",
                    "params": {"icon": ""},
                    "styling": {
                        "params": {
                            "icon": {
                                "classes": ["fas", "fa-exclamation-triangle", "text-warning"],
                                "tag": "i"
                            }
                        }
                    }
                }
            })

        if evr.success:
            return RenderedStringTemplateContent(**{
                "content_block_type": "string_template",
                "string_template": {
                    "template": "$icon",
                    "params": {"icon": ""},
                    "styling": {
                        "params": {
                            "icon": {
                                "classes": ["fas", "fa-check-circle", "text-success"],
                                "tag": "i"
                            }
                        }
                    }
                },
                "styling": {
                    "parent": {
                        "classes": ["hide-succeeded-validation-target-child"]
                    }
                }
            })
        else:
            return RenderedStringTemplateContent(**{
                "content_block_type": "string_template",
                "string_template": {
                    "template": "$icon",
                    "params": {"icon": ""},
                    "styling": {
                        "params": {
                            "icon": {
                                "tag": "i",
                                "classes": ["fas", "fa-times", "text-danger"]
                            }
                        }
                    }
                }
            })

    @classmethod
    def _get_unexpected_table(cls, evr):
        try:
            result = evr.result
        except KeyError:
            return None

        if result is None:
            return None

        if not result.get("partial_unexpected_list") and not result.get("partial_unexpected_counts"):
            return None

        table_rows = []

        if result.get("partial_unexpected_counts"):
            header_row = ["Unexpected Value", "Count"]
            for unexpected_count in result.get("partial_unexpected_counts"):
                if unexpected_count.get("value"):
                    table_rows.append([unexpected_count.get("value"), unexpected_count.get("count")])
                elif unexpected_count.get("value") == "":
                    table_rows.append(["EMPTY", unexpected_count.get("count")])
                elif unexpected_count.get("value") is not None:
                    table_rows.append([unexpected_count.get("value"), unexpected_count.get("count")])
                else:
                    table_rows.append(["null", unexpected_count.get("count")])
        else:
            header_row = ["Unexpected Value"]
            for unexpected_value in result.get("partial_unexpected_list"):
                if unexpected_value:
                    table_rows.append([unexpected_value])
                elif unexpected_value == "":
                    table_rows.append(["EMPTY"])
                elif unexpected_value is not None:
                    table_rows.append([unexpected_value])
                else:
                    table_rows.append(["null"])

        unexpected_table_content_block = RenderedTableContent(**{
            "content_block_type": "table",
            "table": table_rows,
            "header_row": header_row,
            "styling": {
                "body": {
                    "classes": ["table-bordered", "table-sm", "mt-3"]
                }
            }
        })

        return unexpected_table_content_block

    @classmethod
    def _get_unexpected_statement(cls, evr):
        success = evr.success
        result = evr.result

        if evr.exception_info["raised_exception"]:
            exception_message_template_str = "\n\n$expectation_type raised an exception:\n$exception_message"

            exception_message = RenderedStringTemplateContent(**{
                "content_block_type": "string_template",
                "string_template": {
                    "template": exception_message_template_str,
                    "params": {
                        "expectation_type": evr.expectation_config.expectation_type,
                        "exception_message": evr.exception_info["exception_message"]
                    },
                    "tag": "strong",
                    "styling": {
                        "classes": ["text-danger"],
                        "params": {
                            "exception_message": {
                                "tag": "code"
                            },
                            "expectation_type": {
                                "classes": ["badge", "badge-danger", "mb-2"]
                            }
                        }
                    }
                },
            })

            exception_traceback_collapse = CollapseContent(**{
                "collapse_toggle_link": "Show exception traceback...",
                "collapse": [
                    RenderedStringTemplateContent(**{
                        "content_block_type": "string_template",
                        "string_template": {
                            "template": evr.exception_info["exception_traceback"],
                            "tag": "code"
                        }
                    })
                ]
            })

            return [exception_message, exception_traceback_collapse]

        if success or not result.get("unexpected_count"):
            return []
        else:
            unexpected_count = num_to_str(result["unexpected_count"], use_locale=True, precision=20)
            unexpected_percent = num_to_str(result["unexpected_percent"], precision=4) + "%"
            element_count = num_to_str(result["element_count"], use_locale=True, precision=20)

            template_str = "\n\n$unexpected_count unexpected values found. " \
                           "$unexpected_percent of $element_count total rows."

            return [
                RenderedStringTemplateContent(**{
                "content_block_type": "string_template",
                "string_template": {
                    "template": template_str,
                    "params": {
                        "unexpected_count": unexpected_count,
                        "unexpected_percent": unexpected_percent,
                        "element_count": element_count
                    },
                    "tag": "strong",
                    "styling": {
                        "classes": ["text-danger"]
                    }
                }})
            ]

    @classmethod
    def _get_kl_divergence_observed_value(cls, evr):
        if not evr.result.get("details"):
            return "--"

        observed_partition_object = evr.result["details"]["observed_partition"]
        observed_distribution = super(
            ValidationResultsTableContentBlockRenderer, cls)._get_kl_divergence_chart(observed_partition_object)

        observed_value = num_to_str(evr.result.get("observed_value")) if evr.result.get("observed_value") \
            else evr.result.get("observed_value")

        observed_value_content_block = RenderedStringTemplateContent(**{
            "content_block_type": "string_template",
            "string_template": {
                "template": "KL Divergence: $observed_value",
                "params": {
                    "observed_value": str(
                        observed_value) if observed_value else "None (-infinity, infinity, or NaN)",
                },
                "styling": {
                    "classes": ["mb-2"]
                }
            },
        })

        return RenderedContentBlockContainer(**{
            "content_block_type": "content_block_container",
            "content_blocks": [
                observed_value_content_block,
                observed_distribution
            ]
        })

    @classmethod
    def _get_quantile_values_observed_value(cls, evr):
        if evr.result is None or evr.result.get("observed_value") is None:
            return "--"

        quantiles = evr.result.get("observed_value", {}).get("quantiles", [])
        value_ranges = evr.result.get("observed_value", {}).get("values", [])

        table_header_row = ["Quantile", "Value"]
        table_rows = []

        quantile_strings = {
            .25: "Q1",
            .75: "Q3",
            .50: "Median"
        }

        for idx, quantile in enumerate(quantiles):
            quantile_string = quantile_strings.get(quantile)
            table_rows.append([
                quantile_string if quantile_string else "{:3.2f}".format(quantile),
                str(value_ranges[idx])
            ])

        return RenderedTableContent(**{
            "content_block_type": "table",
            "header_row": table_header_row,
            "table": table_rows,
            "styling": {
                "body": {
                    "classes": ["table", "table-sm", "table-unbordered", "col-4"],
                }
            }
        })

    @classmethod
    def _get_observed_value(cls, evr):
        result = evr.result
        if result is None:
            return "--"

        expectation_type = evr.expectation_config["expectation_type"]

        if expectation_type == "expect_column_kl_divergence_to_be_less_than":
            return cls._get_kl_divergence_observed_value(evr)
        elif expectation_type == "expect_column_quantile_values_to_be_between":
            return cls._get_quantile_values_observed_value(evr)

        if result.get("observed_value"):
            observed_value = result.get("observed_value")
            if isinstance(observed_value, (integer_types, float)) and not isinstance(observed_value, bool):
                return num_to_str(observed_value, precision=10, use_locale=True)
            return str(observed_value)
        elif expectation_type == "expect_column_values_to_be_null":
            try:
                notnull_percent = result["unexpected_percent"]
                return num_to_str(100 - notnull_percent, precision=5, use_locale=True) + "% null"
            except KeyError:
                return "unknown % null"
        elif expectation_type == "expect_column_values_to_not_be_null":
            try:
                null_percent = result["unexpected_percent"]
                return num_to_str(100 - null_percent, precision=5, use_locale=True) + "% not null"
            except KeyError:
                return "unknown % not null"
        elif result.get("unexpected_percent") is not None:
            return num_to_str(result.get("unexpected_percent"), precision=5) + "% unexpected"
        else:
            return "--"

    @classmethod
    def _process_content_block(cls, content_block, has_failed_evr):
        super(ValidationResultsTableContentBlockRenderer, cls)._process_content_block(content_block, has_failed_evr)
        content_block.header_row = ["Status", "Expectation", "Observed Value"]

        if has_failed_evr is False:
            styling = copy.deepcopy(content_block.styling) if content_block.styling else {}
            if styling.get("classes"):
                styling["classes"].append("hide-succeeded-validations-column-section-target-child")
            else:
                styling["classes"] = ["hide-succeeded-validations-column-section-target-child"]

            content_block.styling = styling

    @classmethod
    def _get_content_block_fn(cls, expectation_type):
        expectation_string_fn = getattr(cls, expectation_type, None)
        if expectation_string_fn is None:
            expectation_string_fn = getattr(cls, "_missing_content_block_fn")

        #This function wraps expect_* methods from ExpectationStringRenderer to generate table classes
        def row_generator_fn(evr, styling=None, include_column_name=True):
            expectation = evr.expectation_config
            expectation_string_cell = expectation_string_fn(expectation, styling, include_column_name)

            status_cell = [cls._get_status_icon(evr)]
            unexpected_statement = []
            unexpected_table = None
            observed_value = ["--"]

            try:
                unexpected_statement = cls._get_unexpected_statement(evr)
            except Exception as e:
                logger.error("Exception occurred during data docs rendering: ", e, exc_info=True)
            try:
                unexpected_table = cls._get_unexpected_table(evr)
            except Exception as e:
                logger.error("Exception occurred during data docs rendering: ", e, exc_info=True)
            try:
                observed_value = [cls._get_observed_value(evr)]
            except Exception as e:
                logger.error("Exception occurred during data docs rendering: ", e, exc_info=True)

            # If the expectation has some unexpected values...:
            if unexpected_statement:
                expectation_string_cell += unexpected_statement
            if unexpected_table:
                expectation_string_cell.append(unexpected_table)

            if len(expectation_string_cell) > 1:
                return [status_cell + [expectation_string_cell] + observed_value]
            else:
                return [status_cell + expectation_string_cell + observed_value]

        return row_generator_fn


class ColumnSectionRenderer(Renderer):
    @classmethod
    def _get_column_name(cls, ge_object):
        # This is broken out for ease of locating future validation here
        if isinstance(ge_object, list):
            candidate_object = ge_object[0]
        else:
            candidate_object = ge_object
        try:
            if isinstance(candidate_object, ExpectationConfiguration):
                return candidate_object.kwargs["column"]
            elif isinstance(candidate_object, ExpectationValidationResult):
                return candidate_object.expectation_config.kwargs["column"]
            else:
                raise ValueError(
                    "Provide a column section renderer an expectation, list of expectations, evr, or list of evrs.")
        except KeyError:
            return "Table-Level Expectations"


class ProfilingResultsOverviewSectionRenderer(Renderer):

    @classmethod
    def render(cls, evrs, section_name=None):

        content_blocks = []
        # NOTE: I don't love the way this builds content_blocks as a side effect.
        # The top-level API is clean and scannable, but the function internals are counterintutitive and hard to test.
        # I wonder if we can enable something like jquery chaining for this. Tha would be concise AND testable.
        # Pressing on for now...
        cls._render_header(evrs, content_blocks)
        cls._render_dataset_info(evrs, content_blocks)
        cls._render_variable_types(evrs, content_blocks)
        cls._render_warnings(evrs, content_blocks)
        cls._render_expectation_types(evrs, content_blocks)

        return RenderedSectionContent(**{
            "section_name": section_name,
            "content_blocks": content_blocks
        })

    @classmethod
    def _render_header(cls, evrs, content_blocks):
        content_blocks.append(RenderedHeaderContent(**{
            "content_block_type": "header",
            "header": RenderedStringTemplateContent(**{
                "content_block_type": "string_template",
                "string_template": {
                    "template": 'Overview',
                    "tag": "h5",
                    "styling": {
                        "classes": ["m-0"]
                    }
                }
            }),
            "styling": {
                "classes": ["col-12", "p-0"],
                "header": {
                    "classes": ["alert", "alert-secondary"]
                }
            }
        }))

    @classmethod
    def _render_dataset_info(cls, evrs, content_blocks):
        expect_table_row_count_to_be_between_evr = cls._find_evr_by_type(evrs['results'],
                                                                         "expect_table_row_count_to_be_between")

        table_rows = []
        table_rows.append(["Number of variables", len(cls._get_column_list_from_evrs(evrs)), ])

        table_rows.append([
            RenderedStringTemplateContent(**{
                "content_block_type": "string_template",
                "string_template": {
                    "template": "Number of observations",
                    "tooltip": {
                        "content": "expect_table_row_count_to_be_between"
                    },
                    "params": {
                        "tooltip_text": "Number of observations"
                    },
                }
            }),
            "--" if not expect_table_row_count_to_be_between_evr else expect_table_row_count_to_be_between_evr.result["observed_value"]
        ])

        table_rows += [
            ["Missing cells", cls._get_percentage_missing_cells_str(evrs), ],
            # ["Duplicate rows", "0 (0.0%)", ], #TODO: bring back when we have an expectation for this
        ]

        content_blocks.append(RenderedTableContent(**{
            "content_block_type": "table",
            "header": RenderedStringTemplateContent(**{
                "content_block_type": "string_template",
                "string_template": {
                    "template": 'Dataset info',
                    "tag": "h6"
                }
            }),
            "table": table_rows,
            "styling": {
                "classes": ["col-6", "mt-1", "p-1"],
                "body": {
                    "classes": ["table", "table-sm"]
                }
            },
        }))

    @classmethod
    def _render_variable_types(cls, evrs, content_blocks):

        column_types = cls._get_column_types(evrs)
        # TODO: check if we have the information to make this statement. Do all columns have type expectations?
        column_type_counter = Counter(column_types.values())
        table_rows = [[type, str(column_type_counter[type])] for type in ["int", "float", "string", "unknown"]]

        content_blocks.append(RenderedTableContent(**{
            "content_block_type": "table",
            "header": RenderedStringTemplateContent(**{
                "content_block_type": "string_template",
                "string_template": {
                    "template": 'Variable types',
                    "tag": "h6"
                }
            }),
            "table": table_rows,
            "styling": {
                "classes": ["col-6", "table-responsive", "mt-1", "p-1"],
                "body": {
                    "classes": ["table", "table-sm"]
                }
            },
        }))

    @classmethod
    def _render_expectation_types(cls, evrs, content_blocks):

        type_counts = defaultdict(int)

        for evr in evrs.results:
            type_counts[evr.expectation_config.expectation_type] += 1

        bullet_list_items = sorted(type_counts.items(), key=lambda kv: -1 * kv[1])

        bullet_list_items = [
            RenderedStringTemplateContent(**{
                "content_block_type": "string_template",
                "string_template": {
                    "template": "$expectation_type $expectation_count",
                    "params": {
                        "expectation_type": tr[0],
                        "expectation_count": tr[1],
                    },
                    "styling": {
                        "classes": ["list-group-item", "d-flex", "justify-content-between", "align-items-center"],
                        "params": {
                            "expectation_count": {
                                "classes": ["badge", "badge-secondary", "badge-pill"],
                            }
                        }
                    }
                },
                "styling": {
                    'parent': {
                        'styles': {
                            'list-style-type': 'none'
                        }
                    }
                }
            }) for tr in bullet_list_items]

        bullet_list = RenderedBulletListContent(**{
            "content_block_type": "bullet_list",
            "bullet_list": bullet_list_items,
            "styling": {
                "classes": ["col-12", "mt-1"],
                "body": {
                    "classes": ["list-group"],
                },
            },
        })

        bullet_list_collapse = CollapseContent(**{
            "collapse_toggle_link": "Show Expectation Types...",
            "collapse": [bullet_list],
            "styling": {
                "classes": ["col-12", "p-1"]
            }
        })

        content_blocks.append(bullet_list_collapse)

    @classmethod
    def _render_warnings(cls, evrs, content_blocks):
        return

        # def render_warning_row(template, column, n, p, badge_label):
        #     return [{
        #         "template": template,
        #         "params": {
        #             "column": column,
        #             "n": n,
        #             "p": p,
        #         },
        #         "styling": {
        #             "params": {
        #                 "column": {
        #                     "classes": ["badge", "badge-primary", ]
        #                 }
        #             }
        #         }
        #     }, {
        #         "template": "$badge_label",
        #         "params": {
        #             "badge_label": badge_label,
        #         },
        #         "styling": {
        #             "params": {
        #                 "badge_label": {
        #                     "classes": ["badge", "badge-warning", ]
        #                 }
        #             }
        #         }
        #     }]

        # table_rows = [
        #     render_warning_row(
        #         "$column has $n ($p%) missing values", "Age", 177, 19.9, "Missing"),
        #     render_warning_row(
        #         "$column has a high cardinality: $n distinct values", "Cabin", 148, None, "Warning"),
        #     render_warning_row(
        #         "$column has $n ($p%) missing values", "Cabin", 687, 77.1, "Missing"),
        #     render_warning_row(
        #         "$column has $n (< $p%) zeros", "Fare", 15, "0.1", "Zeros"),
        #     render_warning_row(
        #         "$column has $n (< $p%) zeros", "Parch", 678, "76.1", "Zeros"),
        #     render_warning_row(
        #         "$column has $n (< $p%) zeros", "SibSp", 608, "68.2", "Zeros"),
        # ]

        # content_blocks.append({
        #     "content_block_type": "table",
        #     "header": "Warnings",
        #     "table": table_rows,
        #     "styling": {
        #         "classes": ["col-12"],
        #         "styles": {
        #             "margin-top": "20px"
        #         },
        #         "body": {
        #             "classes": ["table", "table-sm"]
        #         }
        #     },
        # })

    @classmethod
    def _get_percentage_missing_cells_str(cls, evrs):

        columns = cls._get_column_list_from_evrs(evrs)
        if not columns or len(columns) == 0:
            warnings.warn("Cannot get % of missing cells - column list is empty")
            return "?"

        expect_column_values_to_not_be_null_evrs = cls._find_all_evrs_by_type(evrs.results,
                                                                              "expect_column_values_to_not_be_null")

        if len(columns) > len(expect_column_values_to_not_be_null_evrs):
            warnings.warn(
                "Cannot get % of missing cells - not all columns have expect_column_values_to_not_be_null expectations")
            return "?"

        # assume 100.0 missing for columns where ["result"]["unexpected_percent"] is not available
        return "{0:.2f}%".format(sum([evr.result["unexpected_percent"] if "unexpected_percent" in evr.result and
                                                                             evr.result["unexpected_percent"] is not None else 100.0
                                      for evr in expect_column_values_to_not_be_null_evrs]) / len(columns))

    @classmethod
    def _get_column_types(cls, evrs):
        columns = cls._get_column_list_from_evrs(evrs)

        type_evrs = cls._find_all_evrs_by_type(evrs.results, "expect_column_values_to_be_in_type_list") + \
                    cls._find_all_evrs_by_type(evrs.results, "expect_column_values_to_be_of_type")

        column_types = {}
        for column in columns:
            column_types[column] = "unknown"

        for evr in type_evrs:
            column = evr.expectation_config.kwargs["column"]
            if evr.expectation_config.expectation_type == "expect_column_values_to_be_in_type_list":
                if evr.expectation_config.kwargs["type_list"] is None:
                    column_types[column] = "unknown"
                    continue
                else:
                    expected_types = set(evr.expectation_config.kwargs["type_list"])
            else:  # assuming expect_column_values_to_be_of_type
                expected_types = {[evr.expectation_config.kwargs["type_"]]}

            if expected_types.issubset(BasicDatasetProfiler.INT_TYPE_NAMES):
                column_types[column] = "int"
            elif expected_types.issubset(BasicDatasetProfiler.FLOAT_TYPE_NAMES):
                column_types[column] = "float"
            elif expected_types.issubset(BasicDatasetProfiler.STRING_TYPE_NAMES):
                column_types[column] = "string"
            elif expected_types.issubset(BasicDatasetProfiler.DATETIME_TYPE_NAMES):
                column_types[column] = "datetime"
            elif expected_types.issubset(BasicDatasetProfiler.BOOLEAN_TYPE_NAMES):
                column_types[column] = "bool"
            else:
                warnings.warn("The expected type list is not a subset of any of the profiler type sets: {0:s}".format(
                    str(expected_types)))
                column_types[column] = "unknown"

        return column_types


class ProfilingResultsColumnSectionRenderer(ColumnSectionRenderer):

    def __init__(self, overview_table_renderer=None, expectation_string_renderer=None, runtime_environment=None):
        if overview_table_renderer is None:
            overview_table_renderer = {
                "class_name": "ProfilingOverviewTableContentBlockRenderer"
            }
        if expectation_string_renderer is None:
            expectation_string_renderer = {
                "class_name": "ExpectationStringRenderer"
            }
        self._overview_table_renderer = instantiate_class_from_config(
            config=overview_table_renderer,
            runtime_environment=runtime_environment,
            config_defaults={
                #"module_name": "great_expectations.render.renderer.content_block"
                "module_name": "great_expectations.common"
            }
        )
        self._expectation_string_renderer = instantiate_class_from_config(
            config=expectation_string_renderer,
            runtime_environment=runtime_environment,
            config_defaults={
                #"module_name": "great_expectations.render.renderer.content_block"
                "module_name": "great_expectations.common"
            }
        )

        self.content_block_function_names = [
            "_render_header",
            "_render_overview_table",
            "_render_quantile_table",
            "_render_stats_table",
            "_render_values_set",
            "_render_histogram",
            "_render_bar_chart_table",
            "_render_failed",
        ]

    #Note: Seems awkward to pass section_name and column_type into this renderer.
    #Can't we figure that out internally?
    def render(self, evrs, section_name=None, column_type=None):
        if section_name is None:
            column = self._get_column_name(evrs)
        else:
            column = section_name

        content_blocks = []

        for content_block_function_name in self.content_block_function_names:
            try:
                if content_block_function_name == "_render_header":
                    content_blocks.append(getattr(self, content_block_function_name)(evrs, column_type))
                else:
                    content_blocks.append(getattr(self, content_block_function_name)(evrs))
            except Exception as e:
                logger.error("Exception occurred during data docs rendering: ", e, exc_info=True)

        # NOTE : Some render* functions return None so we filter them out
        populated_content_blocks = list(filter(None, content_blocks))

        return RenderedSectionContent(**{
            "section_name": column,
            "content_blocks": populated_content_blocks,
        })

    @classmethod
    def _render_header(cls, evrs, column_type=None):
        # NOTE: This logic is brittle
        try:
            column_name = evrs[0].expectation_config.kwargs["column"]
        except KeyError:
            column_name = "Table-level expectations"

        return RenderedHeaderContent(**{
            "content_block_type": "header",
            "header": RenderedStringTemplateContent(**{
                        "content_block_type": "string_template",
                        "string_template": {
                            "template": convert_to_string_and_escape(column_name),
                            "tooltip": {
                                "content": "expect_column_to_exist",
                                "placement": "top"
                            },
                            "tag": "h5",
                            "styling": {
                                "classes": ["m-0", "p-0"]
                            }
                        }
                    }),
            "subheader": RenderedStringTemplateContent(**{
                        "content_block_type": "string_template",
                        "string_template": {
                            "template": "Type: {column_type}".format(column_type=column_type),
                            "tooltip": {
                              "content":
                              "expect_column_values_to_be_of_type <br>expect_column_values_to_be_in_type_list",
                            },
                            "tag": "h6",
                            "styling": {
                                "classes": ["mt-1", "mb-0"]
                            }
                        }
                    }),
            # {
            #     "template": column_type,
            # },
            "styling": {
                "classes": ["col-12", "p-0"],
                "header": {
                    "classes": ["alert", "alert-secondary"]
                }
            }
        })

    @classmethod
    def _render_expectation_types(cls, evrs, content_blocks):
        # NOTE: The evr-fetching function is an kinda similar to the code other_section_
        # renderer.ProfilingResultsOverviewSectionRenderer._render_expectation_types

        # type_counts = defaultdict(int)

        # for evr in evrs:
        #     type_counts[evr.expectation_config.expectation_type] += 1

        # bullet_list = sorted(type_counts.items(), key=lambda kv: -1*kv[1])

        bullet_list = [{
            "content_block_type": "string_template",
            "string_template": {
                "template": "$expectation_type $is_passing",
                "params": {
                    "expectation_type": evr.expectation_config.expectation_type,
                    "is_passing": str(evr.success),
                },
                "styling": {
                    "classes": ["list-group-item", "d-flex", "justify-content-between", "align-items-center"],
                    "params": {
                        "is_passing": {
                            "classes": ["badge", "badge-secondary", "badge-pill"],
                        }
                    },
                }
            }
        } for evr in evrs]

        content_blocks.append(RenderedBulletListContent(**{
            "content_block_type": "bullet_list",
            "header": RenderedStringTemplateContent(**{
                "content_block_type": "string_template",
                "string_template": {
                    "template": 'Expectation types <span class="mr-3 triangle"></span>',
                    "tag": "h6"
                }
            }),
            "bullet_list": bullet_list,
            "styling": {
                "classes": ["col-12", "mt-1"],
                "header": {
                    "classes": ["collapsed"],
                    "attributes": {
                        "data-toggle": "collapse",
                        "href": "#{{content_block_id}}-body",
                        "role": "button",
                        "aria-expanded": "true",
                        "aria-controls": "collapseExample",
                    },
                    "styles": {
                        "cursor": "pointer",
                    }
                },
                "body": {
                    "classes": ["list-group", "collapse"],
                },
            },
        }))

    def _render_overview_table(self, evrs):
        unique_n = self._find_evr_by_type(
            evrs,
            "expect_column_unique_value_count_to_be_between"
        )
        unique_proportion = self._find_evr_by_type(
            evrs,
            "expect_column_proportion_of_unique_values_to_be_between"
        )
        null_evr = self._find_evr_by_type(
            evrs,
            "expect_column_values_to_not_be_null"
        )
        evrs = [evr for evr in [unique_n, unique_proportion, null_evr] if (evr is not None)]

        if len(evrs) > 0:
            new_content_block = self._overview_table_renderer.render(evrs)
            new_content_block.header = RenderedStringTemplateContent(**{
                "content_block_type": "string_template",
                "string_template": {
                    "template": 'Properties',
                    "tag": "h6"
                }
            })
            new_content_block.styling = {
                "classes": ["col-3", "mt-1", "pl-1", "pr-1"],
                "body": {
                    "classes": ["table", "table-sm", "table-unbordered"],
                    "styles": {
                        "width": "100%"
                    },
                }

            }
            return new_content_block

    @classmethod
    def _render_quantile_table(cls, evrs):
        table_rows = []

        quantile_evr = cls._find_evr_by_type(
            evrs,
            "expect_column_quantile_values_to_be_between"
        )

        if not quantile_evr or quantile_evr.exception_info["raised_exception"]:
            return

        quantiles = quantile_evr.result["observed_value"]["quantiles"]
        quantile_ranges = quantile_evr.result["observed_value"]["values"]

        quantile_strings = {
            .25: "Q1",
            .75: "Q3",
            .50: "Median"
        }

        for idx, quantile in enumerate(quantiles):
            quantile_string = quantile_strings.get(quantile)
            table_rows.append([
                {
                    "content_block_type": "string_template",
                    "string_template": {
                        "template": quantile_string if quantile_string else "{:3.2f}".format(quantile),
                        "tooltip": {
                            "content": "expect_column_quantile_values_to_be_between \n expect_column_median_to_be_between" if quantile == 0.50 else "expect_column_quantile_values_to_be_between"
                        }
                    }
                },
                quantile_ranges[idx],
            ])

        return RenderedTableContent(**{
            "content_block_type": "table",
            "header": RenderedStringTemplateContent(**{
                "content_block_type": "string_template",
                "string_template": {
                    "template": 'Quantiles',
                    "tag": "h6"
                }
            }),
            "table": table_rows,
            "styling": {
                "classes": ["col-3", "mt-1", "pl-1", "pr-1"],
                "body": {
                    "classes": ["table", "table-sm", "table-unbordered"],
                }
            },
        })

    @classmethod
    def _render_stats_table(cls, evrs):
        table_rows = []

        mean_evr = cls._find_evr_by_type(
            evrs,
            "expect_column_mean_to_be_between"
        )

        if not mean_evr or mean_evr.exception_info["raised_exception"]:
            return

        mean_value = "{:.2f}".format(
            mean_evr.result['observed_value']) if mean_evr else None
        if mean_value:
            table_rows.append([
                {
                    "content_block_type": "string_template",
                    "string_template": {
                        "template": "Mean",
                        "tooltip": {
                            "content": "expect_column_mean_to_be_between"
                        }
                    }
                },
                mean_value
            ])

        min_evr = cls._find_evr_by_type(
            evrs,
            "expect_column_min_to_be_between"
        )
        min_value = "{:.2f}".format(
            min_evr.result['observed_value']) if min_evr else None
        if min_value:
            table_rows.append([
                {
                    "content_block_type": "string_template",
                    "string_template": {
                        "template": "Minimum",
                        "tooltip": {
                            "content": "expect_column_min_to_be_between"
                        }
                    }
                },
                min_value,
            ])

        max_evr = cls._find_evr_by_type(
            evrs,
            "expect_column_max_to_be_between"
        )
        max_value = "{:.2f}".format(
            max_evr.result['observed_value']) if max_evr else None
        if max_value:
            table_rows.append([
                {
                    "content_block_type": "string_template",
                    "string_template": {
                        "template": "Maximum",
                        "tooltip": {
                            "content": "expect_column_max_to_be_between"
                        }
                    }
                },
                max_value
            ])

        if len(table_rows) > 0:
            return RenderedTableContent(**{
                "content_block_type": "table",
                "header": RenderedStringTemplateContent(**{
                    "content_block_type": "string_template",
                    "string_template": {
                        "template": 'Statistics',
                        "tag": "h6"
                    }
                }),
                "table": table_rows,
                "styling": {
                    "classes": ["col-3", "mt-1", "pl-1", "pr-1"],
                    "body": {
                        "classes": ["table", "table-sm", "table-unbordered"],
                    }
                },
            })
        else:
            return

    @classmethod
    def _render_values_set(cls, evrs):
        set_evr = cls._find_evr_by_type(
            evrs,
            "expect_column_values_to_be_in_set"
        )

        if not set_evr or set_evr.exception_info["raised_exception"]:
            return

        if set_evr and "partial_unexpected_counts" in set_evr.result:
            partial_unexpected_counts = set_evr.result["partial_unexpected_counts"]
            values = [str(v["value"]) for v in partial_unexpected_counts]
        elif set_evr and "partial_unexpected_list" in set_evr.result:
            values = [str(item) for item in set_evr.result["partial_unexpected_list"]]
        else:
            return

        classes = ["col-3", "mt-1", "pl-1", "pr-1"]

        if any(len(value) > 80 for value in values):
            content_block_type = "bullet_list"
            content_block_class = RenderedBulletListContent
        else:
            content_block_type = "value_list"
            content_block_class = ValueListContent

        new_block = content_block_class(**{
            "content_block_type": content_block_type,
            "header": RenderedStringTemplateContent(**{
                "content_block_type": "string_template",
                "string_template": {
                    "template": "Example Values",
                    "tooltip": {
                        "content": "expect_column_values_to_be_in_set"
                    },
                    "tag": "h6"
                }
            }),
            content_block_type: [{
                "content_block_type": "string_template",
                "string_template": {
                    "template": "$value",
                    "params": {
                        "value": value
                    },
                    "styling": {
                        "default": {
                            "classes": ["badge", "badge-info"] if content_block_type == "value_list" else [],
                            "styles": {
                                "word-break": "break-all"
                            }
                        },
                    }
                }
            } for value in values],
            "styling": {
                "classes": classes,
            }
        })

        return new_block

    def _render_histogram(self, evrs):
        # NOTE: This code is very brittle
        kl_divergence_evr = self._find_evr_by_type(
            evrs,
            "expect_column_kl_divergence_to_be_less_than"
        )
        # print(json.dumps(kl_divergence_evr, indent=2))
        if kl_divergence_evr is None or kl_divergence_evr.result is None or "details" not in kl_divergence_evr.result:
            return

        observed_partition_object = kl_divergence_evr.result["details"]["observed_partition"]
        weights = observed_partition_object["weights"]
        if len(weights) > 60:
            return None

        header = RenderedStringTemplateContent(**{
            "content_block_type": "string_template",
            "string_template": {
                "template": "Histogram",
                "tooltip": {
                    "content": "expect_column_kl_divergence_to_be_less_than"
                },
                "tag": "h6"
            }
        })

        return self._expectation_string_renderer._get_kl_divergence_chart(observed_partition_object, header)

    @classmethod
    def _render_bar_chart_table(cls, evrs):
        distinct_values_set_evr = cls._find_evr_by_type(
            evrs,
            "expect_column_distinct_values_to_be_in_set"
        )
        if not distinct_values_set_evr or distinct_values_set_evr.exception_info["raised_exception"]:
            return

        value_count_dicts = distinct_values_set_evr.result['details']['value_counts']
        if isinstance(value_count_dicts, pd.Series):
            values = value_count_dicts.index.tolist()
            counts = value_count_dicts.tolist()
        else:
            values = [value_count_dict['value'] for value_count_dict in value_count_dicts]
            counts = [value_count_dict['count'] for value_count_dict in value_count_dicts]

        df = pd.DataFrame({
            "value": values,
            "count": counts,
        })

        if len(values) > 60:
            return None
        else:
            chart_pixel_width = (len(values) / 60.0) * 500
            if chart_pixel_width < 250:
                chart_pixel_width = 250
            chart_container_col_width = round((len(values) / 60.0) * 6)
            if chart_container_col_width < 4:
                chart_container_col_width = 4
            elif chart_container_col_width >= 5:
                chart_container_col_width = 6
            elif chart_container_col_width >= 4:
                chart_container_col_width = 5

        mark_bar_args = {}
        if len(values) == 1:
            mark_bar_args["size"] = 20

        bars = alt.Chart(df).mark_bar(**mark_bar_args).encode(
            y='count:Q',
            x="value:O",
            tooltip=["value", "count"]
        ).properties(height=400, width=chart_pixel_width, autosize="fit")

        chart = bars.to_json()

        new_block = RenderedGraphContent(**{
            "content_block_type": "graph",
            "header": RenderedStringTemplateContent(**{
                        "content_block_type": "string_template",
                        "string_template": {
                            "template": "Value Counts",
                            "tooltip": {
                                "content": "expect_column_distinct_values_to_be_in_set"
                            },
                            "tag": "h6"
                        }
                    }),
            "graph": chart,
            "styling": {
                "classes": ["col-" + str(chart_container_col_width), "mt-1"],
            }
        })

        return new_block

    @classmethod
    def _render_failed(cls, evrs):
        return ExceptionListContentBlockRenderer.render(evrs, include_column_name=False)

    @classmethod
    def _render_unrecognized(cls, evrs, content_blocks):
        unrendered_blocks = []
        new_block = None
        for evr in evrs:
            if evr.expectation_config.expectation_type not in [
                "expect_column_to_exist",
                "expect_column_values_to_be_of_type",
                "expect_column_values_to_be_in_set",
                "expect_column_unique_value_count_to_be_between",
                "expect_column_proportion_of_unique_values_to_be_between",
                "expect_column_values_to_not_be_null",
                "expect_column_max_to_be_between",
                "expect_column_mean_to_be_between",
                "expect_column_min_to_be_between"
            ]:
                new_block = TextContent(**{
                    "content_block_type": "text",
                    "text": []
                })
                new_block["content"].append("""
    <div class="alert alert-primary" role="alert">
        Warning! Unrendered EVR:<br/>
    <pre>"""+json.dumps(evr, indent=2)+"""</pre>
    </div>
                """)

        if new_block is not None:
            unrendered_blocks.append(new_block)

        # print(unrendered_blocks)
        content_blocks += unrendered_blocks


class ValidationResultsColumnSectionRenderer(ColumnSectionRenderer):

    def __init__(self, table_renderer=None):
        if table_renderer is None:
            table_renderer = {
                "class_name": "ValidationResultsTableContentBlockRenderer"
            }
        self._table_renderer = load_class(
            class_name=table_renderer.get("class_name"),
            #module_name=table_renderer.get("module_name", "great_expectations.render.renderer.content_block")
            module_name=table_renderer.get("module_name", "great_expectations.common")
        )

    @classmethod
    def _render_header(cls, validation_results):
        column = cls._get_column_name(validation_results)

        new_block = RenderedHeaderContent(**{
            "header": RenderedStringTemplateContent(**{
                "content_block_type": "string_template",
                "string_template": {
                    "template": convert_to_string_and_escape(column),
                    "tag": "h5",
                    "styling": {
                        "classes": ["m-0"]
                    }
                }
            }),
            "styling": {
                "classes": ["col-12", "p-0"],
                "header": {
                    "classes": ["alert", "alert-secondary"]
                }
            }
        })

        return validation_results, new_block

    def _render_table(self, validation_results):
        new_block = self._table_renderer.render(
            validation_results,
            include_column_name=False
        )

        return [], new_block

    def render(self, validation_results):
        column = self._get_column_name(validation_results)
        content_blocks = []
        remaining_evrs, content_block = self._render_header(validation_results)
        content_blocks.append(content_block)
        remaining_evrs, content_block = self._render_table(remaining_evrs)
        content_blocks.append(content_block)

        return RenderedSectionContent(**{
            "section_name": column,
            "content_blocks": content_blocks
        })


class ExpectationSuiteColumnSectionRenderer(ColumnSectionRenderer):

    def __init__(self, bullet_list_renderer=None):
        if bullet_list_renderer is None:
            bullet_list_renderer = {
                "class_name": "ExpectationSuiteBulletListContentBlockRenderer"
            }
        self._bullet_list_renderer = load_class(
            class_name=bullet_list_renderer.get("class_name"),
            module_name=bullet_list_renderer.get("module_name", "great_expectations.common")
        )

    @classmethod
    def _render_header(cls, expectations):
        column = cls._get_column_name(expectations)

        new_block = RenderedHeaderContent(**{
            "header": RenderedStringTemplateContent(**{
                "content_block_type": "string_template",
                "string_template": {
                    "template": convert_to_string_and_escape(column),
                    "tag": "h5",
                    "styling": {
                        "classes": ["m-0"]
                    }
                }
            }),
            "styling": {
                "classes": ["col-12"],
                "header": {
                    "classes": ["alert", "alert-secondary"]
                }
            }
        })

        return expectations, new_block

    def _render_bullet_list(self, expectations):

        new_block = self._bullet_list_renderer.render(
            expectations,
            include_column_name=False,
        )

        return [], new_block

    def render(self, expectations):
        column = self._get_column_name(expectations)

        content_blocks = []
        remaining_expectations, header_block = self._render_header(expectations)
        content_blocks.append(header_block)
        # remaining_expectations, content_blocks = cls._render_column_type(
        # remaining_expectations, content_blocks)
        remaining_expectations, bullet_block = self._render_bullet_list(remaining_expectations)
        content_blocks.append(bullet_block)

        # NOTE : Some render* functions return None so we filter them out
        populated_content_blocks = list(filter(None, content_blocks))
        return RenderedSectionContent(
            section_name=column,
            content_blocks=populated_content_blocks
        )


class ExpectationSuiteIdentifier(DataContextKey):

    def __init__(self, expectation_suite_name):
        super(ExpectationSuiteIdentifier, self).__init__()
        self._expectation_suite_name = expectation_suite_name

    @property
    def expectation_suite_name(self):
        return self._expectation_suite_name

    def to_tuple(self):
        return tuple(self.expectation_suite_name.split("."))

    def to_fixed_length_tuple(self):
        return self.expectation_suite_name,

    @classmethod
    def from_tuple(cls, tuple_):
        return cls(".".join(tuple_))

    @classmethod
    def from_fixed_length_tuple(cls, tuple_):
        return cls(expectation_suite_name=tuple_[0])


class Store(object):
    """A store is responsible for reading and writing Great Expectations objects
    to appropriate backends. It provides a generic API that the DataContext can
    use independently of any particular ORM and backend.

    An implementation of a store will generally need to define the following:
      - serialize
      - deserialize
      - _key_class (class of expected key type)

    All keys must have a to_tuple() method.
    """
    _key_class = DataContextKey

    def __init__(self, store_backend=None, runtime_environment=None):
        """Runtime environment may be necessary to instantiate store backend elements."""
        if store_backend is None:
            store_backend = {
                "class_name": "InMemoryStoreBackend"
            }
        logger.debug("Building store_backend.")
        self._store_backend = instantiate_class_from_config(
            config=store_backend,
            runtime_environment=runtime_environment or {},
            config_defaults={
                #"module_name": "great_expectations.data_context.store"
                "module_name": "great_expectations.common"
            }
        )
        if not isinstance(self._store_backend, StoreBackend):
            raise DataContextError("Invalid StoreBackend configuration: expected a StoreBackend instance.")
        self._use_fixed_length_key = self._store_backend.fixed_length_key

    def _validate_key(self, key):
        if not isinstance(key, self._key_class):
            raise TypeError("key must be an instance of %s, not %s" % (self._key_class.__name__, type(key)))

    # noinspection PyMethodMayBeStatic
    def serialize(self, key, value):
        return value

    # noinspection PyMethodMayBeStatic
    def key_to_tuple(self, key):
        if self._use_fixed_length_key:
            return key.to_fixed_length_tuple()
        return key.to_tuple()

    def tuple_to_key(self, tuple_):
        if self._use_fixed_length_key:
            return self._key_class.from_fixed_length_tuple(tuple_)
        return self._key_class.from_tuple(tuple_)

    # noinspection PyMethodMayBeStatic
    def deserialize(self, key, value):
        return value

    def get(self, key):
        self._validate_key(key)
        return self.deserialize(key, self._store_backend.get(self.key_to_tuple(key)))

    def set(self, key, value):
        self._validate_key(key)
        return self._store_backend.set(self.key_to_tuple(key), self.serialize(key, value))

    def list_keys(self):
        return [self.tuple_to_key(key) for key in self._store_backend.list_keys()]

    def has_key(self, key):
        return self._store_backend.has_key(key.to_tuple())


class MetricStore(Store):
    _key_class = ValidationMetricIdentifier

    def __init__(self, store_backend=None):
        if store_backend is not None:
            store_backend_module_name = store_backend.get("module_name", "great_expectations.data_context.store")
            store_backend_class_name = store_backend.get("class_name", "InMemoryStoreBackend")
            store_backend_class = load_class(store_backend_class_name, store_backend_module_name)

            if issubclass(store_backend_class, DatabaseStoreBackend):
                # Provide defaults for this common case
                store_backend["table_name"] = store_backend.get("table_name", "ge_metrics")
                store_backend["key_columns"] = store_backend.get(
                    "key_columns", [
                        "run_id",
                        "expectation_suite_identifier",
                        "metric_name",
                        "metric_kwargs_id",
                    ]
                )

        super(MetricStore, self).__init__(store_backend=store_backend)

    # noinspection PyMethodMayBeStatic
    def _validate_value(self, value):
        # Values must be json serializable since they must be inputs to expectation configurations
        ensure_json_serializable(value)

    def serialize(self, key, value):
        return json.dumps({"value": value})

    def deserialize(self, key, value):
        if value:
            return json.loads(value)["value"]


class ValidationsStore(Store):
    _key_class = ValidationResultIdentifier

    def __init__(self, store_backend=None, runtime_environment=None):
        self._expectationSuiteValidationResultSchema = ExpectationSuiteValidationResultSchema(strict=True)

        if store_backend is not None:
            store_backend_module_name = store_backend.get("module_name", "great_expectations.data_context.store")
            store_backend_class_name = store_backend.get("class_name", "InMemoryStoreBackend")
            store_backend_class = load_class(store_backend_class_name, store_backend_module_name)

            if issubclass(store_backend_class, TupleStoreBackend):
                # Provide defaults for this common case
                store_backend["filepath_suffix"] = store_backend.get("filepath_suffix", ".json")
            elif issubclass(store_backend_class, DatabaseStoreBackend):
                # Provide defaults for this common case
                store_backend["table_name"] = store_backend.get("table_name", "ge_validations_store")
                store_backend["key_columns"] = store_backend.get(
                    "key_columns", [
                        "expectation_suite_name",
                        "run_id",
                        "batch_identifier"
                    ]
                )
        super(ValidationsStore, self).__init__(store_backend=store_backend, runtime_environment=runtime_environment)

    def serialize(self, key, value):
        return self._expectationSuiteValidationResultSchema.dumps(value).data

    def deserialize(self, key, value):
        return self._expectationSuiteValidationResultSchema.loads(value).data


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


class EvaluationParameterStore(MetricStore):

    def __init__(self, store_backend=None):
        if store_backend is not None:
            store_backend_module_name = store_backend.get("module_name", "great_expectations.data_context.store")
            store_backend_class_name = store_backend.get("class_name", "InMemoryStoreBackend")
            store_backend_class = load_class(store_backend_class_name, store_backend_module_name)

            if issubclass(store_backend_class, DatabaseStoreBackend):
                # Provide defaults for this common case
                store_backend["table_name"] = store_backend.get("table_name", "ge_evaluation_parameters")
        super(EvaluationParameterStore, self).__init__(store_backend=store_backend)

    def get_bind_params(self, run_id):
        params = {}
        for k in self._store_backend.list_keys((run_id,)):
            key = self.tuple_to_key(k)
            params[key.to_evaluation_parameter_urn()] = self.get(key)
        return params


class StoreBackend(object):
    __metaclass__ = ABCMeta
    """A store backend acts as a key-value store that can accept tuples as keys, to abstract away
    reading and writing to a persistence layer.

    In general a StoreBackend implementation must provide implementations of:
      - _get
      - _set
      - list_keys
      - _has_key
    """

    def __init__(self, fixed_length_key=False):
        self._fixed_length_key = fixed_length_key

    @property
    def fixed_length_key(self):
        return self._fixed_length_key

    def get(self, key):
        self._validate_key(key)
        value = self._get(key)
        return value

    def set(self, key, value, **kwargs):
        self._validate_key(key)
        self._validate_value(value)
        # Allow the implementing setter to return something (e.g. a path used for its key)
        return self._set(key, value, **kwargs)

    def has_key(self, key):
        self._validate_key(key)
        return self._has_key(key)

    def get_url_for_key(self, key, protocol=None):
        raise NotImplementedError(
            "Store backend of type {0:s} does not have an implementation of get_url_for_key".format(
                type(self).__name__))

    def _validate_key(self, key):
        if isinstance(key, tuple):
            for key_element in key:
                if not isinstance(key_element, string_types):
                    raise TypeError(
                        "Elements within tuples passed as keys to {0} must be instances of {1}, not {2}".format(
                            self.__class__.__name__,
                            string_types,
                            type(key_element),
                        ))
        else:
            raise TypeError("Keys in {0} must be instances of {1}, not {2}".format(
                self.__class__.__name__,
                tuple,
                type(key),
            ))

    def _validate_value(self, value):
        pass

    @abstractmethod
    def _get(self, key):
        raise NotImplementedError

    @abstractmethod
    def _set(self, key, value, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def list_keys(self, prefix=()):
        raise NotImplementedError

    def _has_key(self, key):
        raise NotImplementedError


class InMemoryStoreBackend(StoreBackend):
    """Uses an in-memory dictionary as a store backend.
    """

    # noinspection PyUnusedLocal
    def __init__(self, runtime_environment=None, fixed_length_key=False):
        super(InMemoryStoreBackend, self).__init__(fixed_length_key=fixed_length_key)
        self._store = {}

    def _get(self, key):
        return self._store[key]

    def _set(self, key, value, **kwargs):
        self._store[key] = value

    def list_keys(self, prefix=()):
        return [key for key in self._store.keys() if key[:len(prefix)] == prefix]

    def _has_key(self, key):
        return key in self._store


class DatabaseStoreBackend(StoreBackend):

    def __init__(self, credentials, table_name, key_columns, fixed_length_key=True):
        super(DatabaseStoreBackend, self).__init__(fixed_length_key=fixed_length_key)
        if not sqlalchemy:
            raise ge_exceptions.DataContextError("ModuleNotFoundError: No module named 'sqlalchemy'")

        if not self.fixed_length_key:
            raise ValueError("DatabaseStoreBackend requires use of a fixed-length-key")

        meta = MetaData()
        self.key_columns = key_columns
        # Dynamically construct a SQLAlchemy table with the name and column names we'll use
        cols = []
        for column in key_columns:
            if column == "value":
                raise ValueError("'value' cannot be used as a key_element name")
            cols.append(Column(column, String, primary_key=True))

        cols.append(Column("value", String))
        self._table = Table(
            table_name, meta,
            *cols
        )

        drivername = credentials.pop("drivername")
        options = URL(drivername, **credentials)
        self.engine = create_engine(options)
        meta.create_all(self.engine)

    def _get(self, key):
        sel = select([column("value")]).select_from(self._table).where(
            and_(
                *[getattr(self._table.columns, key_col) == val for key_col, val in zip(self.key_columns, key)]
            )
        )
        res = self.engine.execute(sel).fetchone()
        if res:
            return self.engine.execute(sel).fetchone()[0]

    def _set(self, key, value, **kwargs):
        cols = {k: v for (k, v) in zip(self.key_columns, key)}
        cols["value"] = value
        ins = self._table.insert().values(**cols)
        self.engine.execute(ins)

    def _has_key(self, key):
        pass

    def list_keys(self, prefix=()):
        sel = select([column(col) for col in self.key_columns]).select_from(self._table).where(
            and_(
                *[getattr(self._table.columns, key_col) == val for key_col, val in
                  zip(self.key_columns[:len(prefix)], prefix)]
            )
        )
        return [tuple(row) for row in self.engine.execute(sel).fetchall()]


class TupleStoreBackend(StoreBackend):
    __metaclass__ = ABCMeta
    """
    If filepath_template is provided, the key to this StoreBackend abstract class must be a tuple with 
    fixed length equal to the number of unique components matching the regex r"{\d+}"

    For example, in the following template path: expectations/{0}/{1}/{2}/prefix-{2}.json, keys must have
    three components.
    """

    def __init__(self, filepath_template=None, filepath_prefix=None, filepath_suffix=None, forbidden_substrings=None,
                 platform_specific_separator=True, fixed_length_key=False):
        super(TupleStoreBackend, self).__init__(fixed_length_key=fixed_length_key)
        if forbidden_substrings is None:
            forbidden_substrings = ["/", "\\"]
        self.forbidden_substrings = forbidden_substrings
        self.platform_specific_separator = platform_specific_separator

        if filepath_template is not None and filepath_suffix is not None:
            raise ValueError("filepath_suffix may only be used when filepath_template is None")

        self.filepath_template = filepath_template
        if filepath_prefix and len(filepath_prefix) > 0:
            # Validate that the filepath prefix does not end with a forbidden substring
            if filepath_prefix[-1] in self.forbidden_substrings:
                raise StoreBackendError("Unable to initialize TupleStoreBackend: filepath_prefix may not end with a "
                                        "forbidden substring. Current forbidden substrings are " +
                                        str(forbidden_substrings))
        self.filepath_prefix = filepath_prefix
        self.filepath_suffix = filepath_suffix

        if filepath_template is not None:
            # key length is the number of unique values to be substituted in the filepath_template
            self.key_length = len(
                set(
                    re.findall(r"{\d+}", filepath_template)
                )
            )

            self.verify_that_key_to_filepath_operation_is_reversible()
            self._fixed_length_key = True

    def _validate_key(self, key):
        super(TupleStoreBackend, self)._validate_key(key)

        for key_element in key:
            for substring in self.forbidden_substrings:
                if substring in key_element:
                    raise ValueError("Keys in {0} must not contain substrings in {1} : {2}".format(
                        self.__class__.__name__,
                        self.forbidden_substrings,
                        key,
                    ))

    def _validate_value(self, value):
        if not isinstance(value, string_types) and not isinstance(value, bytes):
            raise TypeError("Values in {0} must be instances of {1} or {2}, not {3}".format(
                self.__class__.__name__,
                string_types,
                bytes,
                type(value),
            ))

    def _convert_key_to_filepath(self, key):
        # NOTE: This method uses a hard-coded forward slash as a separator,
        # and then replaces that with a platform-specific separator if requested (the default)
        self._validate_key(key)
        if self.filepath_template:
            converted_string = self.filepath_template.format(*list(key))
        else:
            converted_string = '/'.join(key)

        if self.filepath_prefix:
            converted_string = self.filepath_prefix + "/" + converted_string
        if self.filepath_suffix:
            converted_string += self.filepath_suffix
        if self.platform_specific_separator:
            converted_string = os.path.normpath(converted_string)

        return converted_string

    def _convert_filepath_to_key(self, filepath):
        if self.platform_specific_separator:
            filepath = os.path.normpath(filepath)

        if self.filepath_prefix:
            if not filepath.startswith(self.filepath_prefix) and len(filepath) >= len(self.filepath_prefix) + 1:
                # If filepath_prefix is set, we expect that it is the first component of a valid filepath.
                raise ValueError("filepath must start with the filepath_prefix when one is set by the store_backend")
            else:
                # Remove the prefix before processing
                # Also remove the separator that was added, which may have been platform-dependent
                filepath = filepath[len(self.filepath_prefix) + 1:]

        if self.filepath_suffix:
            if not filepath.endswith(self.filepath_suffix):
                # If filepath_suffix is set, we expect that it is the last component of a valid filepath.
                raise ValueError("filepath must end with the filepath_suffix when one is set by the store_backend")
            else:
                # Remove the suffix before processing
                filepath = filepath[:-len(self.filepath_suffix)]

        if self.filepath_template:
            # filepath_template is always specified with forward slashes, but it is then
            # used to (1) dynamically construct and evaluate a regex, and (2) split the provided (observed) filepath
            if self.platform_specific_separator:
                filepath_template = os.path.join(*self.filepath_template.split('/'))
                filepath_template = filepath_template.replace('\\', '\\\\')
            else:
                filepath_template = self.filepath_template

            # Convert the template to a regex
            indexed_string_substitutions = re.findall(r"{\d+}", filepath_template)
            tuple_index_list = ["(?P<tuple_index_{0}>.*)".format(i, ) for i in range(len(indexed_string_substitutions))]
            intermediate_filepath_regex = re.sub(
                r"{\d+}",
                lambda m, r=iter(tuple_index_list): next(r),
                filepath_template
            )
            filepath_regex = intermediate_filepath_regex.format(*tuple_index_list)

            # Apply the regex to the filepath
            matches = re.compile(filepath_regex).match(filepath)
            if matches is None:
                return None

            # Map key elements into the appropriate parts of the tuple
            new_key = [None] * self.key_length
            for i in range(len(tuple_index_list)):
                tuple_index = int(re.search(r'\d+', indexed_string_substitutions[i]).group(0))
                key_element = matches.group('tuple_index_' + str(i))
                new_key[tuple_index] = key_element

            new_key = tuple(new_key)
        else:
            filepath = os.path.normpath(filepath)
            new_key = tuple(filepath.split(os.sep))

        return new_key

    def verify_that_key_to_filepath_operation_is_reversible(self):
        def get_random_hex(size=4):
            return "".join([random.choice(list("ABCDEF0123456789")) for _ in range(size)])

        key = tuple([get_random_hex() for _ in range(self.key_length)])
        filepath = self._convert_key_to_filepath(key)
        new_key = self._convert_filepath_to_key(filepath)
        if key != new_key:
            raise ValueError(
                "filepath template {0} for class {1} is not reversible for a tuple of length {2}. "
                "Have you included all elements in the key tuple?".format(
                    self.filepath_template,
                    self.__class__.__name__,
                    self.key_length,
                ))


class TupleFilesystemStoreBackend(TupleStoreBackend):
    """Uses a local filepath as a store.

    The key to this StoreBackend must be a tuple with fixed length based on the filepath_template,
    or a variable-length tuple may be used and returned with an optional filepath_suffix (to be) added.
    The filepath_template is a string template used to convert the key to a filepath.
    """

    def __init__(self,
                 base_directory,
                 filepath_template=None,
                 filepath_prefix=None,
                 filepath_suffix=None,
                 forbidden_substrings=None,
                 platform_specific_separator=True,
                 root_directory=None,
                 fixed_length_key=False):
        super(TupleFilesystemStoreBackend, self).__init__(
            filepath_template=filepath_template,
            filepath_prefix=filepath_prefix,
            filepath_suffix=filepath_suffix,
            forbidden_substrings=forbidden_substrings,
            platform_specific_separator=platform_specific_separator,
            fixed_length_key=fixed_length_key
        )
        if os.path.isabs(base_directory):
            self.full_base_directory = base_directory
        else:
            if root_directory is None:
                raise ValueError("base_directory must be an absolute path if root_directory is not provided")
            elif not os.path.isabs(root_directory):
                raise ValueError("root_directory must be an absolute path. Got {0} instead.".format(root_directory))
            else:
                self.full_base_directory = os.path.join(root_directory, base_directory)

        safe_mmkdir(str(os.path.dirname(self.full_base_directory)))

    def _get(self, key):
        filepath = os.path.join(
            self.full_base_directory,
            self._convert_key_to_filepath(key)
        )
        with open(filepath, 'r') as infile:
            return infile.read()

    def _set(self, key, value, **kwargs):
        if not isinstance(key, tuple):
            key = key.to_tuple()
        filepath = os.path.join(
            self.full_base_directory,
            self._convert_key_to_filepath(key)
        )
        path, filename = os.path.split(filepath)

        safe_mmkdir(str(path))
        with open(filepath, "wb") as outfile:
            if isinstance(value, string_types):
                # Following try/except is to support py2, since both str and bytes objects pass above condition
                try:
                    outfile.write(value.encode("utf-8"))
                except UnicodeDecodeError:
                    outfile.write(value)
            else:
                outfile.write(value)
        return filepath

    def list_keys(self, prefix=()):
        key_list = []
        for root, dirs, files in os.walk(os.path.join(self.full_base_directory, *prefix)):
            for file_ in files:
                full_path, file_name = os.path.split(os.path.join(root, file_))
                relative_path = os.path.relpath(
                    full_path,
                    self.full_base_directory,
                )
                if relative_path == ".":
                    filepath = file_name
                else:
                    filepath = os.path.join(
                        relative_path,
                        file_name
                    )

                if self.filepath_prefix and not filepath.startswith(self.filepath_prefix):
                    continue
                elif self.filepath_suffix and not filepath.endswith(self.filepath_suffix):
                    continue
                else:
                    key = self._convert_filepath_to_key(filepath)
                if key:
                    key_list.append(key)

        return key_list

    def get_url_for_key(self, key, protocol=None):
        path = self._convert_key_to_filepath(key)
        full_path = os.path.join(self.full_base_directory, path)
        if protocol is None:
            protocol = "file:"
        url = protocol + "//" + full_path

        return url

    def _has_key(self, key):
        return os.path.isfile(os.path.join(self.full_base_directory, self._convert_key_to_filepath(key)))


class TupleS3StoreBackend(TupleStoreBackend):
    """
    Uses an S3 bucket as a store.

    The key to this StoreBackend must be a tuple with fixed length based on the filepath_template,
    or a variable-length tuple may be used and returned with an optional filepath_suffix (to be) added.
    The filepath_template is a string template used to convert the key to a filepath.
    """

    def __init__(
            self,
            bucket,
            prefix="",
            filepath_template=None,
            filepath_prefix=None,
            filepath_suffix=None,
            forbidden_substrings=None,
            platform_specific_separator=False,
            fixed_length_key=False
    ):
        super(TupleS3StoreBackend, self).__init__(
            filepath_template=filepath_template,
            filepath_prefix=filepath_prefix,
            filepath_suffix=filepath_suffix,
            forbidden_substrings=forbidden_substrings,
            platform_specific_separator=platform_specific_separator,
            fixed_length_key=fixed_length_key
        )
        self.bucket = bucket
        self.prefix = prefix

    def _get(self, key):
        s3_object_key = os.path.join(
            self.prefix,
            self._convert_key_to_filepath(key)
        )

        import boto3
        s3 = boto3.client('s3')
        s3_response_object = s3.get_object(Bucket=self.bucket, Key=s3_object_key)
        return s3_response_object['Body'].read().decode(s3_response_object.get("ContentEncoding", 'utf-8'))

    def _set(self, key, value, content_encoding='utf-8', content_type='application/json'):
        s3_object_key = os.path.join(
            self.prefix,
            self._convert_key_to_filepath(key)
        )

        import boto3
        s3 = boto3.resource('s3')
        result_s3 = s3.Object(self.bucket, s3_object_key)
        if isinstance(value, string_types):
            # Following try/except is to support py2, since both str and bytes objects pass above condition
            try:
                result_s3.put(Body=value.encode(content_encoding), ContentEncoding=content_encoding,
                              ContentType=content_type)
            except TypeError:
                result_s3.put(Body=value, ContentType=content_type)
        else:
            result_s3.put(Body=value, ContentType=content_type)
        return s3_object_key

    def list_keys(self):
        key_list = []

        import boto3
        s3 = boto3.client('s3')

        s3_objects = s3.list_objects(Bucket=self.bucket, Prefix=self.prefix)
        if "Contents" in s3_objects:
            objects = s3_objects["Contents"]
        elif "CommonPrefixes" in s3_objects:
            logger.warning("TupleS3StoreBackend returned CommonPrefixes, but delimiter should not have been set.")
            objects = []
        else:
            # No objects found in store
            objects = []

        for s3_object_info in objects:
            s3_object_key = s3_object_info['Key']
            s3_object_key = os.path.relpath(
                s3_object_key,
                self.prefix,
            )
            if self.filepath_prefix and not s3_object_key.startswith(self.filepath_prefix):
                # There can be other keys located in the same bucket; they are *not* our keys
                continue

            key = self._convert_filepath_to_key(s3_object_key)
            if key:
                key_list.append(key)

        return key_list

    def get_url_for_key(self, key, protocol=None):
        import boto3

        location = boto3.client('s3').get_bucket_location(Bucket=self.bucket)['LocationConstraint']
        if location is None:
            location = "s3"
        else:
            location = "s3-" + location
        s3_key = self._convert_key_to_filepath(key)
        return "https://%s.amazonaws.com/%s/%s%s" % (location, self.bucket, self.prefix, s3_key)

    def _has_key(self, key):
        all_keys = self.list_keys()
        return key in all_keys


class TupleGCSStoreBackend(TupleStoreBackend):
    """
    Uses a GCS bucket as a store.

    The key to this StoreBackend must be a tuple with fixed length based on the filepath_template,
    or a variable-length tuple may be used and returned with an optional filepath_suffix (to be) added.

    The filepath_template is a string template used to convert the key to a filepath.
    """

    def __init__(
            self,
            bucket,
            prefix,
            project,
            filepath_template=None,
            filepath_prefix=None,
            filepath_suffix=None,
            forbidden_substrings=None,
            platform_specific_separator=False,
            fixed_length_key=False
    ):
        super(TupleGCSStoreBackend, self).__init__(
            filepath_template=filepath_template,
            filepath_prefix=filepath_prefix,
            filepath_suffix=filepath_suffix,
            forbidden_substrings=forbidden_substrings,
            platform_specific_separator=platform_specific_separator,
            fixed_length_key=fixed_length_key
        )
        self.bucket = bucket
        self.prefix = prefix
        self.project = project

    def _get(self, key):
        gcs_object_key = os.path.join(
            self.prefix,
            self._convert_key_to_filepath(key)
        )

        from google.cloud import storage
        gcs = storage.Client(project=self.project)
        bucket = gcs.get_bucket(self.bucket)
        gcs_response_object = bucket.get_blob(gcs_object_key)
        return gcs_response_object.download_as_string().decode("utf-8")

    def _set(self, key, value, content_encoding='utf-8', content_type='application/json'):
        gcs_object_key = os.path.join(
            self.prefix,
            self._convert_key_to_filepath(key)
        )

        from google.cloud import storage
        gcs = storage.Client(project=self.project)
        bucket = gcs.get_bucket(self.bucket)
        blob = bucket.blob(gcs_object_key)
        if isinstance(value, string_types):
            # Following try/except is to support py2, since both str and bytes objects pass above condition
            try:
                blob.upload_from_string(value.encode(content_encoding), content_encoding=content_encoding,
                                        content_type=content_type)
            except TypeError:
                blob.upload_from_string(value, content_type=content_type)
        else:
            blob.upload_from_string(value, content_type=content_type)
        return gcs_object_key

    def list_keys(self):
        key_list = []

        from google.cloud import storage
        gcs = storage.Client(self.project)

        for blob in gcs.list_blobs(self.bucket, prefix=self.prefix):
            gcs_object_name = blob.name
            gcs_object_key = os.path.relpath(
                gcs_object_name,
                self.prefix,
            )

            key = self._convert_filepath_to_key(gcs_object_key)
            if key:
                key_list.append(key)

        return key_list

    def _has_key(self, key):
        all_keys = self.list_keys()
        return key in all_keys


class ValidationOperator(object):
    """
    The base class of all validation operators.

    It defines the signature of the public run method - this is the only
    contract re operators' API. Everything else is up to the implementors
    of validation operator classes that will be the descendants of this base class.
    """

    def run(self, assets_to_validate, run_id):
        raise NotImplementedError


class ActionListValidationOperator(ValidationOperator):
    """
    ActionListValidationOperator is a validation operator
    that validates each batch in the list that is passed to its run
    method and then invokes a list of configured actions on every
    validation result.

    A user can configure the list of actions to invoke.

    Each action in the list must be an instance of ValidationAction
    class (or its descendants).

    Below is an example of this operator's configuration::

        action_list_operator:
            class_name: ActionListValidationOperator
            action_list:
              - name: store_validation_result
                action:
                  class_name: StoreValidationResultAction
                  target_store_name: validations_store
              - name: store_evaluation_params
                action:
                  class_name: StoreEvaluationParametersAction
                  target_store_name: evaluation_parameter_store
              - name: send_slack_notification_on_validation_result
                action:
                  class_name: SlackNotificationAction
                  # put the actual webhook URL in the uncommitted/config_variables.yml file
                  slack_webhook: ${validation_notification_slack_webhook}
                 notify_on: all # possible values: "all", "failure", "success"
                  renderer:
                    module_name: great_expectations.render.renderer.slack_renderer
                    class_name: SlackRenderer
    """

    def __init__(self, data_context, action_list):
        self.data_context = data_context

        self.action_list = action_list
        self.actions = {}
        for action_config in action_list:
            assert isinstance(action_config, dict)
            #NOTE: Eugene: 2019-09-23: need a better way to validate an action config:
            if not set(action_config.keys()) == {"name", "action"}:
                raise KeyError('Action config keys must be ("name", "action"). Instead got {}'.format(action_config.keys()))

            new_action = instantiate_class_from_config(
                config=action_config["action"],
                runtime_environment={
                    "data_context": self.data_context,
                },
                config_defaults={
                    #"module_name": "great_expectations.validation_operators"
                    "module_name": "great_expectations.common"
                }
            )
            self.actions[action_config["name"]] = new_action

    def _build_batch_from_item(self, item):
        """Internal helper method to take an asset to validate, which can be either:
          (1) a DataAsset; or
          (2) a tuple of data_asset_name, expectation_suite_name, and batch_kwargs (suitable for passing to get_batch)

        Args:
            item: The item to convert to a batch (see above)

        Returns:
            A batch of data

        """
        if not isinstance(item, DataAsset):
            if not (isinstance(item, tuple) and len(item) == 2 and isinstance(item[0], dict) and isinstance(
                    item[1], string_types)):
                raise ValueError("Unable to build batch from item.")
            batch = self.data_context.get_batch(
                batch_kwargs=item[0],
                expectation_suite_name=item[1]
            )
        else:
            batch = item

        return batch

    def run(self, assets_to_validate, run_id):
        result_object = {
            "success": None,
            "details": {}
        }

        for item in assets_to_validate:
            batch = self._build_batch_from_item(item)
            expectation_suite_identifier = ExpectationSuiteIdentifier(
                expectation_suite_name=batch._expectation_suite.expectation_suite_name
            )
            # validation_result_id = ValidationResultIdentifier(
            #     batch_identifier=BatchIdentifier(batch.batch_id),
            #     expectation_suite_identifier=expectation_suite_identifier,
            #     run_id=run_id,
            # )
            result_object["details"][expectation_suite_identifier] = {}
            batch_validation_result = batch.validate(run_id=run_id, result_format="SUMMARY")
            result_object["details"][expectation_suite_identifier]["validation_result"] = batch_validation_result
            batch_actions_results = self._run_actions(batch, expectation_suite_identifier, batch._expectation_suite, batch_validation_result, run_id)
            result_object["details"][expectation_suite_identifier]["actions_results"] = batch_actions_results

        result_object["success"] = all([val["validation_result"].success for val in result_object["details"].values()])

        return result_object

    def _run_actions(self, batch, expectation_suite_identifier, expectation_suite, batch_validation_result, run_id):
        """
        Runs all actions configured for this operator on the result of validating one
        batch against one expectation suite.

        If an action fails with an exception, the method does not continue.

        :param batch:
        :param expectation_suite:
        :param batch_validation_result:
        :param run_id:
        :return: a dictionary: {action name -> result returned by the action}
        """
        batch_actions_results = {}
        for action in self.action_list:
            # NOTE: Eugene: 2019-09-23: log the info about the batch and the expectation suite
            logger.debug("Processing validation action with name {}".format(action["name"]))

            validation_result_id = ValidationResultIdentifier(
                expectation_suite_identifier=expectation_suite_identifier,
                run_id=run_id,
                batch_identifier=batch.batch_id
            )
            try:
                action_result = self.actions[action["name"]].run(
                                                validation_result_suite_identifier=validation_result_id,
                                                validation_result_suite=batch_validation_result,
                                                data_asset=batch
                )

                batch_actions_results[action["name"]] = {} if action_result is None else action_result
            except Exception as e:
                logger.exception("Error running action with name {}".format(action["name"]))
                exception_traceback = traceback.format_exc()
                exception_message = "[ERROR] {}: {}".format(type(e).__name__, str(e)) + "ACTION_NAME: '{}'".format(action["name"]) + " [TRACEBACK] " + exception_traceback + " [ADDITIONAL INFORMATION] " + "Error running action with name {}".format(action["name"])
                raise Exception(exception_message)

                #raise e

        return batch_actions_results

        result_object = {}

        for item in assets_to_validate:
            batch = self._build_batch_from_item(item)
            expectation_suite_identifier = ExpectationSuiteIdentifier(
                expectation_suite_name=batch.expectation_suite_name
            )
            validation_result_id = ValidationResultIdentifier(
                expectation_suite_identifier=expectation_suite_identifier,
                run_id=run_id,
                batch_identifier=batch.batch_id
            )
            result_object[validation_result_id] = {}
            batch_validation_result = batch.validate(result_format="SUMMARY")
            result_object[validation_result_id]["validation_result"] = batch_validation_result
            batch_actions_results = self._run_actions(batch, batch._expectation_suite, batch_validation_result, run_id)
            result_object[validation_result_id]["actions_results"] = batch_actions_results

        # NOTE: Eugene: 2019-09-24: Need to define this result object. Discussion required!
        return result_object


class WarningAndFailureExpectationSuitesValidationOperator(ActionListValidationOperator):
    """WarningAndFailureExpectationSuitesValidationOperator is a validation operator
    that accepts a list batches of data assets (or the information necessary to fetch these batches).
    The operator retrieves 2 expectation suites for each data asset/batch - one containing
    the critical expectations ("failure") and the other containing non-critical expectations
    ("warning"). By default, the operator assumes that the first is called "failure" and the
    second is called "warning", but "base_expectation_suite_name" attribute can be specified
    in the operator's configuration to make sure it searched for "{base_expectation_suite_name}.failure"
    and {base_expectation_suite_name}.warning" expectation suites for each data asset.

    The operator validates each batch against its "failure" and "warning" expectation suites and
    invokes a list of actions on every validation result.

    The list of these actions is specified in the operator's configuration

    Each action in the list must be an instance of ValidationAction
    class (or its descendants).

    The operator sends a Slack notification (if "slack_webhook" is present in its
    config). The "notify_on" config property controls whether the notification
    should be sent only in the case of failure ("failure"), only in the case
    of success ("success"), or always ("all").

    Below is an example of this operator's configuration::


        run_warning_and_failure_expectation_suites:
            class_name: WarningAndFailureExpectationSuitesValidationOperator
            # put the actual webhook URL in the uncommitted/config_variables.yml file
            slack_webhook: ${validation_notification_slack_webhook}
            action_list:
              - name: store_validation_result
                action:
                  class_name: StoreValidationResultAction
                  target_store_name: validations_store
              - name: store_evaluation_params
                action:
                  class_name: StoreEvaluationParametersAction
                  target_store_name: evaluation_parameter_store


    The operator returns an object that looks like the example below.

    The value of "success" is True if no critical expectation suites ("failure")
    failed to validate (non-critial ("warning") expectation suites
    are allowed to fail without affecting the success status of the run::


        {
            "batch_identifiers": [list, of, batch, identifiers],
            "success": True/False,
            "failure": {
                "expectation_suite_identifier": {
                    "validation_result": validation_result,
                    "action_results": {
                        "action name": "action result object"
                    }
                }
            },
            "warning": {
                "expectation_suite_identifier": {
                    "validation_result": validation_result,
                    "action_results": {
                        "action name": "action result object"
                    }
                }
            }
        }

    """


    def __init__(self,
        data_context,
        action_list,
        base_expectation_suite_name=None,
        expectation_suite_name_suffixes=[".failure", ".warning"],
        stop_on_first_error=False,
        slack_webhook=None,
        notify_on="all"
    ):
        super(WarningAndFailureExpectationSuitesValidationOperator, self).__init__(
            data_context,
            action_list,
        )

        self.stop_on_first_error = stop_on_first_error
        self.base_expectation_suite_name = base_expectation_suite_name

        assert len(expectation_suite_name_suffixes) == 2
        for suffix in expectation_suite_name_suffixes:
            assert isinstance(suffix, string_types)
        self.expectation_suite_name_suffixes = expectation_suite_name_suffixes
        
        self.slack_webhook = slack_webhook
        self.notify_on = notify_on

    def _build_slack_query(self, run_return_obj):
        timestamp = datetime.datetime.strftime(datetime.datetime.now(), "%x %X")
        success = run_return_obj.get("success")
        status_text = "Success :tada:" if success else "Failed :x:"
        run_id = run_return_obj.get("run_id")
        batch_identifiers = run_return_obj.get("batch_identifiers")
        failed_data_assets = []

        if run_return_obj.get("failure"):
            failed_data_assets = [
                validation_result_identifier.expectation_suite_identifier.expectation_suite_name + "-" +
                validation_result_identifier.batch_identifier
                for
                validation_result_identifier, value in run_return_obj.get("failure").items()
                if not value["validation_result"].success
            ]
    
        title_block = {
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": "*FailureVsWarning Validation Operator Completed.*",
            },
        }
        divider_block = {
            "type": "divider"
        }

        query = {"blocks": [divider_block, title_block, divider_block]}

        status_element = {
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": "*Status*: {}".format(status_text)},
        }
        query["blocks"].append(status_element)
        
        batch_identifiers_element = {
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": "*Batch Id List:* {}".format(batch_identifiers)
            }
        }
        query["blocks"].append(batch_identifiers_element)
    
        if not success:
            failed_data_assets_element = {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": "*Failed Batches:* {}".format(failed_data_assets)
                }
            }
            query["blocks"].append(failed_data_assets_element)
    
        run_id_element = {
            "type": "section",
            "text":
                {
                    "type": "mrkdwn",
                    "text": "*Run ID:* {}".format(run_id),
                }
            ,
        }
        query["blocks"].append(run_id_element)
        
        timestamp_element = {
            "type": "section",
            "text":
                {
                    "type": "mrkdwn",
                    "text": "*Timestamp:* {}".format(timestamp),
                }
            ,
        }
        query["blocks"].append(timestamp_element)
        query["blocks"].append(divider_block)

        documentation_url = "https://docs.greatexpectations.io/en/latest/reference/validation_operators/warning_and_failure_expectation_suites_validation_operator.html"
        footer_section = {
            "type": "context",
            "elements": [
                {
                    "type": "mrkdwn",
                    "text": "Learn about FailureVsWarning Validation Operators at {}".format(documentation_url),
                }
            ],
        }
        query["blocks"].append(footer_section)
        
        return query

    def run(self, assets_to_validate, run_id, base_expectation_suite_name=None):
        if base_expectation_suite_name is None:
            if self.base_expectation_suite_name is None:
                raise ValueError("base_expectation_suite_name must be configured in the validation operator or passed at runtime")
            base_expectation_suite_name = self.base_expectation_suite_name

        return_obj = {
            "batch_identifiers": [],
            "success": None,
            "failure": {},
            "warning": {},
            "run_id": run_id
        }

        for item in assets_to_validate:
            batch = self._build_batch_from_item(item)

            batch_id = batch.batch_id
            run_id = run_id

            assert not batch_id is None
            assert not run_id is None

            return_obj["batch_identifiers"].append(batch_id)

            failure_expectation_suite_identifier = ExpectationSuiteIdentifier(
                expectation_suite_name=base_expectation_suite_name + self.expectation_suite_name_suffixes[0]
            )

            failure_validation_result_id = ValidationResultIdentifier(
                expectation_suite_identifier=failure_expectation_suite_identifier,
                run_id=run_id,
                batch_identifier=batch_id
            )

            failure_expectation_suite = None
            try:
                failure_expectation_suite = self.data_context.stores[self.data_context.expectations_store_name].get(
                    failure_expectation_suite_identifier
                )

            # NOTE : Abe 2019/09/17 : I'm concerned that this may be too permissive, since
            # it will catch any error in the Store, not just KeyErrors. In the longer term, a better
            # solution will be to have the Stores catch other known errors and raise KeyErrors,
            # so that methods like this can catch and handle a single error type.
            except Exception as e:
                logger.debug("Failure expectation suite not found: {}".format(failure_expectation_suite_identifier))

            if failure_expectation_suite:
                return_obj["failure"][failure_validation_result_id] = {}
                failure_validation_result = batch.validate(failure_expectation_suite, result_format="SUMMARY")
                return_obj["failure"][failure_validation_result_id]["validation_result"] = failure_validation_result
                failure_actions_results = self._run_actions(
                    batch,
                    failure_expectation_suite_identifier,
                    failure_expectation_suite,
                    failure_validation_result,
                    run_id
                )
                return_obj["failure"][failure_validation_result_id]["actions_results"] = failure_actions_results

                if not failure_validation_result.success and self.stop_on_first_error:
                    break


            warning_expectation_suite_identifier = ExpectationSuiteIdentifier(
                expectation_suite_name=base_expectation_suite_name + self.expectation_suite_name_suffixes[1]
            )

            warning_validation_result_id = ValidationResultIdentifier(
                expectation_suite_identifier=warning_expectation_suite_identifier,
                run_id=run_id,
                batch_identifier=batch.batch_id
            )

            warning_expectation_suite = None
            try:
                warning_expectation_suite = self.data_context.stores[self.data_context.expectations_store_name].get(
                    warning_expectation_suite_identifier
                )
            except Exception as e:
                logger.debug("Warning expectation suite not found: {}".format(warning_expectation_suite_identifier))

            if warning_expectation_suite:
                return_obj["warning"][warning_validation_result_id] = {}
                warning_validation_result = batch.validate(warning_expectation_suite, result_format="SUMMARY")
                return_obj["warning"][warning_validation_result_id]["validation_result"] = warning_validation_result
                warning_actions_results = self._run_actions(
                    batch,
                    warning_expectation_suite_identifier,
                    warning_expectation_suite,
                    warning_validation_result,
                    run_id
                )
                return_obj["warning"][warning_validation_result_id]["actions_results"] = warning_actions_results

        return_obj["success"] = all([val["validation_result"].success for val in return_obj["failure"].values()])

        # NOTE: Eugene: 2019-09-24: Update the data doc sites?
        if self.slack_webhook:
            if self.notify_on == "all" or \
                    self.notify_on == "success" and return_obj.success or \
                    self.notify_on == "failure" and not return_obj.success:
                slack_query = self._build_slack_query(run_return_obj=return_obj)
                send_slack_notification(query=slack_query, slack_webhook=self.slack_webhook)

        return return_obj


class ValidationAction(object):
    """
    This is the base class for all actions that act on validation results
    and are aware of a data context namespace structure.

    The data context is passed to this class in its constructor.
    """

    def __init__(self, data_context):
        self.data_context = data_context

    def run(self, validation_result_suite, validation_result_suite_identifier, data_asset, **kwargs):
        """

        :param validation_result_suite:
        :param validation_result_suite_identifier:
        :param data_asset:
        :param: kwargs - any additional arguments the child might use
        :return:
        """
        return self._run(validation_result_suite, validation_result_suite_identifier, data_asset, **kwargs)

    def _run(self, validation_result_suite, validation_result_suite_identifier, data_asset):
        return NotImplementedError


class NoOpAction(ValidationAction):

    def __init__(self, data_context,):
        super(NoOpAction, self).__init__(data_context)
    
    def _run(self, validation_result_suite, validation_result_suite_identifier, data_asset):
        print("Happily doing nothing")


class SlackNotificationAction(ValidationAction):
    """
    SlackNotificationAction sends a Slack notification to a given webhook.
    
    Example config:
    {
        "renderer": {
            #"module_name": "great_expectations.render.renderer.slack_renderer",
            "module_name": "great_expectations.common",
            "class_name": "SlackRenderer",
        },
        "slack_webhook": "https://example_webhook",
        "notify_on": "all"
    }
    """
    def __init__(
            self,
            data_context,
            renderer,
            slack_webhook,
            notify_on="all",
    ):
        """Construct a SlackNotificationAction

        Args:
            data_context:
            renderer: dictionary specifying the renderer used to generate a query consumable by Slack API, for example:
                {
                   #"module_name": "great_expectations.render.renderer.slack_renderer",
                   "module_name": "great_expectations.common",
                   "class_name": "SlackRenderer",
               }
            slack_webhook: incoming Slack webhook to which to send notification
            notify_on: "all", "failure", "success" - specifies validation status that will trigger notification
        """
        super(SlackNotificationAction, self).__init__(data_context)
        self.renderer = instantiate_class_from_config(
            config=renderer,
            runtime_environment={},
            config_defaults={},
        )
        self.slack_webhook = slack_webhook
        assert slack_webhook, "No Slack webhook found in action config."
        self.notify_on = notify_on
        
    def _run(self, validation_result_suite, validation_result_suite_identifier, data_asset=None):
        logger.debug("SlackNotificationAction.run")
    
        if validation_result_suite is None:
            return
        
        if not isinstance(validation_result_suite_identifier, ValidationResultIdentifier):
            raise TypeError("validation_result_suite_id must be of type ValidationResultIdentifier, not {0}".format(
                type(validation_result_suite_identifier)
            ))

        validation_success = validation_result_suite.success
        
        if self.notify_on == "all" or \
                self.notify_on == "success" and validation_success or \
                self.notify_on == "failure" and not validation_success:
            query = self.renderer.render(validation_result_suite)
            return send_slack_notification(query, slack_webhook=self.slack_webhook)
        else:
            return


class StoreValidationResultAction(ValidationAction):
    """
    StoreValidationResultAction stores a validation result in the ValidationsStore.
    """

    def __init__(self,
                 data_context,
                 target_store_name=None,
                 ):
        """

        :param data_context: data context
        :param target_store_name: the name of the param_store in the data context which
                should be used to param_store the validation result
        """

        super(StoreValidationResultAction, self).__init__(data_context)
        if target_store_name is None:
            self.target_store = data_context.stores[data_context.validations_store_name]
        else:
            self.target_store = data_context.stores[target_store_name]

    def _run(self, validation_result_suite, validation_result_suite_identifier, data_asset):
        logger.debug("StoreValidationResultAction.run")

        if validation_result_suite is None:
            return

        if not isinstance(validation_result_suite_identifier, ValidationResultIdentifier):
            raise TypeError("validation_result_id must be of type ValidationResultIdentifier, not {0}".format(
                type(validation_result_suite_identifier)
            ))

        self.target_store.set(validation_result_suite_identifier, validation_result_suite)


class StoreEvaluationParametersAction(ValidationAction):
    """
    StoreEvaluationParametersAction is a namespeace-aware validation action that
    extracts evaluation parameters from a validation result and stores them in the param_store
    configured for this action.

    Evaluation parameters allow expectations to refer to statistics/metrics computed
    in the process of validating other prior expectations.
    """

    def __init__(self, data_context, target_store_name=None):
        """

        Args:
            data_context: data context
            target_store_name: the name of the store in the data context which
                should be used to store the evaluation parameters
        """
        super(StoreEvaluationParametersAction, self).__init__(data_context)

        if target_store_name is None:
            self.target_store = data_context.evaluation_parameter_store
        else:
            self.target_store = data_context.stores[target_store_name]

    def _run(self, validation_result_suite, validation_result_suite_identifier, data_asset):
        logger.debug("StoreEvaluationParametersAction.run")

        if validation_result_suite is None:
            return

        if not isinstance(validation_result_suite_identifier, ValidationResultIdentifier):
            raise TypeError("validation_result_id must be of type ValidationResultIdentifier, not {0}".format(
                type(validation_result_suite_identifier)
            ))

        self.data_context.store_evaluation_parameters(validation_result_suite)


class StoreMetricsAction(ValidationAction):
    def __init__(self, data_context, requested_metrics, target_store_name="metrics_store"):
        """

        Args:
            data_context: data context
            requested_metrics: dictionary of metrics to store. Dictionary should have the following structure:

                expectation_suite_name:
                    metric_name:
                        - metric_kwargs_id

                You may use "*" to denote that any expectation suite should match.
            target_store_name: the name of the store in the data context which
                should be used to store the metrics
        """
        super(StoreMetricsAction, self).__init__(data_context)
        self._requested_metrics = requested_metrics
        self._target_store_name = target_store_name
        try:
            store = data_context.stores[target_store_name]
        except KeyError:
            raise DataContextError("Unable to find store {} in your DataContext configuration.".format(
                target_store_name))
        if not isinstance(store, MetricStore):
            raise DataContextError("StoreMetricsAction must have a valid MetricsStore for its target store.")

    def _run(self, validation_result_suite, validation_result_suite_identifier, data_asset):
        logger.debug("StoreMetricsAction.run")

        if validation_result_suite is None:
            return

        if not isinstance(validation_result_suite_identifier, ValidationResultIdentifier):
            raise TypeError("validation_result_id must be of type ValidationResultIdentifier, not {0}".format(
                type(validation_result_suite_identifier)
            ))

        self.data_context.store_validation_result_metrics(
            self._requested_metrics,
            validation_result_suite,
            self._target_store_name
        )


class UpdateDataDocsAction(ValidationAction):
    """
    UpdateDataDocsAction is a namespace-aware validation action that
    notifies the site builders of all the data docs sites of the data context
    that a validation result should be added to the data docs.
    """

    def __init__(self, data_context):
        """
        :param data_context: data context
        """
        super(UpdateDataDocsAction, self).__init__(data_context)

    def _run(self, validation_result_suite, validation_result_suite_identifier, data_asset):
        logger.debug("UpdateDataDocsAction.run")

        if validation_result_suite is None:
            return

        if not isinstance(validation_result_suite_identifier, ValidationResultIdentifier):
            raise TypeError("validation_result_id must be of type ValidationResultIdentifier, not {0}".format(
                type(validation_result_suite_identifier)
            ))

        self.data_context.build_data_docs(
            resource_identifiers=[validation_result_suite_identifier]
        )


class DataAssetProfiler(object):

    @classmethod
    def validate(cls, data_asset):
        return isinstance(data_asset, DataAsset)


class DatasetProfiler(DataAssetProfiler):

    @classmethod
    def validate(cls, dataset):
        return isinstance(dataset, Dataset)

    @classmethod
    def add_expectation_meta(cls, expectation):
        expectation.meta[str(cls.__name__)] = {
            "confidence": "very low"
        }
        return expectation

    @classmethod
    def add_meta(cls, expectation_suite, batch_kwargs=None):
        class_name = str(cls.__name__)
        expectation_suite.meta[class_name] = {
            "created_by": class_name,
            "created_at": time.time(),
        }

        if batch_kwargs is not None:
            expectation_suite.meta[class_name]["batch_kwargs"] = batch_kwargs

        new_expectations = [cls.add_expectation_meta(
            exp) for exp in expectation_suite.expectations]
        expectation_suite.expectations = new_expectations

        if not "notes" in expectation_suite.meta:
            expectation_suite.meta["notes"] = {
                "format": "markdown",
                "content": [
                    "_To add additional notes, edit the <code>meta.notes.content</code> field in the appropriate Expectation json file._"
                    #TODO: be more helpful to the user by piping in the filename.
                    #This will require a minor refactor to make more DataContext information accessible from this method.
                    # "_To add additional notes, edit the <code>meta.notes.content</code> field in <code>expectations/mydb/default/movies/BasicDatasetProfiler.json</code>_"
                ]
            }
        return expectation_suite

    @classmethod
    def profile(cls, data_asset, run_id=None):
        if not cls.validate(data_asset):
            raise GreatExpectationsError("Invalid data_asset for profiler; aborting")

        expectation_suite = cls._profile(data_asset)

        batch_kwargs = data_asset.batch_kwargs
        expectation_suite = cls.add_meta(expectation_suite, batch_kwargs)
        validation_results = data_asset.validate(expectation_suite, run_id=run_id, result_format="SUMMARY")
        expectation_suite.add_citation(
            comment=str(cls.__name__) + " added a citation based on the current batch.",
            batch_kwargs=data_asset.batch_kwargs,
            batch_markers=data_asset.batch_markers,
            batch_parameters=data_asset.batch_parameters
        )
        return expectation_suite, validation_results

    @classmethod
    def _profile(cls, dataset):
        raise NotImplementedError


class BasicDatasetProfilerBase(DatasetProfiler):
    """BasicDatasetProfilerBase provides basic logic of inferring the type and the cardinality of columns
    that is used by the dataset profiler classes that extend this class.
    """

    INT_TYPE_NAMES = {"INTEGER", "int", "INT", "TINYINT", "BYTEINT", "SMALLINT", "BIGINT", "IntegerType", "LongType", "DECIMAL"}
    FLOAT_TYPE_NAMES = {"FLOAT", "FLOAT4", "FLOAT8", "DOUBLE_PRECISION", "NUMERIC", "FloatType", "DoubleType", "float"}
    STRING_TYPE_NAMES = {"CHAR", "VARCHAR", "TEXT", "StringType", "string", "str"}
    BOOLEAN_TYPE_NAMES = {"BOOLEAN", "BOOL", "bool", "BooleanType"}
    DATETIME_TYPE_NAMES = {"DATETIME", "DATE", "TIMESTAMP", "DateType", "TimestampType", "datetime64", "Timestamp"}

    @classmethod
    def _get_column_type(cls, df, column):

        # list of types is used to support pandas and sqlalchemy
        df.set_config_value("interactive_evaluation", True)
        try:
            if df.expect_column_values_to_be_in_type_list(column, type_list=sorted(list(cls.INT_TYPE_NAMES))).success:
                type_ = "int"

            elif df.expect_column_values_to_be_in_type_list(column, type_list=sorted(list(cls.FLOAT_TYPE_NAMES))).success:
                type_ = "float"

            elif df.expect_column_values_to_be_in_type_list(column, type_list=sorted(list(cls.STRING_TYPE_NAMES))).success:
                type_ = "string"

            elif df.expect_column_values_to_be_in_type_list(column, type_list=sorted(list(cls.BOOLEAN_TYPE_NAMES))).success:
                type_ = "bool"

            elif df.expect_column_values_to_be_in_type_list(column, type_list=sorted(list(cls.DATETIME_TYPE_NAMES))).success:
                type_ = "datetime"

            else:
                df.expect_column_values_to_be_in_type_list(column, type_list=None)
                type_ = "unknown"
        except NotImplementedError:
            type_ = "unknown"

        df.set_config_value('interactive_evaluation', False)
        return type_

    @classmethod
    def _get_column_cardinality(cls, df, column):
        num_unique = None
        pct_unique = None
        df.set_config_value("interactive_evaluation", True)

        try:
            num_unique = df.expect_column_unique_value_count_to_be_between(column, None, None).result['observed_value']
            pct_unique = df.expect_column_proportion_of_unique_values_to_be_between(
                column, None, None).result['observed_value']
        except KeyError:  # if observed_value value is not set
            logger.error("Failed to get cardinality of column {0:s} - continuing...".format(column))

        if num_unique is None or num_unique == 0 or pct_unique is None:
            cardinality = "none"

        elif pct_unique == 1.0:
            cardinality = "unique"

        elif pct_unique > .1:
            cardinality = "very many"

        elif pct_unique > .02:
            cardinality = "many"

        else:
            cardinality = "complicated"
            if num_unique == 1:
                cardinality = "one"

            elif num_unique == 2:
                cardinality = "two"

            elif num_unique < 60:
                cardinality = "very few"

            elif num_unique < 1000:
                cardinality = "few"

            else:
                cardinality = "many"
        # print('col: {0:s}, num_unique: {1:s}, pct_unique: {2:s}, card: {3:s}'.format(column, str(num_unique), str(pct_unique), cardinality))

        df.set_config_value('interactive_evaluation', False)

        return cardinality


class BasicDatasetProfiler(BasicDatasetProfilerBase):
    """BasicDatasetProfiler is inspired by the beloved pandas_profiling project.

    The profiler examines a batch of data and creates a report that answers the basic questions
    most data practitioners would ask about a dataset during exploratory data analysis.
    The profiler reports how unique the values in the column are, as well as the percentage of empty values in it.
    Based on the column's type it provides a description of the column by computing a number of statistics,
    such as min, max, mean and median, for numeric columns, and distribution of values, when appropriate.
    """

    @classmethod
    def _profile(cls, dataset):
        df = dataset

        df.set_default_expectation_argument("catch_exceptions", True)

        df.expect_table_row_count_to_be_between(min_value=0, max_value=None)
        df.expect_table_columns_to_match_ordered_list(None)
        df.set_config_value('interactive_evaluation', False)

        columns = df.get_table_columns()

        meta_columns = {}
        for column in columns:
            meta_columns[column] = {"description": ""}

        number_of_columns = len(columns)
        for i, column in enumerate(columns):
            logger.info("            Preparing column {} of {}: {}".format(i+1, number_of_columns, column))

            # df.expect_column_to_exist(column)

            type_ = cls._get_column_type(df, column)
            cardinality = cls._get_column_cardinality(df, column)
            df.expect_column_values_to_not_be_null(column, mostly=0.5) # The renderer will show a warning for columns that do not meet this expectation
            df.expect_column_values_to_be_in_set(column, [], result_format="SUMMARY")

            if type_ == "int":
                if cardinality == "unique":
                    df.expect_column_values_to_be_unique(column)
                elif cardinality in ["one", "two", "very few", "few"]:
                    df.expect_column_distinct_values_to_be_in_set(column, value_set=None, result_format="SUMMARY")
                elif cardinality in ["many", "very many", "unique"]:
                    df.expect_column_min_to_be_between(column, min_value=None, max_value=None)
                    df.expect_column_max_to_be_between(column, min_value=None, max_value=None)
                    df.expect_column_mean_to_be_between(column, min_value=None, max_value=None)
                    df.expect_column_median_to_be_between(column, min_value=None, max_value=None)
                    df.expect_column_stdev_to_be_between(column, min_value=None, max_value=None)
                    df.expect_column_quantile_values_to_be_between(column,
                                                                   quantile_ranges={
                                                                       "quantiles": [0.05, 0.25, 0.5, 0.75, 0.95],
                                                                       "value_ranges": [[None, None], [None, None], [None, None], [None, None], [None, None]]
                                                                   }
                                                                   )
                    df.expect_column_kl_divergence_to_be_less_than(column, partition_object=None,
                                                           threshold=None, result_format='COMPLETE')
                else: # unknown cardinality - skip
                    pass
            elif type_ == "float":
                if cardinality == "unique":
                    df.expect_column_values_to_be_unique(column)

                elif cardinality in ["one", "two", "very few", "few"]:
                    df.expect_column_distinct_values_to_be_in_set(column, value_set=None, result_format="SUMMARY")

                elif cardinality in ["many", "very many", "unique"]:
                    df.expect_column_min_to_be_between(column, min_value=None, max_value=None)
                    df.expect_column_max_to_be_between(column, min_value=None, max_value=None)
                    df.expect_column_mean_to_be_between(column, min_value=None, max_value=None)
                    df.expect_column_median_to_be_between(column, min_value=None, max_value=None)
                    df.expect_column_quantile_values_to_be_between(column,
                                                                   quantile_ranges={
                                                                       "quantiles": [0.05, 0.25, 0.5, 0.75, 0.95],
                                                                       "value_ranges": [[None, None], [None, None], [None, None], [None, None], [None, None]]
                                                                   }
                                                                   )
                    df.expect_column_kl_divergence_to_be_less_than(column, partition_object=None,

                                                           threshold=None, result_format='COMPLETE')
                else:  # unknown cardinality - skip
                    pass

            elif type_ == "string":
                # Check for leading and trailing whitespace.
                #!!! It would be nice to build additional Expectations here, but
                #!!! the default logic for remove_expectations prevents us.
                df.expect_column_values_to_not_match_regex(column, r"^\s+|\s+$")

                if cardinality == "unique":
                    df.expect_column_values_to_be_unique(column)

                elif cardinality in ["one", "two", "very few", "few"]:
                    df.expect_column_distinct_values_to_be_in_set(column, value_set=None, result_format="SUMMARY")
                else:
                    # print(column, type_, cardinality)
                    pass

            elif type_ == "datetime":
                df.expect_column_min_to_be_between(column, min_value=None, max_value=None)

                df.expect_column_max_to_be_between(column, min_value=None, max_value=None)

                # Re-add once kl_divergence has been modified to support datetimes
                # df.expect_column_kl_divergence_to_be_less_than(column, partition_object=None,
                #                                            threshold=None, result_format='COMPLETE')

                if cardinality in ["one", "two", "very few", "few"]:
                    df.expect_column_distinct_values_to_be_in_set(column, value_set=None, result_format="SUMMARY")



            else:
                if cardinality == "unique":
                    df.expect_column_values_to_be_unique(column)

                elif cardinality in ["one", "two", "very few", "few"]:
                    df.expect_column_distinct_values_to_be_in_set(column, value_set=None, result_format="SUMMARY")
                else:
                    # print(column, type_, cardinality)
                    pass

        df.set_config_value("interactive_evaluation", True)
        expectation_suite = df.get_expectation_suite(suppress_warnings=True, discard_failed_expectations=False)
        expectation_suite.meta["columns"] = meta_columns

        return expectation_suite


LOCAL_MODULE_TO_CLASS_MAPPING_DICT = allocate_module_to_class_mappings()
LOCAL_MODULE_NAMES_LIST = LOCAL_MODULE_TO_CLASS_MAPPING_DICT.keys()

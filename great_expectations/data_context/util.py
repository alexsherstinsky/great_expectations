# -*- coding: utf-8 -*-

import sys
import logging
import os
import errno
import six
from six import string_types
from six import integer_types
from marshmallow import ValidationError
import importlib
from pydoc import locate
import copy
import re
import inspect
import warnings
from collections import (
    OrderedDict,
    defaultdict,
    Counter
)

import json
from string import Template as pTemplate
import datetime
from uuid import uuid4

#from builtins import str  # PY2 compatibility

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
    contextfilter
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
    RenderedGraphContent
)

from great_expectations.render.util import ordinal, num_to_str
from great_expectations.core.id_dict import BatchKwargs

from great_expectations.profile.basic_dataset_profiler import BasicDatasetProfiler

from great_expectations.data_context.types.base import DataContextConfig
import great_expectations.exceptions as exceptions
from great_expectations.exceptions import (
    PluginModuleNotFoundError,
    PluginClassNotFoundError,
    InvalidConfigError
)

from great_expectations.cli.datasource import DATASOURCE_TYPE_BY_DATASOURCE_CLASS
from great_expectations.data_context.store.html_site_store import (
    HtmlSiteStore,
    SiteSectionIdentifier,
)
from great_expectations.data_context.types.resource_identifiers import (
    ExpectationSuiteIdentifier,
    ValidationResultIdentifier
)

from great_expectations.core import (
    ExpectationConfiguration,
    ExpectationValidationResult,
)

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

logger = logging.getLogger(__name__)

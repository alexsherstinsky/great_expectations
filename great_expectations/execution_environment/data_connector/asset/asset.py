# -*- coding: utf-8 -*-

import logging

logger = logging.getLogger(__name__)


class Asset(object):
    def __init__(
        self,
        name: str,
        **kwargs
    ):
        self._name = name

        for k, v in kwargs.items():
            setattr(self, k, v)

    @property
    def name(self) -> str:
        return self._name

# -*- coding: utf-8 -*-
# @Time    : 2023/10/8 10:57
# @Author  : Patrick
# @Email   : firechecking@gmail.com
# @File    : user_overrides.py
# @Software: CleanParallel
# @Description: user_overrides

from .. import utils

_USER_CAST_REGISTRY = set()
_USER_PROMOTE_REGISTRY = set()


def register_half_function(module, name):
    if not hasattr(module, name):
        raise ValueError('No function named {} in module {}.'.format(
            name, module))
    _USER_CAST_REGISTRY.add((module, name, utils.maybe_half))


def register_float_function(module, name):
    if not hasattr(module, name):
        raise ValueError('No function named {} in module {}.'.format(
            name, module))
    _USER_CAST_REGISTRY.add((module, name, utils.maybe_float))


def register_promote_function(module, name):
    if not hasattr(module, name):
        raise ValueError('No function named {} in module {}.'.format(
            name, module))
    _USER_PROMOTE_REGISTRY.add((module, name))


if __name__ == "__main__":
    pass

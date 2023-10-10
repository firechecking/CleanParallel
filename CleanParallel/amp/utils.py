# -*- coding: utf-8 -*-
# @Time    : 2023/10/8 10:59
# @Author  : Patrick
# @Email   : firechecking@gmail.com
# @File    : utils.py
# @Software: CleanParallel
# @Description: utils

import itertools

import torch


def is_cuda_enabled():
    return torch.version.cuda is not None


def get_cuda_version():
    return tuple(int(x) for x in torch.version.cuda.split('.'))


def collect_fp_tensor_types(args, kwargs):
    def collect_types(x, types):
        if is_nested(x):
            for y in x:
                collect_types(y, types)
        else:
            types.add(type_string(x))

    all_args = itertools.chain(args, kwargs.values())
    types = set()
    for x in all_args:
        if is_fp_tensor(x):
            collect_types(x, types)
    return types


def tensor_is_float_tensor():
    x = torch.Tensor()
    return type(x) == torch.FloatTensor


def is_tensor_like(x):
    return torch.is_tensor(x) or isinstance(x, torch.autograd.Variable)


def is_floating_point(x):
    if hasattr(torch, 'is_floating_point'):
        return torch.is_floating_point(x)
    try:
        torch_type = x.type()
        return torch_type.endswith('FloatTensor') or \
               torch_type.endswith('HalfTensor') or \
               torch_type.endswith('DoubleTensor')
    except AttributeError:
        return False


def is_fp_tensor(x):
    if is_nested(x):
        # Fast-fail version of all(is_fp_tensor)
        for y in x:
            if not is_fp_tensor(y):
                return False
        return True
    return is_tensor_like(x) and is_floating_point(x)


def casted_args(cast_func, args, kwargs):
    new_args = []
    for x in args:
        if is_fp_tensor(x):
            new_args.append(cast_func(x))
        else:
            new_args.append(x)
    for k in kwargs:
        val = kwargs[k]
        if is_fp_tensor(val):
            kwargs[k] = cast_func(val)
    return new_args


def has_func(mod, name):
    if isinstance(mod, dict):
        return name in mod
    else:
        return hasattr(mod, name)


def get_func(mod, name):
    if isinstance(mod, dict):
        return mod[name]
    else:
        return getattr(mod, name)


def set_func(mod, name, new_func):
    if isinstance(mod, dict):
        mod[name] = new_func
    else:
        setattr(mod, name, new_func)


def type_string(x):
    return x.type().split('.')[-1]


def is_nested(x):
    return isinstance(x, tuple) or isinstance(x, list)


def maybe_half(x, name='', verbose=False):
    if is_nested(x):
        return type(x)([maybe_half(y) for y in x])

    if not x.is_cuda or type_string(x) == 'HalfTensor':
        return x
    else:
        if verbose:
            print('Float->Half ({})'.format(name))
        return x.half()


def maybe_float(x, name='', verbose=False):
    if is_nested(x):
        return type(x)([maybe_float(y) for y in x])

    if not x.is_cuda or type_string(x) == 'FloatTensor':
        return x
    else:
        if verbose:
            print('Half->Float ({})'.format(name))
        return x.float()


def as_inplace(fns):
    for x in fns:
        yield x + '_'


if __name__ == "__main__":
    pass

# -*- coding: utf-8 -*-
# @Time    : 2023/9/18 20:54
# @Author  : Patrick
# @Email   : firechecking@gmail.com
# @File    : amp.py.py
# @Software: CleanParallel
# @Description: amp.py

import functools

import torch
from .amp_state import amp_state, maybe_print


class O0:
    def __init__(self):
        self.opt_level = 'O0'

        self.cast_model_type = torch.float32
        self.master_weights = False
        self.patch_torch_functions = False
        self.loss_scale = 1.0


class O1(O0):
    def __init__(self):
        super(O1, self).__init__()
        self.opt_level = 'O1'

        self.cast_model_type = None
        self.patch_torch_functions = True
        self.loss_scale = 'dynamic'


class O2(O0):
    def __init__(self):
        super(O2, self).__init__()
        self.opt_level = 'O2'

        self.cast_model_type = torch.float16
        self.master_weights = True
        self.loss_scale = 'dynamic'


class O3(O0):
    def __init__(self):
        super(O3, self).__init__()
        self.opt_level = 'O3'

        self.cast_model_type = torch.float16


def update_opt_properities(opt_properties, custom_properties):
    for k, v in custom_properties:
        if v is not None:
            opt_properties.__dict__[k] = v
    return opt_properties


def to_type(v, dtype):
    if isinstance(v, torch.Tensor) and v.is_floating_point():
        return v.to(dtype)
    return v


def patch_forward(old_forward, input_caster=None, output_caster=None):
    input_caster = (lambda x: x) if input_caster is None else input_caster
    output_caster = (lambda x: x) if output_caster is None else output_caster

    def new_forward(*args, **kwargs):
        args = input_caster(args)
        kwargs = input_caster(kwargs)
        output = old_forward(*args, **kwargs)
        return output_caster(output)

    return new_forward


class O2StateDictHook(object):
    def __init__(self, fn):
        self.fn = fn

    def __call__(self, module, state_dict, prefix, local_metadata):
        for key in state_dict:
            param = state_dict[key]
            if 'Half' in param.type():
                state_dict[key] = self.fn(param)


def initialize(
        models,
        optimizers=None,
        enabled=True,
        verbosity=1,
        min_loss_scale=None,
        max_loss_scale=2. ** 24,
        opt_level="O1",
        cast_model_outputs=None,
        custom_properties={},
):
    amp_state.enabled = enabled
    amp_state.verbosity = verbosity
    amp_state.min_loss_scale = min_loss_scale
    amp_state.max_loss_scale = max_loss_scale

    ############### 获取默认参数 ###############
    maybe_print(f'opt_level: {opt_level}')
    amp_state.opt_properties = eval(opt_level)()
    ############### 更新自定义参数 ###############
    amp_state.opt_properties = update_opt_properities(amp_state.opt_properties, custom_properties)
    for k, v in amp_state.opt_properties.__dict__.items():
        maybe_print('{:22}: {}'.format(k, v))

    ############### 处理cast_model_type ###############
    models, models_was_list = (models, True) if isinstance(models, list) else ([models, ], False)
    # TODO: 校验models不能是parallel或重复amp模型
    if amp_state.opt_properties.cast_model_type:
        input_caster = functools.partial(to_type, dtype=amp_state.opt_properties.cast_model_type)
        for model in models:
            ############### 转换模型: parameters ###############
            model.to(amp_state.opt_properties.cast_model_type)

            ############### 转换模型: inputs ###############
            model.forward = patch_forward(model.forward, input_caster=input_caster)

            ############### 转换模型: state_dict ###############
            for module in model.modules():
                module._register_state_dict_hook(O2StateDictHook(functools.partial(to_type, dtype=torch.float32)))

    ############### 处理cast_model_outputs ###############
    if cast_model_outputs is not None:
        output_caster = functools.partial(to_type, dtype=cast_model_outputs)
        for model in models:
            model.forward = patch_forward(model.forward, output_caster=output_caster)


if __name__ == "__main__":
    pass

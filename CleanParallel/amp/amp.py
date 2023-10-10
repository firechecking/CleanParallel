# -*- coding: utf-8 -*-
# @Time    : 2023/9/18 20:54
# @Author  : Patrick
# @Email   : firechecking@gmail.com
# @File    : amp.py.py
# @Software: CleanParallel
# @Description: amp.py

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


if __name__ == "__main__":
    pass

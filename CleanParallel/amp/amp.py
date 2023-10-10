# -*- coding: utf-8 -*-
# @Time    : 2023/9/18 20:54
# @Author  : Patrick
# @Email   : firechecking@gmail.com
# @File    : amp.py.py
# @Software: CleanParallel
# @Description: amp.py

import functools, types, itertools

import torch
from .amp_state import amp_state, maybe_print
from .lists import user_overrides, functional_overrides, tensor_overrides, torch_overrides
from . import utils
from .scaler import LossScaler


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


class AmpOptimizerState(object):
    pass


def _process_optimizer(optimizer, master_weigths):
    optimizer._amp_stash = AmpOptimizerState()
    ############### 复制master_weights ###############
    optimizer._amp_stash.fp16_to_fp32_params = {}

    if master_weigths:
        for group in optimizer.param_groups:
            for i, param in enumerate(group['params']):
                if param.requires_grad:
                    if param.type() in ('torch.cuda.HalfTensor', 'torch.HalfTensor'):
                        group['params'][i] = master_weight = param.detach().clone().float()
                        master_weight.requires_grad = True
                        optimizer._amp_stash.fp16_to_fp32_params[param] = master_weight

                        if param in optimizer.state:
                            optimizer.state[master_weight] = optimizer.state.pop(param)

    ############### 处理step ###############
    old_step = optimizer.step

    def new_step(self, closure=None):
        if closure is not None:
            raise RuntimeError("Currently, Amp does not support closure use with optimizers.")
        ############### 将计算图中half类型的grad复制到master_weight.grad ###############
        for half_weight, master_weight in self._amp_stash.fp16_to_fp32_params.items():
            if half_weight.grad is not None and master_weight.grad is None:
                master_weight.grad = torch.empty_like(master_weight)
                master_weight.grad.copy_(half_weight.grad)
        ############### 常规的用grad更新weight ###############
        ret_val = old_step()
        ############### 将master_weight复制到half ###############
        for half_weight, master_weight in self._amp_stash.fp16_to_fp32_params.items():
            half_weight.data.copy_(master_weight.data)

        return ret_val

    optimizer.step = types.MethodType(new_step, optimizer)

    ############### 处理zero_grad ###############
    old_zero_grad = optimizer.zero_grad

    def new_zero_grad(self):
        old_zero_grad()
        for param in optimizer._amp_stash.fp16_to_fp32_params.keys():  # 计算图中的half类型也需要清除grad
            if param.grad is not None:
                param.grad.detach_()
                param.grad.zero_()

    optimizer.zero_grad = types.MethodType(new_zero_grad, optimizer)

    return optimizer


class Wrapper():
    @classmethod
    def cast(cls, mod, name, cast_func):
        if not utils.has_func(mod, name):
            return
        orig_func = utils.get_func(mod, name)

        @functools.wraps(orig_func)
        def new_func(*args, **kwargs):
            new_args = utils.casted_args(cast_func, args, kwargs)
            return orig_func(*new_args, **kwargs)

        utils.set_func(mod, name, new_func)

    @classmethod
    def promote(cls, mod, name):
        if not utils.has_func(mod, name):
            return
        orig_func = utils.get_func(mod, name)

        @functools.wraps(orig_func)
        def new_func(*args, **kwargs):
            types = utils.collect_fp_tensor_types(args, kwargs)
            if len(types) <= 1: return orig_func(*args, **kwargs)
            if types == set(['HalfTensor', 'FloatTensor']):
                new_args = utils.casted_args(utils.maybe_float,
                                             args,
                                             kwargs)
                return orig_func(*new_args, **kwargs)
            raise NotImplementedError('Do not know how to handle ' +
                                      'these types to promote: {}'
                                      .format(types))

        utils.set_func(mod, name, new_func)

    @classmethod
    def sequence_promote(cls, mod, name):
        if not utils.has_func(mod, name):
            return
        orig_func = utils.get_func(mod, name)

        @functools.wraps(orig_func)
        def new_func(seq, *args, **kwargs):
            types = set([utils.type_string(x) for x in seq])

            if len(types) <= 1: return orig_func(seq, *args, **kwargs)
            if types == set(['HalfTensor', 'FloatTensor']):
                cast_seq = utils.casted_args(utils.maybe_float,
                                             seq,
                                             {})
                return orig_func(cast_seq, *args, **kwargs)
            return orig_func(seq, *args, **kwargs)

        utils.set_func(mod, name, new_func)

    @classmethod
    def err_if_any_half(cls, mod, name):
        if not utils.has_func(mod, name):
            return
        orig_fn = utils.get_func(mod, name)

        @functools.wraps(orig_fn)
        def new_func(*args, **kwargs):
            types = utils.collect_fp_tensor_types(args, kwargs)
            if 'HalfTensor' in types:
                raise NotImplementedError('Cannot call in-place function ' +
                                          '{} with fp16 arguments.'.format(name))
            else:
                return orig_fn(*args, **kwargs)

        utils.set_func(mod, name, new_func)

    @classmethod
    def err_if_arg0_half(cls, mod, name):
        if not utils.has_func(mod, name):
            return
        orig_fn = utils.get_func(mod, name)

        @functools.wraps(orig_fn)
        def new_func(arg0, *args, **kwargs):
            assert utils.is_tensor_like(arg0)
            if utils.type_string(arg0) == 'HalfTensor':
                raise NotImplementedError('Cannot call in-place method ' +
                                          '{} on fp16 Tensors.'.format(name))
            else:
                new_args = utils.casted_args(utils.maybe_float, args, kwargs)
                return orig_fn(arg0, *new_args, **kwargs)

        utils.set_func(mod, name, new_func)

    @classmethod
    def promote_match_arg0(cls, mod, name):
        if not utils.has_func(mod, name):
            return
        orig_fn = utils.get_func(mod, name)

        @functools.wraps(orig_fn)
        def new_func(arg0, *args, **kwargs):
            assert utils.is_tensor_like(arg0)
            if utils.type_string(arg0) == 'HalfTensor':
                cast_fn = utils.maybe_half
            elif utils.type_string(arg0) == 'FloatTensor':
                cast_fn = utils.maybe_float
            else:
                return orig_fn(arg0, *args, **kwargs)
            new_args = utils.casted_args(cast_fn, args, kwargs)
            return orig_fn(arg0, *new_args, **kwargs)

        utils.set_func(mod, name, new_func)


def _patch_torch_functions(allow_banned=False):
    ############### fp16、fp32白名单 ###############
    for module in [functional_overrides, torch_overrides, tensor_overrides]:
        for name in getattr(module, 'FP16_FUNCS'):
            Wrapper.cast(module.MODULE, name, utils.maybe_half)
        for name in getattr(module, 'FP32_FUNCS'):
            Wrapper.cast(module.MODULE, name, utils.maybe_float)

    ############### 将混合输入转成float ###############
    for module in [torch_overrides, tensor_overrides]:
        for name in getattr(module, 'CASTS'):
            Wrapper.promote(module.MODULE, name)
        for name in getattr(module, 'SEQUENCE_CASTS'):
            Wrapper.sequence_promote(module.MODULE, name)

    ############### cuda.tensor ###############
    if utils.tensor_is_float_tensor():
        for name in tensor_overrides.FP16_FUNCS:
            Wrapper.cast(torch.cuda.FloatTensor, name, utils.maybe_half)
        for name in tensor_overrides.FP32_FUNCS:
            Wrapper.cast(torch.cuda.HalfTensor, name, utils.maybe_float)
        for name in tensor_overrides.CASTS:
            Wrapper.promote(torch.cuda.FloatTensor, name)
            Wrapper.promote(torch.cuda.HalfTensor, name)
        for name in tensor_overrides.SEQUENCE_CASTS:
            Wrapper.sequence_promote(torch.cuda.FloatTensor, name)
            Wrapper.sequence_promote(torch.cuda.HalfTensor, name)

    ############### inplace操作 ###############
    for name in utils.as_inplace(torch_overrides.FP32_FUNCS):
        Wrapper.err_if_any_half(torch_overrides.MODULE, name)
    for name in utils.as_inplace(tensor_overrides.FP32_FUNCS):
        Wrapper.err_if_arg0_half(tensor_overrides.MODULE, name)
    for name in utils.as_inplace(itertools.chain(
            tensor_overrides.FP16_FUNCS,
            tensor_overrides.CASTS)):
        Wrapper.promote_match_arg0(tensor_overrides.MODULE, name)
        if utils.tensor_is_float_tensor():
            Wrapper.promote_match_arg0(torch.cuda.HalfTensor, name)
            Wrapper.promote_match_arg0(torch.cuda.FloatTensor, name)

    ############### 对受限funcs操作 ###############
    for name, err_msg in functional_overrides.BANNED_FUNCS:
        if allow_banned:
            Wrapper.cast(functional_overrides.MODULE, name, utils.maybe_float)
        else:
            Wrapper.err_if_any_half(functional_overrides.MODULE, name)

    ############### 用户自定义functions ###############
    for mod, name, cast_func in user_overrides._USER_CAST_REGISTRY:
        Wrapper.cast(mod, name, cast_func)
    for mod, name in user_overrides._USER_PROMOTE_REGISTRY:
        Wrapper.promote(mod, name)
    user_overrides._USER_CAST_REGISTRY.clear()
    user_overrides._USER_PROMOTE_REGISTRY.clear()


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

    ############### 处理optimizer及master_weights ###############
    optimizers, optimizers_was_list = (optimizers, True) if isinstance(optimizers, list) else ([optimizers, ], False)
    for i, optimizer in enumerate(optimizers):
        optimizers[i] = _process_optimizer(optimizer, amp_state.opt_properties.master_weights)

    ############### 处理patch_torch_functions ###############
    _patch_torch_functions()

    ############### 初始化loss scaler ###############
    amp_state.loss_scaler = LossScaler(loss_scale=amp_state.opt_properties.loss_scale, min_loss_scale=min_loss_scale, max_loss_scale=max_loss_scale)

    ############### 结果返回 ###############
    if not models_was_list: models = models[0]
    if not optimizers_was_list: optimizers = optimizers[0]
    return models, optimizers


if __name__ == "__main__":
    pass

# -*- coding: utf-8 -*-
# @Time    : 2023/10/8 16:29
# @Author  : Patrick
# @Email   : firechecking@gmail.com
# @File    : scaler.py
# @Software: CleanParallel
# @Description: scaler

import contextlib
import types
import torch

from .amp_state import amp_state, maybe_print


@contextlib.contextmanager
def scale_loss(loss, optimizers):
    if not amp_state.enabled:
        yield loss
        return

    loss_scaler = amp_state.loss_scaler
    if (loss_scaler.loss_scale == 1.0) and (not loss_scaler.dynamic):
        yield loss.float()
        return

    ############### scaled loss ###############
    yield loss.float() * loss_scaler.loss_scale

    ############### 更新loss_scale，并判断是否有溢出 ###############
    if not isinstance(optimizers, list):
        optimizers = [optimizers, ]

    loss_scaler.clear_overflow_state()
    for optimizer in optimizers:
        loss_scaler.unscale(optimizer)
    should_skip = loss_scaler.update_scale()

    ############### 如果有溢出，跳过本次权重更新 ###############
    if should_skip:
        for optimizer in optimizers:
            if not hasattr(optimizer._amp_stash, 'skip_patched'):
                optimizer._amp_stash.skip_patched = False

            if not optimizer._amp_stash.skip_patched:
                old_step = optimizer.step

                def patch_skep_step(self, closure=None):
                    if closure is not None:
                        raise RuntimeError("Currently, Amp does not support closure use with optimizers.")

                    maybe_print(("Gradient overflow.  Skipping step, loss scaler " +
                                 "reducing loss scale to {}").format(loss_scaler.loss_scale))
                    self.step = old_step
                    self._amp_stash.skip_patched = False

                optimizer.step = types.MethodType(patch_skep_step, optimizer)
                optimizer._amp_stash.skip_patched = True


class LossScaler():
    def __init__(self,
                 loss_scale,
                 init_scale=2. ** 16,
                 scale_factor=2.,
                 scale_window=2000,
                 min_loss_scale=None,
                 max_loss_scale=2. ** 24):
        self.dynamic, self.loss_scale = (True, init_scale) if loss_scale == 'dynamic' else (False, loss_scale)
        self.min_loss_scale, self.max_loss_scale = min_loss_scale, max_loss_scale
        self.scale_factor = scale_factor
        self.scale_window = scale_window
        self.unskipped = 0
        self.has_overflow = False

    def clear_overflow_state(self):
        self.has_overflow = False

    def unscale(self, optimizer):
        if self.has_overflow: return
        if self.loss_scale == 1.0: return
        for half_weight, master_weight in optimizer._amp_stash.fp16_to_fp32_params.items():
            ############### 判断grad是否有溢出 ###############
            _sum = float(half_weight.grad.float().sum())
            if _sum == float('inf') or _sum == -float('inf') or _sum != _sum:
                self.has_overflow = True
                return
            ############### 将计算图中half类型的grad复制到master_weight.grad ###############
            if master_weight.grad is None:
                master_weight.grad = torch.empty_like(master_weight)
            if master_weight.grad is not half_weight.grad:
                master_weight.grad.copy_(half_weight.grad)
            ############### loss_scale还原 ###############
            master_weight.grad.mul_(1.0 / self.loss_scale)

    def update_scale(self):
        should_skip = False
        self.unskipped += 1
        ############### 梯度溢出，降低scale，并跳过本次更新 ###############
        if self.has_overflow and self.dynamic:
            should_skip = True
            self.unskipped = 0
            self.loss_scale /= self.scale_factor
            if self.min_loss_scale is not None:
                self.loss_scale = max(self.min_loss_scale, self.loss_scale)

        ############### 多次更新无溢出，增大scale ###############
        if self.unskipped >= self.scale_window and self.dynamic:
            self.loss_scale *= self.scale_factor
            if self.max_loss_scale is not None:
                self.loss_scale = min(self.max_loss_scale, self.loss_scale)
                self.unskipped = 0

        return should_skip


if __name__ == "__main__":
    pass

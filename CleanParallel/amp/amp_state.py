# -*- coding: utf-8 -*-
# @Time    : 2023/9/18 21:11
# @Author  : Patrick
# @Email   : firechecking@gmail.com
# @File    : amp_state.py
# @Software: CleanParallel
# @Description: amp_state

import torch


class AmpState():
    def __init__(self):
        self.verbosity = 1
        self.enabled = True


############### 如果是分布式训练，只在rank0打印 ###############
def maybe_print(msg, rank0=True, verbosity=True):
    if not verbosity: return

    distributed = torch.distributed.is_available() and \
                  torch.distributed.is_initialized() and \
                  torch.distributed.get_world_size() > 1

    if not distributed:
        print(msg)
    elif not rank0:
        print(msg)
    elif torch.distributed.get_rank() == 0:
        print(msg)


############### 全局amp_state ###############
amp_state = AmpState()

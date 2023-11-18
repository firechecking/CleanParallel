# -*- coding: utf-8 -*-
# @Time    : 2023/11/16 21:26
# @Author  : Patrick
# @Email   : firechecking@gmail.com
# @File    : checkpointing.py.py
# @Software: CleanParallel
# @Description: checkpointing.py

import torch
from CleanParallel.PipelineParallel.functions import copy, wait, Join


class Checkpointing():
    def __init__(self, stream, func, *args):
        self.stream = stream
        self.func = func
        self.args = args

        ############### 保存Checkpoint和Recompute共享的rng_state、计算图 ###############
        self.share_parameters = {}

    def checkpoint(self):
        with torch.cuda.stream(self.stream):
            ############### phony的作用是让Checkpoint返回值的requires_grad=True ###############
            phony = torch.tensor(0.0, device=self.args[0].device, requires_grad=True)
            return Checkpoint.apply(self.func, self.share_parameters, phony, *self.args)

    def recompute(self, data):
        with torch.cuda.stream(self.stream):
            ############### phony的作用是在之后的代码中构建计算图 ###############
            phony = Recompute.apply(self.func, self.share_parameters, data, *self.args)
            return phony


class Checkpoint(torch.autograd.Function):
    @staticmethod
    def forward(ctx, func, share_parameters, phony, *args):
        ctx.func = func
        ctx.share_parameters = share_parameters
        ctx.save_for_backward(*args)

        ############### 保存rng_state ###############
        ctx.cpu_rng_state = torch.get_rng_state()
        ctx.gpu_rng_state = torch.cuda.get_rng_state(args[0].device) if args[0].is_cuda else None
        ctx.share_parameters['rng_state'] = (ctx.cpu_rng_state, ctx.gpu_rng_state)

        ############### with no_grad下进行forward ###############
        with torch.no_grad():
            output = func(*args)
        return output

    @staticmethod
    def backward(ctx, *grad_input):
        ############### 重新进行backward ###############
        recomputed_output, leaf_args = ctx.share_parameters['recomputed']
        torch.autograd.backward(recomputed_output, grad_tensors=grad_input)
        return (None, None, None,) + tuple(tensor.grad for tensor in leaf_args)


class Recompute(torch.autograd.Function):
    @staticmethod
    def forward(ctx, function, share_parameters, data, *args):
        ctx.func = function
        ctx.share_parameters = share_parameters
        ctx.save_for_backward(*args)

        phony = torch.tensor(0.0, device=args[0].device)
        return phony

    @staticmethod
    def backward(ctx, *grad_input):
        args = ctx.saved_tensors

        with torch.random.fork_rng(devices=[args[0].device] if args[0].is_cuda else None):
            ############### 恢复rng_state ###############
            cpu_rng_state, gpu_rng_state = ctx.share_parameters['rng_state']
            torch.set_rng_state(cpu_rng_state)
            if gpu_rng_state is not None:
                torch.cuda.set_rng_state(gpu_rng_state)

            ############### with grad下进行forward ###############
            # 对leaf节点的backward才会保存grad，所以创建一个leaf拷贝
            leaf_args = [tensor.detach().requires_grad_(tensor.requires_grad) for tensor in args]
            with torch.enable_grad():
                recomputed_output = ctx.func(*leaf_args)

        ctx.share_parameters['recomputed'] = (recomputed_output, leaf_args)

        ############### Recompute只需要提供计算图不提供梯度，所以梯度返回None ###############
        return (None, None, None,) + tuple([None for _ in leaf_args])


def calc_func(a, b):
    c = a * b + a + b
    d = c * c * c
    return d


def calc_func2(a):
    b = a * a * a - 2 * a * a - 5 * a
    c = b - a
    return c


if __name__ == '__main__':
    torch.manual_seed(999)

    a = torch.tensor(2.0, requires_grad=True, device='cuda:0')
    b = torch.tensor(4.0, requires_grad=True, device='cuda:0')

    copy_stream0 = torch.cuda.Stream('cuda:0')
    calc_stream0 = torch.cuda.default_stream('cuda:0')
    copy_stream1 = torch.cuda.Stream('cuda:1')
    calc_stream1 = torch.cuda.default_stream('cuda:1')

    ############### checkpoint->recompute->wait->copy ###############
    chk = Checkpointing(calc_stream0, calc_func, a, b)
    c = chk.checkpoint()
    c = wait(c, copy_stream0, calc_stream0)
    phony = chk.recompute(c)
    c = Join.apply(c, phony)  # 将checkpoint输出和recompute输出同时输入后续节点
    c = copy(c, copy_stream0, copy_stream1)

    ############### wait->checkpoint->recompute ###############
    c = wait(c, calc_stream1, copy_stream1)
    chk = Checkpointing(calc_stream1, calc_func2, c)
    d = chk.checkpoint()
    phony = chk.recompute(d)
    output = Join.apply(d, phony)  # 将checkpoint输出和recompute输出同时输入后续节点

    output.backward()
    print(output)  # tensor(2.0646e+10, device='cuda:0', grad_fn=<CheckpointingBackward>)
    print(a.grad)  # tensor(6.6378e+10, device='cuda:0')
    print(b.grad)  # tensor(3.9827e+10, device='cuda:0')

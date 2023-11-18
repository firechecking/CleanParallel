# -*- coding: utf-8 -*-
# @Time    : 2023/11/16 21:34
# @Author  : Patrick
# @Email   : firechecking@gmail.com
# @File    : functions.py.py
# @Software: CleanParallel
# @Description: functions.py

import torch
from CleanParallel.PipelineParallel.stream import use_stream, record_stream, current_stream, wait_stream


def copy(data, prev_stream, next_stream):
    if isinstance(data, (list, tuple)):
        data = Copy.apply(prev_stream, next_stream, *data)
    else:
        data = Copy.apply(prev_stream, next_stream, data)[0]
    return data


class Copy(torch.autograd.Function):
    @staticmethod
    def forward(ctx, prev_stream, next_stream, *data):
        ctx.prev_stream = prev_stream
        ctx.next_stream = next_stream

        output = []
        current_next_stream = current_stream(next_stream.device)

        ############### 将复制操作放在prev_stream和next_stream上 ###############
        with use_stream(prev_stream), use_stream(next_stream):
            ############### 数据复制 ###############
            for x in data:
                y = x.to(next_stream.device)
                output.append(y)

                ############### 防止数据被释放 ###############
                record_stream(x, next_stream)
                record_stream(y, current_next_stream)

        return tuple(output)

    @staticmethod
    def backward(ctx, *grad_input):
        prev_stream = ctx.prev_stream
        next_stream = ctx.next_stream

        grad_out = []
        current_prev_stream = current_stream(prev_stream.device)

        ############### 将复制操作放在prev_stream和next_stream上 ###############
        with use_stream(prev_stream), use_stream(next_stream):
            ############### 梯度复制 ###############
            for x in grad_input:
                y = x.to(prev_stream.device)
                grad_out.append(y)

                ############### 防止梯度被释放 ###############
                record_stream(x, next_stream)
                record_stream(y, current_prev_stream)

        return (None, None) + tuple(grad_out)


def wait(batch, current_stream, prev_stream):
    if isinstance(batch, (list, tuple)):
        batch = Wait.apply(current_stream, prev_stream, *batch)
    else:
        batch = Wait.apply(current_stream, prev_stream, batch)[0]
    return batch


class Wait(torch.autograd.Function):
    @staticmethod
    def forward(ctx, current_stream, prev_stream, *data):
        ctx.current_stream = current_stream
        ctx.prev_stream = prev_stream

        wait_stream(current_stream, prev_stream)  # 相当于current_steam.wait_stream(prev_stream)

        return tuple(x.detach() for x in data)

    @staticmethod
    def backward(ctx, *grad_input):
        current_stream = ctx.current_stream
        prev_stream = ctx.prev_stream

        wait_stream(prev_stream, current_stream)  # 相当于prev_stream.wait_stream(current_steam)

        return (None, None) + grad_input


class Join(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, phony):
        return input.detach()

    @staticmethod
    def backward(ctx, grad_input):
        return grad_input, None


if __name__ == "__main__":
    pass

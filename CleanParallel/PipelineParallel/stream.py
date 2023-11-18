# -*- coding: utf-8 -*-
# @Time    : 2023/11/16 21:36
# @Author  : Patrick
# @Email   : firechecking@gmail.com
# @File    : stream.py
# @Software: CleanParallel
# @Description: stream


from contextlib import contextmanager
import torch


class CPUStream():
    device = 'cpu'


cpu_stream = CPUStream()


def new_stream(device):
    if isinstance(device, str):
        device = torch.device(device)
    if device.type != 'cuda':
        return cpu_stream
    return torch.cuda.Stream(device)


def default_stream(device):
    if isinstance(device, str):
        device = torch.device(device)
    if device.type != 'cuda':
        return cpu_stream
    return torch.cuda.default_stream(device)


def current_stream(device):
    if isinstance(device, str):
        device = torch.device(device)

    if device.type != 'cuda':
        return cpu_stream
    return torch.cuda.current_stream(device)


@contextmanager
def use_stream(stream):
    if isinstance(stream, CPUStream):
        yield
        return
    with torch.cuda.stream(stream):
        yield


def record_stream(tensor, stream):
    if isinstance(stream, CPUStream):
        return

    tensor = tensor.new_empty([0]).set_(tensor.storage())
    tensor.record_stream(stream)


def wait_stream(source, target):
    if isinstance(target, CPUStream):
        return
    if isinstance(source, CPUStream):
        target.synchronize()
    else:
        source.wait_stream(target)


if __name__ == "__main__":
    pass

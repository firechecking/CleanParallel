# -*- coding: utf-8 -*-
# @Time    : 2023/11/17 21:54
# @Author  : Patrick
# @Email   : firechecking@gmail.com
# @File    : parallel.py.py
# @Software: CleanParallel
# @Description: parallel.py

from collections import OrderedDict

import torch

from CleanParallel.PipelineParallel.stream import new_stream, default_stream
from CleanParallel.PipelineParallel.functions import copy, wait, Join
from CleanParallel.PipelineParallel.checkpointing import Checkpointing


def split_module(module, partition_size, devices):
    '''
    将模型切分成多个partition，并转移到不同device
    '''
    layers = OrderedDict()
    partitions = []
    i = 0
    for name, layer in module.named_children():
        layers[name] = layer
        if len(layers) == partition_size[i]:
            partitions.append(torch.nn.Sequential(layers).to(devices[i]))
            layers.clear()
            i += 1
    return torch.nn.ModuleList(partitions)


def split_data(data, chunks):
    '''
    将数据切分成多个micro-batch
    '''
    if isinstance(data, torch.Tensor):
        datas = data.chunk(chunks)
    else:
        split_tensors = []
        for tensor in data:
            tensors = tensor.chunk(chunks)
            split_tensors.append(tensors)
        datas = zip(*split_tensors)
    return list(datas)


def merge_data(datas):
    if isinstance(datas[0], torch.Tensor):
        data = torch.cat([tensor for tensor in datas])
    else:
        rotated = [tensors for tensors in datas]
        data_buf = []
        for tensors in zip(*rotated):
            data_buf.append(torch.cat(tensors))
        data = tuple(data_buf)
    return data


def task_schedule(num_micro_batch, num_partition):
    '''
    :return:  (i_micro_batch, i_partition)
    '''
    schedules = []
    for step in range(num_micro_batch + num_partition - 1):
        schedule = []
        for i_partition in range(max(0, step - num_micro_batch + 1), 1 + min(step, num_partition - 1)):
            schedule.append((step - i_partition, i_partition))
        schedules.append(schedule)
    return schedules


class GPipe(torch.nn.Module):
    def __init__(self, module, balance, chunks, devices=None, checkpoint='except_last'):
        super().__init__()

        if devices is None:
            if torch.cuda.is_available():
                devices = [torch.device('cuda:{}'.format(i % torch.cuda.device_count())) for i in range(len(balance))]
            else:
                devices = [torch.device('cpu') for _ in balance]
        self.devices = devices

        ############### 模型切分 ###############
        self.partitions = split_module(module, balance, devices)

        ############### 记录需要checkpointing的模型分片 ###############
        if checkpoint == 'always':
            self.checkpoint_step = list(range(len(balance)))
        elif checkpoint == 'except_last':
            self.checkpoint_step = list(range(len(balance) - 1))
        elif checkpoint == 'never':
            self.checkpoint_step = []
        else:
            raise ValueError('checkpoint must be one of "always", "except_last", or "never"')

        self.chunks = chunks

        ############### 创建streams ###############
        self.copy_streams = [[new_stream(device) for _ in range(chunks)] for device in devices]
        self.calc_streams = [default_stream(device) for device in devices]

    def forward(self, input):
        ############### 将数据切分成micro-batches ###############
        micro_batches = split_data(input, self.chunks)

        ############### 为每个partition分配任务 ###############
        for schedule in task_schedule(len(micro_batches), len((self.partitions))):
            for i_micro_batch, i_partition in schedule:
                micro_batch = micro_batches[i_micro_batch]

                if i_partition > 0:
                    ############### 执行copy_stream ###############
                    micro_batch = copy(micro_batch,
                                       self.copy_streams[i_partition - 1][i_micro_batch],
                                       self.copy_streams[i_partition][i_micro_batch])

                    ############### calc_stream等待copy_stream(输入) ###############
                    micro_batch = wait(micro_batch,
                                       self.calc_streams[i_partition],
                                       self.copy_streams[i_partition][i_micro_batch])

                ############### 执行compute/checkpoint ###############
                if i_partition in self.checkpoint_step:
                    ckp = Checkpointing(self.calc_streams[i_partition], self.partitions[i_partition], micro_batch)
                    micro_batch = ckp.checkpoint()
                else:
                    ckp = None
                    micro_batch = self.partitions[i_partition](micro_batch)

                ############### copy_stream等待calc_stream(输出) ###############
                if i_partition < len(self.partitions) - 1:
                    micro_batch = wait(micro_batch,
                                       self.copy_streams[i_partition][i_micro_batch],
                                       self.calc_streams[i_partition])

                ############### 执行recompute ###############
                if i_partition in self.checkpoint_step:
                    phony = ckp.recompute(micro_batch)
                    micro_batch = Join.apply(micro_batch, phony)

                ############### 更新micro_batches的阶段性结果 ###############
                micro_batches[i_micro_batch] = micro_batch

        output = merge_data(micro_batches)
        return output


if __name__ == "__main__":
    pass

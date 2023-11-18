# -*- coding: utf-8 -*-
# @Time    : 2023/10/31 19:53
# @Author  : Patrick
# @Email   : firechecking@gmail.com
# @File    : DistributedDataParallel.py
# @Software: CleanParallel
# @Description: DistributedDataParallel

from collections import OrderedDict
from itertools import chain

import torch
import torch.distributed as dist


def print_rank_0(*args, **kwargs):
    if dist.get_rank() == 0:
        print(*args, **kwargs)


class BucketAllReducer():
    def __init__(self, names_and_parameters_list, bucket_size):
        self.names_and_parameters_list = names_and_parameters_list
        self.param_to_idx = {param: i for i, (_, param) in enumerate(names_and_parameters_list)}
        self.idx_to_name_param = {i: (name, param) for i, (name, param) in enumerate(names_and_parameters_list)}

        self.bucket_size = bucket_size
        self.need_rebuild_buckets = True
        self.temp_buckets = {}
        self.temp_bucket_size = {}
        self.buckets = []

        self.buckets_ready_count = {}

    def delay_all_reduce(self, name, tensor):
        # print_rank_0(f'delay_all_reduce on tensor: {name}')
        if self.need_rebuild_buckets:
            ############### 不同数据类型的tensor需要分开 ###############
            tp = tensor.type()
            if tp not in self.temp_buckets: self.temp_buckets[tp] = []
            self.temp_buckets[tp].append((name, tensor))

            ############### 对bucket内的tensor大小进行累加 ###############
            self.temp_bucket_size[tp] = self.temp_bucket_size.get(tp, 0) + tensor.numel() * tensor.element_size()

            ############### 如果满足bucket尺寸，存入正式buckets中 ###############
            if self.temp_bucket_size[tp] >= self.bucket_size:
                self.buckets.append(self.temp_buckets[tp])
                self.temp_buckets[tp] = []
                self.temp_bucket_size[tp] = 0
        else:
            ############### 找到param对应的bucket, 记录bucket已经获取的grad次数 ###############
            bucket_id = self._param_name_to_bucket_id(name)
            self.buckets_ready_count[bucket_id] = self.buckets_ready_count.get(bucket_id, 0) + 1

            ############### 当某个bucket已获取所有grad后，对bucket进行all-reduce ###############
            if self.buckets_ready_count[bucket_id] == len(self.buckets[bucket_id]):
                # print_rank_0('all reduce on tensors: {}'.format([name for name, param in self.buckets[bucket_id]]))
                self.all_reduce([param.grad for name, param in self.buckets[bucket_id]])

    def _param_name_to_bucket_id(self, param_name):
        if not hasattr(self, 'param_to_bucket'): self.param_to_bucket_id = {}
        if len(self.param_to_bucket_id) < 1:
            for bucket_id, bucket in enumerate(self.buckets):
                for name, _ in bucket:
                    self.param_to_bucket_id[name] = bucket_id
        return self.param_to_bucket_id[param_name]

    def on_backwards_end(self):
        # print_rank_0('on_backwards_end')
        if self.need_rebuild_buckets:
            ############### 确保所有临时buckets都已转为正式buckets ###############
            for tp, bucket in self.temp_buckets.items():
                if len(bucket) > 0:
                    self.buckets.append(bucket)
            self.need_rebuild_buckets = False  # buckets构建完成后，不需要再次重建
            # print_rank_0('build buckets: {}'.format([[name for name, _ in bucket] for bucket in self.buckets]))

            ############### 同步buckets，确保所有rank的buckets结构相同 ###############
            self._sync_buckets_structure()

            ############### 同步所有tensor的grad ###############
            tensors = [param.grad for name, param in chain(*self.buckets)]
            self.all_reduce(tensors)
            # print_rank_0('all-reduce all tensors')
        else:
            ############### 确保所有buckets都已经同步 ###############
            # print_rank_0('ensure that all buckets are allreduced')
            for i, bucket in enumerate(self.buckets):
                if len(bucket) < 1: continue
                if self.buckets_ready_count[i] != len(bucket):
                    print(f'ready_count: {self.buckets_ready_count[i]}, bucket_size: {len(bucket)}')
                    raise RuntimeError("Some param buckets were not allreduced.")

    def all_reduce(self, tensors, process_group=None):
        buckets = self._split_tensors_to_buckets_by_type_and_device(tensors)
        for _, bucket in buckets.items():
            ############### 将bucket中的多个tensor展平，并合并成1个tensor ###############
            coalesced = self.flatten_bucket(bucket)
            dist.all_reduce(coalesced, group=process_group)  # TODO: 在side-stream中执行
            coalesced = coalesced / dist.get_world_size()

        ############### 用已同步的tensor值替换原始值 ###############
        for synced_tensor, tensor in zip(self.unflatten_bucket(coalesced, bucket), bucket):
            tensor.data.copy_(synced_tensor.data)

    def _sync_buckets_structure(self):
        ############### 组织buckets的结构信息 ###############
        num_buckets = len(self.buckets)
        bucket_sizes = [len(bucket) for bucket in self.buckets]
        bucket_ids = [self.param_to_idx[param] for _, param in chain(*self.buckets)]

        buckets_structure_tensor = torch.IntTensor([num_buckets] + bucket_sizes + bucket_ids)

        ############### 从rank0广播buckets结构信息 ###############
        dist.broadcast(buckets_structure_tensor, 0)

        ############### 使用buckets结构信息重建buckets ###############
        info = [int(entry) for entry in buckets_structure_tensor]

        num_buckets = info[0]
        bucket_sizes = info[1:num_buckets + 1]
        flattened_bucket_ids = info[num_buckets + 1:]

        self.buckets = []
        flat_i = 0
        for i in range(num_buckets):
            bucket = []
            for _ in range(bucket_sizes[i]):
                bucket.append(self.idx_to_name_param[flattened_bucket_ids[flat_i]])
                flat_i += 1
            self.buckets.append(bucket)

    @classmethod
    def broadcast(cls, tensors, src=0, process_group=None):
        ############### 按照tensor类型切分，确保bucket中只有一种数据类型 ###############
        buckets = cls._split_tensors_to_buckets_by_type_and_device(tensors)
        for tp, bucket in buckets.items():
            ############### 将bucket中的多个tensor展平，并合并成1个tensor ###############
            coalesced = cls.flatten_bucket(bucket)
            dist.broadcast(coalesced, src, group=process_group)

        ############### 将已同步的合并tensor还原成原来的多个tensor，并将原值替换为合并后的值 ###############
        for synced_tensor, tensor in zip(cls.unflatten_bucket(coalesced, bucket), bucket):
            tensor.data.copy_(synced_tensor.data)

    @classmethod
    def _split_tensors_to_buckets_by_type_and_device(cls, tensors):
        buckets = OrderedDict()
        for tensor in tensors:
            tp, device = tensor.type(), tensor.device
            if (tp, device) not in buckets:
                buckets[(tp, device)] = []
            buckets[(tp, device)].append(tensor)
        return buckets

    @classmethod
    def flatten_bucket(cls, bucket):
        return torch._utils._flatten_dense_tensors(bucket)

    @classmethod
    def unflatten_bucket(cls, coalesced, bucket):
        return torch._utils._unflatten_dense_tensors(coalesced, bucket)


class DistributedDataParallel(torch.nn.Module):
    def __init__(self, module, bucket_size_mb=25):
        super(DistributedDataParallel, self).__init__()
        world_size = dist.get_world_size()
        self.module = module

        ############### 同步模型parameters、buffers ###############
        if world_size > 1:
            module_states = []
            for name, param in module.named_parameters():
                module_states.append(param.detach())
            for name, buffer in module.named_buffers():
                module_states.append(buffer.detach())

            BucketAllReducer.broadcast(module_states)

        ############### 实例化BucketAllReducer，broadcast、all-reduce都封装在BucketAllReducer中 ###############
        params = set()
        names_and_parameters = []
        for name, param in module.named_parameters():
            if (param.requires_grad == True) and (param not in params):
                params.add(param)
                names_and_parameters.append((name, param))
        self.bucket_all_reducer = BucketAllReducer(names_and_parameters_list=names_and_parameters, bucket_size=bucket_size_mb * 1024 * 1024)

        ############### 注册grad_hook ###############
        self.need_queue_callback = True
        self.grad_accs = []
        for name, param in names_and_parameters:
            grad_fn = self._register_grad_hook(name, param)
            self.grad_accs.append(grad_fn)  # 将grad_fn保存下来，防止变量被释放掉

    def _register_grad_hook(self, name, param):
        def grad_hook(*unused_grads):
            ############### 将已获得grad的param传给bucket_all_reducer实例 ###############
            self.bucket_all_reducer.delay_all_reduce(name, param)
            if self.need_queue_callback:
                ############### queue_callback的方法在backward完成后触发，一次计算图只需要注册一次 ###############
                torch.autograd.Variable._execution_engine.queue_callback(self.bucket_all_reducer.on_backwards_end)
                self.need_queue_callback = False

        ############### 叶子节点的grad_fn为None，需要通过一个临时的上游节点，来获取叶子节点的AccumulateGrad ###############
        tmp_param = param.expand_as(param)
        grad_fn = tmp_param.grad_fn.next_functions[0][0]
        grad_fn.register_hook(grad_hook)
        return grad_fn

    def forward(self, *args, **kwargs):
        self.need_queue_callback = True
        self.bucket_all_reducer.buckets_ready_count = {}
        return self.module(*args, **kwargs)


if __name__ == "__main__":
    pass

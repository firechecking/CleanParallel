# -*- coding: utf-8 -*-
# @Time    : 2023/10/31 19:07
# @Author  : Patrick
# @Email   : firechecking@gmail.com
# @File    : exp_mnist_ddp.py
# @Software: CleanParallel
# @Description: exp_mnist_ddp

import os, random
import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision import transforms

use_ddp = 'CleanParallel'
if use_ddp == "torch":
    from torch.nn.parallel.distributed import DistributedDataParallel as DDP

    use_ddp = 'torch.nn.parallel.distributed'
else:
    from CleanParallel.DataParallel.DistributedDataParallel import DistributedDataParallel as DDP

    use_ddp = 'CleanParallel.DataParallel.DistributedDataParallel'
from CleanParallel.DataParallel.DistributedDataParallel import print_rank_0


class DemoModel(torch.nn.Module):
    def __init__(self):
        super(DemoModel, self).__init__()

        self.linear1 = torch.nn.Linear(28 * 28, 28)
        self.linear2 = torch.nn.Linear(28, 10)

    def forward(self, input, target=None):
        x = input.view([-1, 28 * 28])
        x = self.linear1(x)
        x = torch.nn.functional.relu(x)
        output = self.linear2(x)
        if target is not None:
            criterion = torch.nn.CrossEntropyLoss()
            loss = criterion(output, target)
            return output, loss
        return output, None


def train(data_root, save_checkpoint):
    rank = dist.get_rank()

    ############### 根据rank区分不同的device (gpu) ###############
    device = torch.device(f'cuda:{rank}' if torch.cuda.is_available() else 'cpu')
    model = DemoModel().to(device)
    model.train()
    # print(f'rank: {rank}, model-device: {next(model.parameters()).device}')
    print_rank_0(f'use_ddp: {use_ddp}')
    model = DDP(model)

    optimzier = torch.optim.AdamW(model.parameters(), lr=0.001)

    mnist = MNIST(root=data_root,
                  train=True, download=True, transform=transforms.ToTensor())
    datasampler = torch.utils.data.distributed.DistributedSampler(mnist, shuffle=True)
    train_loader = DataLoader(mnist, batch_size=128, sampler=datasampler)

    steps = 0
    for i in range(3):
        train_loader.sampler.set_epoch(i)
        for batch in train_loader:
            steps += 1
            # print_rank_0(f'\n====== step: {steps} ======')

            ############### 数据和模型要位于同一个device ###############
            output, loss = model(batch[0].to(device), batch[1].to(device))

            optimzier.zero_grad()
            loss.backward()
            optimzier.step()

        print(f'rank: {rank}, epoch: {i}, step: {steps}, loss: {loss.data}')
        if rank == 0:
            torch.save(model.module.state_dict(), save_checkpoint)


def eval(data_root, checkpoint):
    ############### 这里简单处理,模型和数据直接在cpu上进行eval ###############
    model = DemoModel()
    model.eval()
    model.load_state_dict(torch.load(checkpoint, map_location=torch.device('cpu')))

    mnist = MNIST(root=data_root, train=False, download=False, transform=transforms.ToTensor())
    eval_loader = DataLoader(mnist, batch_size=1, shuffle=True)

    result = {}
    for i, batch in enumerate(eval_loader):
        output, loss = model(batch[0])
        if torch.argmax(output, dim=-1).item() == batch[1].item():
            result['right'] = result.get('right', 0) + 1
        else:
            result['wrong'] = result.get('wrong', 0) + 1
    print(result)


def init_parallel(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12345'
    dist.init_process_group('nccl' if torch.cuda.is_available() else 'gloo', rank=rank, world_size=world_size)


def init_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def parallel_main(rank, world_size):
    init_seed(999)
    init_parallel(rank, world_size)
    # print(f'进程启动，rank={rank}')

    data_root, save_checkpoint = './data', 'mnist_model.bin'
    train(data_root, save_checkpoint)
    if rank == 0:
        eval(data_root, save_checkpoint)


if __name__ == '__main__':
    world_size = 4
    mp.spawn(parallel_main,
             args=(world_size,),
             nprocs=world_size,
             join=True)

# -*- coding: utf-8 -*-
# @Time    : 2023/9/18 19:53
# @Author  : Patrick
# @Email   : firechecking@gmail.com
# @File    : exp_mnist_amp.py
# @Software: CleanParallel
# @Description: exp_mnist_amp

import random
import numpy as np
import torch
from torch.utils.data import DataLoader

from torchvision.datasets import MNIST
from torchvision import transforms


class ApexWrapper():
    def __init__(self, opt_level):
        self.opt_level = opt_level

    def process_model_optimizer_fn(self, model, optimizer):
        from apex import amp
        model, optimizer = amp.initialize(model, optimizer, opt_level=self.opt_level)
        return model, optimizer

    def process_loss_backward(self, loss, optimizer):
        from apex import amp
        with amp.scale_loss(loss, optimizer) as scaled_loss:
            scaled_loss.backward()
        self.scaled_loss = scaled_loss


class AmpWrapper():
    def __init__(self, opt_level):
        self.opt_level = opt_level

    def process_model_optimizer_fn(self, model, optimizer):
        from CleanParallel import amp
        model, optimizer = amp.initialize(model, optimizer, opt_level=self.opt_level)
        return model, optimizer

    def process_loss_backward(self, loss, optimizer):
        from CleanParallel import amp
        with amp.scale_loss(loss, optimizer) as scaled_loss:
            scaled_loss.backward()
        self.scaled_loss = scaled_loss


class NullWrapper():
    def __init__(self, opt_level='O0'):
        self.opt_level = opt_level

    def process_model_optimizer_fn(self, model, optimizer):
        return model, optimizer

    def process_loss_backward(self, loss, optimizer):
        loss.backward()
        self.scaled_loss = loss


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


def train(save_checkpoint, amp_wrapper):
    model = DemoModel().cuda() if torch.cuda.is_available() else DemoModel()
    model.train()
    optimzier = torch.optim.AdamW(model.parameters(), lr=0.001)

    model, optimizer = amp_wrapper.process_model_optimizer_fn(model, optimzier)

    mnist = MNIST(root='./data', train=True, download=True, transform=transforms.ToTensor())
    train_loader = DataLoader(mnist, batch_size=128, shuffle=True)
    steps = 0
    for i in range(10):
        for batch in train_loader:
            steps += 1
            output, loss = model(batch[0].cuda(), batch[1].cuda()) if torch.cuda.is_available() else model(batch[0], batch[1])
            optimzier.zero_grad()

            amp_wrapper.process_loss_backward(loss, optimizer)

            optimzier.step()

        print(f'epoch: {i}, step: {steps}, loss: {loss.data}, scaled_loss: {amp_wrapper.scaled_loss.data}')

        torch.save(model.state_dict(), save_checkpoint)


def eval(checkpoint):
    model = DemoModel()
    model.eval()
    model.load_state_dict(torch.load(checkpoint))

    mnist = MNIST(root='./data', train=False, download=False, transform=transforms.ToTensor())
    eval_loader = DataLoader(mnist, batch_size=1, shuffle=True)

    result = {}
    for i, batch in enumerate(eval_loader):
        output, loss = model(batch[0])
        if torch.argmax(output, dim=-1).item() == batch[1].item():
            result['right'] = result.get('right', 0) + 1
        else:
            result['wrong'] = result.get('wrong', 0) + 1
    print(result)


def init_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


if __name__ == "__main__":
    init_seed(999)
    save_checkpoint = 'mnist_model.bin'
    # train(save_checkpoint, ApexWrapper('O2'))
    # train(save_checkpoint, NullWrapper())
    train(save_checkpoint, AmpWrapper('O2'))
    eval(save_checkpoint)

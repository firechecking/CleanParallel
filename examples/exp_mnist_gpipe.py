import random
import numpy as np

import torch
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision import transforms

# from torchgpipe import GPipe
from CleanParallel.PipelineParallel.parallel import GPipe


def build_model():
    linear1 = torch.nn.Linear(28 * 28, 28)
    relu = torch.nn.ReLU()
    linear2 = torch.nn.Linear(28, 10)
    return torch.nn.Sequential(linear1, relu, linear2)


def train(save_checkpoint, data_dir):
    model = build_model()
    model.train()
    model = GPipe(model, chunks=4, balance=[1, 1, 1],
                  devices=['cuda:0', 'cuda:1', 'cuda:2'] if torch.cuda.is_available() else ['cpu', 'cpu', 'cpu'],
                  checkpoint='except_last')

    optimzier = torch.optim.AdamW(model.parameters(), lr=0.001)

    mnist = MNIST(root=data_dir, train=True, download=True, transform=transforms.ToTensor())

    train_loader = DataLoader(mnist, batch_size=128, shuffle=True)

    steps = 0
    criterion = torch.nn.CrossEntropyLoss()
    for i in range(3):
        for batch in train_loader:
            steps += 1
            input, target = (batch[0].cuda(), batch[1].cuda()) if torch.cuda.is_available() else (batch[0], batch[1])
            input = input.view([-1, 28 * 28])
            output = model(input)

            if hasattr(model, 'devices'):
                target = target.to(model.devices[-1])
            loss = criterion(output, target)

            optimzier.zero_grad()

            loss.backward()

            optimzier.step()

        print(f'epoch: {i}, step: {steps}, loss: {loss.data}')

        torch.save(model.state_dict(), save_checkpoint)


def eval(checkpoint, data_dir):
    model = build_model()
    model.eval()
    model = GPipe(model, chunks=4, balance=[1, 1, 1],
                  devices=['cuda:0', 'cuda:1', 'cuda:2'] if torch.cuda.is_available() else ['cpu', 'cpu', 'cpu'],
                  checkpoint='except_last')

    model.load_state_dict(torch.load(checkpoint))

    mnist = MNIST(root=data_dir, train=False, download=False, transform=transforms.ToTensor())
    eval_loader = DataLoader(mnist, batch_size=1, shuffle=True)

    result = {}
    for i, batch in enumerate(eval_loader):
        input = batch[0].view([-1, 28 * 28])
        input = input.cuda() if torch.cuda.is_available() else input
        output = model(input)
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
    data_dir = './data'
    train(save_checkpoint, data_dir)
    eval(save_checkpoint, data_dir)

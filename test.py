import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import SGD, Adam
from spikingjelly.activation_based import learning, layer, neuron, functional
from torchvision import datasets, transforms
from tqdm import tqdm

T = 8
N = 2
C = 3
H = 32
W = 32
lr = 0.1
tau_pre = 2.
tau_post = 100.
step_mode = 'm'


def f_weight(x):
    return torch.clamp(x, -1, 1.)


net = nn.Sequential(
    layer.Conv2d(1, 16, kernel_size=3, stride=1, padding=1, bias=False),
    neuron.IFNode(),
    layer.MaxPool2d(2, 2),
    layer.Conv2d(16, 16, kernel_size=3, stride=1, padding=1, bias=False),
    neuron.IFNode(),
    layer.MaxPool2d(2, 2),
    layer.Flatten(),
    layer.Linear(16 * 7 * 7, 64, bias=False),
    neuron.IFNode(),
    layer.Linear(64, 10, bias=False),
    neuron.IFNode(),
)

functional.set_step_mode(net, step_mode)

if __name__ == '__main__':
    x = torch.rand(size=[5, 32, 1, 28, 28])
    print(net(x).shape)
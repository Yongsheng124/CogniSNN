import sys
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
lr = 0.001
tau_pre = 2.
tau_post = 100.
step_mode = 'm'


def f_weight(x):
    return torch.clamp(x, -1, 1.)


# class Net(nn.Module):
#     def __init__(self):
#         super(Net, self).__init__()
#         self.conv1 = nn.Sequential(
#             layer.Conv2d(1, 16, kernel_size=3, stride=1, padding=1, bias=False),
#             neuron.IFNode(),
#             layer.MaxPool2d(2, 2)
#         )
#         self.conv2 = nn.Sequential(
#             layer.Conv2d(16, 16, kernel_size=3, stride=1, padding=1, bias=False),
#             neuron.IFNode(),
#             layer.MaxPool2d(2, 2)
#         )
#         self.conv3 = nn.Sequential(
#             layer.Flatten(),
#             layer.Linear(16 * 7 * 7, 64, bias=False),
#             neuron.IFNode(),
#             layer.Linear(64, 10, bias=False),
#             neuron.IFNode(),
#         )
#
#     def forward(self, x):
#         x = self.conv1(x)
#         x = self.conv2(x)
#         x = self.conv3(x)
#         return x


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
# x = torch.rand(size=[5, 1, 1, 28, 28])
# y = net(x)
# print(y.shape)

# 禁用 TorchScript 编译
# neuron.IFNode.jit_hard_reset = False

instances_stdp = (layer.Conv2d,)

stdp_learners = []

for i in range(net.__len__()):
    if isinstance(net[i], instances_stdp):
        stdp_learners.append(
            learning.STDPLearner(step_mode=step_mode, synapse=net[i], sn=net[i + 1], tau_pre=tau_pre, tau_post=tau_post,
                                 f_pre=f_weight, f_post=f_weight)
        )

params_stdp = []
for m in net.modules():
    if isinstance(m, instances_stdp):
        for p in m.parameters():
            params_stdp.append(p)

params_stdp_set = set(params_stdp)
params_gradient_descent = []
for p in net.parameters():
    if p not in params_stdp_set:
        params_gradient_descent.append(p)

device = torch.device("cuda")
epochs = 30

transform = transforms.Compose([
    transforms.ToTensor()
])
net = net.to(device)
trainset = datasets.MNIST(root="E:\\datasets\\MNIST", train=True, download=False, transform=transform)
testset = datasets.MNIST(root="E:\\datasets\\MNIST", train=False, download=False, transform=transform)

train_data_loader = torch.utils.data.DataLoader(
    trainset, batch_size=32, num_workers=0, pin_memory=True)

test_data_loader = torch.utils.data.DataLoader(
    testset, batch_size=32, num_workers=0, pin_memory=True)

loss_fn = nn.CrossEntropyLoss().to(device)

optimizer_gd = Adam(params_gradient_descent, lr=lr)
optimizer_stdp = SGD(params_stdp, lr=lr, momentum=0.)

# lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=64, gamma=0.1)
epoch_list = []
test_acc_list = []
train_acc_list = []
train_loss_list = []
test_loss_list = []
max_test_acc = 0.
total = sum([param.nelement() for param in net.parameters()])
print("Number of parameter: % .2fM" % (total / 1e6))
time1 = time.time()
for epoch in range(epochs):

    epoch_list.append(epoch + 1)
    start_time = time.time()
    net.train()
    step = 0
    train_loss = 0
    train_acc = 0
    train_samples = 0
    print("---------第{}轮训练开始--------".format(epoch + 1))
    for data in tqdm(train_data_loader, desc="epoch " + str(epoch + 1), mininterval=1):

        optimizer_gd.zero_grad()
        optimizer_stdp.zero_grad()
        img, label = data
        img = img.to(device)
        img = img.repeat(5, 1, 1, 1, 1)  # [N, C, H, W] -> [T, N, C, H, W]

        #print(img.shape)
        label = label.to(device)
        output = net(img).mean(0)  # 这里不需要再进行T仿真时长是因为在model中的forward方法里面已经做了
        #print(output.shape)
        #sys.exit()

        loss = loss_fn(output, label)
        #print(loss)
        loss.backward()

        optimizer_stdp.zero_grad()  # ************************************ 不懂为什么要清零，等一下试一下 ************************************

        for i in range(stdp_learners.__len__()):
            stdp_learners[i].step(on_grad=True)
        optimizer_gd.step()
        optimizer_stdp.step()

        functional.reset_net(net)
        for i in range(stdp_learners.__len__()):
            stdp_learners[i].reset()

        train_samples += label.numel()
        train_loss += loss.item() * label.numel()
        train_acc += (output.argmax(1) == label).float().sum().item()
        step += 1
        if step % 100 == 0:
            print("[Epoch {0:4d}] Loss: {1:2.3f} Acc: {2:.3f}%".format(epoch + 1, loss.data,
                                                                       (train_acc * 100 / train_samples)), end='\n')
    train_time = time.time()
    train_speed = train_samples / (train_time - start_time)  # 一秒多少张图
    train_loss /= train_samples  # 每张图的平均损失
    train_acc /= train_samples
    print(
        "Epoch{}: Train_acc {}; Train_loss {}; Time of train {}; Speed of train {};".format(
            epoch + 1, train_acc, train_loss,
            (train_time - start_time), train_speed))

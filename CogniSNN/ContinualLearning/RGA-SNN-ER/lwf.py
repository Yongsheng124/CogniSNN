import argparse
import copy
import os.path
import re
import sys
import time
import torch.nn.functional as F
import torch
from torch import nn
from tqdm import tqdm
from torchvision import datasets, transforms
from smodel import Model
from spikingjelly.clock_driven import functional
from spikingjelly.datasets.dvs128_gesture import DVS128Gesture
import matplotlib.pyplot as plt
import os
from torch.cuda import amp

_seed_ = 2020
import random

random.seed(2020)
torch.manual_seed(_seed_)  # use torch.manual_seed() to seed the RNG for all devices (both CPU and CUDA)
torch.cuda.manual_seed_all(_seed_)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
import numpy as np
import json

np.random.seed(_seed_)


def kaiming_normal_init(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
    elif isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, nonlinearity='sigmoid')


def test(net, testloader, device, out_features):
    net.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data in tqdm(testloader, desc="test", mininterval=1):
            inputs, targets = data
            inputs, targets = inputs.to(device), targets.to(device)
            inputs = inputs.repeat(4, 1, 1, 1, 1)
            targets += out_features
            outputs = net(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            functional.reset_net(net)
    print("CIFAR10's Test Acc = ", correct / total)
    # Save checkpoint.
    acc = correct / total

    return acc


def val(net, valloader, device, T):
    net.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data in tqdm(valloader, desc="evaluation", mininterval=1):
            inputs, targets = data
            inputs, targets = inputs.to(device), targets.to(device)
            inputs = inputs.repeat(T, 1, 1, 1, 1)
            outputs = net(inputs)
            _, predicted_old = outputs.max(1)
            total += targets.size(0)
            correct += predicted_old.eq(targets).sum().item()
            functional.reset_net(net)
    print("CIFAR100's Acc = ", correct / total)
    return correct / total


def train(net, old_net, epoch, trainloader, out_features, device, optimizer, T, criterion, alpha):
    print('\nEpoch: %d' % (epoch + 1))
    net.eval()
    old_net.eval()
    train_loss = 0
    correct = 0
    total = 0
    step = 0
    for data in tqdm(trainloader, desc="train", mininterval=1):
        inputs, targets = data
        inputs, targets = inputs.to(device), targets.to(device)
        inputs = inputs.repeat(4, 1, 1, 1, 1)
        targets += out_features
        optimizer.zero_grad()
        outputs = net(inputs)
        soft_target = old_net(inputs)
        output_dict = net.monitor()
        old_output_dict = old_net.monitor()
        loss_other_path = 0
        freeze_nodes = [22, 24, 27, 29, 30]
        for freeze_node in freeze_nodes:
            loss_other_path += F.mse_loss(old_output_dict[freeze_node], output_dict[freeze_node])
            # for key, value in output_dict.items():
        #     print(key, value.shape)
        # output_23 = output_dict[23]
        # output_24 = output_dict[24]
        # print(output_24.shape)
        # print(output_23.shape)
        # import torch.nn.functional as F
        # mse_loss = F.mse_loss(output_24, output_23)
        # print(mse_loss)
        #
        # p_tensor = output_24.mean(0)
        # q_tensor = output_23.mean(0)
        # p_tensor = F.softmax(p_tensor, dim=-1)
        # q_tensor = F.softmax(q_tensor, dim=-1)
        # kl_divergence = F.kl_div(q_tensor.log(), p_tensor, reduction='mean')
        # print(f"The KL divergence D(P || Q) is: {kl_divergence.item()}")
        #
        # loss_other_path = 0
        loss3 = loss_other_path
        # Cross entropy between output of the new task and label
        loss1 = criterion(outputs, targets)
        # Using the new softmax to handle outputs
        outputs_S = F.softmax(outputs[:, :out_features] / T, dim=1)
        outputs_T = F.softmax(soft_target[:, :out_features] / T, dim=1)
        # Cross entropy between output of the old task and output of the old model
        loss2 = outputs_T.mul(-1 * torch.log(outputs_S))
        loss2 = loss2.sum(1)
        loss2 = loss2.mean() * T * T
        loss = loss1 * alpha + loss2 * (1 - alpha)/2 + loss3 * (1 - alpha)/2
        #
        loss.backward(retain_graph=True)
        optimizer.step()
        functional.reset_net(net)
        functional.reset_net(old_net)

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        step += 1
        if step % 100 == 0:
            print(correct / total)
    print("CIFAR10's Train Acc = ", correct / total)
    return train_loss / total


def main(node_list, edge_list):
    parser = argparse.ArgumentParser('parameters')

    parser.add_argument('--T', type=int, default=4, help='time step of neuron, (default: 5)')
    parser.add_argument('--epochs', type=int, default=200, help='number of epochs, (default: 100)')
    parser.add_argument('--p', type=float, default=0.2, help='graph probability, (default: 0.75)')
    parser.add_argument('--c', type=int, default=109,
                        help='channel count for each node, (example: 78, 109, 154), (default: 154)')
    parser.add_argument('--k', type=int, default=4,
                        help='each node is connected to k nearest neighbors in ring topology, (default: 4)')
    parser.add_argument('--m', type=int, default=3,
                        help='number of edges to attach from a new node to existing nodes, (default: 5)')
    parser.add_argument('--graph-mode', type=str, default="ER",
                        help="random graph, (Example: ER, WS, BA), (default: ER)")
    parser.add_argument('--node-num', type=int, default=30, help="Number of graph node (default n=32)")
    parser.add_argument('--learning-rate', type=float, default=0.001, help='learning rate, (default: 1e-1)')

    T = 2
    alpha = 0.05

    parser.add_argument('--batch-size', type=int, default=48, help='batch size, (default: 100)')
    parser.add_argument('--model-mode', type=str, default="CIFAR10",
                        help='CIFAR10, CIFAR100, SMALL_REGIME, REGULAR_REGIME, (default: CIFAR10)')
    parser.add_argument('--dataset-mode', type=str, default="CIFAR10",
                        help='Which dataset to use? (Example, CIFAR10, CIFAR100, MNIST), (default: CIFAR10)')
    parser.add_argument('--is-train', type=bool, default=True, help="True if training, False if test. (default: True)")
    parser.add_argument('--load-model', type=bool, default=False)
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='Momentum for SGD. Adam will not use momentum')
    parser.add_argument('--wd', '--weight-decay', default=5e-4, type=float,
                        metavar='W', help='weight decay (default: 0)',
                        dest='weight_decay')

    parser.add_argument('--device', default="cuda:0", type=str)
    parser.add_argument('--workers', default="16", type=int)

    parser.add_argument('--data-dir', default="/home/hys/datasets/CIFAR100", type=str)

    args = parser.parse_args()

    device = args.device

    net = Model(args.node_num, args.p, args.c, args.c, args.graph_mode, args.model_mode, args.dataset_mode,
                args.is_train).to(device)

    net_orig = copy.deepcopy(net)

    orr = torch.load("Origin_Net.pth.pth")
    net.load_state_dict(orr)
    net_orig.load_state_dict(orr)

    for name, param in net.named_parameters():
        # 先关系第一层卷积层，不让他继续学习
        conv_pattern = r"conv1\.*"
        if re.match(conv_pattern, name):
            param.requires_grad = False
        # 在关闭随机网络，也就是特征提取层，不让他继续学习
        pattern1 = r"randwire1\.module_list\.(\d+)\.*"
        if re.match(pattern1, name):
            param.requires_grad = False

    graph_matrix = {0: [], 1: [0], 2: [0], 3: [1], 4: [0], 5: [2], 6: [5], 7: [3], 8: [3], 9: [3, 4], 10: [4, 5],
                    11: [0], 12: [9], 13: [0], 14: [3, 11], 15: [10], 16: [1, 9], 17: [12, 13], 18: [12],
                    19: [7, 12, 13], 20: [6, 12, 13], 21: [12], 22: [1, 4, 8, 16, 17, 20], 23: [1, 9, 16, 18],
                    24: [7, 8, 17, 21], 25: [13, 15, 19], 26: [4, 11, 19, 23, 25], 27: [4, 8, 9, 11, 14],
                    28: [2, 5, 10, 11, 13, 16, 17, 19, 20, 21, 26], 29: [1, 4, 5, 7, 13, 17, 21, 28],
                    30: [11, 15, 17, 25], 31: [22, 24, 27, 29, 30]}
    # 除开路径和分类层之外，公共的地方，也就是第一层卷积层，第一个初始节点和最后一个输出节点都不应该训练
    for train_node in node_list:
        pattern1 = r"randwire1\.module_list\." + str(train_node) + r"\.unit\..*"
        for name, param in net.named_parameters():
            if train_node == 0:
                pattern_0 = r"randwire1\.module_list\.0\.pooling_unit\..*"
                if re.match(pattern_0, name):
                    param.requires_grad = True
            if re.match(pattern1, name):
                param.requires_grad = True
    for train_edge in edge_list:
        in_node = train_edge[0]
        out_node = train_edge[1]
        index = graph_matrix[out_node].index(in_node)
        for name, param in net.named_parameters():
            if name == "randwire1.module_list." + str(out_node) + ".weights" \
                    or name == "randwire1.module_list." + str(out_node) + ".weights." + str(index):
                param.requires_grad = True
    # for name, param in net.named_parameters():
    #     pattern1 = r"randwire1\.module_list\.(\d+)\.*"
    #     if re.match(pattern1, name):
    #         param.requires_grad = True
    for name, param in net.named_parameters():
        print(name, param.requires_grad)

    data_name = "CIFAR10"  # or MNIST
    if data_name == "CIFAR10":
        transform = transforms.Compose([
            # transforms.RandomHorizontalFlip(),  # 随机水平翻转
            # transforms.RandomCrop(32, padding=4),  # 随机裁剪并填充
            transforms.ToTensor(),  # 转为tensor
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),  # 标准化
        ])

        test_transform = transforms.Compose([
            transforms.ToTensor(),  # 转为tensor
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),  # 标准化
        ])

        # 加载训练集
        train_dataset = datasets.CIFAR10(root="/home/hys/datasets/CIFAR10", train=True, download=False,
                                         transform=transform)

        # 加载测试集
        test_dataset = datasets.CIFAR10(root="/home/hys/datasets/CIFAR10", train=False, download=False,
                                        transform=test_transform)
    elif data_name == "MNIST":
        train_transform = transforms.Compose([
            # 该函数用于对图像进行随机裁剪,crop表示随机裁剪的尺寸,scale表示随机裁剪的尺寸范围
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
            transforms.Resize((32, 32)),
            transforms.RandomHorizontalFlip(),
        ])
        test_transform = transforms.Compose([
            # 该函数用于对图像进行随机裁剪,crop表示随机裁剪的尺寸,scale表示随机裁剪的尺寸范围
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
            transforms.Resize((32, 32)),
        ])

        train_dataset = datasets.MNIST(root="/home/hys/datasets/mnist", train=True, download=False,
                                       transform=train_transform)
        test_dataset = datasets.MNIST(root="/home/hys/datasets/mnist", train=False, download=False,
                                      transform=test_transform)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True
    )

    CIFAR100_TRAIN_MEAN = (0.5070751592371323, 0.48654887331495095, 0.4409178433670343)
    CIFAR100_TRAIN_STD = (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)

    mean = CIFAR100_TRAIN_MEAN
    std = CIFAR100_TRAIN_STD

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    val_set = datasets.CIFAR100(root=args.data_dir, train=False, download=False, transform=transform_test)

    val_loader = torch.utils.data.DataLoader(
        val_set, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)

    num_new_class = 10
    print(net.CIFAR_classifier)
    in_features = net.CIFAR_classifier.in_features
    out_features = net.CIFAR_classifier.out_features

    weight = net.CIFAR_classifier.weight.data
    bias = net.CIFAR_classifier.bias.data

    new_out_features = num_new_class + out_features
    print(new_out_features)
    new_fc = nn.Linear(in_features, new_out_features, bias=True)
    kaiming_normal_init(new_fc.weight)

    new_fc.weight.data[:out_features] = weight
    new_fc.bias.data[:out_features] = bias

    net.CIFAR_classifier = new_fc
    net = net.to(device)
    net_orig = net_orig.to(device)

    print(net.CIFAR_classifier)

    # val(net, val_loader, device, args.T)
    loss_fn = nn.CrossEntropyLoss().to(device)

    epochs = args.epochs

    # 5e-4
    optimizer = torch.optim.SGD(
        filter(lambda p: p.requires_grad, net.parameters()), lr=args.learning_rate, momentum=args.momentum,
        weight_decay=args.weight_decay)
    # lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=64, gamma=0.1)
    print(out_features)
    max_acc = 0.
    for epoch in range(epochs):
        train(net=net, old_net=net_orig, epoch=epoch, trainloader=train_loader, optimizer=optimizer, criterion=loss_fn,
              device=device, T=T, alpha=alpha, out_features=out_features)
        new_data_acc = test(net=net, testloader=test_loader, device=device, out_features=out_features)
        old_data_acc = val(net=net, valloader=val_loader, device=device, T=args.T)
        total_acc = new_data_acc + old_data_acc
        if total_acc > max_acc:
            max_acc = total_acc
            print([old_data_acc, new_data_acc])
        # if old_data_acc < new_data_acc:
        #     break


if __name__ == "__main__":
    # [0, 1, 3, 9, 12, 19, 25, 26, 28, 29, 31]
    #  examine if need the start node and end node
    # node = {1, 3, 9, 12, 19, 25, 26, 28, 29}
    # edge = {(0, 1), (1, 3), (3, 9), (9, 12), (12, 19), (19, 25), (25, 26), (26, 28), (28, 29), (29, 31)}
    node = {11, 27}
    edge = {(0, 11), (11, 27), (27, 31)}

    # node = {}
    # edge = {}
    main(node, edge)

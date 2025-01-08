import argparse
import os.path
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


def save_args_to_file(*args):
    with open('config_and_results.txt', 'w') as file:
        for arg in args:
            para = str(arg)
            para = para.replace(", ", ",\n")
            para = 'config: ' + para
            file.write(para + '\n')


def draw_plot(epoch_list, train_loss_list, train_acc_list, val_acc_list, test_loss_list):
    plt.figure(figsize=(10, 4))
    plt.subplot(121)
    plt.plot(epoch_list, test_loss_list, label='test loss')
    plt.plot(epoch_list, train_loss_list, label='training loss')
    plt.legend()

    plt.subplot(122)
    plt.plot(epoch_list, train_acc_list, label='train acc')
    plt.plot(epoch_list, val_acc_list, label='validation acc')
    plt.legend()

    if os.path.isdir('./plot'):
        plt.savefig('./plot/conv1-or.png')

    else:
        os.makedirs('./plot')
        plt.savefig('./plot/conv1-or.png')
    plt.close()


def main():
    parser = argparse.ArgumentParser('parameters')

    parser.add_argument('--T', type=int, default=4, help='time step of neuron, (default: 5)')
    parser.add_argument('--epochs', type=int, default=192, help='number of epochs, (default: 100)')
    parser.add_argument('--p', type=float, default=0.75, help='graph probability, (default: 0.75)')
    parser.add_argument('--c', type=int, default=109,
                        help='channel count for each node, (example: 78, 109, 154), (default: 154)')
    parser.add_argument('--k', type=int, default=4,
                        help='each node is connected to k nearest neighbors in ring topology, (default: 4)')
    parser.add_argument('--m', type=int, default=3,
                        help='number of edges to attach from a new node to existing nodes, (default: 5)')
    parser.add_argument('--graph-mode', type=str, default="WS",
                        help="random graph, (Example: ER, WS, BA), (default: ER)")
    parser.add_argument('--node-num', type=int, default=30, help="Number of graph node (default n=32)")
    parser.add_argument('--learning-rate', type=float, default=0.1, help='learning rate, (default: 1e-1)')
    parser.add_argument('--batch-size', type=int, default=16, help='batch size, (default: 100)')
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

    parser.add_argument('--data-dir', default="E:\\datasets\\CIFAR100", type=str)

    args = parser.parse_args()

    config_and_results = {}
    save_args_to_file(args)

    device = args.device

    net = Model(args.node_num, args.p, args.c, args.c, args.graph_mode, args.model_mode, args.dataset_mode,
                args.is_train).to(device)

    CIFAR100_TRAIN_MEAN = (0.5070751592371323, 0.48654887331495095, 0.4409178433670343)
    CIFAR100_TRAIN_STD = (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)

    mean = CIFAR100_TRAIN_MEAN
    std = CIFAR100_TRAIN_STD

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    trainset = datasets.CIFAR100(root=args.data_dir, train=True, download=False, transform=transform_train)
    testset = datasets.CIFAR100(root=args.data_dir, train=False, download=False, transform=transform_test)
    train_data_loader = torch.utils.data.DataLoader(
        trainset, batch_size=args.batch_size, num_workers=0, pin_memory=True)

    test_data_loader = torch.utils.data.DataLoader(
        testset, batch_size=args.batch_size, num_workers=0, pin_memory=True)

    loss_fn = nn.CrossEntropyLoss().to(device)

    epochs = args.epochs
    # 5e-4
    optimizer = torch.optim.SGD(
        net.parameters(), lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=64, gamma=0.1)

    use_amp = True
    if use_amp:
        print("使用了混合精度训练")
        scaler = amp.GradScaler()
    else:
        scaler = None

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
            optimizer.zero_grad()
            img, label = data
            img = img.to(device)
            img = img.repeat(args.T, 1, 1, 1, 1)  # [N, C, H, W] -> [T, N, C, H, W]
            label = label.to(device)
            if use_amp:
                with amp.autocast():
                    output = net(img)  # 这里不需要再进行T仿真时长是因为在model中的forward方法里面已经做了
                    loss = loss_fn(output, label)
            else:
                output = net(img)  # 这里不需要再进行T仿真时长是因为在model中的forward方法里面已经做了
                loss = loss_fn(output, label)

            if use_amp:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
            functional.reset_net(net)

            train_samples += label.numel()
            train_loss += loss.item() * label.numel()
            train_acc += (output.argmax(1) == label).float().sum().item()
            step += 1
            if step % 100 == 0:
                print("[Epoch {0:4d}] Loss: {1:2.3f} Acc: {2:.3f}%".format(epoch + 1, loss.data,
                                                                           (train_acc * 100 / train_samples)), end='')
                for param_group in optimizer.param_groups:
                    print(",  Current learning rate is: {}".format(param_group['lr']))

        train_time = time.time()
        train_speed = train_samples / (train_time - start_time)  # 一秒多少张图
        train_loss /= train_samples  # 每张图的平均损失
        train_acc /= train_samples
        print(
            "Epoch{}: Train_acc {}; Train_loss {}; Time of train {}; Speed of train {};".format(
                epoch + 1, train_acc, train_loss,
                (train_time - start_time), train_speed))
        for param_group in optimizer.param_groups:
            print("Current learning rate is: {}".format(param_group['lr']))
        lr_scheduler.step()

        net.eval()
        test_samples = 0
        test_acc = 0.
        test_loss = 0.
        with torch.no_grad():
            for data in tqdm(test_data_loader, desc="evaluation", mininterval=1):
                frame, label = data
                frame = frame.to(device)
                frame = frame.repeat(args.T, 1, 1, 1, 1)  # [N, C, H, W] -> [T, N, C, H, W]
                label = label.to(device)
                output = net(frame)
                loss = loss_fn(output, label)
                test_loss += loss.item() * label.numel()
                test_acc += (output.argmax(1) == label).float().sum().item()
                test_samples += label.numel()
                functional.reset_net(net)

            test_acc /= test_samples
            test_loss /= test_samples
            test_acc_list.append(test_acc)
            train_loss_list.append(train_loss)
            train_acc_list.append(train_acc)
            test_loss_list.append(test_loss)
            print('Epoch{}: Test set accuracy: {:.5f}, Best accuracy: {:.5f}'.format(epoch + 1, test_acc,
                                                                                     max_test_acc))
            if max_test_acc < test_acc:
                max_test_acc = test_acc
                torch.save(net.state_dict(), "Origin_Net.pth")

        draw_plot(epoch_list, train_loss_list, train_acc_list, test_acc_list, test_loss_list)

    time2 = time.time()
    train_time = time2 - time1
    # 将时间差转换为时、分、秒
    hours = int(train_time / 3600)
    minutes = int((train_time % 3600) / 60)
    seconds = int(train_time % 60)
    # 格式化为 "hours:minutes:seconds" 的字符串
    time_str = f"{hours:02d}:{minutes:02d}:{seconds:02d}"

    config_and_results['train_time'] = time_str
    config_and_results['max_test_acc'] = max_test_acc
    config_and_results['train_loss_list'] = train_loss_list
    config_and_results['train_acc_list'] = train_acc_list
    config_and_results['test_loss_list'] = test_loss_list
    config_and_results['test_acc_list'] = test_acc_list
    print("train_time:", time_str)

    # 将字符串写入文件
    with open('config_and_results.txt', 'a') as file:
        for key, value in config_and_results.items():
            dict_str = json.dumps({key: value}, indent=4)
            file.write(dict_str + '\n')


if __name__ == "__main__":
    main()

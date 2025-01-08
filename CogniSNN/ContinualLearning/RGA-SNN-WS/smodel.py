import copy

import torch
import torch.nn as nn
from spikingjelly.clock_driven import layer, encoding
from spikingjelly.clock_driven.neuron import MultiStepParametricLIFNode
# 使用泊松编码器
from thop import profile

from graph import RandomGraph


# 假设 x 是你的输入 Tensor, 形状为 [T, B, C, N, N]
# 将其分成小块，每个块的大小为 [T, B, C, M, M]
def split_tensor(x, input_size, output_size):
    # 假设 x 是你的输入 Tensor, 形状为 [T, B, C, N, N]
    N = input_size
    M = output_size

    # 确定行和列需要分割的次数
    rows, cols = N // M, N // M

    # 如果不能被M整除，那么我们需要添加更多的行/列
    if N % M != 0:
        rows += 1
        cols += 1

    # 判断是否需要对tensor进行填充
    if x.shape[-2] < M * rows or x.shape[-1] < M * cols:
        # 需要填充的维度，格式为: (左, 右, 上, 下)
        padding = (0, M * cols - x.shape[-1], 0, M * rows - x.shape[-2])
        x = torch.nn.functional.pad(x, padding, "constant", 0)

    slices = []
    for i in range(rows):
        for j in range(cols):
            tensor_slice = x[..., i * M: (i + 1) * M, j * M: (j + 1) * M]
            slices.append(tensor_slice)

    tensors = 0.
    # 现在，slices列表包含了多个[T, B, C, M, M]张量
    for tensor in slices:
        tensors += tensor
    tensors /= len(slices)
    return tensors


def is_pulse_sequence(tensor):
    return torch.all(torch.logical_or(tensor.eq(0), tensor.eq(1)))


encoder = encoding.PoissonEncoder()


def test_CIFAR_classifier():
    return nn.Linear(2048, 11, bias=True)


def max():
    return layer.SeqToANNContainer(nn.AvgPool2d((4, 4)))


def conv3x3(in_channels, out_channels):
    return nn.Sequential(
        layer.SeqToANNContainer(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=1, bias=False),
            nn.BatchNorm2d(out_channels),
        ),
        MultiStepParametricLIFNode(init_tau=2.0, detach_reset=True)
    )


def Flatten():
    return nn.Flatten(2)


class SEWBlock(nn.Module):
    def __init__(self, in_channels, mid_channels, connect_f=None):
        super(SEWBlock, self).__init__()
        self.connect_f = connect_f
        self.conv1 = nn.Sequential(
            conv3x3(in_channels, mid_channels),
        )
        self.conv2 = nn.Sequential(
            conv3x3(mid_channels, mid_channels),
        )

    def forward(self, x: torch.Tensor):
        out = self.conv1(x)
        identity = out
        out = self.conv2(out)
        out = 1. - ((1. - identity) * (1. - out))
        return out


class Unit(nn.Module):
    def __init__(self, in_channels, mid_channels, connect_f):
        super(Unit, self).__init__()
        self.sewblock = SEWBlock(in_channels, mid_channels, connect_f)

    def forward(self, x):
        out = self.sewblock(x)
        return out


class Pooling_Unit(nn.Module):
    def __init__(self, in_channels, mid_channels, connect_f):
        super(Pooling_Unit, self).__init__()
        self.sewblock = SEWBlock(in_channels, mid_channels, connect_f)
        self.avgp = layer.SeqToANNContainer(nn.AvgPool2d(2, 2))

    def forward(self, x):
        out = self.sewblock(x)
        out = self.avgp(out)
        return out


class Node(nn.Module):
    def __init__(self, in_degree, in_channels, out_channels):
        super(Node, self).__init__()
        self.in_degree = in_degree
        # in_degree是入度，对于一个结点来说，为指向自己的所有有向边的序号
        if len(self.in_degree) > 1:
            # self.weights = nn.Parameter(torch.zeros(len(self.in_degree), requires_grad=True))
            self.weights = nn.ParameterList(
                [nn.Parameter(torch.ones(1, requires_grad=True)) for _ in range(len(self.in_degree))])
            # 如果一个结点的入度大于1，就给他赋予可训练的权重参数，长度为入度数，值为1
        else:
            self.weights = nn.Parameter(torch.ones(1, requires_grad=True))
        self.unit = Unit(in_channels, out_channels, connect_f='ADD')
        # unit对应的即是文章中所说的Transformation

    def forward(self, *input):
        if len(self.in_degree) > 1:
            x = (input[0] * torch.sigmoid(self.weights[0]))
            for index in range(1, len(input)):
                x += (input[index] * torch.sigmoid(self.weights[index]))
            # 这里其实就是对每一维的输入*可训练的参数权重，然后，加权求和
            out = self.unit(x)
        else:
            out = self.unit(input[0])
        return out

    def get_weights(self):
        return self.weights


# Reporting 2,
# In the paper, they said "The aggregation is done by weighted sum with learnable positive weights".
class Pooling_Node(nn.Module):
    def __init__(self, in_degree, in_channels, out_channels):
        super(Pooling_Node, self).__init__()
        self.in_degree = in_degree
        # in_degree是入度，对于一个结点来说，为指向自己的所有有向边的序号
        if len(self.in_degree) > 1:
            # self.weights = nn.Parameter(torch.zeros(len(self.in_degree), requires_grad=True))
            self.weights = nn.ParameterList(
                [nn.Parameter(torch.ones(1, requires_grad=True)) for _ in range(len(self.in_degree))])
            # 如果一个结点的入度大于1，就给他赋予可训练的权重参数，长度为入度数，值为1
        else:
            self.weights = nn.Parameter(torch.ones(1, requires_grad=True))
            # nn.Parameter(torch.ones(1, requires_grad=True))
        self.pooling_unit = Pooling_Unit(in_channels, out_channels, connect_f='ADD')
        # unit对应的即是文章中所说的Transformation

    def forward(self, *input):
        if len(self.in_degree) > 1:
            x = (input[0] * torch.sigmoid(self.weights[0]))
            for index in range(1, len(input)):
                x += (input[index] * torch.sigmoid(self.weights[index]))
            # 这里其实就是对每一维的输入*可训练的参数权重，然后，加权求和
            out = self.pooling_unit(x)
        else:
            out = self.pooling_unit(input[0])
        return out

    def get_weights(self):
        return self.weights
    # 这里的node其实就是一个单元体节点，就是文章里面说的单元体，不过出度复制的问题貌似暂时没有实现


class RandWire(nn.Module):
    def __init__(self, node_num, p, in_channels, out_channels, graph_mode, is_train, name):
        super(RandWire, self).__init__()
        self.node_num = node_num
        self.p = p
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.graph_mode = graph_mode
        self.is_train = is_train
        self.name = name
        self.memory = {}  # memory 是一个字典统计每一个模块的输出
        self.wire_weights = {}

        graph_node = RandomGraph(self.node_num, self.p, graph_mode=graph_mode)
        if self.is_train is True:
            print("is_train: True")
            graph = graph_node.make_graph()
            self.nodes, self.in_edges = graph_node.get_graph_info(graph)
            # nodes是一个结点list，包括每一个节点的序号
            # 可视化图
            # graph_node.visualization_graph(self.in_edges)

            print(self.in_edges)

        # define input Node
        # in_edges[i] 是第i号节点的有向边的序列
        self.module_list = nn.ModuleList(
            [Pooling_Node(self.in_edges[0], self.in_channels, self.out_channels)])
        for node in self.nodes:
            if node > 0 and node != len(self.nodes) - 1:
                self.module_list.append(Node(self.in_edges[node], self.out_channels, self.out_channels))
            if node == len(self.nodes) - 1:
                self.module_list.append(Node(self.in_edges[node], self.out_channels, self.out_channels))

        # define the rest Node
        # self.module_list.extend(
        #     [Node(self.in_edges[node], self.out_channels, self.out_channels) for node in self.nodes if node > 0])
        # 把每个节点模块都累积到moduleList这个容器里面

    def forward(self, x):
        # memory 是一个字典统计每一个模块的输出
        # start vertex
        out = self.module_list[0].forward(x)
        # print(out.shape)
        self.memory[0] = out
        # memory保存Node 0的输出

        # 剩余的中间Node， 现在没有管最后一个Node
        for node in range(1, len(self.nodes)):
            # print(node, self.in_edges[node][0], self.in_edges[node])
            if len(self.in_edges[node]) > 1:
                # 如果Node的入度是大于0的
                # 他们的输出应该是他所有入度节点的输出然后送进自己的forward里（加权在自己forward里面管）
                out = self.module_list[node].forward(*[self.memory[in_vertex] for in_vertex in self.in_edges[node]])
            else:
                # 如果Node只有一个入度
                # 只需要把这个节点唯一的那个入度的输出传递给他就好了。
                out = self.module_list[node].forward(self.memory[self.in_edges[node][0]])
            # 保存到字典里面
            self.memory[node] = out

        # 现在memory里面有除了最后一个Output Node的所有输出。

        # Reporting 3,
        # How do I handle the last part?
        # It has two kinds of methods.
        # first, Think of the last module as a Node and collect the data by proceeding in the same way as the previous operation.
        # second, simply sum the data and export the output.

        # My Opinion
        # out = self.module_list[self.node_num + 1].forward(*[memory[in_vertex] for in_vertex in self.in_edges[self.node_num + 1]])

        # In paper
        # print("self.in_edges: ", self.in_edges[self.node_num + 1], self.in_edges[self.node_num + 1][0])
        # print("self.in_edges[self.node_num + 1][0]",self.in_edges[self.node_num + 1][0])
        # 我们现在取出output上一个节点，在这里是节点5的out出来
        out = self.memory[self.node_num + 1]

        for node in range(len(self.nodes)):
            self.wire_weights[node] = self.module_list[node].get_weights()
        return out

    def monitor(self):
        return self.memory

    def get_weights(self):
        return self.wire_weights


class Model(nn.Module):
    def __init__(self, node_num, p, in_channels, out_channels, graph_mode, model_mode, dataset_mode, is_train):
        super(Model, self).__init__()
        self.node_num = node_num
        self.p = p
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.graph_mode = graph_mode
        self.model_mode = model_mode
        self.is_train = is_train
        self.dataset_mode = dataset_mode

        self.num_classes = 11
        self.dropout_rate = 0.2
        self.memory = {}

        if self.model_mode is "CIFAR10":
            self.conv1 = nn.Sequential(
                layer.SeqToANNContainer(
                    nn.Conv2d(in_channels=3, out_channels=self.out_channels, kernel_size=3, padding=1, stride=1,
                              bias=False),
                    nn.BatchNorm2d(self.out_channels),
                ),
                MultiStepParametricLIFNode(init_tau=2.0, detach_reset=True))
            self.avgpool1 = layer.SeqToANNContainer(nn.AvgPool2d(2, 2))
            self.avgpool2 = layer.SeqToANNContainer(nn.AvgPool2d(4, 4))

            self.randwire1 = RandWire(self.node_num, self.p, self.in_channels, self.out_channels, self.graph_mode,
                                      self.is_train, name="conv1")
            # self.randwire2 = RandWire(self.node_num * 2, self.p, self.out_channels, self.out_channels * 2,
            #                           self.graph_mode,
            #                           self.is_train, name="conv2")
            #
            # self.randwire3 = RandWire(self.node_num * 2, self.p, self.out_channels * 2, self.out_channels * 4,
            #                           self.graph_mode,
            #                           self.is_train, name="conv3")

            # self.conv2 = nn.Sequential(
            #     layer.SeqToANNContainer(
            #         nn.Conv2d(in_channels=self.out_channels * 4, out_channels=1280, kernel_size=1),
            #         nn.BatchNorm2d(1280),
            #     ),
            #     MultiStepParametricLIFNode(init_tau=2.0, detach_reset=True))

            self.flatten = nn.Flatten(2)

            self.CIFAR_classifier = nn.Linear(self.out_channels * 2 * 2, 100, bias=True)

    def forward(self, x):
        # print(x.shape)
        out = self.conv1(x)
        out = self.avgpool1(out)
        # print(out.shape)
        self.memory['conv1'] = out
        out = self.randwire1(out)
        out = self.avgpool2(out)
        out = self.flatten(out)
        # print("flatten", out.shape)
        out = out.mean(0)
        out = self.CIFAR_classifier(out)
        # print(out.shape)
        return out

    def monitor(self):
        randwire_memory = self.randwire1.monitor()
        self.memory.update(randwire_memory)
        return self.memory

    def get_weights(self):
        return self.randwire.get_weights()


def cal_firing_rate(spike_seq):
    return spike_seq.flatten().mean(0)


if __name__ == "__main__":
    _seed_ = 2020
    import random

    random.seed(2020)
    torch.manual_seed(_seed_)  # use torch.manual_seed() to seed the RNG for all devices (both CPU and CUDA)
    torch.cuda.manual_seed_all(_seed_)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    x2 = torch.round(torch.rand((4, 1, 3, 32, 32))).to("cuda")
    net = Model(30, 0.75, 109, 109, 'WS', "CIFAR10", "CIFAR10", True).to(device="cuda")
    # print(net)
    print(net(x2).shape)
    Flops, params = profile(net, inputs=(x2,))
    print('Flops: % .4fG' % (Flops))  # 计算量
    print('params参数量: % .4fM' % (params / 1000000))  # 参数量：等价与上面的summary输出的Total params值

    output_dict = net.monitor()
    for key, value in output_dict.items():
        print(key, value.shape)
    output_23 = output_dict[23]
    output_24 = output_dict[24]
    print(output_24.shape)
    print(output_23.shape)
    import torch.nn.functional as F
    mse_loss = F.mse_loss(output_24,output_23)
    print(mse_loss)

    net_orig = copy.deepcopy(net)
    output_dict_old = net_orig.monitor()
    for key, value in output_dict_old.items():
        print(key, value.shape)

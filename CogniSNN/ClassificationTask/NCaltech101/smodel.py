import sys
import torch
import torch.nn as nn
from spikingjelly.clock_driven import layer, encoding
from spikingjelly.clock_driven.neuron import MultiStepParametricLIFNode
from thop import profile
from graph import RandomGraph


def conv3x3(in_channels, out_channels):
    return nn.Sequential(
        layer.SeqToANNContainer(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=1, bias=False),
            nn.BatchNorm2d(out_channels),
        ),
        MultiStepParametricLIFNode(init_tau=2.0, detach_reset=True)
    )


class SEWBlock(nn.Module):
    def __init__(self, in_channels, mid_channels):
        super(SEWBlock, self).__init__()
        self.conv1 = nn.Sequential(
            conv3x3(in_channels, mid_channels),
        )
        self.conv2 = nn.Sequential(
            conv3x3(mid_channels, in_channels),
        )

    def forward(self, x: torch.Tensor):
        out = self.conv1(x)
        identity = out
        out = self.conv2(out)
        out = 1. - ((1. - identity) * (1. - out))
        return out


class Unit(nn.Module):
    def __init__(self, in_channels, mid_channels):
        super(Unit, self).__init__()
        self.sewblock = SEWBlock(in_channels, mid_channels)

    def forward(self, x):
        out = self.sewblock(x)
        return out


class Pooling_Unit(nn.Module):
    def __init__(self, in_channels, mid_channels):
        super(Pooling_Unit, self).__init__()
        self.sewblock = SEWBlock(in_channels, mid_channels)
        self.avgp = layer.SeqToANNContainer(nn.AvgPool2d(2, 2))

    def forward(self, x):
        out = self.sewblock(x)
        out = self.avgp(out)
        return out


class Pooling(nn.Module):
    def __init__(self, input_size, output_size):
        super(Pooling, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.kernal = self.input_size // self.output_size
        self.pooling = layer.SeqToANNContainer(nn.AvgPool2d(self.kernal, self.kernal))

    def forward(self, x):
        out = self.pooling(x)
        return out


# Reporting 2,
# In the paper, they said "The aggregation is done by weighted sum with learnable positive weights".
class Pooling_Node(nn.Module):
    def __init__(self, in_degree, in_channels, out_channels):
        super(Pooling_Node, self).__init__()
        self.in_degree = in_degree
        # in_degree是入度，对于一个结点来说，为指向自己的所有有向边的序号
        if len(self.in_degree) > 1:

            self.weights = nn.Parameter(torch.ones(len(self.in_degree), requires_grad=True))
            # 如果一个结点的入度大于1，就给他赋予可训练的权重参数，长度为入度数，值为1
        else:
            self.weights = torch.ones(1)
        self.pooling_unit = Pooling_Unit(in_channels, out_channels)
        self.unit = Unit(in_channels, out_channels)
        # unit对应的即是文章中所说的Transformation

    def forward(self, *input):
        if len(self.in_degree) > 1:
            input = list(input)
            output_size = input[-1].shape[-1]
            for i in range(len(input) - 1):
                _, _, _, H, _ = input[i].shape
                if H == output_size:
                    continue
                pooling_layer = Pooling(H, output_size)
                input[i] = pooling_layer(input[i])
            x = (input[0] * torch.sigmoid(self.weights[0]))
            for index in range(1, len(input)):
                x += (input[index] * torch.sigmoid(self.weights[index]))
            # 这里其实就是对每一维的输入*可训练的参数权重，然后，加权求和
            if input[0].shape[-1] == 1:
                out = self.unit(x)
            else:
                out = self.pooling_unit(x)
            # 进入Transformation

            # different paper, add identity mapping
            # out += x
        else:
            if input[0].shape[-1] == 1:
                out = self.unit(input[0])
            else:
                out = self.pooling_unit(input[0])
        return out

    def get_weights(self):
        return self.weights
    # 这里的node其实就是一个单元体节点，就是文章里面说的单元体，不过出度复制的问题貌似暂时没有实现


class RandWire(nn.Module):
    def __init__(self, node_num, p, in_channels, out_channels, graph_mode, output_dir):
        super(RandWire, self).__init__()
        self.node_num = node_num
        self.p = p
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.graph_mode = graph_mode
        self.memory = {}  # memory 是一个字典统计每一个模块的输出
        self.wire_weights = {}
        self.output_dir = output_dir

        graph_node = RandomGraph(self.node_num, self.p, graph_mode=graph_mode)
        graph = graph_node.make_graph()
        self.nodes, self.in_edges = graph_node.get_graph_info(graph)
        # nodes是一个结点list，包括每一个节点的序号
        # 可视化图
        graph_node.visualization_graph(self.in_edges, self.output_dir)

        print(self.in_edges)
        # in_edges是一个字典，包括每一个节点是key，然后对应的value值即指向自己的有向边
        # define input Node
        # in_edges[i] 是第i号节点的有向边的序列
        self.module_list = nn.ModuleList(
            [Pooling_Node(self.in_edges[0], self.in_channels, self.out_channels)])
        for node in self.nodes:
            if node > 0:
                self.module_list.append(Pooling_Node(self.in_edges[node], self.out_channels, self.out_channels))

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
    def __init__(self, node_num, p, in_channels, out_channels, graph_mode, output_dir):
        super(Model, self).__init__()
        self.node_num = node_num
        self.p = p
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.graph_mode = graph_mode
        self.output_dir = output_dir

        self.num_classes = 101
        self.dropout_rate = 0.2
        self.memory = {}
        self.conv1 = nn.Sequential(
            layer.SeqToANNContainer(
                nn.Conv2d(in_channels=2, out_channels=self.out_channels, kernel_size=3, padding=1, stride=1,
                          bias=False),
                nn.BatchNorm2d(self.out_channels),
            ),
            MultiStepParametricLIFNode(init_tau=2.0, detach_reset=True))

        self.randwire = RandWire(self.node_num, self.p, self.in_channels, self.out_channels, self.graph_mode,
                                 self.output_dir)

        self.flatten = nn.Flatten(2)

        self.CIFAR_classifier = nn.Linear(self.out_channels, self.num_classes, bias=True)

    def forward(self, x):
        out = self.conv1(x)
        out = self.randwire(out)
        out = self.flatten(out)
        out = out.mean(0)
        out = self.CIFAR_classifier(out)
        return out

    def monitor(self):
        randwire_memory = self.randwire.monitor()
        self.memory.update(randwire_memory)
        return self.memory

    def get_weights(self):
        return self.randwire.get_weights()


def cal_firing_rate(spike_seq):
    return spike_seq.flatten().mean(0)


if __name__ == "__main__":
    x2 = torch.round(torch.rand((5, 1, 2, 48, 48))).to("cuda")
    net = Model(5, 0.75, 109, 109, 'WS', './test').to(device="cuda")

    Flops, params = profile(net, inputs=(x2,))
    print('Flops: %.4fG' % (Flops / 1e9))  # 将 FLOPs 转换为 GFLOPs
    print('params参数量: % .4fM' % (params / 1000000))
    # print(net)
    print(net(x2).shape)

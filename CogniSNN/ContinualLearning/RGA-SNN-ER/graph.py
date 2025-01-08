import networkx as nx
import os


class RandomGraph(object):
    def __init__(self, node_num, p, k=4, m=5, graph_mode="WS"):
        self.node_num = node_num
        self.p = p
        self.k = k
        self.m = m
        self.graph_mode = graph_mode

    def make_graph(self):
        # reference
        # https://networkx.github.io/documentation/networkx-1.9/reference/generators.html

        # Code details,
        # In the case of the nx.random_graphs module, we can give the random seeds as a parameter.
        # But I have implemented it to handle it in the module.
        print(self.graph_mode)
        print("k=", self.k)
        print("m=", self.m)
        if self.graph_mode is "ER":
            graph = nx.random_graphs.erdos_renyi_graph(self.node_num, self.p)
        elif self.graph_mode is "WS":
            graph = nx.random_graphs.connected_watts_strogatz_graph(self.node_num, self.k, self.p)
        elif self.graph_mode is "BA":
            graph = nx.random_graphs.barabasi_albert_graph(self.node_num, 3)

        return graph

    def get_graph_info(self, graph):
        in_edges = {}
        in_edges[0] = []
        nodes = [0]
        end = []
        for node in graph.nodes():
            neighbors = list(graph.neighbors(node))
            neighbors.sort()

            edges = []
            check = []
            for neighbor in neighbors:
                if node > neighbor:
                    edges.append(neighbor + 1)
                    check.append(neighbor)
            if not edges:
                edges.append(0)
            in_edges[node + 1] = edges
            if check == neighbors:
                end.append(node + 1)
            nodes.append(node + 1)
        in_edges[self.node_num + 1] = end
        nodes.append(self.node_num + 1)

        return nodes, in_edges
    def visualization_graph(self, in_edges):
        result = "digraph G {\n"
        for node, edges in in_edges.items():
            for edge in edges:
                result += f"    {edge} -> {node};\n"
        result += "}"
        # 保存到txt文件
        with open("graph.dot", "w") as file:
            file.write(result)

        import subprocess

        dot_file = "graph.dot"
        png_file = "example.png"

        # 执行命令
        command = f'dot -Tpng "{dot_file}" -o "{png_file}"'
        subprocess.run(command, shell=True, check=True)
        print("dot文件已成功转换为PNG图像文件。")


if __name__ == "__main__":
    _seed_ = 2020
    import random
    import torch
    random.seed(2020)
    torch.manual_seed(_seed_)  # use torch.manual_seed() to seed the RNG for all devices (both CPU and CUDA)
    torch.cuda.manual_seed_all(_seed_)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    graph_node = RandomGraph(30, 0.75, graph_mode="WS")
    graph = graph_node.make_graph()
    nodes, in_edges = graph_node.get_graph_info(graph)
    print(nodes)
    print(in_edges)
    # in_edges = {0: [], 1: [0], 2: [1], 3: [1, 2], 4: [3], 5: [2, 3, 4], 6: [5]}
    #
    #
    graph_node.visualization_graph(in_edges)

    # def convert_to_digraph(graph):
    #     result = "digraph G {\n"
    #     for node, edges in graph.items():
    #         for edge in edges:
    #             result += f"    {node} -> {edge};\n"
    #     result += "}"
    #
    #     return result
    #
    #
    # digraph = convert_to_digraph(in_edges)
    # print(digraph)
    # # 保存到txt文件
    # with open("graph.dot", "w") as file:
    #     file.write(digraph)
    #
    # import subprocess
    #
    # dot_file = "graph.dot"
    # png_file = "example.png"
    #
    # # 执行命令
    # command = f'dot -Tpng "{dot_file}" -o "{png_file}"'
    # subprocess.run(command, shell=True, check=True)
    #
    # print("dot文件已成功转换为PNG图像文件。")
    #

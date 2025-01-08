import argparse
import sys
import networkx as nx
import numpy as np
import torch


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
        if self.graph_mode == "ER":
            graph = nx.random_graphs.erdos_renyi_graph(self.node_num, self.p)
        elif self.graph_mode == "WS":
            graph = nx.random_graphs.connected_watts_strogatz_graph(self.node_num, self.k, self.p)
        elif self.graph_mode == "BA":
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


def get_path_node_and_edge(path_list):
    path_node = set()
    for path in path_list:
        path_node.update(path)
    path_node = set(sorted(path_node))
    path_edge = set()
    for path in path_list:
        for i in range(len(path) - 1):
            path_edge.add((path[i], path[i + 1]))
    return [path_node, path_edge]


def find_all_paths(matrix, start, end, path=[]):
    path = path + [start]
    if start == end:
        return [path]
    if start >= len(matrix):
        return []
    paths = []
    for node in range(len(matrix)):
        if matrix[start][node] == 1 and node not in path:
            new_paths = find_all_paths(matrix, node, end, path)
            for p in new_paths:
                paths.append(p)
    return paths


def calculate_centrality(adjacency_matrix):
    all_paths = find_all_paths(adjacency_matrix, 0, 31)
    total_paths = len(all_paths)

    node_centrality = {i: 0 for i in range(len(adjacency_matrix))}
    edge_centrality = {(i, j): 0 for i in range(len(adjacency_matrix)) for j in range(len(adjacency_matrix)) if
                       adjacency_matrix[i][j] == 1}

    for path in all_paths:
        for i in range(len(path)):
            node_centrality[path[i]] += 1
            if i < len(path) - 1:
                edge_centrality[(path[i], path[i + 1])] += 1
    for node in node_centrality:
        node_centrality[node] /= total_paths
    for edge in edge_centrality:
        edge_centrality[edge] /= total_paths

    return node_centrality, edge_centrality


def calculate_citial_path_high_low(graph_matrix):
    adjacency_matrix = [[0 for _ in range(32)] for _ in range(32)]

    for node, in_nodes in graph_matrix.items():
        if node == 0:
            continue
        for in_node in in_nodes:
            adjacency_matrix[in_node][node] = 1

    node_centrality, edge_centrality = calculate_centrality(adjacency_matrix)
    all_paths = find_all_paths(adjacency_matrix, 0, 31)
    print(len(all_paths))
    path_node_edge_score = []
    for path in all_paths:
        node_score = 0
        edge_score = 0
        for i in range(len(path)):
            node_score += node_centrality[path[i]]
            if i < len(path) - 1:
                edge_score += edge_centrality[(path[i], path[i + 1])]
        path_node_edge_score.append((path, node_score + edge_score))
    # print(path_node_edge_score)
    # print(len(path_node_edge_score))
    sorted_array = sorted(path_node_edge_score, key=lambda x: x[1], reverse=True)
    return sorted_array


if __name__ == '__main__':

    _seed_ = 2020
    import random

    random.seed(2020)
    graph_mode = "ER"  # ER

    p = None
    if graph_mode == "WS":
        p = 0.75
    elif graph_mode == "ER":
        p = 0.2
    graph_node = RandomGraph(30, p, graph_mode=graph_mode)
    # graph_node = RandomGraph(30, 0.2, graph_mode="ER")
    graph = graph_node.make_graph()

    nodes, in_edges = graph_node.get_graph_info(graph)
    print(in_edges)

    centrality_path = calculate_citial_path_high_low(in_edges)
    print("The critical path with high betweenness centrality:", centrality_path[0])
    print("The critical path with lwo betweenness centrality:", centrality_path[-1])

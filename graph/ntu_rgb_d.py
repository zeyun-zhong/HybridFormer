import sys
import numpy as np

sys.path.extend(['../'])
from graph import tools

num_node = 25
self_link = [(i, i) for i in range(num_node)]

# InfoGCN
# inward_ori_index = [
#     (2, 1), (2, 21), (21, 3), (3, 4), #head
#     (21, 5), (5, 6), (6, 7), (7, 8), (8, 23), (23, 22), # left arm
#     (21, 9), (9, 10), (10, 11), (11, 12), (12, 25), (25, 24), # right arm
#     (1, 13), (13, 14), (14, 15),(15, 16), # left leg
#     (1, 17), (17, 18),  (18, 19),  (19, 20) # right leg
# ]

# Hyperformer
inward_ori_index = [(1, 2), (2, 21), (3, 21), (4, 3), (5, 21), (6, 5), (7, 6),
                    (8, 7), (9, 21), (10, 9), (11, 10), (12, 11), (13, 1),
                    (14, 13), (15, 14), (16, 15), (17, 1), (18, 17), (19, 18),
                    (20, 19), (22, 8), (23, 8), (24, 12), (25, 12)]

hops = np.array(
    [[0., 1., 3., 4., 3., 4., 5., 6., 3., 4., 5., 6., 1., 2., 3., 4., 1., 2., 3., 4., 2., 7., 7., 7., 7.],
    [1., 0., 2., 3., 2., 3., 4., 5., 2., 3., 4., 5., 2., 3., 4., 5., 2., 3., 4., 5., 1., 6., 6., 6., 6.],
    [3., 2., 0., 1., 2., 3., 4., 5., 2., 3., 4., 5., 4., 5., 6., 7., 4., 5., 6., 7., 1., 6., 6., 6., 6.],
    [4., 3., 1., 0., 3., 4., 5., 6., 3., 4., 5., 6., 5., 6., 7., 8., 5., 6., 7., 8., 2., 7., 7., 7., 7.],
    [3., 2., 2., 3., 0., 1., 2., 3., 2., 3., 4., 5., 4., 5., 6., 7., 4., 5., 6., 7., 1., 4., 4., 6., 6.],
    [4., 3., 3., 4., 1., 0., 1., 2., 3., 4., 5., 6., 5., 6., 7., 8., 5., 6., 7., 8., 2., 3., 3., 7., 7.],
    [5., 4., 4., 5., 2., 1., 0., 1., 4., 5., 6., 7., 6., 7., 8., 9., 6., 7., 8., 9., 3., 2., 2., 8., 8.],
    [6., 5., 5., 6., 3., 2., 1., 0., 5., 6., 7., 8., 7., 8., 9., 10., 7., 8., 9., 10., 4., 1., 1., 9., 9.],
    [3., 2., 2., 3., 2., 3., 4., 5., 0., 1., 2., 3., 4., 5., 6., 7., 4., 5., 6., 7., 1., 6., 6., 4., 4.],
    [4., 3., 3., 4., 3., 4., 5., 6., 1., 0., 1., 2., 5., 6., 7., 8., 5., 6., 7., 8., 2., 7., 7., 3., 3.],
    [5., 4., 4., 5., 4., 5., 6., 7., 2., 1., 0., 1., 6., 7., 8., 9., 6., 7., 8., 9., 3., 8., 8., 2., 2.],
    [6., 5., 5., 6., 5., 6., 7., 8., 3., 2., 1., 0., 7., 8., 9., 10., 7., 8., 9., 10., 4., 9., 9., 1., 1.],
    [1., 2., 4., 5., 4., 5., 6., 7., 4., 5., 6., 7., 0., 1., 2., 3., 2., 3., 4., 5., 3., 8., 8., 8., 8.],
    [2., 3., 5., 6., 5., 6., 7., 8., 5., 6., 7., 8., 1., 0., 1., 2., 3., 4., 5., 6., 4., 9., 9., 9., 9.],
    [3., 4., 6., 7., 6., 7., 8., 9., 6., 7., 8., 9., 2., 1., 0., 1., 4., 5., 6., 7., 5., 10., 10., 10., 10.],
    [4., 5., 7., 8., 7., 8., 9., 10., 7., 8., 9., 10., 3., 2., 1., 0., 5., 6., 7., 8., 6., 11., 11., 11., 11.],
    [1., 2., 4., 5., 4., 5., 6., 7., 4., 5., 6., 7., 2., 3., 4., 5., 0., 1., 2., 3., 3., 8., 8., 8., 8.],
    [2., 3., 5., 6., 5., 6., 7., 8., 5., 6., 7., 8., 3., 4., 5., 6., 1., 0., 1., 2., 4., 9., 9., 9., 9.],
    [3., 4., 6., 7., 6., 7., 8., 9., 6., 7., 8., 9., 4., 5., 6., 7., 2., 1., 0., 1., 5., 10., 10., 10., 10.],
    [4., 5., 7., 8., 7., 8., 9., 10., 7., 8., 9., 10., 5., 6., 7., 8., 3., 2., 1., 0., 6., 11., 11., 11., 11.],
    [2., 1., 1., 2., 1., 2., 3., 4., 1., 2., 3., 4., 3., 4., 5., 6., 3., 4., 5., 6., 0., 5., 5., 5., 5.],
    [7., 6., 6., 7., 4., 3., 2., 1., 6., 7., 8., 9., 8., 9., 10., 11., 8., 9., 10., 11., 5., 0., 2., 10., 10.],
    [7., 6., 6., 7., 4., 3., 2., 1., 6., 7., 8., 9., 8., 9., 10., 11., 8., 9., 10., 11., 5., 2., 0., 10., 10.],
    [7., 6., 6., 7., 6., 7., 8., 9., 4., 3., 2., 1., 8., 9., 10., 11., 8., 9., 10., 11., 5., 10., 10., 0., 2.],
    [7., 6., 6., 7., 6., 7., 8., 9., 4., 3., 2., 1., 8., 9., 10., 11., 8., 9., 10., 11., 5., 10., 10., 2., 0.]
])

inward = [(i - 1, j - 1) for (i, j) in inward_ori_index]
outward = [(j, i) for (i, j) in inward]
neighbor = inward + outward

num_node_1 = 11
indices_1 = [0, 3, 5, 7, 9, 11, 13, 15, 17, 19, 20]
self_link_1 = [(i, i) for i in range(num_node_1)]
inward_ori_index_1 = [(1, 11), (2, 11), (3, 11), (4, 3), (5, 11), (6, 5), (7, 1), (8, 7), (9, 1), (10, 9)]
inward_1 = [(i - 1, j - 1) for (i, j) in inward_ori_index_1]
outward_1 = [(j, i) for (i, j) in inward_1]
neighbor_1 = inward_1 + outward_1

num_node_2 = 5
indices_2 = [3, 5, 6, 8, 10]
self_link_2 = [(i ,i) for i in range(num_node_2)]
inward_ori_index_2 = [(0, 4), (1, 4), (2, 4), (3, 4), (0, 1), (2, 3)]
inward_2 = [(i - 1, j - 1) for (i, j) in inward_ori_index_2]
outward_2 = [(j, i) for (i, j) in inward_2]
neighbor_2 = inward_2 + outward_2

class Graph:
    def __init__(self, labeling_mode='spatial', scale=1):
        self.num_node = num_node
        self.self_link = self_link
        self.inward = inward
        self.outward = outward
        self.neighbor = neighbor
        self.A_outward_binary = tools.get_adjacency_matrix(self.outward, self.num_node)
        self.A = self.get_adjacency_matrix(labeling_mode)
        self.A1 = tools.get_spatial_graph(num_node_1, self_link_1, inward_1, outward_1)
        self.A2 = tools.get_spatial_graph(num_node_2, self_link_2, inward_2, outward_2)
        self.A_binary = tools.edge2mat(neighbor, num_node)
        self.A_norm = tools.normalize_adjacency_matrix(self.A_binary + 2*np.eye(num_node))
        self.A_binary_K = tools.get_k_scale_graph(scale, self.A_binary)

        self.A_A1 = ((self.A_binary + np.eye(num_node)) / np.sum(self.A_binary + np.eye(self.A_binary.shape[0]), axis=1, keepdims=True))[indices_1]
        self.A1_A2 = tools.edge2mat(neighbor_1, num_node_1) + np.eye(num_node_1)
        self.A1_A2 = (self.A1_A2 / np.sum(self.A1_A2, axis=1, keepdims=True))[indices_2]

        self.hops = hops


    def get_adjacency_matrix(self, labeling_mode=None):
        if labeling_mode is None:
            return self.A
        if labeling_mode == 'spatial':
            A = tools.get_spatial_graph(num_node, self_link, inward, outward)
        else:
            raise ValueError()
        return A
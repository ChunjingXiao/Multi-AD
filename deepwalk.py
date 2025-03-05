import numpy as np
import networkx as nx
import random
from tqdm import tqdm


def read_edge(edge_txt):
    edges = np.loadtxt(edge_txt, dtype=np.int16)
    edges = [(edges[i, 0], edges[i, 1]) for i in range(edges.shape[0])]
    return edges


class deepwalk_hy():
    def __init__(self, node_num, edge_index, undirected=False) -> None:
        if undirected:
            self.G = nx.Graph()
        else:
            self.G = nx.DiGraph()
        self.G.add_nodes_from(list(range(node_num)))

        edge_list = [tuple(edge_index[:, i].tolist()) for i in range(edge_index.shape[1])]
        self.G.add_edges_from(edge_list)

        self.adjacency = np.array(nx.adjacency_matrix(self.G).todense())
        self.G_neighbor = {}
        for i in range(self.adjacency.shape[0]):
            self.G_neighbor[i] = []
            for j in range(self.adjacency.shape[0]):
                if i == j:
                    continue
                if self.adjacency[i, j] > 0.01:
                    self.G_neighbor[i].append(j)

    def random_walk(self, path_len, alpha=0, rand_iter=random.Random(9931), start=None):
        G = self.G_neighbor
        if start is not None:
            rand_path = [start]
        else:
            rand_path = [rand_iter.choice(list(G.keys()))]

        while len(rand_path) < path_len:
            current_pos = rand_path[-1]
            if len(G[current_pos]) > 0:
                if rand_iter.random() >= alpha:
                    rand_path.append(rand_iter.choice(G[current_pos]))
                else:
                    rand_path.append(rand_path[0])
            else:
                rand_path.append(rand_iter.choice(list(G.keys())))
                break
        return [str(node) for node in rand_path]

    def build_total_corpus(self, num_paths, path_length, alpha=0, rand_iter=random.Random(9999)):
        print("Start randomwalk.")
        total_walks = []
        self.node_sequences = {}
        G = self.G_neighbor
        nodes = list(G.keys())

        for cnt in tqdm(range(num_paths)):
            for node in nodes:
                print("node{}: {} sequence".format(node, cnt))
                walk = self.random_walk(path_length, rand_iter=rand_iter, alpha=alpha, start=node)
                total_walks.append(walk)

                if node not in self.node_sequences:
                    self.node_sequences[node] = []
                self.node_sequences[node].append(walk)

        return total_walks


    def concate(self, deepwalkseqlist, node_num, seq_num, length):
        graphseq = []
        node_sequences_dict = {}
        node_sequences_index_dict = {}

        for i in range(node_num):
            valid_sequences = True
            temp_sequences = []
            temp_indices = []

            for j in range(seq_num):
                deepwalkseqlist[i + j * node_num].reverse()
                sequence = deepwalkseqlist[i + j * node_num][:-1] + deepwalkseqlist[i + (j + 1) * node_num]

                if len(sequence) >= length:
                    temp_sequences.append(sequence)
                    temp_indices.append(len(graphseq) + len(temp_sequences) - 1)
                else:
                    valid_sequences = False
                    break

            if valid_sequences:
                node_sequences_dict[i] = temp_sequences
                node_sequences_index_dict[i] = temp_indices
                graphseq.extend(temp_sequences)

        return graphseq, node_sequences_index_dict


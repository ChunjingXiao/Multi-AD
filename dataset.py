from dgl.data.utils import load_graphs
import networkx as nx
import numpy as np
import torch


class Dataset:
    def __init__(self, name='tfinance', prefix='datasets/'):
        graph = load_graphs(prefix + name)[0][0]
        self.name = name
        self.attr = graph.ndata['feature']
        self.adj = graph.adj().to_dense()
        self.edge_index = torch.stack((graph.edges()[0], graph.edges()[1]))
        self.label = graph.ndata['label']
        self.num_edges = graph.num_edges()
        self.num_nodes = graph.num_nodes()
        self.graph = graph

    def get_node_attribute_matrix(self):
        nodes = list(self.graph.nodes())
        attributes = nx.get_node_attributes(self.graph)
        attribute_matrix = np.zeros((len(nodes), 1))
        for i, node in enumerate(nodes):
            attribute_matrix[i] = attributes.get(node, 0)
        return attribute_matrix

    def split(self, semi_supervised=True, trial_id=0):
        if semi_supervised:
            trial_id += 10
        self.graph.ndata['train_mask'] = self.graph.ndata['train_masks'][:,trial_id]
        self.graph.ndata['val_mask'] = self.graph.ndata['val_masks'][:,trial_id]
        self.graph.ndata['test_mask'] = self.graph.ndata['test_masks'][:,trial_id]
        print(self.graph.ndata['train_mask'].sum(), self.graph.ndata['val_mask'].sum(), self.graph.ndata['test_mask'].sum())


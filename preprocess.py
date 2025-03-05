import pickle
import numpy as np
import os
import torch
import torch.optim as optim
from deepwalk import deepwalk_hy
from dataset import Dataset
from args import args
from util import construct_label, get_seq_only_node
from models.gcn import GCN

def load_model(model_path, model, optimizer=None):
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    model.eval()
    return model, optimizer

def process_data(dataname):
    data_path = f'preprocessed/{dataname}/'

    dataset = Dataset(dataname)
    attr = dataset.attr
    adj = dataset.adj
    node_label = dataset.label
    edge_index = dataset.edge_index

    walker = deepwalk_hy(dataset.num_nodes, edge_index, undirected=True)
    seq_num = args.seq_num
    seq_length = args.seq_length / 2 + 1
    deepwalk_corpus = walker.build_total_corpus(seq_num * 2, seq_length)
    graphseq, graphseq_dict = walker.concate(deepwalk_corpus, node_num=dataset.num_nodes, seq_num=seq_num, length=args.seq_length)

    seq_label = construct_label(node_label, graphseq)

    graph_data = {
        'attr': attr,
        'adj': adj,
        'node_label': node_label,
        'edge_index': edge_index
    }
    os.makedirs(data_path, exist_ok=True)
    with open(os.path.join(data_path, 'graph_data.pkl'), 'wb') as f:
        pickle.dump(graph_data, f)

    graphseq_data = {
        'seq_data': get_seq_only_node(attr, graphseq),
        'seq_label': seq_label
    }
    os.makedirs(data_path, exist_ok=True)
    with open(os.path.join(data_path, 'graphseq_data.pkl'), 'wb') as f:
        pickle.dump(graphseq_data, f)

    mapping_data = {
        'graphseq_dict': graphseq_dict,
        'graphseq': graphseq
    }
    with open(os.path.join(data_path, 'mapping_data.pkl'), 'wb') as f:
        pickle.dump(mapping_data, f)

    node_indices = list(graphseq_dict.keys())
    label_1_indices = [i for i in node_indices if node_label[i] == 1]
    label_0_indices = [i for i in node_indices if node_label[i] == 0]

    np.random.seed(42)
    np.random.shuffle(label_1_indices)
    np.random.shuffle(label_0_indices)

    train_label_1_indices = label_1_indices[:347]
    train_label_0_indices = label_0_indices[:3014]
    train_label_0_indices_dengbi = label_0_indices[:347]

    test_label_1_indices = label_1_indices[347:]
    test_label_0_indices = label_0_indices[3014:]

    train_node_indices = train_label_1_indices + train_label_0_indices
    train_node_indices_dengbi = train_label_1_indices + train_label_0_indices_dengbi
    test_node_indices = test_label_1_indices + test_label_0_indices

    np.random.seed(55)
    np.random.shuffle(train_node_indices)
    np.random.shuffle(test_node_indices)

    split_node_indices = {
        'train_node_indices': train_node_indices,
        'gcn_train_node_indices': train_node_indices_dengbi,
        'test_node_indices': test_node_indices
    }

    with open(os.path.join(data_path, 'split_node_indices.pkl'), 'wb') as f:
        pickle.dump(split_node_indices, f)

    print("Data saved successfully!")

    model = GCN(attr.size(1), args.hidden_channels, 2, dropout=0.5)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    model, optimizer = load_model(f'checkpoints/{dataname}/gcn_model.pth', model, optimizer)

    attr_train = attr[train_node_indices]
    adj_train = adj[train_node_indices][:, train_node_indices]
    node_label_train = node_label[train_node_indices]

    row_indices, col_indices = torch.nonzero(adj_train, as_tuple=True)
    edge_index_train = torch.stack((row_indices, col_indices))

    walker_train = deepwalk_hy(attr_train.size(0), edge_index_train, undirected=True)
    deepwalk_corpus_train = walker_train.build_total_corpus(seq_num * 2, seq_length)
    graphseq_train, _ = walker_train.concate(deepwalk_corpus_train, node_num=attr_train.size(0), seq_num=seq_num, length=seq_length*2 - 1)

    seq_label_train = construct_label(node_label_train, graphseq_train)
    with torch.no_grad():
        _, node_embeddings_train = model(attr_train, adj_train)

    seq_data_train = get_seq_only_node(node_embeddings_train, graphseq_train)

    graphseq_data_train = {
        'seq_data': seq_data_train,
        'seq_label': seq_label_train
    }
    with open(os.path.join(data_path, 'train_graphseq_data.pkl'), 'wb') as f:
        pickle.dump(graphseq_data_train, f)

    print("Training data saved successfully!")

if __name__ == "__main__":
    process_data(args.dataname)

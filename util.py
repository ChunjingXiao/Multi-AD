from sklearn.metrics import roc_auc_score, f1_score, precision_score
import torch
import numpy as np

def evaluatation(result, labels):
    result = result.cpu().detach().numpy()
    labels = labels.cpu().detach().numpy()
    auc = roc_auc_score(labels, result)
    num_anomalies = np.sum(labels)-8
    threshold_index = np.argsort(result)[-num_anomalies]
    threshold = result[threshold_index]
    binary_result = (result >= threshold).astype(int)
    f1 = f1_score(labels, binary_result)
    precision = precision_score(labels, binary_result)
    return auc, f1, precision

def get_sequence_indices(node_indices, graphseq_dict):
    seq_indices = []
    for node_index in node_indices:
        seq_indices.extend(graphseq_dict[node_index])
    return seq_indices

def compute_f1(result, labels):
    result = result.cpu().detach().numpy()
    labels = labels.cpu().detach().numpy()
    return f1_score(labels, result)

def compute_ascore(seq_result, graph_dict, num_sequences_per_node=7):
    seq_result = torch.tensor(seq_result, dtype=torch.float32)
    num_nodes = len(seq_result) // num_sequences_per_node
    ascore = torch.zeros(num_nodes)

    for node_index in range(num_nodes):
        start_index = node_index * num_sequences_per_node
        end_index = start_index + num_sequences_per_node
        scores = seq_result[start_index:end_index]
        ascore[node_index] = torch.mean(scores)
    return ascore


def get_seq(node_embeddings, edge_embeddings, edge_index, seq_lists, padding_value=0):
    edge_dict = {}
    for j in range(edge_index.shape[1]):
        start_node, end_node = edge_index[0, j].item(), edge_index[1, j].item()
        edge_dict[(start_node, end_node)] = j
        edge_dict[(end_node, start_node)] = j

    all_seq_data = []

    for seq_list in seq_lists:
        seq_data = []

        for i in range(len(seq_list)):
            node_idx = int(seq_list[i])
            node_embedding = node_embeddings[node_idx]

            if node_embedding.dim() == 0:
                node_embedding = node_embedding.unsqueeze(0)

            seq_data.append(node_embedding)

            if i < len(seq_list) - 1:
                start_node = int(seq_list[i])
                end_node = int(seq_list[i + 1])
                edge_idx = edge_dict.get((start_node, end_node))

                if edge_idx is not None:
                    edge_embedding = edge_embeddings[edge_idx]

                    if edge_embedding.dim() == 0:
                        edge_embedding = edge_embedding.unsqueeze(0)

                    seq_data.append(edge_embedding)

        seq_data_flat = torch.cat(seq_data, dim=0)
        all_seq_data.append(seq_data_flat)

    all_seq_data_tensor = torch.stack(all_seq_data)

    return all_seq_data_tensor

def get_seq_only_node(node_embeddings, seq_lists, padding_value=0):
    all_seq_data = []

    for seq_list in seq_lists:
        seq_data = []

        for i in range(len(seq_list)):
            node_idx = int(seq_list[i])

            node_embedding = node_embeddings[node_idx]

            if node_embedding.dim() == 0:
                node_embedding = node_embedding.unsqueeze(0)

            seq_data.append(node_embedding)

        seq_data_flat = torch.cat(seq_data, dim=0)
        all_seq_data.append(seq_data_flat)

    all_seq_data_tensor = torch.stack(all_seq_data)

    return all_seq_data_tensor


def compute_edge_representation(node_embeddings, edge_index):
    src, dst = edge_index
    edge_embeddings = node_embeddings[src].clone()
    edge_embeddings.add_(node_embeddings[dst])
    edge_embeddings = edge_embeddings.float()
    edge_embeddings.div_(2)
    return edge_embeddings

def construct_label(node_labels, graph_seq):
    seq_labels = []

    for seq in graph_seq:
        seq_int = list(map(int, seq))
        if any(node_labels[idx] == 1 for idx in seq_int):
            seq_labels.append(1)
        else:
            seq_labels.append(0)

    return torch.tensor(seq_labels, dtype=torch.float32)


def compute_auc(result, labels):
    result = result.cpu().detach().numpy()
    labels = labels.cpu().detach().numpy()
    return roc_auc_score(labels, result)
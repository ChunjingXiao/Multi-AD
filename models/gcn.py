import argparse
from dataset import Dataset
import torch
import torch.nn.functional as F

import torch.nn as nn
import torch.optim as optim
import pickle
from util import compute_auc
from torch_geometric.data import Data
from models.layers import GraphConvolution

class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GCN, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nclass)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        return F.log_softmax(x, dim=1), x


def train(model, data, optimizer, criterion):
    model.train()
    optimizer.zero_grad()
    out, x = model(data.x, data.adj)
    loss = criterion(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return loss.item()


def validate(model, data):
    model.eval()
    with torch.no_grad():
        out, x = model(data.x, data.adj)
        pred = out.argmax(dim=1)
        auc = compute_auc(pred[data.val_mask], data.y[data.val_mask])
        correct = (pred[data.val_mask] == data.y[data.val_mask]).sum()
        acc = int(correct) / int(data.val_mask.sum())
    return acc, auc


def test(model, data):
    model.eval()
    with torch.no_grad():
        out, x = model(data.x, data.adj)
        pred = out.argmax(dim=1)
        auc = compute_auc(pred[data.test_mask], data.y[data.test_mask])
        correct = (pred[data.test_mask] == data.y[data.test_mask]).sum()
        acc = int(correct) / int(data.test_mask.sum())
    return acc, auc


def load_model(model_path, model, optimizer=None):
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    model.eval()
    return model, optimizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--hidden_channels', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=150)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--dropout', type=float, default=0.5)
    args = parser.parse_args()

    dataname = 'weibo'
    dataset = Dataset(dataname)

    x = torch.tensor(dataset.attr, dtype=torch.float)
    edge_index = torch.tensor(dataset.edge_index, dtype=torch.long)
    adj = torch.tensor(dataset.adj, dtype=torch.float)
    y = torch.tensor(dataset.label, dtype=torch.long)

    num_nodes = y.size(0)

    with open('../preprocessed/weibo/split_node_indices.pkl', 'rb') as f:
        data = pickle.load(f)
        train_node_indices = data['gcn_train_node_indices']
        test_node_indices = data['test_node_indices']

    train_indices = train_node_indices
    test_indices = test_node_indices

    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)

    train_mask[train_indices] = True
    test_mask[test_indices] = True

    data = Data(x=x, edge_index=edge_index, adj = adj, y=y, train_mask=train_mask, val_mask=test_mask, test_mask=test_mask)

    model = GCN(nfeat=x.size(1),
                nhid=args.hidden_channels,
                nclass=2,
                dropout=args.dropout)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(args.epochs):
        train_loss = train(model, data, optimizer, criterion)
        val_acc, val_auc = validate(model, data)
        test_acc, test_auc = test(model, data)
        print(f'Epoch: {epoch + 1:03d}, Loss: {train_loss:.4f}, Val Acc: {val_acc:.4f}, Test Acc: {test_acc:.4f}, '
              f'Val AUC: {val_auc:.4f}, Test AUC: {test_auc:.4f}')

    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }, 'checkpoints/'+dataname+'/sameBaseline/gcn_model.pth')
    print('Model saved to checkpoints/weibo/sameBaseline/gcn_model.pth')

if __name__ == '__main__':
    main()

import os.path
import pickle
import numpy as np
from util import evaluatation, compute_ascore, get_sequence_indices
import torch
import torch.utils.data
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import logging
from sklearn.metrics import roc_auc_score
from datetime import datetime
from models.transformer_based_model import TransformerForSequenceClassification
from args import args

def custom_collate_fn(batch):
    seq_data, label = zip(*batch)
    return torch.stack(seq_data), torch.tensor(label, dtype=torch.float).view(-1, 1)

class GraphSeqLoader(torch.utils.data.Dataset):
    def __init__(self, data_path, mode, win_size=None, step=None):
        super(GraphSeqLoader, self).__init__()
        self.data_path = data_path
        self.mode = mode
        self.win_size = win_size
        self.step = step
        self.load_data()

    def load_data(self):
        with open(self.data_path+'graphseq_data.pkl', 'rb') as f:
            data = pickle.load(f)
            self.seq_data = data['seq_data']
            self.seq_label = data['seq_label']

        with open(self.data_path+'graph_data.pkl', 'rb') as f:
            data = pickle.load(f)
            self.node_label = data['node_label']

        with open(self.data_path+'mapping_data.pkl', 'rb') as f:
            data = pickle.load(f)
            self.graphseq_dict = data['graphseq_dict']

        with open(self.data_path+'split_node_indices.pkl', 'rb') as f:
            data = pickle.load(f)
            train_node_indices = data['train_node_indices']
            # val_node_indices = data['val_node_indices']
            test_node_indices = data['test_node_indices']

        if self.mode in ['train', 'valid', 'test']:
            if self.mode == 'train':
                with open(self.data_path + 'train_graphseq_data.pkl', 'rb') as f:
                    data = pickle.load(f)
                    self.seq_data = data['seq_data']
                    self.seq_label = data['seq_label']
                self.node_indices = train_node_indices
                seq_indices = get_sequence_indices(train_node_indices, self.graphseq_dict)
                self.node_label = self.node_label[self.node_indices]
                self.graphseq_dict = {i: self.graphseq_dict[i] for i in self.node_indices}
            elif self.mode == 'valid':
                self.node_indices = test_node_indices
                seq_indices = get_sequence_indices(self.node_indices, self.graphseq_dict)
                self.seq_data = self.seq_data[seq_indices]
                self.seq_label = self.seq_label[seq_indices]
                self.node_label = self.node_label[self.node_indices]
                self.graphseq_dict = {i: self.graphseq_dict[i] for i in self.node_indices}
            elif self.mode == 'test':
                self.node_indices = test_node_indices
                seq_indices = get_sequence_indices(self.node_indices, self.graphseq_dict)
                self.seq_data = self.seq_data[seq_indices]
                self.seq_label = self.seq_label[seq_indices]
                self.node_label = self.node_label[self.node_indices]
                self.graphseq_dict = {i: self.graphseq_dict[i] for i in self.node_indices}

        self.window_data_list = self.seq_data
        self.window_label_list = self.seq_label

        print(f"===> {self.mode} data are loaded successfully!")

    def __len__(self):
        return len(self.window_data_list)

    def __getitem__(self, index):
        return self.window_data_list[index], self.window_label_list[index]

def train_model(dataname, timestamp, model, valid_loader, train_dataloader, valid_dataloader, criterion, optimizer, num_epochs=10, device='cuda'):
    model.to(device)
    best_valid_loss = float('inf')
    checkpoints_dir = "./checkpoints/"+dataname+'/'
    os.makedirs(checkpoints_dir, exist_ok=True)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for seq_data, labels in tqdm(train_dataloader, desc=f"Training Epoch {epoch + 1}/{num_epochs}"):
            seq_data, labels = seq_data.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(seq_data)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * seq_data.size(0)

        epoch_loss = running_loss / len(train_dataloader.dataset)
        logging.info(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}')
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}')

        valid_loss = validate_model(model, valid_loader, valid_dataloader, criterion, device)

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), './checkpoints/'+dataname+'/best_model'+timestamp+'.pth')
            logging.info(f'Saved best model with validation loss: {best_valid_loss:.4f}')
            print(f'Saved best model with validation loss: {best_valid_loss:.4f}')

    final_model_path = os.path.join(checkpoints_dir, 'final_model'+timestamp+'.pth')
    torch.save(model.state_dict(), final_model_path)
    logging.info(f'Saved final model at {final_model_path}')
    print(f'Saved final model at {final_model_path}')

def validate_model(model, valid_loader, valid_dataloader, criterion, device='cuda'):
    model.eval()
    running_loss = 0.0
    correct_predictions = 0
    all_labels = []
    all_outputs = []
    with torch.no_grad():
        for seq_data, labels in tqdm(valid_dataloader, desc="Validating"):
            seq_data, labels = seq_data.to(device), labels.to(device)
            outputs = model(seq_data)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * seq_data.size(0)

            preds = torch.round(torch.sigmoid(outputs))
            correct_predictions += torch.sum(preds == labels).item()

            all_labels.append(labels.cpu().numpy())
            all_outputs.append(torch.sigmoid(outputs).cpu().numpy())

    epoch_loss = running_loss / len(valid_dataloader.dataset)
    accuracy = correct_predictions / len(valid_dataloader.dataset)
    all_labels = np.concatenate(all_labels)
    all_outputs = np.concatenate(all_outputs)
    node_ascore = compute_ascore(all_outputs, valid_loader.graphseq_dict)
    node_auc, f1, pr = evaluatation(node_ascore, valid_loader.node_label)
    logging.info(f'Validation Loss: {epoch_loss:.4f}, Accuracy: {accuracy:.4f}')
    logging.info(f'Node AUC: {node_auc:.4f}, PR: {pr:.4f}')
    print(f'Validation Loss: {epoch_loss:.4f}, Accuracy: {accuracy:.4f}, ')
    print(f'Node AUC: {node_auc:.4f}, PR: {pr:.4f}')
    return epoch_loss

def test_model(dataname, timestamp, model, test_loader, test_dataloader, device='cuda'):
    model.load_state_dict(torch.load('checkpoints/'+dataname+'/final_model.pth'))
    model.to(device)
    model.eval()
    correct_predictions = 0
    all_labels = []
    all_outputs = []
    with torch.no_grad():
        for seq_data, labels in tqdm(test_dataloader, desc="Testing"):
            seq_data, labels = seq_data.to(device), labels.to(device)
            outputs = model(seq_data)
            preds = torch.round(torch.sigmoid(outputs))
            correct_predictions += torch.sum(preds == labels).item()

            all_labels.append(labels.cpu().numpy())
            all_outputs.append(torch.sigmoid(outputs).cpu().numpy())

    all_labels = np.concatenate(all_labels)
    all_outputs = np.concatenate(all_outputs)
    node_ascore = compute_ascore(all_outputs, test_loader.graphseq_dict)
    node_auc, f1, pr = evaluatation(node_ascore, test_loader.node_label)
    accuracy = correct_predictions / len(test_dataloader.dataset)
    auc = roc_auc_score(all_labels, all_outputs)
    logging.info(f'Test Accuracy: {accuracy:.4f}, AUC: {auc:.4f}, ')
    logging.info(f'Test Node_auc: {node_auc:.4f}, Node_f1: {f1:.4f}, Node_pr: {pr:.4f}')
    print(f'Test Accuracy: {accuracy:.4f}, AUC: {auc:.4f}, ')
    print(f'Test Node_auc: {node_auc:.4f}, Node_f1: {f1:.4f}, Node_pr: {pr:.4f}')

if __name__ == "__main__":


    dataname = args.dataname
    timestamp = datetime.now().strftime("%Y%m%d-%H%M")
    # logging.basicConfig(filename=f'./logs/{args.mode}_{timestamp}_{dataname}.log', level=logging.INFO,
    #                     format='%(asctime)s - %(levelname)s - %(message)s')

    if args.mode == 'train':
        graphseq_data_path = './preprocessed/{}/'.format(dataname)
        train_loader = GraphSeqLoader(graphseq_data_path, 'train', win_size=10, step=5)
        valid_loader = GraphSeqLoader(graphseq_data_path, 'valid', win_size=10, step=5)
        train_dataloader = torch.utils.data.DataLoader(train_loader, batch_size=256, shuffle=True, num_workers=4, collate_fn=custom_collate_fn)
        valid_dataloader = torch.utils.data.DataLoader(valid_loader, batch_size=256, shuffle=False, num_workers=4, collate_fn=custom_collate_fn)

        model = TransformerForSequenceClassification()
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        train_model(dataname, timestamp, model, valid_loader, train_dataloader, valid_dataloader, criterion, optimizer, num_epochs=10, device='cuda')
    elif args.mode == 'test':
        graphseq_data_path = './preprocessed/{}/'.format(dataname)
        test_loader = GraphSeqLoader(graphseq_data_path, 'test', win_size=10, step=5)
        test_dataloader = torch.utils.data.DataLoader(test_loader, batch_size=256, shuffle=False, num_workers=4, collate_fn=custom_collate_fn)

        model = TransformerForSequenceClassification()
        test_model(dataname, timestamp, model, test_loader, test_dataloader, device='cuda')

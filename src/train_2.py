
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.datasets import Planetoid
#from torch_geometric.utils import train_test_split
from sklearn.model_selection import train_test_split
from dataset import load_graph, MIMICIIIDataset, collate_text
from models import get_model, get_tokenizer
from utils import load_config, seed_all, get_optimizer, get_scheduler, count_parameters
import argparse

# Define a simple Graph Convolutional Network (GCN) model
class GCN(nn.Module):
    def __init__(self, num_features, num_classes):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(num_features, 16)
        self.conv2 = GCNConv(16, num_classes)

    def forward(self, data):
        print(data)
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)


# Load a dataset (e.g., Cora, Citeseer, or Pubmed)
#dataset = Planetoid(root='data', name='Pubmed', split='full')  # Change name and split as needed

if __name__ == '__main__':
    dataset = MIMICIIIDataset(root='../data/discharge/', split='train')  # Change name and split as needed

    # Split the dataset into train, validation, and test sets
    train_data, test_data = train_test_split(dataset, test_size=0.2)
    train_data, val_data = train_test_split(train_data, test_size=0.25)  # 60% train, 20% validation
    this_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Create the GCN model
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--logdir', type=str, default='./logs')
    parser.add_argument('--savename', type=str, default='test')
    parser.add_argument('--resume', type=str, default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    model = get_model(config.model).to(this_device)

    # Define loss function and optimizer
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    # Training loop
    def train():
        this_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        graph = load_graph(config.datasets.root, config.datasets.threshold, this_device)
        model.train()
        optimizer.zero_grad()
        #outputs = model(train_data)
        outputs = model(graph.x, graph.edge_index)[graph.train_mask]
        loss = criterion(outputs[graph.train_mask], graph.y[graph.train_mask])
        loss.backward()
        optimizer.step()
        return loss.item()

    # Evaluation function
    def evaluate(data):
        model.eval()
        outputs = model(data)
        pred_labels = outputs.argmax(dim=1)
        mask = data.val_mask if data.val_mask is not None else data.test_mask
        acc = (pred_labels[mask] == data.y[mask]).sum().item() / mask.sum().item()
        return acc

    # Training and evaluation
    num_epochs = 10
    for epoch in range(1, num_epochs + 1):
        loss = train()
        val_acc = evaluate(val_data)
        print(f'Epoch [{epoch}/{num_epochs}] Loss: {loss:.4f}, Val Acc: {val_acc:.4f}')

    # Evaluate on the test set
    test_acc = evaluate(test_data)
    print(f'Test Acc: {test_acc:.4f}')

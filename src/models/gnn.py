import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, SAGEConv
from transformers import AutoModel, AutoTokenizer
import torch
from ._base import register_model


class BaseGNN(nn.Module):
    def __init__(self, input_size, hidden_size, out_size, num_layers=2, dropout_p=0., **kwargs):
        super(BaseGNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.out_size = out_size
        self.num_layers = num_layers
        self.dropout_p = dropout_p

        self.convs = nn.ModuleList()
        self.convs.append(self.init_conv(input_size, hidden_size, **kwargs))
        for _ in range(num_layers - 2):
            self.convs.append(self.init_conv(hidden_size, hidden_size, **kwargs))
        self.convs.append(self.init_conv(hidden_size, out_size, **kwargs))

    def init_conv(self, in_size, out_size, **kwargs):
        raise NotImplementedError

    def forward(self, x, edge_index):
        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index)
            if i != self.num_layers - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout_p, training=self.training)
        return x

@register_model('GNNWithClinicalBERT')
class GNNWithClinicalBERT(nn.Module):
    def __init__(self, num_layers, input_size, hidden_size, out_size, num_classes, dropout_p):
        super(GNNWithClinicalBERT, self).__init__()
        self.num_layers = num_layers
        self.dropout_p = dropout_p

        # Define GNN layers
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(input_size, hidden_size))
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_size, hidden_size))
        self.convs.append(GCNConv(hidden_size, out_size))

        # Load and use Clinical BERT
        self.text_embedding = AutoModel.from_pretrained("./checkpoints/clinicalbert/")
        self.tokenizer = AutoTokenizer.from_pretrained("./checkpoints/clinicalbert/")

        # Fully connected layer for classification
        self.fc = nn.Linear(out_size + 30522, num_classes)

    def process_text(self, input_ids, token_type_ids, attention_mask):
        # Process text inputs with Clinical BERT
        text_embeddings = self.text_embedding(
            input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask
        )[0].mean(dim=1)

        return text_embeddings
    def forward(self, x, edge_index):
        # Process graph inputs with GNN
        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index)
            if i != self.num_layers - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout_p, training=self.training)
        return x
@register_model('gcn')
class GCN(BaseGNN):
    def init_conv(self, in_size, out_size, **kwargs):
        return GCNConv(in_size, out_size, **kwargs)


@register_model('gat')
class GAT(BaseGNN):
    def init_conv(self, in_size, out_size, **kwargs):
        return GATConv(in_size, out_size, **kwargs)


@register_model('sage')
class GraphSAGE(BaseGNN):
    def init_conv(self, in_size, out_size, **kwargs):
        return SAGEConv(in_size, out_size, **kwargs)

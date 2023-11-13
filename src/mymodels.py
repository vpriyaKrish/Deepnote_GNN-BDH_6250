# Code re-used from HW4 from 6250-BDH
import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, SAGEConv

##### DO NOT MODIFY OR REMOVE THIS VALUE #####
checksum = '169a9820bbc999009327026c9d76bcf1'
##### DO NOT MODIFY OR REMOVE THIS VALUE #####
import copy

_MODEL_DICT = {}


def register_model(name):
    def decorator(cls):
        _MODEL_DICT[name] = cls
        return cls

    return decorator


def get_model(cfg):
    m_dict = copy.deepcopy(cfg)
    model_type = m_dict.pop('type')
    return _MODEL_DICT[model_type](**m_dict)
class MyVariableRNN(nn.Module):
	def __init__(self, dim_input):
		super(MyVariableRNN, self).__init__()
		# You may use the input argument 'dim_input', which is basically the number of features
		self.embedding = nn.Sequential(nn.Linear(dim_input, 32),nn.Tanh())
		self.gru = nn.GRU(input_size=32, hidden_size=8, num_layers=1, batch_first=False, dropout=0.2, bias=True)
		#self.gru = nn.GRU(input_size=32, hidden_size=128, num_layers=1, batch_first=True, dropout=0.2, bias=False)
		self.output_layer = nn.Linear(in_features=8, out_features=2)

	def forward(self, input_tuple):
		# HINT: Following two methods might be useful
		# 'pack_padded_sequence' and 'pad_packed_sequence' from torch.nn.utils.rnn
		sequences, lengths = input_tuple
		sequences = self.embedding(sequences)
		packed_sequences = pack_padded_sequence(sequences, lengths, batch_first=True, enforce_sorted=False)
		packed_output, h = self.gru(packed_sequences)
		h = h.squeeze(0)
		out = self.output_layer(h)
		return out

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

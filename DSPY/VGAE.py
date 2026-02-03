from torch_geometric.nn.inits import glorot
from torch import nn
from torch.nn import Parameter
from torch_geometric.utils import softmax
from torch_geometric.data import Data
from torch_geometric.utils.convert import to_networkx
from torch_scatter import scatter

from torch.nn import MultiheadAttention
import torch.nn.functional as F
import networkx as nx
import numpy as np
import torch
import math
from torch_geometric.nn import GCNConv, GATConv # Import GCNConv from PyG
from torch_geometric.nn import VGAE

class SparseInputLinear(nn.Module):
    def __init__(self, inp_dim, out_dim):
        super(SparseInputLinear, self).__init__()

        weight = np.zeros((inp_dim, out_dim), dtype=np.float32)
        weight = nn.Parameter(torch.from_numpy(weight))
        bias = np.zeros(out_dim, dtype=np.float32)
        bias = nn.Parameter(torch.from_numpy(bias))
        self.inp_dim, self.out_dim = inp_dim, out_dim
        self.weight, self.bias = weight, bias
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / np.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x):
        return torch.mm(x, self.weight) + self.bias

class RelTemporalEncoding(nn.Module):
    def __init__(self, n_hid, max_len=50, dropout=0.2):
        super(RelTemporalEncoding, self).__init__()

        position = torch.arange(0.0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, n_hid, 2) * -(math.log(10000.0) / n_hid))
        emb = nn.Embedding(max_len, n_hid)
        emb.weight.data[:, 0::2] = torch.sin(position * div_term) / math.sqrt(n_hid)
        emb.weight.data[:, 1::2] = torch.cos(position * div_term) / math.sqrt(n_hid)
        emb.requires_grad = False
        self.emb = emb
        self.lin = nn.Linear(n_hid, n_hid)

    def forward(self, x, t):
        return x + self.lin(self.emb(t))


class LinkPredictor(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, dropout):
        super(LinkPredictor, self).__init__()

        self.lins = torch.nn.ModuleList()
        self.lins.append(torch.nn.Linear(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.lins.append(torch.nn.Linear(hidden_channels, hidden_channels))
        self.lins.append(torch.nn.Linear(hidden_channels, out_channels))

        self.dropout = dropout

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()

    def forward(self, z, e):
        x_i = z[e[0]]
        x_j = z[e[1]]
        x = torch.cat([x_i, x_j], dim=1)
        for lin in self.lins[:-1]:
            x = lin(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lins[-1](x)
        return torch.sigmoid(x).squeeze()


class MultiplyPredictor(torch.nn.Module):
    def __init__(self):
        super(MultiplyPredictor, self).__init__()

    def forward(self, z, e):
        x_i = z[e[0]]
        x_j = z[e[1]]
        x = (x_i * x_j).sum(dim=1)
        return torch.sigmoid(x)


class MergeMultiplyPredictor(torch.nn.Module):
    def __init__(self, input_dim1, input_dim2, hidden_dim): # Removed output_dim as it's effectively 1 after sum
        super(MergeMultiplyPredictor, self).__init__()
        self.fc1 = nn.Linear(input_dim1, hidden_dim)
        self.fc2 = nn.Linear(input_dim2, hidden_dim)
        self.act = nn.ReLU()
    def forward(self, z, e):
        # x_i = self.act(self.fc1(z[e[0]]))
        # x_j = self.act(self.fc2(z[e[1]]))
        x_i = self.act(z[e[0]])
        x_j = self.act(z[e[1]])
        x_merged = x_i * x_j
        x_summed = x_merged.sum(dim=1)
        return torch.sigmoid(x_summed)

class MergeLayer(nn.Module):
    def __init__(self, input_dim1, input_dim2, hidden_dim, output_dim):
        super(MergeLayer, self).__init__()

        super().__init__()
        self.fc1 = nn.Linear(input_dim1 + input_dim2, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.act = nn.ReLU()

    def forward(self, z, e):
        x_i = z[e[0]]
        x_j = z[e[1]]
        x = torch.cat([x_i, x_j], dim=1)
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return torch.sigmoid(x).squeeze()

class NodeClf(nn.Module):
    def __init__(self, args):
        super().__init__()
        clf = nn.ModuleList()
        hid_dim = args.nfeat
        clf_layers = args.clf_layers
        num_classes = args.num_classes
        for i in range(clf_layers - 1):
            clf.append(nn.Sequential(nn.Linear(hid_dim, hid_dim), nn.ReLU()))
        clf.append(nn.Linear(hid_dim, num_classes))
        self.clf = clf

    def forward(self, x):
        for layer in self.clf:
            x = layer(x)
        return x

class GCNEncoder(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.n_layers = args.n_layers
        self.n_factors = args.n_factors
        self.delta_d = args.delta_d
        self.in_dim = args.nfeat

        if self.args.dataset == 'aminer':
            self.n_factors = 8
            self.delta_d = 16
        self.hid_dim = self.n_factors * self.delta_d
        self.norm = args.norm
        self.maxiter = args.maxiter
        self.use_RTE = args.use_RTE
        self.agg_param = args.agg_param
        in_channels = self.in_dim
        hidden_channels = self.hid_dim
        out_channels = self.in_dim
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv_mu = GCNConv(hidden_channels, hidden_channels)
        self.conv_logvar = GCNConv(hidden_channels, hidden_channels)
        if self.args.dataset == 'aminer':
            self.classifier = NodeClf(args)
        # self.linear = SparseInputLinear(self.in_dim, self.hid_dim)

        self.feat = Parameter((torch.ones(args.num_nodes, args.nfeat)).to(args.device), requires_grad=True)
        self.edge_decoder = MultiplyPredictor()
        # self.edge_decoder = MergeLayer(self.hid_dim, self.hid_dim, self.hid_dim, 1)
        # self.edge_decoder = MergeMultiplyPredictor(self.hid_dim, self.hid_dim, self.hid_dim)
    def forward(self,x_list, edge_index):
        # if x_list is None:
        #     x = self.linear(self.feat)
        # else:
        #     x = self.linear(x_list)
        x = F.relu(self.conv1(x_list, edge_index))
        return self.conv_mu(x, edge_index), self.conv_logvar(x, edge_index)




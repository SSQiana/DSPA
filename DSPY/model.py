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
from torch_geometric.nn import GCNConv, GATConv, APPNP


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


class EAConv(nn.Module):
    def __init__(self, dim, n_factors, agg_param, use_RTE=False):
        super(EAConv, self).__init__()

        assert dim % n_factors == 0
        self.d = dim
        self.k = n_factors
        self.delta_d = self.d // self.k
        self.dk = self.d - self.delta_d
        self.use_RTE = use_RTE
        self.rte = RelTemporalEncoding(self.d)
        self.agg_param = agg_param
        self.dropout = 0.1
        self.dropout = nn.Dropout(self.dropout)

        self.norm_layers = nn.ModuleList([
            nn.LayerNorm(self.d),
            nn.LayerNorm(self.d)
        ])

        self.conv1 = GCNConv(self.d, 4 * self.d)
        self.conv2 = GCNConv(4 * self.d, self.d)  # Output dimension for node embeddings

    def time_encoding(self, x_all):
        if self.use_RTE:
            times = len(x_all)
            for t in range(times):
                x_all[t] = self.rte(x_all[t], torch.LongTensor([t]).to(x_all[t].device))
        return x_all

    def forward(self, x_all, max_iter, ix, edge_index, aug_loss):
        d, k, delta_d = self.d, self.k, self.delta_d
        n = x_all.size(0)

        dev = x_all.device
        edge_index = edge_index.to(dev)

        x_all = F.dropout(x_all, p=0.5, training=self.training)

        x = self.conv1(x_all, edge_index)
        z = F.relu(x)
        z = F.dropout(z, p=0.9, training=self.training)  # 现有的 Dropout
        z = self.conv2(z, edge_index)
        z = F.normalize(z.view(n, k, delta_d), dim=2).view(n, d)

        return z.to(dev)


class DyGNN(nn.Module):
    def __init__(self, args=None):
        super(DyGNN, self).__init__()

        self.args = args
        self.n_layers = args.n_layers
        self.n_factors = args.n_factors
        self.delta_d = args.delta_d

        self.n_views = 1

        if self.args.dataset in ['collab', 'yelp', 'act']:
            self.n_factors = 4
            self.delta_d = 8
        elif self.args.dataset == 'aminer':
            self.n_factors = 8
            self.delta_d = 16
        else:
            self.n_factors = 4
            self.delta_d = 16
        self.in_dim = args.nfeat
        self.hid_dim = self.n_factors * self.delta_d
        self.norm = args.norm
        self.maxiter = args.maxiter
        self.use_RTE = args.use_RTE
        self.agg_param = args.agg_param
        self.feat = Parameter((torch.ones(args.num_nodes, args.nfeat)).to(args.device), requires_grad=True)
        self.linear = nn.Linear(self.in_dim, self.hid_dim)
        self.view_encoders = nn.ModuleList()
        for v in range(self.n_views):
            layers = nn.ModuleList([
                EAConv(self.in_dim, self.n_factors, self.agg_param, self.use_RTE)
                for i in range(self.n_layers)
            ])
            self.view_encoders.append(layers)

        self.relu = F.relu
        self.LeakyReLU = nn.LeakyReLU()
        self.dropout = args.dropout
        self.reset_parameter()
        self.device = args.device
        self.edge_decoder = MultiplyPredictor()

        if args.dataset == 'aminer':
            self.classifier = NodeClf(args)

    def reset_parameter(self):
        glorot(self.feat)

    def forward(self, edge_index, x_list, ix, aug_loss, view_idx=0):
        # if x_list is None:
        #     x_list = self.linear(self.feat)
        # else:
        #     x_list = self.linear(x_list)

        current_layers = self.view_encoders[0]

        for i, layer in enumerate(current_layers):
            x_list = layer(x_list, self.maxiter, ix, edge_index, aug_loss)
        return x_list


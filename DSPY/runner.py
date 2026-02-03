from DSPY.utils.mutils import *
from DSPY.utils.inits import prepare
from DSPY.utils.loss import EnvLoss
from DSPY.utils.util import init_logger, logger
from torch_geometric.utils import negative_sampling
from torch_geometric.data import Data
from torch_geometric.utils.convert import to_networkx
from torch.nn import functional as F
from torch import nn
from tqdm import tqdm

import os
import sys
import time
import torch
import numpy as np
import torch.optim as optim
import pandas as pd
import math
import torch.utils.checkpoint as checkpoint

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)


class NeibSampler:
    def __init__(self, graph, nb_size, include_self=False):
        n = graph.number_of_nodes()
        assert 0 <= min(graph.nodes()) and max(graph.nodes()) < n
        if include_self:
            nb_all = torch.zeros(n, nb_size + 1, dtype=torch.int64)
            nb_all[:, 0] = torch.arange(0, n)
            nb = nb_all[:, 1:]
        else:
            nb_all = torch.zeros(n, nb_size, dtype=torch.int64)
            nb = nb_all
        popkids = []
        for v in range(n):
            nb_v = sorted(graph.neighbors(v))
            if len(nb_v) <= nb_size:
                nb_v.extend([-1] * (nb_size - len(nb_v)))
                nb[v] = torch.LongTensor(nb_v)
            else:
                popkids.append(v)
        self.include_self = include_self
        self.g, self.nb_all, self.pk = graph, nb_all, popkids

    def to(self, dev):
        self.nb_all = self.nb_all.to(dev)
        return self

    def sample(self):
        nb = self.nb_all[:, 1:] if self.include_self else self.nb_all
        nb_size = nb.size(1)
        pk_nb = np.zeros((len(self.pk), nb_size), dtype=np.int64)
        for i, v in enumerate(self.pk):
            pk_nb[i] = np.random.choice(sorted(self.g.neighbors(v)), nb_size)
        nb[self.pk] = torch.from_numpy(pk_nb).to(nb.device)
        return self.nb_all


class Runner(object):
    def __init__(self, args, model, data, writer=None, **kwargs):
        self.args = args
        self.data = data
        self.model = model
        self.tau = 0.9

        if args.dataset in ['collab_04', 'collab_06', 'collab_08']:
            self.n_nodes = data["x"].shape[1]
        else:
            self.n_nodes = data["x"].shape[0]

        self.writer = writer
        if args.dataset in ['aminer']:
            self.len = args.len_train
        else:
            self.len = len(data["train"]["edge_index_list"])
        self.len_train = self.len - args.testlength - args.vallength
        self.len_val = args.vallength
        self.len_test = args.testlength
        self.nbsz = args.nbsz
        self.n_factors = args.n_factors
        self.delta_d = args.delta_d
        self.d = self.n_factors * self.delta_d
        self.interv_size_ratio = args.interv_size_ratio
        self.criterion = torch.nn.CrossEntropyLoss()

        x = data["x"].to(args.device).clone().detach()
        self.x = [x for _ in range(self.len)] if len(x.shape) <= 2 else x

        self.loss = EnvLoss(args)
        print("total length: {}, test length: {}".format(self.len, args.testlength))

        if args.dataset == 'aminer':
            self.edge_index_list_pre = [
                data["edge_index"][ix].long().to(args.device)
                for ix in range(self.len)]
        else:
            self.edge_index_list_pre = [
                data["train"]["edge_index_list"][ix].long().to(args.device)
                for ix in range(self.len)
            ]

    def extract_neighbors(self, data, args, ix):

        if args.dataset == 'aminer':
            self.edge_index_list_pre = [
                data["aug_edge_index_list"][0][i].long().to(args.device)
                for i in range(len(data["aug_edge_index_list"][0]))]
        else:
            self.edge_index_list_pre = [
                data["train"]["aug_edge_index_list"][0][i].long().to(args.device)
                for i in range(len(data["train"]["aug_edge_index_list"][0]))]

        neighbors_all = []
        for t in range(len(self.edge_index_list_pre)):
            graph_data = Data(x=self.x[t], edge_index=self.edge_index_list_pre[t])
            graph = to_networkx(graph_data)
            sampler = NeibSampler(graph, self.nbsz)
            neighbors = sampler.sample().to(args.device)
            neighbors_all.append(neighbors)
        return torch.stack(neighbors_all).to(args.device)

    def cal_y(self, embeddings, decoder, edge_index, device):
        preds = torch.tensor([]).to(device)
        z = embeddings
        pred = decoder(z, edge_index)
        preds = torch.cat([preds, pred])
        return preds

    def classification_cal_y(self, embeddings, decoder, node_masks, device, ix):
        z = embeddings
        mask = node_masks[ix]
        pred = decoder(z)[mask]
        # preds = torch.cat([preds, pred])
        return pred

    def accuracy(self, y, label):
        _, predicted = torch.max(y, 1)
        correct = (predicted == label).sum().item()
        total = label.size(0)
        acc = correct / total
        return acc


    def cal_loss(self, y, label):
        criterion = torch.nn.BCELoss()
        return criterion(y, label)

    def run(self):
        args = self.args
        min_epoch = args.min_epoch
        max_patience = args.patience
        patience = 0
        self.optimizer = optim.Adam(
            [p for n, p in self.model.named_parameters() if "ss" not in n],
            lr=args.lr,
            weight_decay=args.weight_decay,
        )

        t_total0 = time.time()
        max_auc = 0
        max_test_auc = 0
        max_train_auc = 0

        with tqdm(range(1, args.max_epoch + 1)) as bar:
            for epoch in bar:
                t0 = time.time()
                epoch_losses, train_auc_list, val_auc_list, test_auc_list = self.train(epoch, self.data["train"])
                average_epoch_loss = epoch_losses
                average_train_auc = np.mean(train_auc_list)
                average_val_auc = np.mean(val_auc_list)
                average_test_auc = np.mean(test_auc_list)
                if average_val_auc > max_auc:
                    max_auc = average_val_auc
                    max_test_auc = average_test_auc
                    max_train_auc = average_train_auc
                    test_results = self.test(epoch, self.data["test"])
                    metrics = "train_auc,val_auc,test_auc,epoch,test_train_auc,test_val_auc,test_test_auc".split(",")
                    measure_dict = dict(zip(metrics, [max_train_auc, max_auc, max_test_auc] + test_results, ))
                    patience = 0
                    filepath = "../checkpoint/" + self.args.dataset + ".pth"
                    torch.save({"model_state_dict": self.model.state_dict()}, filepath)
                else:
                    patience += 1
                    if epoch > min_epoch and patience > max_patience:
                        break
                if epoch == 1 or epoch % self.args.log_interval == 0:
                    print("Epoch:{}, Loss: {:.4f}, Time: {:.3f}".format(epoch, average_epoch_loss, time.time() - t0))
                    print(
                        f"Current: Epoch:{epoch}, Train AUC:{average_train_auc:.4f}, Val AUC: {average_val_auc:.4f}, Test AUC: {average_test_auc:.4f}")
                    print(
                        f"Train: Epoch:{test_results[0]}, Train AUC:{max_train_auc:.4f}, Val AUC: {max_auc:.4f}, Test AUC: {max_test_auc:.4f}")
                    print(
                        f"Test: Epoch:{test_results[0]}, Train AUC:{test_results[1]:.4f}, Val AUC: {test_results[2]:.4f}, Test AUC: {test_results[3]:.4f}")
        epoch_time = (time.time() - t_total0) / (epoch - 1)
        metrics = [max_train_auc, max_auc, max_test_auc] + test_results + [epoch_time]
        metrics_des = "train_auc,val_auc,test_auc,epoch,test_train_auc,test_val_auc,test_test_auc,epoch_time".split(",")
        metrics_dict = dict(zip(metrics_des, metrics))
        df = pd.DataFrame([metrics], columns=metrics_des)
        print(df)
        return metrics_dict

    def _sim(self, z1: torch.Tensor, z2: torch.Tensor):
        z1 = F.normalize(z1)
        z2 = F.normalize(z2)
        return torch.mm(z1, z2.t())

    def contrastive_loss_v2(self, z1, z2, batch_size=1024):
        N = z1.shape[0]
        f = lambda x: torch.exp(x / self.tau)

        all_losses = []

        for i in range(0, N, batch_size):
            z1_batch = z1[i:i + batch_size]
            z2_batch = z2[i:i + batch_size]

            pos_sim = f(torch.sum(z1_batch * z2_batch, dim=-1))  # [B]
            sim_matrix = f(self._sim(z1_batch, z2))  # [B, N]
            denom = sim_matrix.sum(dim=1)  # [B]

            loss = -torch.log(pos_sim / denom)
            all_losses.append(loss)

        return torch.cat(all_losses).mean()

    def contrastive_loss(self, z1, z2, batch_size=1024):
        N = z1.shape[0]
        f = lambda x: torch.exp(x / self.tau)

        all_losses = []

        for i in range(0, N, batch_size):
            z1_batch = z1[i:i + batch_size]

            z2_batch_for_pos = z2[i:i + batch_size]

            def _calculate_loss_for_checkpoint(z1_b, full_z2_tensor, z2_batch_for_pos_tensor):
                # Recalculate pos_sim using the passed batch for z2
                pos_sim = f(torch.sum(z1_b * z2_batch_for_pos_tensor, dim=-1))

                # Use the full z2 for the negative samples
                sim_matrix = f(self._sim(z1_b, full_z2_tensor))
                denom = sim_matrix.sum(dim=1)
                loss_val = -torch.log(pos_sim / denom)
                return loss_val

            loss_batch = checkpoint.checkpoint(_calculate_loss_for_checkpoint, z1_batch, z2, z2_batch_for_pos)
            all_losses.append(loss_batch)

        return torch.cat(all_losses).mean()

    def test(self, epoch, data):
        args = self.args
        train_auc_list = []
        val_auc_list = []
        test_auc_list = []
        self.model.eval()
        embeddings = []
        last_embeddings = None
        for i in range(self.len):

            embeddings = self.model(data['edge_index_list'][i].long().to(args.device), self.x[i],  i, False)

            if i < self.len - 1:
                z = embeddings
                pos_index = data["pedges"][i]  # torch edge index
                neg_index = data["nedges"][i]
                # edge_index, pos_edge, neg_edge = prepare(pos_index, neg_index)[:3]
                edge_index, pos_edge, neg_edge = prepare(data, i + 1)[:3]

                if is_empty_edges(neg_edge):
                    continue
                auc, ap = self.loss.predict(z, pos_edge, neg_edge, self.model.edge_decoder)
                if i < self.len_train - 1:
                    train_auc_list.append(auc)
                elif i < self.len_train + self.len_val - 1:
                    val_auc_list.append(auc)
                else:
                    test_auc_list.append(auc)
        return [
            epoch,
            np.mean(train_auc_list),
            np.mean(val_auc_list),
            np.mean(test_auc_list),
        ]

    def classification_test(self, epoch, data):
        args = self.args

        self.model.eval()
        embeddings = []
        last_embeddings = None
        node_masks = data['node_masks']
        device = self.args.device
        test_list = []
        for i in range(self.len-3, self.len):
            # embeddings = self.model(data['edge_index'][i].long().to(args.device), self.x[i], self.neighbors_all[i], i, False)
            embeddings = self.model(data['edge_index'][i].long().to(args.device), self.x[i], 0, i)

            label = data['y'][node_masks[i]].squeeze()
            predictions_z_I = self.classification_cal_y(embeddings, self.model.classifier, node_masks, device, i)  # [N,C]
            predictions_z_I = predictions_z_I.to('cpu')
            label = label.to('cpu')
            loss_I = self.criterion(predictions_z_I, label)
            acc = self.accuracy(predictions_z_I, label)
            test_list.append(acc)

        return [
            epoch,
            test_list[0],
            test_list[1],
            test_list[2]
        ]

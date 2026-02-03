import sys
import time

import torch

sys.path.append("..")

from DSPY.config import args
from DSPY.utils.mutils import *
from DSPY.utils.data_util import *
from DSPY.utils.util import init_logger
from Augmention.data_augmention import Compose, FeatureAugmentor, SpectralAugmentor
from torch_geometric.utils import to_dense_adj, dense_to_sparse, remove_self_loops
from torch_geometric.loader import GraphSAINTRandomWalkSampler
import torch.optim as optim
from tqdm import tqdm
# from runner1 import test
import warnings
from DSPY.utils.inits import prepare
import pandas as pd

warnings.simplefilter("ignore")
pe = 0.8
aug_lr1 = 200
aug_iter = 7
device = 'cuda:0'
train_bs = 4096
walk_length = 5
num_steps = 10
k = 5
################
args.max_epoch = 120


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# from EAGLE.runner import Runner
from DSPY.runner import Runner
# from EAGLE.model import EADGNN
from DSPY.model import EADGNN

# load data
# for args.dataset in ['collab', 'collab_06', 'collab_08']:
for args.dataset in ['collab', 'yelp', 'act','collab_04']:

    for runs in range(5):
        seed_everything(runs)

        # Runner
        # from EAGLE.runner import Runner
        from DSPY.runner import Runner
        # from EAGLE.model import EADGNN
        from DSPY.model import EADGNN

        args.n_layers = 1
        args, data = load_data(args)
        model = EADGNN(args=args).to(args.device)
        runner = Runner(args, model, data)
        span1 = SpectralAugmentor(ratio=pe,
                                  lr=aug_lr1,
                                  iteration=aug_iter,
                                  dis_type='max',
                                  device=device,
                                  threshold=0.8,
                                  k=k)
        path = osp.join(os.getcwd(), 'augmented_data')
        subdir = osp.join(path, args.dataset)


        results = []
        min_epoch = args.min_epoch
        max_patience = args.patience
        patience = 0
        optimizer = optim.Adam(
            [p for n, p in runner.model.named_parameters() if "ss" not in n],
            lr=args.lr,
            weight_decay=args.weight_decay,
        )
        t_total0 = time.time()
        max_auc = 0
        max_test_auc = 0
        max_train_auc = 0

        data['train']['aug_edge_index_list'] = []


        with tqdm(range(1, args.max_epoch + 1)) as bar:
            t0 = time.time()
            for epoch in bar:
                # ... (initialization code remains the same) ...
                val_auc_list = []
                test_auc_list = []
                train_auc_list = []
                for ix, edge_index in enumerate(data['train']['edge_index_list']):
                    batch_pedges = edge_index
                    batch_nedges = data['train']['nedges'][ix]
                    update_data_path = osp.join(subdir, '{}_{}_{}.pt'.format(ix, aug_lr1, pe))

                    if os.path.exists(update_data_path):  # Load precomputed probability matrix
                        batch_aug_edge_index = torch.load(update_data_path)
                        data['train']['aug_edge_index_list'].append(batch_aug_edge_index)
                        # print('(A): perturbation probability loaded!')
                    else:
                        # print('(A): precomputing probability ...')
                        # data = loader['train'].data
                        # data.edge_index = remove_self_loops(data.edge_index)[0]
                        span1.calc_prob(data, ix)  # note that data is updated with data['max']=ptb_prob1
                        batch_aug_edge_index = data['train']['aug_edge_index_list'][ix]
                        save_path = os.path.join(subdir, f"{ix}_{aug_lr1}_{pe}.pt")
                        torch.save(batch_aug_edge_index, save_path)

                    log_dir = args.log_dir
                    init_logger(prepare_dir(log_dir) + "log_" + args.dataset + ".txt")
                    info_dict = get_arg_dict(args)

                    z_I = None  # To store invariant embeddings
                    z_V_list = []  # To store a list of variant embeddings

                    batch_aug_neighbors = runner.extract_neighbors(data, args, ix)

                    for num_aug, aug_edge_index in enumerate(batch_aug_edge_index):
                        edge_index = aug_edge_index

                        if num_aug == 0:
                            # Invariant embeddings Z_I
                            z_I = runner.model(edge_index.long().to(args.device), runner.x[ix],
                                               runner.neighbors_all[ix], ix, aug_loss=False)
                            if ix <= runner.len - 2:
                                _, pos_edge, neg_edge = prepare(data['train'], ix + 1)[:3]
                                auc, ap = runner.loss.predict(z_I, pos_edge, neg_edge, runner.model.edge_decoder)

                                if ix < runner.len_train - 1:
                                    train_auc_list.append(auc)
                                elif ix < runner.len_train + runner.len_val - 1:
                                    val_auc_list.append(auc)
                                else:
                                    test_auc_list.append(auc)

                        # Embeddings from augmented graphs are the variant patterns ZV
                        else:
                            # Variant embeddings Z_V
                            z_V = runner.model(edge_index.long().to(args.device), runner.x[ix],
                                               batch_aug_neighbors[num_aug], ix, aug_loss=True)
                            z_V_list.append(z_V)

                    if ix <= runner.len - 2 and z_I is not None and len(z_V_list) > 0:
                        # Prepare edges and labels for loss calculation
                        _, pos_edge, neg_edge = prepare(data['train'], ix + 1)[:3]
                        edge_label_index = torch.cat([pos_edge, neg_edge], dim=-1)
                        pos_y = z_I.new_ones(pos_edge.size(1))
                        neg_y = z_I.new_zeros(neg_edge.size(1))
                        edge_label = torch.cat([pos_y, neg_y], dim=0)

                        pred_y_I = runner.cal_y(z_I, runner.model.edge_decoder, edge_label_index, device)
                        loss_I = runner.cal_loss(pred_y_I, edge_label)

                        loss_V_list = []
                        for z_V in z_V_list:
                            pred_y_V = runner.cal_y(z_V, runner.model.edge_decoder, edge_label_index, device)
                            loss_V_list.append(runner.cal_loss(pred_y_V, edge_label))
                        loss_V = torch.stack(loss_V_list).mean()

                        # 3. Calculate contrastive loss (L_C)
                        mask = torch.eye(z_I.size(0), device=z_I.device)
                        loss_C_list = []
                        for z_V in z_V_list:
                            loss_C_list.append(runner.contrastive_loss(z_I, z_V))
                        loss_C = torch.stack(loss_C_list).mean()
                        loss_var = torch.var(torch.stack(loss_V_list), dim=0)  # Variance of variant losses
                        # Final Training Objective from Equation (10) [cite: 189, 190]
                        # Assuming you are minimizing f_V in the same step
                        final_loss = loss_I + 0.9 * loss_C + loss_var
                        ### Backpropagation ###
                        optimizer.zero_grad()
                        final_loss.backward()
                        optimizer.step()
                        average_epoch_loss = final_loss.item()

                        # ... (AUC calculation and logging remain the same) ...
                        auc, ap = runner.loss.predict(z_I, pos_edge, neg_edge, runner.model.edge_decoder)
                        bar.set_description(f"batch_idx: {ix}, AUC: {auc:.4f}, AP: {ap:.4f}, Loss: {final_loss:.4f}, L_I: {loss_I:.4f}, L_C: {loss_C:.4f}, L_V: {loss_var:.4f}")

                average_train_auc = np.mean(train_auc_list)
                average_val_auc = np.mean(val_auc_list)
                average_test_auc = np.mean(test_auc_list)
                if average_val_auc > max_auc:
                    max_auc = average_val_auc
                    max_test_auc = average_test_auc
                    max_train_auc = average_train_auc
                    test_results = runner.test(epoch, data["test"])
                    metrics = "train_auc,val_auc,test_auc,epoch,test_train_auc,test_val_auc,test_test_auc".split(",")
                    measure_dict = dict(zip(metrics, [max_train_auc, max_auc, max_test_auc] + test_results, ))
                    patience = 0
                    filepath = "../checkpoint/" + 'abla' + 'span' + args.dataset + str(runs) + ".pth"
                    torch.save({"model_state_dict": model.state_dict()}, filepath)
                else:
                    patience += 1
                    if epoch > min_epoch and patience > max_patience:
                        break
                if epoch == 1 or epoch % args.log_interval == 0:
                    print("Epoch:{}, Loss: {:.4f}, Time: {:.3f}".format(epoch, average_epoch_loss, time.time() - t0))
                    print(
                        f"Current: Epoch:{epoch}, Train AUC:{average_train_auc:.4f}, Val AUC: {average_val_auc:.4f}, Test AUC: {average_test_auc:.4f}")
                    print(
                        f"Train: Epoch:{test_results[0]}, Train AUC:{max_train_auc:.4f}, Val AUC: {max_auc:.4f}, Test AUC: {max_test_auc:.4f}")
                    print(
                        f"Test: Epoch:{test_results[0]}, Train AUC:{test_results[1]:.4f}, Val AUC: {test_results[2]:.4f}, Test AUC: {test_results[3]:.4f}")

        # post-logs
        measure_dict = results
        info_dict.update(measure_dict)
        filename = "info_" + args.dataset + ".json"
        json.dump(info_dict, open(osp.join(log_dir, filename), "w"))

    # results = runner.re_run()
    # print(f"Final Test: Epoch:{test_results[0]}, Train AUC:{test_results[1]:.4f}, Val AUC: {test_results[2]:.4f}, Test AUC: {test_results[3]:.4f}")


def evaluate(args, runner):
    all_train_auc = []
    all_val_auc = []
    all_test_auc = []

    for runs in range(5):
        data = runner.data["test"]

        runner.optimizer = optim.Adam(
            [p for n, p in runner.model.named_parameters() if "ss" not in n],
            lr=args.lr,
            weight_decay=args.weight_decay, )

        filepath = "../checkpoint/" + 'abla' + 'span' + args.dataset + str(runs) + ".pth"
        checkpoint = torch.load(filepath)
        runner.model.load_state_dict(checkpoint["model_state_dict"])
        runner.model.eval()

        train_auc_list = []
        val_auc_list = []
        test_auc_list = []
        last_embeddings = None
        for i in range(runner.len):

            embeddings = runner.model(data['edge_index_list'][i].long().to(args.device), runner.x[i],
                                      runner.neighbors_all[i], i, aug_loss=False)

            if i < runner.len - 1:
                z = embeddings
                edge_index, pos_edge, neg_edge = prepare(data, i + 1)[:3]
                if is_empty_edges(neg_edge):
                    continue
                auc, _ = runner.loss.predict(z, pos_edge, neg_edge, runner.model.edge_decoder)

                if i < runner.len_train - 1:
                    train_auc_list.append(auc)
                elif i < runner.len_train + runner.len_val - 1:
                    val_auc_list.append(auc)
                else:
                    test_auc_list.append(auc)

        all_train_auc.append(np.mean(train_auc_list))
        all_val_auc.append(np.mean(val_auc_list))
        all_test_auc.append(np.mean(test_auc_list))

    mean_train_auc = np.mean(all_train_auc)
    std_train_auc = np.std(all_train_auc)

    mean_val_auc = np.mean(all_val_auc)
    std_val_auc = np.std(all_val_auc)

    mean_test_auc = np.mean(all_test_auc)
    std_test_auc = np.std(all_test_auc)

    df = pd.DataFrame([{
        "train_auc_mean": mean_train_auc,
        "train_auc_std": std_train_auc,
        "val_auc_mean": mean_val_auc,
        "val_auc_std": std_val_auc,
        "test_auc_mean": mean_test_auc,
        "test_auc_std": std_test_auc
    }])

    return {
        "train_auc_mean": mean_train_auc,
        "train_auc_std": std_train_auc,
        "val_auc_mean": mean_val_auc,
        "val_auc_std": std_val_auc,
        "test_auc_mean": mean_test_auc,
        "test_auc_std": std_test_auc
    }


for args.dataset in ['collab', 'yelp',  'act', 'collab_04']:
    args.n_layers = 1
    args, data = load_data(args)
    model = EADGNN(args=args).to(args.device)
    # model = GCNEncoder(args=args).to(args.device)
    runner = Runner(args, model, data)
    results = evaluate(args, runner)
    print('Results for dataset {}:', args.dataset)
    print(results)

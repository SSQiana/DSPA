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
from torch_geometric.nn import VGAE

warnings.simplefilter("ignore")

device = 'cuda:0'

################
args.lr = 0.0001
args.max_epoch = 200

args.dataset = 'aminer'
args.len_test = 3
args.len_val = 3
args.len_train = 11
args.clf_layers = 2


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# from EAGLE.runner import Runner
from DSPY.runner import Runner
# from EAGLE.model import EADGNN
from DSPY.VGAE import GCNEncoder

# load data
for args.dataset in ['aminer']:
    for runs in range(3):
        seed_everything(runs)

        args.n_layers = 1
        args, data = load_data(args)

        encoder = GCNEncoder(args)
        model = VGAE(encoder).to(args.device)
        runner = Runner(args, model, data)

        path = osp.join(os.getcwd(), 'augmented_data')
        subdir = osp.join(path, args.dataset)
        # update_data_path = osp.join(subdir, 'updated_{}_{}.pt'.format(aug_lr1, pe))
        # update_data_path = osp.join(subdir, '{}_{}_{}.pt'.format(ix,aug_lr1, pe))
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
        node_masks = data['node_masks']

        with tqdm(range(1, args.max_epoch + 1)) as bar:
            t0 = time.time()
            for epoch in bar:
                val_auc_list = []
                test_auc_list = []
                train_auc_list = []
                # data['train']['edge_index_list'] = data['train']['edge_index_list'][:5]
                train_acc_list = []

                train_data = data['edge_index']
                for ix in range(args.len_train):
                    edge_index = train_data[ix]
                    log_dir = args.log_dir
                    init_logger(prepare_dir(log_dir) + "log_" + args.dataset + ".txt")
                    info_dict = get_arg_dict(args)

                    edge_index = edge_index.long().to(args.device)
                    x_input = runner.x[ix].to(args.device)

                    # VGAE前向传播，得到Z
                    z = runner.model.encode(x_input, edge_index)

                    # VGAE损失
                    recon_loss = runner.model.recon_loss(z, edge_index)
                    kl_loss = (1 / x_input.size(0)) * runner.model.kl_loss()
                    vgae_loss = recon_loss + kl_loss

                    # 分类损失
                    pred = encoder.classifier(z[node_masks[ix]])
                    label = data['y'][node_masks[ix]].to(args.device)
                    class_loss = runner.criterion(pred, label)
                    acc = runner.accuracy(pred, label)
                    train_acc_list.append(acc)
                    # 总损失
                    loss = 0.1 * vgae_loss + class_loss

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    bar.set_description(
                        f"batch_idx: {ix}, Recon Loss: {recon_loss:.4f}, KL: {kl_loss:.4f}, Total Loss: {loss:.4f}")

                average_train_auc = np.mean(train_acc_list)

                if average_train_auc > max_auc:
                    max_auc = average_train_auc

                    test_results = runner.classification_vgae_test(epoch, data)
                    metrics = "train_auc,val_auc,test_auc,epoch,test_train_auc,test_val_auc,test_test_auc".split(",")
                    measure_dict = dict(zip(metrics, test_results, ))
                    patience = 0
                    filepath = "../checkpoint/" + 'vgae_class' + args.dataset + str(runs) + ".pth"
                    torch.save({"model_state_dict": model.state_dict()}, filepath)
                else:
                    patience += 1
                    if epoch > min_epoch and patience > max_patience:
                        break
                if epoch == 1 or epoch % args.log_interval == 0:
                    print("Epoch:{}, Time: {:.3f}".format(epoch, time.time() - t0))
                    print(f"Current: Epoch:{epoch}, Train AUC:{average_train_auc:.4f}")
                    print(
                        f"Test: Epoch:{test_results[0]}, Aminer15:{test_results[1]:.4f}, Aminer16: {test_results[2]:.4f}, Aminer17: {test_results[3]:.4f}")


def evaluate(args, runner):
    aminer15 = []
    aminer16 = []
    aminer17 = []
    for runs in range(3):
        data = runner.data
        node_masks = data['node_masks']
        runner.optimizer = optim.Adam(
            [p for n, p in runner.model.named_parameters() if "ss" not in n],
            lr=args.lr,
            weight_decay=args.weight_decay, )

        filepath = "../checkpoint/" + 'vgae_class' + args.dataset + str(runs) + ".pth"
        checkpoint = torch.load(filepath)
        runner.model.load_state_dict(checkpoint["model_state_dict"])
        runner.model.eval()

        test_list = []
        last_embeddings = None

        for i in range(runner.len - 3, runner.len):
            # embeddings = runner.model(data['edge_index'][i].long().to(args.device), runner.x[i], 0, i)
            embeddings = runner.model.encode(runner.x[i], data['edge_index'][i].long().to(args.device))

            label = data['y'][node_masks[i]].squeeze()
            # predictions_z_I = runner.classification_cal_y(embeddings, runner.model.classifier, node_masks, device, i)  # [N,C]
            predictions_z_I = encoder.classifier(embeddings[node_masks[i]])

            predictions_z_I = predictions_z_I.to(device)
            label = label.to(device)
            loss_I = runner.criterion(predictions_z_I, label)
            acc = runner.accuracy(predictions_z_I, label)
            test_list.append(acc)

        aminer15.append(test_list[0])
        aminer16.append(test_list[1])
        aminer17.append(test_list[2])
    # Calculate mean and standard deviation for each dataset
    mean_aminer15_auc = np.mean(aminer15)
    std_aminer15_auc = np.std(aminer15)

    mean_aminer16_auc = np.mean(aminer16)
    std_aminer16_auc = np.std(aminer16)

    mean_aminer17_auc = np.mean(aminer17)
    std_aminer17_auc = np.std(aminer17)

    return {
        'aminer15': (mean_aminer15_auc, std_aminer15_auc),
        'aminer16': (mean_aminer16_auc, std_aminer16_auc),
        'aminer17': (mean_aminer17_auc, std_aminer17_auc)
    }


for args.dataset in ['aminer']:
    args.n_layers = 1
    args, data = load_data(args)
    encoder = GCNEncoder(args)
    model = VGAE(encoder).to(args.device)
    runner = Runner(args, model, data)
    results = evaluate(args, runner)
    print('Results for dataset {}:', args.dataset)
    print(results)

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
args.max_epoch = 200


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


args.dataset = 'aminer'
args.len_test = 3
args.len_val = 3
args.len_train = 11
args.clf_layers = 2

# Runner
# from EAGLE.runner import Runner
from DSPY.runner import Runner
# from EAGLE.model import EADGNN
from DSPY.model import EADGNN

for args.dataset in ['aminer']:  # 或其他数据集
    for runs in range(1):
        seed_everything(runs)

        args.n_layers = 1
        args, data = load_data(args)

        model = EADGNN(args=args).to(args.device)
        runner = Runner(args, model, data)

        # 初始化 Augmentor
        span1 = SpectralAugmentor(ratio=pe,
                                  lr=aug_lr1,
                                  iteration=aug_iter,
                                  dis_type='max',
                                  device=device,
                                  threshold=0.8,
                                  k=k)

        path = osp.join(os.getcwd(), 'augmented_data')
        if not os.path.exists(path):
            os.makedirs(path)
        subdir = osp.join(path, args.dataset)
        if not os.path.exists(subdir):
            os.makedirs(subdir)

        # [修正 1] 初始化 results 字典和 test_results 默认值
        results = {}
        test_results = [0, 0, 0, 0]  # 防止 epoch 1 打印报错

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

        # [修正 2] 列表初始化放在循环外
        data['aug_edge_index_list'] = []
        node_masks = data['node_masks']

        # 定义多少个 snapshot 打印一次日志
        log_step = 5

        # [新增 1] 获取快照总数，用于计算进度的分母
        train_data = data['edge_index']
        num_snapshots = len(train_data)

        with tqdm(range(1, args.max_epoch + 1)) as bar:
            t0 = time.time()
            for epoch in bar:

                train_acc_list = []
                # train_data = data['edge_index'] # 移到了外面获取长度，这里可以直接用

                for ix in range(args.len_train):
                    edge_index = train_data[ix]

                    # [修正 3] 内存/缓存管理逻辑
                    update_data_path = osp.join(subdir, '{}_{}_{}.pt'.format(ix, aug_lr1, pe))

                    if len(data['aug_edge_index_list']) <= ix:
                        # ... (原有的缓存加载逻辑保持不变) ...
                        if os.path.exists(update_data_path):
                            batch_aug_edge_index = torch.load(update_data_path)
                            data['aug_edge_index_list'].append(batch_aug_edge_index)
                        else:
                            span1.calc_prob(args, data, ix)
                            batch_aug_edge_index = data['aug_edge_index_list'][ix]
                            save_path = os.path.join(subdir, f"{ix}_{aug_lr1}_{pe}.pt")
                            torch.save(batch_aug_edge_index, save_path)
                    else:
                        batch_aug_edge_index = data['aug_edge_index_list'][ix]

                    z_I = None
                    z_V_list = []

                    for num_aug, aug_edge_index in enumerate(batch_aug_edge_index):
                        edge_index = aug_edge_index
                        if num_aug == 0:
                            z_I = runner.model(edge_index.long().to(args.device), runner.x[ix], ix, aug_loss=False)
                        else:
                            z_V = runner.model(edge_index.long().to(args.device), runner.x[ix], ix, aug_loss=True)
                            z_V_list.append(z_V)

                    label = data['y'][node_masks[ix]].squeeze().to(args.device)
                    pred_I = runner.classification_cal_y(z_I, runner.model.classifier, node_masks, device, ix)
                    loss_I = runner.criterion(pred_I, label)

                    acc = runner.accuracy(pred_I, label)
                    train_acc_list.append(acc)

                    loss_V_list = []
                    for z_V in z_V_list:
                        pred_V = runner.classification_cal_y(z_V, runner.model.classifier, node_masks, device, ix)
                        loss_V_list.append(runner.criterion(pred_V, label))

                    all_task_losses = [loss_I] + loss_V_list
                    loss_stack = torch.stack(all_task_losses)
                    loss_var = loss_stack.var()

                    loss_C_list = []
                    for z_V in z_V_list:
                        loss_C_list.append(runner.contrastive_loss(z_I, z_V))

                    if loss_C_list:
                        loss_C = torch.stack(loss_C_list).mean()
                    else:
                        loss_C = torch.tensor(0.0, device=device)

                    final_loss = loss_I + loss_var + 0.02 * loss_C

                    ### Backpropagation ###
                    optimizer.zero_grad()
                    final_loss.backward()
                    optimizer.step()

                    # --- [修改点]：实时更新进度条，加入 LR 显示 ---
                    current_avg_acc = sum(train_acc_list) / len(train_acc_list)

                    # 在 bar 的描述中加入了 LR，方便确认动态调整是否生效
                    bar.set_description(
                        f"Ep:{epoch} B:{ix} | L:{final_loss:.3f} | CurAcc:{acc:.3f} AvgAcc:{current_avg_acc:.3f}"
                    )

                    # 更新进度条描述：增加 CurAcc (当前batch) 和 AvgAcc (累计平均)
                    bar.set_description(
                        f"Ep:{epoch} B:{ix} | L:{final_loss:.3f} Var:{loss_var:.3f} | CurAcc:{acc:.3f} AvgAcc:{current_avg_acc:.3f}"
                    )

                    # --- [修改点]：如果你希望每隔 N 个 batch 显式打印一行 ---
                    if ix % log_step == 0:
                        tqdm.write(f"Epoch {epoch} Step {ix}: Loss={final_loss:.4f}, Acc={acc:.4f}")

            # Epoch 结束后的逻辑保持不变
            average_train_auc = np.mean(train_acc_list)

            if average_train_auc > max_auc:
                max_auc = average_train_auc
                test_results = runner.classification_test(epoch, data)

                # [修正 5] 更新 measure_dict
                metrics = "train_auc,val_auc,test_auc,epoch,test_train_auc,test_val_auc,test_test_auc".split(",")
                current_metrics = [max_auc, 0, 0] + test_results
                measure_dict = dict(zip(metrics, current_metrics))
                results = measure_dict

                patience = 0
                filepath = "../checkpoint/" + 'dspy' + args.dataset + str(runs) + ".pth"
                if not os.path.exists("../checkpoint/"):
                    os.makedirs("../checkpoint/")
                torch.save({"model_state_dict": model.state_dict()}, filepath)
            else:
                patience += 1
                if epoch > min_epoch and patience > max_patience:
                    print(f"Early stopping at epoch {epoch}")
                    break

            if epoch == 1 or epoch % args.log_interval == 0:
                print("Epoch:{}, Time: {:.3f}".format(epoch, time.time() - t0))
                print(f"Summary: Epoch:{epoch}, Train Avg Acc:{average_train_auc:.4f}")
                if len(test_results) >= 4:
                    print(
                        f"Test: Epoch:{test_results[0]}, Res1:{test_results[1]:.4f}, Res2: {test_results[2]:.4f}, Res3: {test_results[3]:.4f}")


def evaluate(args, runner):
    aminer15 = []
    aminer16 = []
    aminer17 = []
    node_masks = runner.data['node_masks']
    for runs in range(1):
        data = runner.data

        runner.optimizer = optim.Adam(
            [p for n, p in runner.model.named_parameters() if "ss" not in n],
            lr=args.lr,
            weight_decay=args.weight_decay, )

        filepath = "../checkpoint/" + 'dspy' + args.dataset + str(runs) + ".pth"
        checkpoint = torch.load(filepath)
        runner.model.load_state_dict(checkpoint["model_state_dict"])
        runner.model.eval()

        test_list = []
        last_embeddings = None

        # for i in range(runner.len):
        #
        #     embeddings = runner.model(data['edge_index_list'][i].long().to(args.device), runner.x[i],
        #                               runner.neighbors_all[i], i, aug_loss=False)

        for i in range(runner.len - 3, runner.len):
            embeddings = runner.model(data['edge_index'][i].long().to(args.device), runner.x[i],
                                      i, False)
            label = data['y'][node_masks[i]].squeeze()
            predictions_z_I = runner.classification_cal_y(embeddings, runner.model.classifier, node_masks, device,
                                                          i)  # [N,C]

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
    model = EADGNN(args=args).to(args.device)

    runner = Runner(args, model, data)
    results = evaluate(args, runner)
    print('Results for dataset {}:', args.dataset)
    print(results)

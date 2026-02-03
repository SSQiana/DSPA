import sys
import time

import warnings

import torch.optim as optim
from tqdm import tqdm

sys.path.append("..")
# get arguments
from DSPY.config import args
from DSPY.utils.mutils import *
from DSPY.utils.data_util import *
from DSPY.utils.inits import prepare
from DSPY.runner import Runner
from DSPY.model import DyGNN
from Augmention.data_augmention import Compose, FeatureAugmentor, SpectralAugmentor
from evaluate_link_prediction import evaluate
warnings.simplefilter("ignore")


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


for runs in range(args.num_runs):
    seed_everything(runs)

    # get data for training, validation and testing
    args, data = load_data(args)
    # create model
    model = DyGNN(args=args).to(args.device)
    runner = Runner(args, model, data)

    span = SpectralAugmentor(
        ratio=args.pe,
        lr=args.aug_lr1,
        iteration=args.aug_iter,
        dis_type='max',
        device=args.device,
        threshold=0.8,
        k=args.k
    )

    path = osp.join(os.getcwd(), 'augmented_data')
    if not os.path.exists(path):
        os.makedirs(path)

    subdir = osp.join(path, args.dataset)
    if not os.path.exists(subdir):
        os.makedirs(subdir)

    results = {}
    min_epoch = args.min_epoch
    max_patience = args.patience
    patience = 0

    optimizer = optim.Adam(
        [p for n, p in runner.model.named_parameters() if "ss" not in n],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    data['train']['aug_edge_index_list'] = []

    with tqdm(range(1, args.max_epoch + 1)) as bar:
        t0 = time.time()
        for epoch in bar:
            val_auc_list = []
            test_auc_list = []
            train_auc_list = []

            current_edge_index_list = data['train']['edge_index_list']

            for ix, edge_index in enumerate(current_edge_index_list):
                batch_pedges = edge_index
                batch_nedges = data['train']['nedges'][ix]
                update_data_path = osp.join(subdir, '{}_{}_{}.pt'.format(ix, args.aug_lr1, args.pe))
                # t1 = time.time()
                if len(data['train']['aug_edge_index_list']) <= ix:
                    if os.path.exists(update_data_path):
                        # Load precomputed probability matrix
                        batch_aug_edge_index = torch.load(update_data_path)
                        data['train']['aug_edge_index_list'].append(batch_aug_edge_index)
                    else:
                        span.calc_prob(args, data, ix)

                    if len(data['train']['aug_edge_index_list']) > ix:
                        batch_aug_edge_index = data['train']['aug_edge_index_list'][ix]
                        save_path = os.path.join(subdir, f"{ix}_{args.aug_lr1}_{args.pe}.pt")
                        if not os.path.exists(save_path):
                            torch.save(batch_aug_edge_index, save_path)

                log_dir = args.log_dir
                info_dict = get_arg_dict(args)

                z_I = None  # To store invariant embeddings
                z_V_list = []  # To store a list of variant embeddings

                for num_aug, aug_edge_index in enumerate(batch_aug_edge_index):
                    edge_index_aug = aug_edge_index

                    if num_aug == 0:
                        z_I = runner.model(edge_index_aug.long().to(args.device), runner.x[ix], ix,
                                           aug_loss=False, view_idx=num_aug)

                        if ix <= runner.len - 2:
                            _, pos_edge, neg_edge = prepare(data['train'], ix + 1)[:3]
                            auc, ap = runner.loss.predict(z_I, pos_edge, neg_edge, runner.model.edge_decoder)

                            if ix < runner.len_train - 1:
                                train_auc_list.append(auc)
                            elif ix < runner.len_train + runner.len_val - 1:
                                val_auc_list.append(auc)
                            else:
                                test_auc_list.append(auc)
                    else:
                        z_V = runner.model(edge_index_aug.long().to(args.device), runner.x[ix], ix,
                                           aug_loss=False, view_idx=num_aug)
                        z_V_list.append(z_V)

                # Loss Calculation Logic
                if ix <= runner.len - 2:
                    _, pos_edge, neg_edge = prepare(data['train'], ix + 1)[:3]

                    if args.dataset == "yelp":
                        neg_edge_index = bi_negative_sampling(pos_edge, args.num_nodes, args.shift)
                    else:
                        neg_edge_index = negative_sampling(pos_edge, num_neg_samples=pos_edge.size(
                            1) * args.sampling_times)

                    edge_label_index = torch.cat([pos_edge, neg_edge], dim=-1)
                    pos_y = z_I.new_ones(pos_edge.size(1))
                    neg_y = z_I.new_zeros(neg_edge.size(1))
                    edge_label = torch.cat([pos_y, neg_y], dim=0)

                    pred_y_I = runner.cal_y(z_I, runner.model.edge_decoder, edge_label_index, args.evice)

                    loss_I = runner.cal_loss(pred_y_I, edge_label)

                    loss_V_list = []
                    for z_V in z_V_list:
                        pred_y_V = runner.cal_y(z_V, runner.model.edge_decoder, edge_label_index, args.device)
                        loss_V_list.append(runner.cal_loss(pred_y_V, edge_label))

                    all_task_losses = [loss_I] + loss_V_list
                    if all_task_losses:
                        loss_stack = torch.stack(all_task_losses)
                        loss_mean = loss_stack.mean()
                        loss_var = loss_stack.var()
                    else:
                        loss_mean = loss_I
                        loss_var = torch.tensor(0.0, device=args.device)

                    loss_C_list = []
                    for z_V in z_V_list:
                        loss_C_list.append(runner.contrastive_loss(z_I, z_V))

                    if loss_C_list:
                        loss_C = torch.stack(loss_C_list).mean()
                    else:
                        loss_C = torch.tensor(0.0, device=args.device)

                    final_loss = loss_mean + loss_var + args.alpha * loss_C

                    optimizer.zero_grad()
                    final_loss.backward()
                    optimizer.step()

                    average_epoch_loss = final_loss.item()

                    # Log progress
                    auc, ap = runner.loss.predict(z_I, pos_edge, neg_edge, runner.model.edge_decoder)
                    bar.set_description(
                        f"batch_idx: {ix}, AUC: {auc:.4f}, AP: {ap:.4f}, Loss: {final_loss:.4f}, "
                        f"L_I: {loss_I:.4f},  L_V: {loss_var:.4f}"
                    )

            # Epoch Summary
            average_train_auc = np.mean(train_auc_list) if train_auc_list else 0
            average_val_auc = np.mean(val_auc_list) if val_auc_list else 0
            average_test_auc = np.mean(test_auc_list) if test_auc_list else 0

            if average_val_auc > max_auc:
                max_auc = average_val_auc
                max_test_auc = average_test_auc
                max_train_auc = average_train_auc
                test_results = runner.test(epoch, data["test"])
                metrics = "train_auc,val_auc,test_auc,epoch,test_train_auc,test_val_auc,test_test_auc".split(
                    ",")

                current_best_metrics = [max_train_auc, max_auc, max_test_auc] + test_results
                measure_dict = dict(zip(metrics, current_best_metrics))
                results = measure_dict

                patience = 0
                filepath = "../checkpoint/" + 'dspy' + args.dataset + str(runs)  + ".pth"
                if not os.path.exists("../checkpoint/"):
                    os.makedirs("../checkpoint/")
                torch.save({"model_state_dict": model.state_dict()}, filepath)
            else:
                patience += 1
                if epoch > min_epoch and patience > max_patience:
                    print(f"Early stopping at epoch {epoch}")
                    break

            if epoch == 1 or epoch % args.log_interval == 0:
                print(
                    "Epoch:{}, Loss: {:.4f}, Time: {:.3f}".format(epoch, average_epoch_loss, time.time() - t0))
                print(
                    f"Current: Epoch:{epoch}, Train AUC:{average_train_auc:.4f}, Val AUC: {average_val_auc:.4f}, Test AUC: {average_test_auc:.4f}")
                print(
                    f"Train: Epoch:{test_results[0]}, Train AUC:{max_train_auc:.4f}, Val AUC: {max_auc:.4f}, Test AUC: {max_test_auc:.4f}")
                print(
                    f"Test: Epoch:{test_results[0]}, Train AUC:{test_results[1]:.4f}, Val AUC: {test_results[2]:.4f}, Test AUC: {test_results[3]:.4f}")

    if results:
        info_dict.update(results)
        filename = "info_" + args.dataset + ".json"
        with open(osp.join(log_dir, filename), "w") as f:
            json.dump(info_dict, f, indent=4)

    del model, runner, optimizer
    torch.cuda.empty_cache()


args, data = load_data(args)
model = DyGNN(args=args).to(args.device)
runner = Runner(args, model, data)

res = evaluate(args, runner)
print("=" * 50)
print(f"Evaluation Results")
print(f"Dataset    : {args.dataset}")
print(f"Model      : DyGNN")
print(f"Runs       : {args.num_runs}")
print(f"n_layers   : {args.n_layers}")
print("-" * 50)

print(f"Train AP : {res['train_ap_mean']:.4f} ± {res['train_ap_std']:.4f}")
print(f"Val   AP : {res['val_ap_mean']:.4f} ± {res['val_ap_std']:.4f}")
print(f"Test  AP : {res['test_ap_mean']:.4f} ± {res['test_ap_std']:.4f}")

print("=" * 50)

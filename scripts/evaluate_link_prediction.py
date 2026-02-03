import sys
sys.path.append("..")
from DSPY.utils.mutils import *
from DSPY.utils.data_util import *
import torch.optim as optim
import warnings
from DSPY.utils.inits import prepare



warnings.simplefilter("ignore")

def evaluate(args, runner):
    all_train_ap = []
    all_val_ap = []
    all_test_ap = []

    for runs in range(args.num_runs):
        data = runner.data["test"]

        runner.optimizer = optim.Adam(
            [p for n, p in runner.model.named_parameters() if "ss" not in n],
            lr=args.lr,
            weight_decay=args.weight_decay,
        )

        filepath = "../checkpoint/" + 'dspy' + args.dataset + str(runs) + ".pth"

        if not os.path.exists(filepath):
            print(f"Checkpoint not found: {filepath}")
            continue

        checkpoint = torch.load(filepath)
        runner.model.load_state_dict(checkpoint["model_state_dict"])
        runner.model.eval()

        train_ap_list = []
        val_ap_list = []
        test_ap_list = []

        for i in range(runner.len):
            embeddings = runner.model(
                data['edge_index_list'][i].long().to(args.device), runner.x[i], i, aug_loss=False
            )
            if i < runner.len - 1:
                z = embeddings
                edge_index, pos_edge, neg_edge = prepare(data, i + 1)[:3]
                if is_empty_edges(neg_edge):
                    continue
                _, ap = runner.loss.predict(
                    z, pos_edge, neg_edge, runner.model.edge_decoder
                )

                if i < runner.len_train - 1:
                    train_ap_list.append(ap)
                elif i < runner.len_train + runner.len_val - 1:
                    val_ap_list.append(ap)
                else:
                    test_ap_list.append(ap)

        all_train_ap.append(np.mean(train_ap_list) if train_ap_list else 0)
        all_val_ap.append(np.mean(val_ap_list) if val_ap_list else 0)
        all_test_ap.append(np.mean(test_ap_list) if test_ap_list else 0)

    return {
        "train_ap_mean": np.mean(all_train_ap),
        "train_ap_std": np.std(all_train_ap),
        "val_ap_mean": np.mean(all_val_ap),
        "val_ap_std": np.std(all_val_ap),
        "test_ap_mean": np.mean(all_test_ap),
        "test_ap_std": np.std(all_test_ap)
    }

import argparse
import csv
import random
import time
import gc
import os
import numpy as np
import torch
from torch.utils.data import DataLoader

from models.models import GroupIM
from models.pretrain import seed_group_predictor, user_pretrain_loss, user_pretrain_parameters
from utils.user_utils import TrainUserDataset, EvalUserDataset
from utils.group_utils import TrainGroupDataset, EvalGroupDataset
from eval.evaluate import evaluate_user, evaluate_group

parser = argparse.ArgumentParser(description='PyTorch GroupIM: Group Information Maximization for Group Recommendation')
parser.add_argument('--dataset', type=str, default='weeplaces', help='Name of dataset')

# Training settings.
parser.add_argument('--lr', type=float, default=5e-3, help='initial learning rate')
parser.add_argument('--wd', type=float, default=0.00, help='weight decay coefficient')
parser.add_argument('--lambda_mi', type=float, default=1.0, help='MI lambda hyper param')
parser.add_argument('--drop_ratio', type=float, default=0.4, help='Dropout ratio')
parser.add_argument('--batch_size', type=int, default=256, help='batch size')
parser.add_argument('--epochs', type=int, default=200, help='maximum # training epochs')
parser.add_argument('--eval_freq', type=int, default=5, help='frequency to evaluate performance on validation set')

# Model settings.
parser.add_argument('--emb_size', type=int, default=64, help='layer size')
parser.add_argument('--aggregator', type=str, default='attention', help='choice of group preference aggregator',
                    choices=['maxpool', 'meanpool', 'attention'])
parser.add_argument('--negs_per_group', type=int, default=5, help='# negative users sampled per group')

# Pre-training settings.
parser.add_argument('--pretrain_user', action='store_true', help='Pre-train user encoder on user-item interactions')
parser.add_argument('--pretrain_mi', action='store_true', help='Pre-train MI estimator for a few epochs')
parser.add_argument('--pretrain_epochs', type=int, default=100, help='# pre-train epochs for user encoder layer')

parser.add_argument('--cuda', action='store_true', help='use CUDA')
parser.add_argument('--seed', type=int, default=1111, help='random seed for reproducibility')
parser.add_argument('--encoder', choices=['mlp', 'lightgcn'], default='mlp')
parser.add_argument('--k', type=int, default=0, help='LightGCN layers. Ignored for the MLP encoder.')
parser.add_argument('--losses', choices=['full'], default='full', help='full is L_G + λ L_U + L_MI')
parser.add_argument('--mi-epochs', type=int, default=10, help='MI pretrain epochs')
parser.add_argument('--tied-item-embeddings', action=argparse.BooleanOptionalAction, default=True)
parser.add_argument('--artifact-dir', type=str, default='', help='Write per-group test records here')

# Model save file parameters.
parser.add_argument('--save', type=str, default='', help='path to save the user encoder; default is <artifact-dir>/model_user.pt')
parser.add_argument('--save_group', type=str, default='', help='path to save the group model; default is <artifact-dir>/model_group.pt')

args = parser.parse_args()
if not args.artifact_dir:
    raise SystemExit('pass --artifact-dir; checkpoints and records go in the run directory')
os.makedirs(args.artifact_dir, exist_ok=True)
if not args.save:
    args.save = os.path.join(args.artifact_dir, 'model_user.pt')
if not args.save_group:
    args.save_group = os.path.join(args.artifact_dir, 'model_group.pt')
for save_path in (args.save, args.save_group):
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)

run_started = time.time()
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(args.seed)

if args.cuda and not torch.cuda.is_available():
    raise SystemExit("CUDA was requested with --cuda, but no CUDA device is available")
if torch.cuda.is_available() and not args.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

###############################################################################
# Load data
###############################################################################

# The author used 6 workers. On this WSL box that many workers densify group
# batches and the OOM killer sends SIGKILL. The main process is enough.
num_workers = 0
train_params = {'batch_size': args.batch_size, 'shuffle': False, 'num_workers': num_workers, 'pin_memory': True}
eval_params = {'batch_size': args.batch_size, 'shuffle': False, 'num_workers': num_workers, 'pin_memory': True}
device = torch.device("cuda" if args.cuda else "cpu")
if args.cuda:
    free_bytes = [torch.cuda.mem_get_info(i)[0] for i in range(torch.cuda.device_count())]
    torch.cuda.set_device(int(np.argmax(free_bytes)))

# Define train/val/test datasets on user interactions.
train_dataset = TrainUserDataset(args.dataset)  # train dataset for user-item interactions.
n_items, n_users = train_dataset.n_items, train_dataset.n_users
val_dataset = EvalUserDataset(args.dataset, n_items, datatype='val')
test_dataset = EvalUserDataset(args.dataset, n_items, datatype='test')

# Define train/val/test datasets on group and user interactions.
train_group_dataset = TrainGroupDataset(args.dataset, n_items, args.negs_per_group)
padding_idx = train_group_dataset.padding_idx
val_group_dataset = EvalGroupDataset(args.dataset, n_items, padding_idx, datatype='val')
test_group_dataset = EvalGroupDataset(args.dataset, n_items, padding_idx, datatype='test')

# Define data loaders on user interactions.
train_loader = DataLoader(train_dataset, **train_params)
val_loader = DataLoader(val_dataset, **eval_params)
test_loader = DataLoader(test_dataset, **eval_params)

# Define data loaders on group interactions.
train_group_loader = DataLoader(train_group_dataset, **train_params)
val_group_loader = DataLoader(val_group_dataset, **eval_params)
test_group_loader = DataLoader(test_group_dataset, **eval_params)

###############################################################################
# Build the model
###############################################################################

user_layers = [args.emb_size]  # user encoder layer configuration is tunable.

if args.encoder == 'lightgcn':
    from models.lightgcn import normalized_adjacency, read_foldin_edges
    edge_users, edge_items, n_users_graph = read_foldin_edges(os.path.join('data', args.dataset), n_items)
    norm_adj = normalized_adjacency(edge_users, edge_items, n_users_graph, n_items)
    model = GroupIM(
        n_items, user_layers, drop_ratio=args.drop_ratio, aggregator_type=args.aggregator,
        lambda_mi=args.lambda_mi, encoder_type='lightgcn', n_layers=args.k,
        n_users=n_users_graph, norm_adj=norm_adj,
    ).to(device)
else:
    model = GroupIM(n_items, user_layers, drop_ratio=args.drop_ratio, aggregator_type=args.aggregator,
                    lambda_mi=args.lambda_mi).to(device)
optimizer_gr = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)

best_user_n100, best_group_n100 = -np.inf, -np.inf

print("args", args)
# At any point you can hit Ctrl + C to break out of training early.
try:
    if args.pretrain_user:
        optimizer_ur = torch.optim.Adam(user_pretrain_parameters(model), lr=0.01, weight_decay=args.wd)
        print("Pre-training model on user-item interactions")
        for epoch in range(0, args.pretrain_epochs):
            epoch_start_time = time.time()
            model.train()
            train_user_loss = 0.0
            start_time = time.time()

            for batch_index, data in enumerate(train_loader):
                optimizer_ur.zero_grad()
                data = [x.to(device, non_blocking=True) for x in data]
                (train_users, train_items) = data
                user_loss = user_pretrain_loss(model, train_users, train_items)
                user_loss.backward()
                train_user_loss += user_loss.item()
                optimizer_ur.step()
                del train_users, train_items, user_loss
            elapsed = time.time() - start_time
            print('| epoch {:3d} |  time {:4.2f} | loss {:4.2f}'.format(epoch + 1, elapsed,
                                                                        train_user_loss / len(train_loader)))
            if epoch % args.eval_freq == 0:
                val_scores = evaluate_user(model, val_loader, device, mode='pretrain')

                if val_scores.n100 > best_user_n100:
                    torch.save(model.state_dict(), args.save)
                    best_user_n100 = val_scores.n100

        print("Load best pre-trained user encoder")
        model.load_state_dict(torch.load(args.save, weights_only=False))
        model = model.to(device)

        val_scores = evaluate_user(model, val_loader, device, mode='pretrain')
        print('=' * 89)
        print('| User evaluation | val loss {:4.4f} | n100 {:4.4f} | r20 {:4.4f} | '
              'r50 {:4.4f}'.format(val_scores.loss, val_scores.n100, val_scores.r20, val_scores.r50))
        print("Initializing group recommender with pre-train user encoder")
        if args.tied_item_embeddings:
            # Initialize the group predictor (item embedding) weight based on the pre-trained user predictor.
            seed_group_predictor(model)

    if args.pretrain_mi:
        # pre-train MI estimator.
        for epoch in range(0, args.mi_epochs):
            model.train()
            t = time.time()
            mi_epoch_loss = 0.0
            for batch_index, data in enumerate(train_group_loader):
                data = [x.to(device, non_blocking=True) for x in data]
                group, group_users, group_mask, group_items, user_items, corrupted_user_items, corrupted_users = data
                optimizer_gr.zero_grad()
                model.zero_grad()
                model.train()
                _, group_embeds, _ = model(group, group_users, group_mask, user_items)
                obs_user_embed = model.user_preference_encoder(user_items, group_users).detach()  # [B, G, D]
                corrupted_user_embed = model.user_preference_encoder(
                    corrupted_user_items, corrupted_users
                ).detach()  # [B, # negs, D]

                scores_observed = model.discriminator(group_embeds, obs_user_embed, group_mask)  # [B, G]
                scores_corrupted = model.discriminator(group_embeds, corrupted_user_embed, group_mask)  # [B, # negs]

                mi_loss = model.discriminator.mi_loss(scores_observed, group_mask, scores_corrupted, device=device)
                mi_loss.backward()
                optimizer_gr.step()
                mi_epoch_loss += mi_loss.item()
                del group, group_users, group_mask, group_items, user_items, corrupted_user_items, \
                    obs_user_embed, corrupted_user_embed
            gc.collect()
            print("MI loss: {}".format(float(mi_epoch_loss) / len(train_group_loader)))

    optimizer_gr = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)

    for epoch in range(0, args.epochs):
        epoch_start_time = time.time()
        model.train()
        train_group_epoch_loss = 0.0
        for batch_index, data in enumerate(train_group_loader):
            data = [x.to(device, non_blocking=True) for x in data]
            group, group_users, group_mask, group_items, user_items, corrupted_user_items, corrupted_users = data
            optimizer_gr.zero_grad()
            model.zero_grad()
            group_logits, group_embeds, scores_ug = model(group.squeeze(), group_users, group_mask, user_items)
            group_loss = model.loss(group_logits, group_embeds, scores_ug, group_mask, group_items, user_items,
                                    corrupted_user_items, group_users, corrupted_users, device=device)
            group_loss.backward()
            train_group_epoch_loss += group_loss.item()
            optimizer_gr.step()
            del group, group_users, group_mask, group_items, user_items, corrupted_user_items, \
                group_logits, group_embeds, scores_ug

        gc.collect()

        print("Train loss: {}".format(float(train_group_epoch_loss) / len(train_group_loader)))

        if epoch % args.eval_freq == 0:
            # Group evaluation.
            val_group = evaluate_group(model, val_group_loader, device)

            print('-' * 89)
            print('| end of epoch {:3d} | time: {:4.2f}s | n100 (group) {:5.4f} | r20 (group) {:5.4f} | r50 (group) '
                  '{:5.4f}'.format(epoch + 1, time.time() - epoch_start_time, val_group.n100, val_group.r20, val_group.r50))
            print('-' * 89)

            # Save the model if the n100 is the best we've seen so far. Author checkpoint metric.
            if val_group.n100 > best_group_n100:
                with open(args.save_group, 'wb') as f:
                    torch.save(model, f)
                best_group_n100 = val_group.n100

except KeyboardInterrupt:
    print('-' * 89)
    print('Exiting from training early')

# Load the best saved model.
with open(args.save_group, 'rb') as f:
    model = torch.load(f, map_location=device, weights_only=False)
    model = model.to(device)

def _metric(value):
    return float(value.detach().cpu())


# Best validation evaluation
val_scores = evaluate_user(model, val_loader, device, mode='group')
print('=' * 89)
print('| User evaluation | val loss {:4.4f} | n100 {:4.4f} | r20 {:4.4f} | r50 {:4.4f}'
      .format(val_scores.loss, val_scores.n100, val_scores.r20, val_scores.r50))

# Test evaluation
test_scores = evaluate_user(model, test_loader, device, mode='group')
print('=' * 89)
print('| User evaluation | test loss {:4.4f} | n100 {:4.4f} | r20 {:4.4f} | r50 {:4.4f}'
      .format(test_scores.loss, test_scores.n100, test_scores.r20, test_scores.r50))

print('=' * 89)
val_group = evaluate_group(model, val_group_loader, device)
print('| Group evaluation (val) | n20 {:4.4f} | n50 {:4.4f} | n100 {:4.4f} | r20 {:4.4f} | r50 {:4.4f}'
      .format(val_group.n20, val_group.n50, val_group.n100, val_group.r20, val_group.r50))

print('=' * 89)
test_group = evaluate_group(model, test_group_loader, device)
print('| Group evaluation (test) | n20 {:4.4f} | n50 {:4.4f} | n100 {:4.4f} | r20 {:4.4f} | r50 {:4.4f}'
      .format(test_group.n20, test_group.n50, test_group.n100, test_group.r20, test_group.r50))

metrics_path = os.path.join(os.path.dirname(args.save_group) or ".", "metrics.csv")
wrote_header = not os.path.exists(metrics_path)
with open(metrics_path, "a", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=[
        "dataset", "encoder", "aggregator", "emb_size", "seed", "pretrain_user", "pretrain_mi",
        "epochs", "pretrain_epochs", "n20", "n50", "r20", "r50", "n100", "seconds",
    ])
    if wrote_header:
        writer.writeheader()
    writer.writerow({
        "dataset": args.dataset,
        "encoder": args.encoder,
        "aggregator": args.aggregator,
        "emb_size": args.emb_size,
        "seed": args.seed,
        "pretrain_user": int(args.pretrain_user),
        "pretrain_mi": int(args.pretrain_mi),
        "epochs": args.epochs,
        "pretrain_epochs": args.pretrain_epochs,
        "n20": _metric(test_group.n20),
        "n50": _metric(test_group.n50),
        "r20": _metric(test_group.r20),
        "r50": _metric(test_group.r50),
        "n100": _metric(test_group.n100),
        "seconds": round(time.time() - run_started, 1),
    })
print(f"wrote {metrics_path}")
if args.artifact_dir:
    from groupim_lightgcn_ablation.group_stats import dump_test_groups
    n_groups = dump_test_groups(
        model,
        test_group_loader,
        device,
        os.path.join(args.artifact_dir, "test_groups.csv"),
        os.path.join(args.artifact_dir, "member_scores.csv"),
    )
    print(f"wrote {n_groups} test groups to {args.artifact_dir}")

"""LightGCN user pretraining: L_U updates embeddings, recall moves, group predictor is seeded."""

from pathlib import Path

import numpy as np
import pandas as pd
import torch

from models.lightgcn import normalized_adjacency
from models.models import GroupIM
from models.pretrain import seed_group_predictor, user_pretrain_loss, user_pretrain_parameters
from utils.user_utils import EvalUserDataset


def _lightgcn(n_users: int, n_items: int, dim: int, k: int) -> GroupIM:
    users = np.arange(n_users)
    items = np.arange(n_users)
    adj = normalized_adjacency(users, items, n_users, n_items)
    return GroupIM(
        n_items, [dim], encoder_type="lightgcn", n_layers=k, n_users=n_users, norm_adj=adj, aggregator_type="meanpool"
    )


def _recall_at_1(model: GroupIM, user_ids: torch.Tensor, fold_in: torch.Tensor, held_out: torch.Tensor) -> float:
    model.eval()
    with torch.no_grad():
        logits, _embeds = model.user_preference_encoder.pre_train_forward(fold_in, user_ids)
        scores = logits.clone()
        scores[torch.arange(scores.shape[0]), fold_in.argmax(dim=1)] = -torch.inf
        predicted = scores.argmax(dim=-1)
        targets = held_out.argmax(dim=-1)
        return float((predicted == targets).float().mean())


def test_mlp_pretrain_still_owns_every_parameter():
    model = GroupIM(5, [4])
    assert [id(param) for param in user_pretrain_parameters(model)] == [id(param) for param in model.parameters()]


def test_lightgcn_pretrain_updates_embeddings_and_lifts_recall():
    torch.manual_seed(0)
    n_users, n_items, dim = 4, 12, 16
    model = _lightgcn(n_users, n_items, dim, k=1)
    encoder = model.user_preference_encoder
    before_user = encoder.user_embedding.weight.detach().clone()
    before_item = encoder.user_predictor.weight.detach().clone()
    before_group = model.group_predictor.weight.detach().clone()
    before_disc = model.discriminator.fc_layer.weight.detach().clone()
    before_agg = next(model.preference_aggregator.parameters()).detach().clone()

    user_ids = torch.arange(n_users).view(-1, 1)
    fold_in = torch.zeros(n_users, n_items)
    held_out = torch.zeros(n_users, n_items)
    fold_in[torch.arange(n_users), torch.arange(n_users)] = 1
    held_out[torch.arange(n_users), torch.arange(n_users) + n_users] = 1
    recall_before = _recall_at_1(model, user_ids, fold_in, held_out)

    optimizer = torch.optim.Adam(user_pretrain_parameters(model), lr=0.1)
    model.train()
    for _ in range(80):
        optimizer.zero_grad()
        loss = user_pretrain_loss(model, user_ids, held_out)
        loss.backward()
        optimizer.step()

    assert not torch.allclose(encoder.user_embedding.weight, before_user)
    assert not torch.allclose(encoder.user_predictor.weight, before_item)
    assert torch.allclose(model.group_predictor.weight, before_group)
    assert torch.allclose(model.discriminator.fc_layer.weight, before_disc)
    assert torch.allclose(next(model.preference_aggregator.parameters()), before_agg)
    recall_after = _recall_at_1(model, user_ids, fold_in, held_out)
    assert recall_after > recall_before
    assert recall_after == 1.0

    seed_group_predictor(model)
    assert torch.equal(model.group_predictor.weight, encoder.user_predictor.weight)
    assert model.group_predictor.weight.data_ptr() == encoder.user_predictor.weight.data_ptr()


def test_eval_users_keep_their_csv_ids(tmp_path: Path, monkeypatch):
    data_dir = tmp_path / "data" / "tiny"
    data_dir.mkdir(parents=True)
    pd.DataFrame({"user": [10, 11], "item": [0, 1]}).to_csv(data_dir / "val_ui_tr.csv", index=False)
    pd.DataFrame({"user": [10, 11], "item": [2, 3]}).to_csv(data_dir / "val_ui_te.csv", index=False)
    monkeypatch.chdir(tmp_path)
    dataset = EvalUserDataset("tiny", n_items=4, datatype="val")
    assert [dataset[index][0] for index in range(len(dataset))] == [10, 11]

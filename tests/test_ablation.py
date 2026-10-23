"""Table 3 rows are launcher configs, for both encoders."""

import numpy as np
import torch

from groupim_lightgcn_ablation.launch import build_parser, config_from_args, train_command
from groupim_lightgcn_ablation.run_record import run_id
from models.aggregators import CosineAggregator, MeanPoolAggregator, UniformAggregator
from models.lightgcn import normalized_adjacency
from models.models import GroupIM


def _config(**overrides):
    args = build_parser().parse_args([])
    for key, value in overrides.items():
        setattr(args, key, value)
    return config_from_args(args)


TABLE3_ROWS = (
    {"losses": "lg", "aggregator": "attention", "pretrain_user": True, "pretrain_mi": True},
    {"losses": "lg+lu", "aggregator": "attention", "pretrain_user": True, "pretrain_mi": True},
    {"losses": "lg+lmi", "aggregator": "attention", "pretrain_user": True, "pretrain_mi": True},
    {"losses": "full", "aggregator": "uniform", "pretrain_user": True, "pretrain_mi": True},
    {"losses": "full", "aggregator": "cosine", "pretrain_user": True, "pretrain_mi": True},
    {"losses": "full", "aggregator": "attention", "pretrain_user": False, "pretrain_mi": False},
    {"losses": "full", "aggregator": "attention", "pretrain_user": True, "pretrain_mi": True},
)


def _model(encoder, losses, aggregator="attention"):
    if encoder == "lightgcn":
        users = np.arange(4)
        adj = normalized_adjacency(users, users, 4, 6)
        return GroupIM(
            6, [4], lambda_mi=0.3, aggregator_type=aggregator, losses=losses,
            encoder_type="lightgcn", n_layers=1, n_users=4, norm_adj=adj,
        )
    return GroupIM(6, [4], lambda_mi=0.3, aggregator_type=aggregator, losses=losses)


def _loss_on(model):
    group = torch.tensor([0, 1])
    group_users = torch.tensor([[0, 1], [2, 3]])
    mask = torch.zeros(2, 2)
    user_items = torch.zeros(2, 2, 6)
    user_items[:, :, 0] = 1
    group_items = torch.zeros(2, 6)
    group_items[:, 1] = 1
    corrupted_items = torch.zeros(2, 2, 6)
    corrupted_items[:, :, 2] = 1
    corrupted_users = torch.tensor([[0, 1], [1, 0]])
    model.eval()
    logits, embeds, scores = model(group, group_users, mask, user_items)
    return model.loss(
        logits, embeds, scores, mask, group_items, user_items, corrupted_items,
        group_users, corrupted_users,
    )


def test_seven_rows_are_distinct_configs_for_both_encoders():
    for encoder in ("mlp", "lightgcn"):
        ids = [run_id(_config(encoder=encoder, **row)) for row in TABLE3_ROWS]
        assert len(set(ids)) == 7
    full = TABLE3_ROWS[-1]
    assert run_id(_config(encoder="mlp", **full)) != run_id(_config(encoder="lightgcn", **full))


def test_pretraining_off_is_the_absence_of_both_flags(tmp_path):
    command = train_command(
        _config(losses="full", aggregator="attention", pretrain_user=False, pretrain_mi=False),
        tmp_path,
    )
    assert "--pretrain_user" not in command
    assert "--pretrain_mi" not in command
    assert command[command.index("--losses") + 1] == "full"


def test_uniform_matches_the_mean_and_cosine_uses_similarity():
    members = torch.tensor([[[1.0, 0.0], [0.2, 0.8], [9.0, -4.0]]])
    mask = torch.tensor([[0.0, 0.0, float("-inf")]])
    uniform = UniformAggregator(2, 2)
    mean = MeanPoolAggregator(2, 2)
    cosine = CosineAggregator(2, 2)
    assert list(uniform.parameters()) == []
    assert list(cosine.parameters()) == []
    assert torch.allclose(uniform(members, mask), mean(members, mask))
    changed = members.clone()
    changed[0, 2] = torch.tensor([3.0, 3.0])
    assert torch.allclose(cosine(members, mask), cosine(changed, mask))
    assert not torch.allclose(cosine(members, mask), uniform(members, mask))


def test_loss_flags_keep_the_same_terms_for_both_encoders():
    for encoder in ("mlp", "lightgcn"):
        torch.manual_seed(0)
        model = _model(encoder, "full")
        model.losses = "lg"
        base = _loss_on(model)
        model.losses = "lg+lu"
        with_user = _loss_on(model)
        model.losses = "lg+lmi"
        with_mi = _loss_on(model)
        model.losses = "full"
        full = _loss_on(model)
        group = torch.tensor([0, 1])
        group_users = torch.tensor([[0, 1], [2, 3]])
        mask = torch.zeros(2, 2)
        user_items = torch.zeros(2, 2, 6)
        user_items[:, :, 0] = 1
        group_items = torch.zeros(2, 6)
        group_items[:, 1] = 1
        corrupted_items = torch.zeros(2, 2, 6)
        corrupted_items[:, :, 2] = 1
        corrupted_users = torch.tensor([[0, 1], [1, 0]])
        logits, embeds, scores = model(group, group_users, mask, user_items)
        mi, user, group_loss = model.infomax_group_loss(
            logits, embeds, scores, mask, group_items, user_items, corrupted_items,
            group_users, corrupted_users,
        )
        assert torch.allclose(base, group_loss)
        assert torch.allclose(with_user, group_loss + model.lambda_mi * user)
        assert torch.allclose(with_mi, group_loss + mi)
        assert torch.allclose(full, group_loss + model.lambda_mi * user + mi)
        assert not torch.allclose(base, full)

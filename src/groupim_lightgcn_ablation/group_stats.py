"""Per-group test records. Analysis tickets read these files and do not retrain.

Coherence is the mean pairwise Pearson correlation of member item vectors, the
paper's "mean pair-wise correlation of personal activities". Aggregate diversity
is the number of distinct items across those members. Discriminator scores are
the bilinear logits for present members only.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import torch

from eval import metrics


def mean_pairwise_pearson(vectors: np.ndarray) -> float:
    """Mean upper-triangle Pearson correlation. NaN when fewer than two varying rows."""
    if vectors.shape[0] < 2:
        return float("nan")
    keep = vectors.std(axis=1) > 0
    vectors = vectors[keep]
    if vectors.shape[0] < 2:
        return float("nan")
    corr = np.corrcoef(vectors)
    upper = corr[np.triu_indices(corr.shape[0], k=1)]
    upper = upper[np.isfinite(upper)]
    if upper.size == 0:
        return float("nan")
    return float(upper.mean())


def aggregate_diversity(vectors: np.ndarray) -> int:
    """Distinct items with a positive interaction for any present member."""
    if vectors.size == 0:
        return 0
    return int((vectors > 0).any(axis=0).sum())


def _present(mask: torch.Tensor) -> torch.Tensor:
    return torch.isfinite(mask) & (mask == 0)


def dump_test_groups(model, loader, device, group_path: Path, member_path: Path) -> int:
    """Write one test-group row and one discriminator row per present member."""
    group_path = Path(group_path)
    member_path = Path(member_path)
    model.eval()
    groups: list[dict] = []
    members: list[dict] = []
    with torch.no_grad():
        for data in loader:
            data = [x.to(device, non_blocking=True) for x in data]
            group, group_users, group_mask, group_items, user_items = data
            groups_batch, members_batch = records_from_batch(
                model, group, group_users, group_mask, group_items, user_items
            )
            groups.extend(groups_batch)
            members.extend(members_batch)
    _write_csv(
        group_path,
        ["group_id", "group_size", "coherence", "aggregate_diversity", "n20", "n50", "r20", "r50"],
        groups,
    )
    _write_csv(member_path, ["group_id", "user_id", "discriminator_score"], members)
    return len(groups)


def records_from_batch(model, group, group_users, group_mask, group_items, user_items):
    """Per-group metrics and per-member discriminator logits for one batch."""
    logits = _forward_eval(model, group, group_users, group_mask, user_items)
    scores = logits.softmax(1)
    device = scores.device
    n20 = metrics.ndcg_binary_at_k_batch_torch(scores, group_items, 20, device=device)
    n50 = metrics.ndcg_binary_at_k_batch_torch(scores, group_items, 50, device=device)
    r20 = metrics.recall_at_k_batch_torch(scores, group_items, 20)
    r50 = metrics.recall_at_k_batch_torch(scores, group_items, 50)
    member_logits = _member_logits(model, user_items, group_users, group_mask)

    group_rows = []
    member_rows = []
    group_ids = group.view(-1).detach().cpu().tolist()
    for index, group_id in enumerate(group_ids):
        present = _present(group_mask[index]).detach().cpu()
        vectors = user_items[index].detach().cpu().numpy()[present.numpy()]
        size = int(present.sum())
        group_rows.append(
            {
                "group_id": int(group_id),
                "group_size": size,
                "coherence": mean_pairwise_pearson(vectors),
                "aggregate_diversity": aggregate_diversity(vectors),
                "n20": float(n20[index].detach().cpu()),
                "n50": float(n50[index].detach().cpu()),
                "r20": float(r20[index].detach().cpu()),
                "r50": float(r50[index].detach().cpu()),
            }
        )
        users = group_users[index].detach().cpu()
        logits_row = member_logits[index].detach().cpu()
        for slot in torch.where(present)[0].tolist():
            member_rows.append(
                {
                    "group_id": int(group_id),
                    "user_id": int(users[slot]),
                    "discriminator_score": float(logits_row[slot]),
                }
            )
    return group_rows, member_rows


def _forward_eval(model, group, group_users, group_mask, user_items):
    was_training = model.training
    model.eval()
    try:
        output = model(group, group_users, group_mask, user_items)
        return output[0]
    finally:
        model.train(was_training)


def _member_logits(model, user_items, group_users, group_mask):
    embeds = model.user_preference_encoder(user_items, group_users)
    group_embed = model.preference_aggregator(embeds, group_mask, mlp=False)
    scores = model.discriminator(group_embed, embeds, group_mask)
    return scores.squeeze(-1)


def _write_csv(path: Path, fields: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

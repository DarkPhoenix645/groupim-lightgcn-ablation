"""LightGCN encoder: fold-in graph, reference propagation, tiny-batch overfit."""

from pathlib import Path

import numpy as np
import pandas as pd
import torch

from models.lightgcn import LightGCNEncoder, normalized_adjacency, read_foldin_edges


def _write(directory: Path, name: str, rows: list[tuple[int, int]]) -> None:
    pd.DataFrame(rows, columns=["user", "item"]).to_csv(directory / name, index=False)


def test_foldin_graph_skips_held_out_items(tmp_path: Path):
    _write(tmp_path, "train_ui.csv", [(0, 0)])
    _write(tmp_path, "val_ui_tr.csv", [(1, 1)])
    _write(tmp_path, "test_ui_tr.csv", [(2, 2)])
    _write(tmp_path, "test_ui_te.csv", [(3, 3)])
    users, items, n_users = read_foldin_edges(tmp_path, n_items=4)
    assert set(zip(users.tolist(), items.tolist())) == {(0, 0), (1, 1), (2, 2)}
    assert n_users == 3


def test_propagation_matches_a_reference_lightgcn():
    users = np.array([0, 1])
    items = np.array([0, 1])
    n_users, n_items, dim, k = 2, 2, 3, 2
    adj = normalized_adjacency(users, items, n_users, n_items)
    encoder = LightGCNEncoder(n_users, n_items, dim, k, adj)
    encoder.eval()
    with torch.no_grad():
        ego = torch.cat([encoder.user_embedding.weight, encoder.user_predictor.weight], dim=0)
        layers = [ego]
        hidden = ego
        for _ in range(k):
            hidden = torch.sparse.mm(encoder.norm_adj, hidden)
            layers.append(hidden)
        expected = torch.stack(layers, dim=0).mean(dim=0)
        got_users, got_items = encoder.propagate()
        got = torch.cat([got_users, got_items], dim=0)
    assert torch.allclose(got, expected, atol=1e-5)
    assert got.shape == (n_users + 1 + n_items, dim)


def test_tiny_batch_overfit():
    torch.manual_seed(0)
    n_users, n_items, dim, k = 4, 6, 8, 1
    users = np.arange(n_users)
    items = np.arange(n_users)
    adj = normalized_adjacency(users, items, n_users, n_items)
    encoder = LightGCNEncoder(n_users, n_items, dim, k, adj)
    optimizer = torch.optim.Adam(encoder.parameters(), lr=0.1)
    targets = torch.zeros(n_users, n_items)
    targets[torch.arange(n_users), torch.arange(n_users)] = 1
    ids = torch.arange(n_users).view(-1, 1)
    losses = []
    for _ in range(60):
        optimizer.zero_grad()
        logits, _embeds = encoder.pre_train_forward(targets, ids)
        loss = -(torch.log_softmax(logits, dim=-1) * targets).sum(dim=-1).mean()
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach()))
    assert losses[-1] < losses[0] * 0.5
    with torch.no_grad():
        logits, _embeds = encoder.pre_train_forward(targets, ids)
        assert torch.argmax(logits, dim=-1).tolist() == [0, 1, 2, 3]

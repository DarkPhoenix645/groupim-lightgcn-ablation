"""LightGCN user encoder: ID embeddings, K-layer propagation, mean readout.

The graph is the bipartite user-item graph of training interactions plus
fold-in interactions. Held-out user-item files are not edges. Propagation runs
inside the forward pass from the current embeddings.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

FOLDIN_FILES = ("train_ui.csv", "val_ui_tr.csv", "test_ui_tr.csv")


def read_foldin_edges(data_dir: Path, n_items: int) -> tuple[np.ndarray, np.ndarray, int]:
    """User and item indices from train plus fold-in files. Held-out files are not read."""
    frames = [pd.read_csv(Path(data_dir) / name) for name in FOLDIN_FILES]
    pairs = pd.concat(frames, ignore_index=True).drop_duplicates(["user", "item"])
    users = pairs["user"].to_numpy(np.int64)
    items = pairs["item"].to_numpy(np.int64)
    if users.size == 0:
        raise ValueError(f"no fold-in edges under {data_dir}")
    if int(items.max()) >= n_items:
        raise ValueError("fold-in item id is outside the training item range")
    return users, items, int(users.max()) + 1


def normalized_adjacency(users: np.ndarray, items: np.ndarray, n_users: int, n_items: int) -> torch.Tensor:
    """Symmetric normalized bipartite adjacency, with an isolated padding user."""
    n_nodes = n_users + 1 + n_items
    item_nodes = items + (n_users + 1)
    src = np.concatenate([users, item_nodes])
    dst = np.concatenate([item_nodes, users])
    indices = torch.tensor(np.vstack([src, dst]), dtype=torch.long)
    ones = torch.ones(indices.shape[1])
    degree = torch.zeros(n_nodes).scatter_add_(0, indices[0], ones)
    scale = degree.pow(-0.5)
    scale[torch.isinf(scale)] = 0
    values = scale[indices[0]] * scale[indices[1]]
    return torch.sparse_coo_tensor(indices, values, (n_nodes, n_nodes)).coalesce()


class LightGCNEncoder(nn.Module):
    """K-layer LightGCN. The readout is the mean of the initial embeddings and the K propagated layers."""

    def __init__(self, n_users: int, n_items: int, embedding_dim: int, n_layers: int, norm_adj: torch.Tensor):
        super().__init__()
        if n_layers < 0:
            raise ValueError("LightGCN K must be >= 0")
        self.n_items = n_items
        self.n_layers = n_layers
        self.embedding_dim = embedding_dim
        self.user_embedding = nn.Embedding(n_users + 1, embedding_dim)
        self.user_predictor = nn.Linear(embedding_dim, n_items, bias=False)
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.user_predictor.weight)
        with torch.no_grad():
            self.user_embedding.weight[-1].zero_()
        self.register_buffer("norm_adj", norm_adj)

    def propagate(self) -> tuple[torch.Tensor, torch.Tensor]:
        """One propagation from the current ID embeddings. Returns user and item readouts."""
        ego = torch.cat([self.user_embedding.weight, self.user_predictor.weight], dim=0)
        layers = [ego]
        hidden = ego
        for _ in range(self.n_layers):
            hidden = torch.sparse.mm(self.norm_adj, hidden)
            layers.append(hidden)
        readout = torch.stack(layers, dim=0).mean(dim=0)
        n_user_rows = self.user_embedding.num_embeddings
        return readout[:n_user_rows], readout[n_user_rows:]

    def pre_train_forward(self, user_items, user_ids=None):
        if user_ids is None:
            raise ValueError("LightGCN pretrain requires user ids")
        user_out, item_out = self.propagate()
        embeds = user_out[user_ids.long().view(-1)]
        logits = embeds @ item_out.t()
        return logits, embeds

    def forward(self, user_items, user_ids=None):
        if user_ids is None:
            raise ValueError("LightGCN requires user ids")
        user_out, _item_out = self.propagate()
        return user_out[user_ids.long()]

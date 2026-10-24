"""Group multi-hot batches are scattered from item indices, not materialized in __getitem__."""

import torch

from utils.group_utils import (
    EvalGroupDataset,
    TrainGroupDataset,
    collate_eval_groups,
    collate_train_groups,
    stack_indicators,
)
from utils.user_utils import TrainUserDataset


def test_stack_indicators_matches_a_hand_built_matrix():
    rows = [torch.tensor([0, 2]), torch.tensor([], dtype=torch.long), torch.tensor([1])]
    cols = [torch.tensor([1, 4]), torch.tensor([], dtype=torch.long), torch.tensor([0])]
    dense = stack_indicators(rows, cols, (3, 3, 5), torch.device("cpu"))
    expected = torch.zeros(3, 3, 5)
    expected[0, 0, 1] = 1
    expected[0, 2, 4] = 1
    expected[2, 1, 0] = 1
    assert torch.equal(dense, expected)

    items = stack_indicators(
        [torch.zeros(2, dtype=torch.long), torch.zeros(0, dtype=torch.long)],
        [torch.tensor([1, 3]), torch.tensor([], dtype=torch.long)],
        (2, 4),
        torch.device("cpu"),
    )
    assert torch.equal(items[0], torch.tensor([0.0, 1.0, 0.0, 1.0]))
    assert torch.equal(items[1], torch.zeros(4))


def test_weeplaces_batch_matches_the_sparse_rows():
    users = TrainUserDataset("weeplaces")
    dataset = TrainGroupDataset("weeplaces", users.n_items, 5)
    samples = [dataset[index] for index in (0, 1, 17)]
    batch = collate_train_groups(
        samples, dataset.n_items, dataset.max_group_size, dataset.negs_per_group, torch.device("cpu"),
    )
    _groups, _users, _masks, group_items, user_items, corrupt_items, corrupt_users = batch
    for row, index in enumerate((0, 1, 17)):
        group = dataset.groups_list[index]
        member_items = torch.from_numpy(dataset.group_inputs[group].toarray())
        assert torch.equal(user_items[row], member_items)
        assert torch.equal(group_items[row], torch.from_numpy(dataset.group_data[group].toarray().reshape(-1)))
        negatives = corrupt_users[row].numpy()
        assert torch.equal(corrupt_items[row], torch.from_numpy(dataset.user_data[negatives].toarray()))

    if torch.cuda.is_available():
        device = torch.device("cuda")
        gpu_batch = collate_train_groups(
            samples, dataset.n_items, dataset.max_group_size, dataset.negs_per_group, device,
        )
        assert gpu_batch[4].is_cuda
        assert torch.equal(gpu_batch[4].cpu(), user_items)


def test_weeplaces_eval_batch_matches_the_sparse_rows():
    users = TrainUserDataset("weeplaces")
    train = TrainGroupDataset("weeplaces", users.n_items, 5)
    dataset = EvalGroupDataset("weeplaces", users.n_items, train.padding_idx, datatype="val")
    samples = [dataset[index] for index in (0, 3)]
    groups, _users, _masks, group_items, user_items = collate_eval_groups(
        samples, dataset.n_items, dataset.max_gsize, torch.device("cpu"),
    )
    assert groups.shape[0] == 2
    for row, index in enumerate((0, 3)):
        group = dataset.eval_groups_list[index]
        member_ids = dataset.eval_group_users[group]
        assert torch.equal(user_items[row], torch.from_numpy(dataset.user_data[member_ids].toarray()))
        assert torch.equal(
            group_items[row], torch.from_numpy(dataset.eval_group_data[group].toarray().reshape(-1)),
        )

"""User-loss pretraining shared by the MLP encoder and the LightGCN encoder.

The author loop trains on user-item pairs with L_U, reports user NDCG and recall,
then assigns the pretrained item embedding to the group predictor. LightGCN uses
that same loss and that same assignment. Its step updates the encoder ID
embeddings only.
"""

from __future__ import annotations


def user_pretrain_parameters(model):
    """Parameters the user-pretrain optimizer is allowed to step."""
    if model.encoder_type == "lightgcn":
        return model.user_preference_encoder.parameters()
    return model.parameters()


def user_pretrain_loss(model, user_ids, user_items):
    """L_U for one user batch. No group loss and no MI loss."""
    user_logits, _user_embeds = model.user_preference_encoder.pre_train_forward(user_items, user_ids)
    return model.user_loss(user_logits, user_items)


def seed_group_predictor(model):
    """Author assignment: the group predictor and the pretrained item embedding share storage."""
    model.group_predictor.weight.data = model.user_preference_encoder.user_predictor.weight.data

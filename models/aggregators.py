import torch
import torch.nn as nn


class MaxPoolAggregator(nn.Module):
    """ Group Preference Aggregator implemented as max pooling over group member embeddings """

    def __init__(self, input_dim, output_dim, drop_ratio=0):
        super(MaxPoolAggregator, self).__init__()

        self.mlp = nn.Sequential(
            nn.Linear(input_dim, output_dim, bias=True),
            nn.ReLU(),
            nn.Dropout(drop_ratio)
        )
        nn.init.xavier_uniform_(self.mlp[0].weight)
        if self.mlp[0].bias is not None:
            self.mlp[0].bias.data.fill_(0.0)

    def forward(self, x, mask, mlp=False):
        """ max pooling aggregator:
            :param x: [B, G, D]  group member embeddings
            :param mask: [B, G]  -inf/0 for absent/present
            :param mlp: flag to add a linear layer before max pooling
        """
        if mlp:
            h = torch.tanh(self.mlp(x))
        else:
            h = x

        if mask is None:
            return torch.max(h, dim=1)
        else:
            res = torch.max(h + mask.unsqueeze(2), dim=1)
            return res.values


# mask:  -inf/0 for absent/present.
class MeanPoolAggregator(nn.Module):
    """ Group Preference Aggregator implemented as mean pooling over group member embeddings """

    def __init__(self, input_dim, output_dim, drop_ratio=0):
        super(MeanPoolAggregator, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, output_dim, bias=True),
            nn.ReLU(),
            nn.Dropout(drop_ratio)
        )
        nn.init.xavier_uniform_(self.mlp[0].weight)
        if self.mlp[0].bias is not None:
            self.mlp[0].bias.data.fill_(0.0)

    def forward(self, x, mask, mlp=False):
        """ mean pooling aggregator:
            :param x: [B, G, D]  group member embeddings
            :param mask: [B, G]  -inf/0 for absent/present
            :param mlp: flag to add a linear layer before mean pooling
        """
        if mlp:
            h = torch.tanh(self.mlp(x))
        else:
            h = x
        if mask is None:
            return torch.mean(h, dim=1)
        else:
            mask = torch.exp(mask)
            res = torch.sum(h * mask.unsqueeze(2), dim=1) / mask.sum(1).unsqueeze(1)
            return res


class AttentionAggregator(nn.Module):
    """ Group Preference Aggregator implemented as attention over group member embeddings """

    def __init__(self, input_dim, output_dim, drop_ratio=0):
        super(AttentionAggregator, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, output_dim, bias=True),
            nn.ReLU(),
            nn.Dropout(drop_ratio)
        )

        self.attention = nn.Linear(output_dim, 1)
        self.drop = nn.Dropout(drop_ratio)
        nn.init.xavier_uniform_(self.mlp[0].weight)
        if self.mlp[0].bias is not None:
            self.mlp[0].bias.data.fill_(0.0)

    def forward(self, x, mask, mlp=False):
        """ attentive aggregator:
            :param x: [B, G, D]  group member embeddings
            :param mask: [B, G]  -inf/0 for absent/present
            :param mlp: flag to add a linear layer before attention
        """
        if mlp:
            h = torch.tanh(self.mlp(x))
        else:
            h = x

        attention_out = torch.tanh(self.attention(h))
        if mask is None:
            weight = torch.softmax(attention_out, dim=1)
        else:
            weight = torch.softmax(attention_out + mask.unsqueeze(2), dim=1)
        ret = torch.matmul(h.transpose(2, 1), weight).squeeze(2)
        return ret


class UniformAggregator(nn.Module):
    """Equal weight on every present member. No learned parameters."""

    def __init__(self, input_dim, output_dim, drop_ratio=0):
        super().__init__()

    def forward(self, x, mask, mlp=False):
        """ :param x: [B, G, D] member embeddings. :param mask: [B, G] -inf/0 for absent/present. """
        if mask is None:
            return torch.mean(x, dim=1)
        present = torch.exp(mask)
        weights = present / present.sum(1, keepdim=True).clamp(min=1)
        return torch.sum(x * weights.unsqueeze(2), dim=1)


class CosineAggregator(nn.Module):
    """Weight members by cosine similarity to the masked mean of the member embeddings."""

    def __init__(self, input_dim, output_dim, drop_ratio=0):
        super().__init__()

    def forward(self, x, mask, mlp=False):
        """ :param x: [B, G, D] member embeddings. :param mask: [B, G] -inf/0 for absent/present. """
        if mask is None:
            present = torch.ones(x.shape[:2], device=x.device, dtype=x.dtype)
            score_mask = 0
        else:
            present = torch.exp(mask)
            score_mask = mask.unsqueeze(2)
        denom = present.sum(1, keepdim=True).clamp(min=1)
        prototype = torch.sum(x * present.unsqueeze(2), dim=1) / denom
        member = torch.nn.functional.normalize(x, dim=-1)
        center = torch.nn.functional.normalize(prototype, dim=-1).unsqueeze(1)
        scores = (member * center).sum(-1, keepdim=True)
        weight = torch.softmax(scores + score_mask, dim=1)
        return torch.matmul(x.transpose(2, 1), weight).squeeze(2)

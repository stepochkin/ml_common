import torch
from torch.nn import Embedding, Module
import torch.nn.functional as F
from stepin.ml.torch_att import PooledAttention, AverageAttention, MmAverageAttention

from stepin.ml.torch_utils import cosine_sim_matrix, topk_scores


class SX(Module):
    def __init__(
        self, user_num, item_num, embed_dim, g, m, w, averaging='mean',
        lambda_u=None, lambda_i=None, att_dim=None, dropout=None
    ):
        super().__init__()
        self.user_num = user_num
        self.item_num = item_num
        self.embedding_dim = embed_dim
        self.g = g
        self.m = m
        self.w = w
        self.user_embeds = Embedding(self.user_num, self.embedding_dim)
        self.item_embeds = Embedding(self.item_num, self.embedding_dim)

        self.lambda_u = lambda_u
        self.lambda_i = lambda_i
        if averaging == 'attention':
            self.user_emb = PooledAttention(
                self.item_embeds, embed_dim if att_dim is None else att_dim
            )
        elif averaging == 'mean':
            self.user_emb = AverageAttention(self.item_embeds)
        elif averaging == 'mean_mm':
            self.user_emb = MmAverageAttention(self.item_embeds)
        else:
            raise Exception('Invalid averaging method')
        if (dropout is not None) and (dropout > 0.0):
            self.dropout = torch.nn.Dropout(dropout)
        else:
            self.dropout = None
        self.initial_weights()

    def initial_weights(self):
        torch.nn.init.xavier_uniform_(self.user_embeds.weight)
        # torch.nn.init.normal_(self.user_embeds.weight, std=1e-4)
        torch.nn.init.xavier_uniform_(self.item_embeds.weight)
        # torch.nn.init.normal_(self.item_embeds.weight, std=1e-4)

    def _build_user_emb(self, users, pos_items, pos_item_offsets):
        user_emb = self.user_embeds(users)
        user_items_emb = self.user_emb(pos_items, pos_item_offsets)
        user_emb = self.g * user_emb + (1 - self.g) * user_items_emb
        if self.dropout is not None:
            user_emb = self.dropout(user_emb)
        return user_emb

    def custom_score(self, items, scores):
        return scores

    def _score(self, user_emb, items, dim=1):
        return self.custom_score(
            items,
            F.cosine_similarity(user_emb, self.item_embeds(items), dim=dim)
        )

    def get_user_embeds(self, ids=None):
        emb = self.user_embeds.weight if ids is None else self.user_embeds.weight[ids]
        return emb.cpu().numpy()

    def get_item_embeds(self, ids=None):
        emb = self.item_embeds.weight if ids is None else self.item_embeds.weight[ids]
        return emb.cpu().numpy()

    def forward(self, users, pos_items, pos_item_offsets, items):
        return self._score(self._build_user_emb(users, pos_items, pos_item_offsets), items)

    def calc_loss(self, users, pos_items, pos_item_offsets, neg_items):
        if isinstance(neg_items, int):
            neg_items = torch.randint(
                0, self.item_num, [pos_items.size(0), neg_items],
                device=pos_items.device
            )
        user_emb = self._build_user_emb(users, pos_items, pos_item_offsets)
        offsets = torch.cat([
            pos_item_offsets,
            torch.unsqueeze(torch.tensor(pos_items.size(0), device=pos_items.device), dim=0)
        ])
        user_emb = torch.repeat_interleave(user_emb, torch.diff(offsets), dim=0)
        pos_scores = self._score(user_emb, pos_items)
        neg_scores = self._score(torch.unsqueeze(user_emb, dim=1), neg_items, dim=2)
        loss = (
            (1.0 - pos_scores) +
            self.w * torch.mean(torch.relu(neg_scores - self.m), dim=1)
        )
        loss = torch.mean(loss)
        # print('loss1', loss)
        if self.lambda_u is not None:
            # print('lambda_u', self.lambda_u)
            loss += self.lambda_u * torch.linalg.norm(self.user_embeds.weight)
        if self.lambda_i is not None:
            # print('lambda_i', self.lambda_i)
            loss += self.lambda_i * torch.linalg.norm(self.item_embeds.weight)
        # print('loss2', loss)
        return loss

    def predict_items(self, users, pos_items, pos_item_offsets, items=None):
        user_emb = self._build_user_emb(users, pos_items, pos_item_offsets)
        if items is None:
            item_emb = self.item_embeds.weight.data
        else:
            item_emb = self.item_embeds(items)
        return cosine_sim_matrix(user_emb, item_emb)

    def predict_topk(
        self, users, pos_items, pos_item_offsets, top_k, items=None, excludes=None
    ):
        scores = self.predict_items(users, pos_items, pos_item_offsets, items=items)
        if excludes is not None:
            scores = scores + excludes
        top_items = torch.topk(scores, min(top_k, scores.size(1))).indices
        return top_items

    def predict_topk_scores(
        self, users, pos_items, pos_item_offsets, top_k, items=None, excludes=None
    ):
        scores = self.predict_items(users, pos_items, pos_item_offsets, items=items)
        if excludes is not None:
            scores = scores + excludes
        return topk_scores(scores, top_k)

    def get_device(self):
        return self.user_embeds.weight.device

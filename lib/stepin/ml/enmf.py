import torch
from torch.nn import Parameter


class ENMF(torch.nn.Module):
    """
    Pytorch implementation of Efficient Neural Matrix Factorization
    without Sampling for Recommendation.
    For details look through (enjoy beautiful and elegant math as i did):
    https://dl.acm.org/doi/pdf/10.1145/3373807
    Reference implementation in tensorflow 1.x:
    https://github.com/chenchongthu/ENMF
    The model beats our favourite ALS and brings a non sampling approach to our help :)
    """
    def __init__(
        self, user_count, item_count, embed_dim,
        neg_weight, lambda_u, lambda_i, dropout_prob=None
    ):
        """
        :param int user_count: user count
        :param int item_count: item count
        :param int embed_dim: embedding dimension
        :param float neg_weight: weight of negative items
        :type neg_weight: float or None
        :param lambda_u: user embedding L2 regularization coefficient
        :type lambda_u: float or None
        :param lambda_i: item embedding L2 regularization coefficient
        :type lambda_i: float or None
        :param dropout_prob: dropout probability
        :type dropout_prob: float or None
        """
        super().__init__()
        self.neg_weight = neg_weight
        self.lambda_u = lambda_u
        self.lambda_i = lambda_i
        self.user_emb = torch.nn.Embedding(user_count, embed_dim)
        torch.nn.init.xavier_uniform_(self.user_emb.weight.data)
        self.item_emb = torch.nn.Embedding(item_count, embed_dim)
        torch.nn.init.xavier_uniform_(self.item_emb.weight.data)
        self.h = Parameter(torch.empty(embed_dim).fill_(0.01))
        if dropout_prob is None:
            self.dropout = None
        else:
            self.dropout = torch.nn.Dropout(dropout_prob)

    def _get_user_emb(self, input_u):
        uid = self.user_emb(input_u)
        if self.dropout is not None:
            uid = self.dropout(uid)
        return uid

    def forward(self, users, items):
        """
        Calculates scoring.
        :param users: users
        :type: torch.tensor[N] of any torch integer type
        :param items: items
        :type: torch.tensor[N] of any torch integer type
        :return: scores for user/item pairs
        :rtype: torch.tensor[N] of torch.float32
        """
        u_emb = self._get_user_emb(users)
        i_emb = self.item_emb(items)
        pos_r = torch.einsum('ac,ac,c->a', u_emb, i_emb, self.h)
        return pos_r

    def calc_loss(self, users, items):
        return self.calc_loss_pos(users, self.forward(users, items))

    def calc_loss_pos(self, users, pos_r):
        """
        Calculates square error loss function for all items.
        :param users: user batch (can be non unique)
        :type: torch.tensor[N] of any torch integer type
        :param pos_r: positive scores for user batch
        :type: torch.tensor[N] of torch.float32
        :return: loss
        :rtype: torch.float32

        Usually training looks like this::
            loss = model.calc_loss(users, model(users, items))
            model.zero_grad()
            loss.backward()
            optimizer.step()
        """
        users = torch.unique(users)
        neg_weight = self.neg_weight
        u_emb = self._get_user_emb(users)
        i_emb = self.item_emb.weight.data
        loss = (
            torch.einsum('ab,ac->bc', i_emb, i_emb) *
            torch.einsum('ab,ac->bc', u_emb, u_emb)
        )
        loss = neg_weight * torch.einsum('a,b,ab->', self.h, self.h, loss)
        loss += torch.sum(
            (1.0 - neg_weight) * torch.square(pos_r) - 2.0 * pos_r
        )
        if self.lambda_u is not None:
            loss += self.lambda_u * torch.linalg.norm(u_emb)
        if self.lambda_i is not None:
            loss += self.lambda_i * torch.linalg.norm(i_emb)
        return loss

    def calc_item_loss(self, items, users):
        return self.calc_item_loss_pos(items, self.forward(users, items))

    def calc_item_loss_pos(self, items, pos_r):
        neg_weight = self.neg_weight
        u_emb = self.user_emb.weight.data
        if self.dropout is not None:
            u_emb = self.dropout(u_emb)
        items = torch.unique(items)
        i_emb = self.item_emb(items)
        loss = (
            torch.einsum('ab,ac->bc', i_emb, i_emb) *
            torch.einsum('ab,ac->bc', u_emb, u_emb)
        )
        loss = neg_weight * torch.einsum('a,b,ab->', self.h, self.h, loss)
        loss += torch.sum(
            (1.0 - neg_weight) * torch.square(pos_r) - 2.0 * pos_r
        )
        if self.lambda_u is not None:
            loss += self.lambda_u * torch.linalg.norm(u_emb)
        if self.lambda_i is not None:
            loss += self.lambda_i * torch.linalg.norm(i_emb)
        return loss

    def predict_items(self, users, items=None):
        """
        Calculates scores for all items for given users.
        Be carefull, can use a lot of memory!
        :param users: user batch
        :type: torch.tensor[N] of any torch integer type
        :param items: items to predict for
        :type: torch.tensor[N] of any torch integer type
        :return: scores
        :rtype: torch.tensor[N, K] of torch.float32
        """
        if items is None:
            i_emb = self.item_emb.weight.data
        else:
            i_emb = self.item_emb(items)
        u_emb = self.user_emb(users)
        return torch.matmul(u_emb, torch.multiply(i_emb, self.h).T)

    def predict_topk(self, users, top_k, items=None, excludes=None):
        """
        Calculates scores for all items for given users.
        Be carefull, can use a lot of memory!
        :param users: user batch
        :type: torch.tensor[N] of any torch integer type
        :param top_k: select top_k maximum scores
        :type: int
        :param items: items to predict for
        :type: torch.tensor[N] of any torch integer type
        :param excludes: items to exclude
        :type: torch.coo_tensor[N, M] of any torch integer type
        :return: scores
        :rtype: torch.tensor[N, K] of torch.int64
        """
        # dot = torch.einsum(
        #     'ac,bc->abc', self._get_user_emb(users), self.item_emb.weight.data
        # )
        # pred = torch.einsum('ajk,k->aj', dot, self.h)
        pred = self.predict_items(users, items=items)
        if excludes is not None:
            pred = pred + excludes
        top_items = torch.topk(pred, min(top_k, pred.size(1))).indices
        return top_items

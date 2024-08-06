import torch
from torch.nn import EmbeddingBag, functional as F

from stepin.ml.torch_utils import build_seq_emb, sum_var_parts_dim2, build_lens, sum_var_parts


def indices2offsets(indices, count):
    if indices.size(0) == 0:
        return torch.zeros(count, dtype=indices.dtype, device=indices.device)
    unique_offsets = torch.cat([
        torch.zeros(1, dtype=indices.dtype, device=indices.device),
        torch.nonzero(indices.diff()).view(-1) + torch.tensor(1, device=indices.device)
    ])
    repeat_cnt = torch.cat([
        torch.tensor([-1], dtype=indices.dtype, device=indices.device),
        indices[unique_offsets]
    ]).diff()
    offsets = torch.repeat_interleave(unique_offsets, repeat_cnt)
    # noinspection PyArgumentList
    if offsets.size(0) < count:
        # noinspection PyArgumentList
        offsets = torch.cat([
            offsets,
            torch.full([count - offsets.size(0)], indices.size(0), device=indices.device)
        ])
    return offsets


def csr_attr_to_coo_indices(indices, indptr):
    offsets_2 = torch.cat(
        [
            indptr,
            torch.tensor(
                [indices.size(0)], dtype=indptr.dtype, device=indptr.device
            )
        ],
        dim=0
    )
    offsets_2 = torch.repeat_interleave(
        torch.arange(indptr.size(0), device=indptr.device), offsets_2.diff()
    )
    indices = torch.cat([
        torch.unsqueeze(offsets_2, dim=0),
        torch.unsqueeze(indices, dim=0),
    ], dim=0)
    return indices


def csr_attr_to_coo(data, indices, indptr, width=None):
    indices = csr_attr_to_coo_indices(indices, indptr)
    return torch.sparse_coo_tensor(
        indices, data,
        size=[
            indptr.size(0),
            indices.max() + 1 if width is None else width
        ]
    )


def csr_attr_softmax(data, indices, indptr, width=None):
    data = csr_attr_to_coo(data, indices, indptr, width=width)
    if not data.is_coalesced():
        data = data.coalesce()
        ind = data.indices()
        indptr = indices2offsets(ind[0], data.size(0))
        indices = ind[1]
    data = torch.sparse.softmax(data, dim=1)
    data = data._values()
    return data, indices, indptr


def csr_attr_softmax_k(data, indices, indptr, width=None):
    sp_indices = csr_attr_to_coo_indices(indices, indptr)
    sp_data = torch.sparse_coo_tensor(
        sp_indices, torch.arange(data.size(0), dtype=torch.long, device=indices.device),
        size=[
            indptr.size(0),
            indices.max() + 1 if width is None else width
        ]
    )
    sp_data = sp_data.coalesce()

    sp_indices = sp_data._indices()
    indptr = indices2offsets(sp_indices[0], sp_data.size(0))
    indices = sp_indices[1]

    data = data[sp_data._values()]
    smax_list = []
    for i in torch.arange(data.size(1), dtype=torch.long, device=data.device):
        sp_data = torch.sparse_coo_tensor(sp_indices, data[:, i])
        smax = torch.sparse.softmax(sp_data, dim=1)
        smax_list.append(smax._values().unsqueeze(-1))
    return torch.cat(smax_list, dim=1), indices, indptr


class L2EmbeddingBag(EmbeddingBag):
    def get_weight(self):
        return self.weight

    def get_l2_reg(self):
        return torch.linalg.norm(self.weight)


class CrossEmbeddingBag(torch.nn.Module):
    def __init__(self, weight):
        super().__init__()
        self.weight = weight

    def forward(self, input_, offsets, per_sample_weights=None):
        return F.embedding_bag(
            input_, self.weight, offsets,
            mode='mean' if per_sample_weights is None else 'sum',
            per_sample_weights=per_sample_weights
        )

    @staticmethod
    def get_l2_reg():
        return 0.0


class AverageAttention(torch.nn.Module):
    def __init__(self, emb):
        super().__init__()
        self.emb = torch.nn.Embedding(*emb) if isinstance(emb, (tuple, list)) else emb

    def set_emb(self, emb):
        self.emb = emb

    def forward(self, input_, offsets, emb_weights=None):
        return F.embedding_bag(
            input_, 
            emb_weights if self.emb is None else self.emb.weight,
            offsets=offsets,
            mode='mean'
        )


class MmAverageAttention(torch.nn.Module):
    def __init__(self, emb):
        super().__init__()
        self.emb = torch.nn.Embedding(*emb) if isinstance(emb, (tuple, list)) else emb

    def set_emb(self, emb):
        self.emb = emb

    def forward(self, input_, offsets, emb_weights=None):
        return sum_var_parts(self.emb(input_), build_lens(input_, offsets), emb_weights)


class PooledAttention(torch.nn.Module):
    def __init__(
        self, emb, att_dim,
        embedding_dim=None,
        device=None, dtype=torch.float32
    ):
        super().__init__()
        # weight = torch.empty((num_embeddings, embedding_dim), device=device, dtype=dtype)
        # torch.nn.init.trunc_normal_(weight, mean=0.0, std=0.01)
        # self.emb = torch.nn.Parameter(weight)
        self.emb = torch.nn.Embedding(*emb) if isinstance(emb, (tuple, list)) else emb
        self.proj = torch.nn.Linear(
            embedding_dim if self.emb is None else self.emb.embedding_dim,
            att_dim, bias=True
        )
        weight = torch.empty(att_dim, device=device, dtype=dtype)
        torch.nn.init.trunc_normal_(weight, mean=0.0, std=0.01)
        self.att_h = torch.nn.Parameter(weight)

    def set_emb(self, emb):
        self.emb = emb

    def forward(self, input_, offsets, emb_weights=None):
        if emb_weights is None:
            att_emb = self.emb(input_)
            num_embeddings = self.emb.num_embeddings
        else:
            att_emb = F.embedding(input_, emb_weights)
            num_embeddings = emb_weights.size(0)
        att = self.proj(att_emb).tanh()
        att = torch.einsum('ab,b->a', att, self.att_h)
        att, input_, offsets = csr_attr_softmax(
            att, input_, offsets, num_embeddings
        )
        return F.embedding_bag(
            input_,
            emb_weights if self.emb is None else self.emb.weight,
            offsets,
            per_sample_weights=att, mode='sum'
        )

    def get_weight(self):
        return self.emb.weight

    def get_l2_reg(self):
        reg = (
            0.0 if self.emb is None else torch.linalg.norm(self.emb.weight) +
            torch.linalg.norm(self.proj.weight) +
            torch.linalg.norm(self.proj.bias) +
            torch.linalg.norm(self.att_h)
        )
        return reg

    @staticmethod
    def get_field_dim():
        return 1


class PooledAttentionDim2(torch.nn.Module):
    def __init__(
        self, emb, att_dim, acc_user_num,
        embedding_dim=None,
        device=None, dtype=torch.float32
    ):
        super().__init__()
        self.emb = torch.nn.Embedding(*emb) if isinstance(emb, (tuple, list)) else emb
        self.proj = torch.nn.Linear(
            embedding_dim if self.emb is None else self.emb.embedding_dim,
            att_dim, bias=True
        )
        weight = torch.empty([att_dim, acc_user_num], device=device, dtype=dtype)
        torch.nn.init.trunc_normal_(weight, mean=0.0, std=0.01)
        self.att_h = torch.nn.Parameter(weight)

    def forward(self, input_, offsets, emb_weights=None):
        if emb_weights is None:
            att_emb = self.emb(input_)
            num_embeddings = self.emb.num_embeddings
        else:
            att_emb = F.embedding(input_, emb_weights)
            num_embeddings = emb_weights.size(0)
        att = self.proj(att_emb).tanh()
        att = att @ self.att_h
        att, input_, offsets = csr_attr_softmax_k(
            att, input_, offsets, width=num_embeddings
        )
        result = sum_var_parts_dim2(att_emb, build_lens(input_, offsets), weights=att)
        return result

    def get_weight(self):
        return self.emb.weight

    def get_l2_reg(self):
        reg = (
            0.0 if self.emb is None else torch.linalg.norm(self.emb.weight) +
            torch.linalg.norm(self.proj.weight) +
            torch.linalg.norm(self.proj.bias) +
            torch.linalg.norm(self.att_h)
        )
        return reg

    @staticmethod
    def get_field_dim():
        return 1


class FullMultiEmbedding(torch.nn.Module):
    def __init__(self, num_emb, embed_dim):
        super().__init__()
        self.embedding = torch.nn.Embedding(num_emb, embed_dim)
        torch.nn.init.xavier_uniform_(self.embedding.weight.data)

    def forward(self, input_, offsets, per_sample_weights=None):
        emb_size = self.embedding.num_embeddings
        emb_dim = self.embedding.embedding_dim
        offsets = offsets.to(torch.long)
        offsets2 = torch.cat([
            offsets,
            torch.tensor([input_.size(0)], dtype=offsets.dtype, device=offsets.device)
        ])
        emb = torch.zeros(
            [offsets.size(0), emb_size, emb_dim],
            dtype=self.embedding.weight.dtype,
            device=input_.device
        )
        iemb = self.embedding(input_)
        if per_sample_weights is not None:
            iemb *= torch.unsqueeze(per_sample_weights, 1)
        # print(offsets.dtype, offsets2.dtype)
        # print(torch.arange(offsets.size(0), dtype=offsets.dtype, device=offsets.device).dtype)
        # print(offsets2.diff().dtype)
        # print(
        #     torch.repeat_interleave(
        #         torch.arange(offsets.size(0), dtype=offsets.dtype, device=offsets.device),
        #         offsets2.diff()
        #     ).dtype
        # )
        # print(input_.dtype)
        emb[
            torch.repeat_interleave(
                torch.arange(offsets.size(0), dtype=offsets.dtype, device=offsets.device),
                offsets2.diff()
            ),
            input_.to(offsets.dtype)
        ] = iemb
        return emb

    @staticmethod
    def get_l2_reg():
        return 0.0

    def get_field_dim(self):
        return self.embedding.num_embeddings


class SeqMultiEmbedding(torch.nn.Module):
    def __init__(self, num_emb, embed_dim):
        super().__init__()
        self.embedding = torch.nn.Embedding(num_emb, embed_dim, padding_idx=0)

    def forward(self, input_, offsets, per_sample_weights=None):
        return build_seq_emb(input_, offsets, self.embedding, weights=per_sample_weights)

    def get_l2_reg(self):
        return torch.linalg.norm(self.embedding.weight)

    def get_field_dim(self):
        return self.embedding.num_embeddings


def std_eb_desc(nm):
    if isinstance(nm, str):
        pp = nm.split(':', 1)
        if len(pp) == 2:
            return dict(name=pp[0], type=pp[1])
        return dict(name=nm, type='ebag')
    if 'type' not in nm:
        nm['type'] = 'ebag'
    return nm


def build_eb_list(eb_dims, embed_dim, att_dim=None, kwargs=None):
    if kwargs is None:
        kwargs = dict()
    item_bemb = []
    for eb_dim in eb_dims:
        if isinstance(eb_dim, dict):
            dim = eb_dim.get('dim')
            type_ = eb_dim.get('type')
            if type_ == 'full':
                item_bemb.append(FullMultiEmbedding(dim, embed_dim))
            elif type_ == 'ebag':
                item_bemb.append(L2EmbeddingBag(dim, embed_dim, **kwargs))
            elif type_ == 'pool_att':
                item_bemb.append(PooledAttention((dim, embed_dim), att_dim, **kwargs))
            else:
                raise Exception('Unknown multivalues type ' + type_)
        else:
            if att_dim is None:
                item_bemb.append(L2EmbeddingBag(eb_dim, embed_dim, **kwargs))
            else:
                item_bemb.append(PooledAttention((eb_dim, embed_dim), att_dim, **kwargs))
    item_bemb = torch.nn.ModuleList(item_bemb)
    return item_bemb


class L2EmbeddingBagAdapter(torch.nn.Module):
    def __init__(self, model: L2EmbeddingBag):
        super().__init__()
        self.model = model

    def forward(self, indices, offsets, weights=None):
        return self.model.forward(indices, offsets, per_sample_weights=weights)

    def get_l2_reg(self):
        return self.model.get_l2_reg()


class PooledAttentionAdapter(torch.nn.Module):
    def __init__(self, model: PooledAttention):
        super().__init__()
        self.model = model

    def forward(self, indices, offsets, _weights=None):
        return self.model.forward(indices, offsets)

    def get_l2_reg(self):
        return self.model.get_l2_reg()


class CrossEmbeddingBagAdapter(torch.nn.Module):
    def __init__(self, model: CrossEmbeddingBag):
        super().__init__()
        self.model = model

    def forward(self, indices, offsets, weights=None):
        return self.model.forward(indices, offsets, per_sample_weights=weights)

    def get_l2_reg(self):
        return self.model.get_l2_reg()


def build_eb_list_ex(eb_dims, embed_dim, cross_emb, cross_eb_emb, att_dim=None, kwargs=None):
    if kwargs is None:
        kwargs = dict()
    item_bemb = []
    for eb_dim in eb_dims:
        if att_dim is None:
            if isinstance(eb_dim, tuple):
                dim, is_cross, is_eb = eb_dim
            else:
                is_cross = False
                is_eb = False
                dim = eb_dim
            if is_cross:
                if is_eb:
                    item_bemb.append(CrossEmbeddingBagAdapter(
                        CrossEmbeddingBag(cross_eb_emb[dim].get_weight())
                    ))
                else:
                    item_bemb.append(CrossEmbeddingBagAdapter(
                        CrossEmbeddingBag(cross_emb.get_weights(dim))
                    ))
            else:
                item_bemb.append(L2EmbeddingBagAdapter(
                    L2EmbeddingBag(dim, embed_dim, mode='sum', **kwargs)
                ))
        else:
            item_bemb.append(PooledAttentionAdapter(
                PooledAttention((eb_dim, embed_dim), att_dim, **kwargs)
            ))
    item_bemb = torch.nn.ModuleList(item_bemb)
    return item_bemb

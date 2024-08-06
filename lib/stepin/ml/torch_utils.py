from collections import OrderedDict

import numpy as np
import torch


def struct_to_device(struct, device):
    if device is None:
        return struct
    if torch.is_tensor(struct):
        return struct.to(device)
    if isinstance(struct, dict):
        return {n: struct_to_device(t, device) for n, t in struct.items()}
    if isinstance(struct, (tuple, list)):
        return tuple(struct_to_device(t, device) for t in struct)
    return struct


def save_model_as_numpy(model, stream, compressed=False):
    save_dict = {}
    names = []
    for n, t in model.state_dict().items():
        save_dict[n] = t.cpu().numpy()
        names.append(n)
    save_dict['__saved_names__'] = names
    if compressed:
        np.savez_compressed(stream, **save_dict)
    else:
        np.savez(stream, **save_dict)


def load_model_as_numpy(model, path):
    with np.load(path) as f:
        state_dict = OrderedDict(
            (fname, torch.from_numpy(f[fname])) for fname in f['__saved_names__']
        )
    model.load_state_dict(state_dict)


# pylint: disable=E1101
def np2csr(csr, device=None):
    indptr = torch.from_numpy(csr.indptr.astype(np.int64, copy=False))
    indices = torch.from_numpy(csr.indices.astype(np.int64, copy=False))
    data = torch.from_numpy(csr.data)
    if device is not None:
        if isinstance(device, str):
            device = torch.device(device)
        indptr = indptr.to(device)
        indices = indices.to(device)
        data = data.to(device)
    # pylint: disable=W0212
    return torch.sparse_csr_tensor(indptr, indices, data)


# pylint: disable=E1101
def np2coo(coo, dtype=None, device=None):
    indices = np.vstack([coo.row, coo.col])
    indices = torch.from_numpy(indices.astype(np.int64, copy=False))
    return torch.sparse_coo_tensor(
        indices,
        torch.from_numpy(coo.data) if dtype is None else torch.tensor(coo.data, dtype=dtype),
        size=coo.shape, device=device
    )


def csr2torch(d, return_data=False):
    if return_data:
        return (
            torch.from_numpy(d.indices),
            torch.from_numpy(d.indptr[:-1]),
            torch.from_numpy(d.data),
        )
    return (
        torch.from_numpy(d.indices),
        torch.from_numpy(d.indptr[:-1]),
    )


def sparse_lens_ind(lens, col):
    return torch.cat([
        torch.repeat_interleave(torch.arange(lens.size(0)), lens).view(1, -1),
        col.view(1, -1)
    ], dim=0)


def csr2coo(csr, device=None):
    lens = torch.tensor(csr.getnnz(axis=1), device=device)
    # noinspection PyTypeChecker
    ind = sparse_lens_ind(lens, torch.tensor(csr.indices, device=device))
    values = torch.tensor(csr.data, device=device)
    return torch.sparse_coo_tensor(ind, values, size=csr.shape)


def sum_var_parts(t, lens, weights=None):
    t_size_0 = t.size(0)
    ind_x = torch.repeat_interleave(torch.arange(lens.size(0), device=t.device), lens)
    indices = torch.cat(
        [
            torch.unsqueeze(ind_x, dim=0),
            torch.unsqueeze(torch.arange(t_size_0, device=t.device), dim=0)
        ],
        dim=0
    )
    if weights is None:
        weights = torch.ones(t_size_0, dtype=t.dtype, device=t.device)
    m = torch.sparse_coo_tensor(
        indices, weights, size=[lens.size(0), t_size_0]
    )
    return m @ t


def coo_indices_dim3(indices, repeat):
    return torch.cat([
        torch.repeat_interleave(
            torch.arange(repeat, dtype=indices.dtype, device=indices.device), indices.size(1)
        ).unsqueeze(0),
        indices.repeat(1, repeat)
    ], 0)


def sum_var_parts_dim2(t, lens, weights):
    t_size_0 = t.size(0)
    ind_x = torch.repeat_interleave(torch.arange(lens.size(0), device=t.device), lens)
    indices = torch.cat(
        [
            torch.unsqueeze(ind_x, dim=0),
            torch.unsqueeze(torch.arange(t_size_0, device=t.device), dim=0)
        ],
        dim=0
    )
    indices3 = coo_indices_dim3(indices, weights.size(1))
    m = torch.sparse_coo_tensor(
        indices3, weights.T.reshape(-1),
        size=[weights.size(1), lens.size(0), t_size_0]
    )
    return torch.bmm(m, t.unsqueeze(0).expand(weights.size(1), -1, -1))


def sum_var_parts_dim2_loop(t, lens, weights):
    t_size_0 = t.size(0)
    ind_x = torch.repeat_interleave(torch.arange(lens.size(0), device=t.device), lens)
    indices = torch.cat(
        [
            torch.unsqueeze(ind_x, dim=0),
            torch.unsqueeze(torch.arange(t_size_0, device=t.device), dim=0)
        ],
        dim=0
    )
    slist = []
    for i in range(weights.size(1)):
        m = torch.sparse_coo_tensor(
            indices, weights[:, i], size=[lens.size(0), t_size_0]
        )
        # print('++++++++++++++++++++')
        # print(m.to_dense())
        slist.append((m @ t).unsqueeze(0))
    return torch.cat(slist, 0)


def build_mlp(
    name, in_dim, dims, activation=torch.nn.ReLU, dropout_prob=None,
    add_sigmoid=False
):
    prev_dim = in_dim
    mlp = torch.nn.Sequential()
    for i, dim in enumerate(dims):
        mlp.add_module(name + '_' + str(i), torch.nn.Linear(prev_dim, dim, bias=False))
        if i < (len(dims) - 1):
            mlp.add_module(name + '_' + str(i) + '_activation', activation())
            if (dropout_prob is not None) and (dropout_prob > 0.0):
                mlp.add_module(
                    name + '_' + str(i) + '_dropout',
                    torch.nn.Dropout(dropout_prob)
                )
        prev_dim = dim
    if add_sigmoid:
        mlp.add_module(name + '_sigmoid', torch.nn.Sigmoid())
    return mlp


def row_dots(a, b):
    a = a.unsqueeze(dim=1)
    b = b.unsqueeze(dim=1)
    return torch.bmm(a, b.permute(0, 2, 1))


def cosine_sim_matrix(a, b, eps=1e-8):
    a_n, b_n = a.norm(dim=1)[:, None], b.norm(dim=1)[:, None]
    a_norm = a / torch.clamp(a_n, min=eps)
    b_norm = b / torch.clamp(b_n, min=eps)
    sim_mt = torch.mm(a_norm, b_norm.transpose(0, 1))
    return sim_mt


def cosine_sim_tensor(a, b, eps=1e-8):
    a_n = a.norm(dim=-1).unsqueeze(-1)
    b_n = b.norm(dim=-1).unsqueeze(-1)
    a_norm = a / torch.clamp(a_n, min=eps)
    b_norm = b / torch.clamp(b_n, min=eps)
    b_norm = b_norm.transpose(0, 1)
    if a.dim() > 2:
        b_norm = b_norm.unsqueeze(0).expand(a_norm.size(0), -1, -1)
        sim_mt = torch.bmm(a_norm, b_norm)
    else:
        sim_mt = torch.mm(a_norm, b_norm)
    return sim_mt


def multirange(lens):
    return (
        torch.arange(lens.sum(), dtype=lens.dtype, device=lens.device) -
        torch.repeat_interleave(torch.cat([
            torch.zeros(1, dtype=lens.dtype, device=lens.device),
            lens[:-1].cumsum(0)
        ]), lens)
    )


def offsets_with_end(input_, offsets):
    return torch.cat([
        offsets,
        torch.tensor([input_.size(0)], dtype=offsets.dtype, device=offsets.device)
    ])


def build_lens(indices, starts):
    return offsets_with_end(indices, starts).diff()


def build_seq_ind_len(lens, on_right=False, width=None):
    ind0 = torch.repeat_interleave(
        torch.arange(lens.size(0), dtype=torch.long, device=lens.device), lens
    )
    ind1 = multirange(lens)
    if width is None:
        width = lens.max()
    if on_right:
        ind1 += torch.repeat_interleave(width - lens, lens)
    return ind0, ind1, width


def _build_seq_ind(indices, starts, on_right=False, width=None):
    lens = build_lens(indices, starts)
    ind0, ind1, width = build_seq_ind_len(lens, on_right=on_right, width=width)
    return ind0, ind1, lens, width


def masked_array(values, lens, width=None, on_right=False, return_mask=False):
    if width is None:
        width = lens.max()
    rows = torch.arange(lens.sum(), dtype=lens.dtype, device=lens.device)
    rows[lens[0]:] -= torch.repeat_interleave(lens[:-1].cumsum(0), lens[1:])
    if on_right:
        rows += torch.repeat_interleave(width - lens, lens)
    cols = torch.repeat_interleave(
        torch.arange(lens.size(0), dtype=lens.dtype, device=lens.device), lens
    )
    t = torch.zeros([lens.size(0), width], dtype=values.dtype, device=values.device)
    t[cols, rows] = values
    if not return_mask:
        return t
    mask = torch.zeros([lens.size(0), width], dtype=torch.bool, device=lens.device)
    mask[cols, rows] = True
    return t, mask


def build_mask_len(lens, width=None, on_right=False):
    ind0, ind1, width = build_seq_ind_len(lens, width=width, on_right=on_right)
    mask = torch.zeros([lens.size(0), width], dtype=torch.bool, device=lens.device)
    mask[ind0, ind1] = True
    return mask


def build_mask(values, offsets, width=None, on_right=False):
    ind0, ind1, lens, width = _build_seq_ind(values, offsets, width=width, on_right=on_right)
    mask = torch.zeros([lens.size(0), width], dtype=torch.bool, device=lens.device)
    mask[ind0, ind1] = True
    return mask


def masked_array2(values, offsets, width=None, on_right=False, return_mask=False):
    ind0, ind1, lens, width = _build_seq_ind(values, offsets, on_right=on_right, width=width)
    t = torch.zeros([lens.size(0), width], dtype=values.dtype, device=values.device)
    t[ind0, ind1] = values
    if not return_mask:
        return t
    mask = torch.zeros([lens.size(0), width], dtype=torch.bool, device=lens.device)
    mask[ind0, ind1] = True
    return t, mask


def build_seq_indices(indices, starts, weights=None, width=None, on_right=False):
    ind0, ind1, lens, width = _build_seq_ind(indices, starts, on_right=on_right, width=width)
    res = torch.zeros([lens.size(0), width], dtype=indices.dtype, device=indices.device)
    res[ind0, ind1] = indices
    if weights is None:
        return res
    wres = torch.zeros([lens.size(0), width], dtype=weights.dtype, device=weights.device)
    wres[ind0, ind1] = weights
    return res, wres


def build_seq_emb(indices, starts, emb_model, weights=None, width=None, on_right=False):
    ind0, ind1, lens, width = _build_seq_ind(indices, starts, on_right=on_right, width=width)
    emb = emb_model(indices)
    if weights is not None:
        emb *= weights.unsqueeze(1)
    res = torch.zeros(
        [lens.size(0), width, emb_model.embedding_dim],
        dtype=emb.dtype, device=emb.device
    )
    res[ind0, ind1] = emb
    return res


def build_seq_embs(starts, *emb_models, width=None, on_right=False):
    ind0, ind1, lens, width = _build_seq_ind(
        emb_models[0]['indices'], starts, on_right=on_right, width=width
    )
    embs = []
    for em in emb_models:
        emb = em['emb'](em['indices'])
        if 'weights' in em:
            emb *= em['weights'].unsqueeze(1)
        res = torch.zeros(
            [lens.size(0), width, em['emb'].embedding_dim],
            dtype=emb.dtype, device=emb.device
        )
        res[ind0, ind1] = emb
        embs.append(res)
    return embs


def build_seq_pos_emb(
    indices, starts, emb_model, pos_emb_model,
    width=None, on_right=False, return_mask=False
):
    ind0, ind1, lens, width = _build_seq_ind(indices, starts, on_right=on_right, width=width)

    emb = emb_model(indices)
    pad_emb = torch.zeros(
        [lens.size(0), width, emb_model.embedding_dim],
        dtype=emb.dtype, device=emb.device
    )
    pad_emb[ind0, ind1] = emb

    pos_emb = pos_emb_model(ind1)
    pad_pos_emb = torch.zeros(
        [lens.size(0), width, pos_emb_model.embedding_dim],
        dtype=pos_emb.dtype, device=pos_emb.device
    )
    pad_pos_emb[ind0, ind1] = pos_emb
    if not return_mask:
        return pad_emb, pad_pos_emb
    mask = torch.zeros([lens.size(0), width], dtype=torch.bool, device=pad_pos_emb.device)
    mask[ind0, ind1] = True
    return pad_emb, pad_pos_emb, mask


def multi_tile(data, offsets, repeats):
    lens = build_lens(data, offsets)
    mr = multirange(torch.repeat_interleave(lens, repeats))
    ind = mr + torch.repeat_interleave(offsets, lens * repeats)
    offsets2 = torch.cat([
        torch.tensor([0], dtype=offsets.dtype, device=offsets.device),
        torch.repeat_interleave(lens, repeats).cumsum(0)[:-1]
    ])
    return data[ind], offsets2


def tile_offsets(data, offsets, repeat):
    lens = build_lens(data, offsets)
    lens = torch.tile(lens, (repeat,))
    return torch.cat([
        torch.tensor([0], dtype=offsets.dtype, device=offsets.device),
        lens[:-1].cumsum(dim=0)
    ])


def topk_scores(scores, top_k):
    top_items = torch.topk(scores, min(top_k, scores.size(1))).indices
    # si = torch.tile(
    #     torch.unsqueeze(torch.arange(top_items.size(0)), dim=1),
    #     [1, top_items.size(1)]
    # )
    si = torch.arange(top_items.size(0)).unsqueeze(-1).expand(top_items.size())
    scores = scores[si, top_items]
    return top_items, scores

from collections import OrderedDict

import numpy as np
import torch


def save_model_as_numpy(model, path):
    save_dict = {}
    names = []
    for n, t in model.state_dict().items():
        save_dict[n] = t.cpu().numpy()
        names.append(n)
    save_dict['__saved_names__'] = names
    np.savez_compressed(path, **save_dict)


def load_model_as_numpy(model, path):
    with np.load(path) as f:
        state_dict = OrderedDict(
            (fname, torch.from_numpy(f[fname])) for fname in f['__saved_names__']
        )
    model.load_state_dict(state_dict)


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
    return torch._sparse_csr_tensor(indptr, indices, data)


def np2coo(coo, device=None):
    indices = np.vstack([coo.row, coo.col])
    indices = torch.from_numpy(indices.astype(np.int64, copy=False))
    return torch.sparse_coo_tensor(
        indices,
        torch.from_numpy(coo.data),
        size=coo.shape, device=device
    )


def csr2torch(d):
    return (
        torch.from_numpy(d.indices),
        torch.from_numpy(d.indptr[:-1]),
    )


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
    M = torch.sparse_coo_tensor(
        indices, weights, size=[lens.size(0), t_size_0]
    )
    return M @ t


def first_cols(lengths):
    return torch.nonzero(
        lengths.unsqueeze(dim=1) > torch.arange(lengths.max()),
        as_tuple=True
    )[1]


def sizes2coo_indices(sizes):
    row = torch.repeat_interleave(torch.arange(sizes.size(0)), sizes)
    col = first_cols(sizes)
    return torch.cat(
        [row.unsqueeze(dim=0), col.unsqueeze(dim=0)], dim=0
    )


def emb_bag_offsets2sizes(offsets, x):
    return torch.cat([
        torch.diff(offsets),
        torch.tensor(
            [x.size(0) - offsets[-1]],
            dtype=offsets.dtype, device=offsets.device
        )
    ], dim=0)


def sizes2emb_bag_offsets(sizes):
    return torch.cat([
        torch.zeros(1, dtype=sizes.dtype, device=sizes.device),
        torch.cumsum(sizes, dim=0)[: -1]
    ])


def range_repeated(sizes):
    return torch.repeat_interleave(
        torch.arange(sizes.size(0), dtype=sizes.dtype, device=sizes.device),
        sizes
    )


def values_sizes2coo(values, sizes):
    indices = torch.cat([
        torch.unsqueeze(range_repeated(sizes), dim=0),
        torch.unsqueeze(values, dim=0)
    ])
    return torch.sparse_coo_tensor(
        indices, torch.ones_like(values, device=values.device)
    )


def var_repeat(values, sizes, repeat_lens):
    m = values_sizes2coo(values, sizes)
    m = torch.index_select(m, 0, range_repeated(repeat_lens))
    return m._indices()[1], torch.repeat_interleave(sizes, repeat_lens)

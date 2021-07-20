import numpy as np
import torch


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

def np2coo(coo):
    indices = np.vstack([coo.row, coo.col])
    indices = torch.from_numpy(indices.astype(np.int64, copy=False))
    return torch.sparse_coo_tensor(indices, coo.data, size=coo.shape)

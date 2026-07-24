import torch
from mmasim_kernels import MMAKernel


def probe_summation_order(kernel: MMAKernel) -> torch.Tensor:
    Ua, Ub = 2.0**7, 2.0**7
    va, vb = 2.0**-9, 2.0**-9
    k = kernel.k
    a = torch.full([k], va, dtype=kernel.a_type)
    b = torch.full([k], vb, dtype=kernel.b_type)
    c = torch.tensor(va * vb, dtype=kernel.c_type)
    res = torch.zeros([k + 1, k + 1], dtype=torch.int32)
    # aibi = U, ajbj = -U
    for i in range(k):
        a[i], b[i] = Ua, Ub
        for j in range(i):
            a[j], b[j] = Ua, -Ub
            res[i, j] = int(kernel.dpa(a, b, c).item() / (va * vb))
            a[j], b[j] = va, vb
        a[i], b[i] = va, vb
    # c = U, ajbj = -U
    c[None] = Ua * Ub
    for j in range(k):
        a[j], b[j] = Ua, -Ub
        res[k, j] = int(kernel.dpa(a, b, c).item() / (va * vb))
        a[j], b[j] = va, vb
    return res

import torch
from mmasim_kernels import MMAKernel


def is_negative_zero_supported(kernel: MMAKernel) -> bool:
    a = -torch.zeros([kernel.k], dtype=kernel.a_type)
    b = torch.zeros([kernel.k], dtype=kernel.b_type)
    c = torch.tensor(-0.0, dtype=kernel.c_type)
    return kernel.dpa(a, b, c).item().hex() == (-0.0).hex()


def is_infinity_supported(kernel: MMAKernel) -> bool:
    inf = float("inf")
    a = torch.zeros([kernel.k], dtype=kernel.a_type)
    b = torch.zeros([kernel.k], dtype=kernel.b_type)
    c = torch.tensor(inf, dtype=kernel.c_type)
    if kernel.dpa(a, b, c).item() != inf:
        return False
    c[None] = 0.0
    a[0], b[0] = inf, 1.0
    if kernel.dpa(a, b, c).item() != inf:
        return False
    return True


def is_nan_supported(kernel: MMAKernel) -> bool:
    a = torch.zeros([kernel.k], dtype=kernel.a_type)
    b = torch.zeros([kernel.k], dtype=kernel.b_type)
    c = torch.tensor(float("nan"), dtype=kernel.c_type)
    if not torch.isnan(kernel.dpa(a, b, c)):
        return False
    c[None] = 0.0
    a[0], b[0] = float("nan"), 0.0
    if not torch.isnan(kernel.dpa(a, b, c)):
        return False
    inf = float("inf")
    if kernel.a_type in [
        torch.float32,
        torch.float16,
        torch.bfloat16,
        torch.float8_e5m2,
    ]:
        a[0], b[0] = inf, 0.0
        if not torch.isnan(kernel.dpa(a, b, c)):
            return False
        a[0], b[0] = inf, 1.0
        c[None] = -inf
        if not torch.isnan(kernel.dpa(a, b, c)):
            return False
    return True


def is_subnormal_supported(kernel: MMAKernel) -> bool:
    a = torch.zeros([kernel.k], dtype=kernel.a_type)
    b = torch.zeros([kernel.k], dtype=kernel.b_type)
    c = torch.tensor(0.0, dtype=kernel.c_type)
    c[None] = 0.5 * torch.finfo(kernel.c_type).smallest_normal
    if kernel.dpa(a, b, c).item() == 0.0:
        return False
    c[None] = 0.0
    a[0], b[0] = 0.5 * torch.finfo(kernel.a_type).smallest_normal, 1.0
    if kernel.dpa(a, b, c).item() == 0.0:
        return False
    a[0], b[0] = torch.finfo(kernel.a_type).smallest_normal, 0.5
    if kernel.dpa(a, b, c).item() == 0.0:
        return False
    return True

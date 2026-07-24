import torch
from mmasim.nv_ptx.sim import MMA
from mmasim_kernels.nv_ptx.turing import mma_kernels

instr = "m16n8k8.f16.f16.f16.f16"
sim = MMA("Turing", instr)
kernel = mma_kernels[instr]


def get_random_tensor(m, n, dtype: torch.dtype, device="cuda", packing=1):
    tensor = torch.randint(
        -(2**31),
        2**31,
        [m * n // packing * dtype.itemsize // 4],
        dtype=torch.int32,
        device=device,
    )
    return tensor.view(dtype).view(m, n // packing)


m, n, k = kernel.m, kernel.n, kernel.k
for _ in range(1000000):
    A = get_random_tensor(m, k, kernel.a_type)
    B = get_random_tensor(n, k, kernel.b_type).T
    C = get_random_tensor(m, n, kernel.c_type)
    D_sim = sim(A, B, C)
    D_gpu = kernel(A, B, C)
    assert torch.equal(D_sim.view(torch.int32), D_gpu.view(torch.int32))


print("Step 4 passed.")

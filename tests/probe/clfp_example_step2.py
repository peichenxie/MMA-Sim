from mmasim_kernels.nv_ptx.turing import mma_kernels
from features.summation_order import probe_summation_order

kernel = mma_kernels["m16n8k8.f16.f16.f16.f16"]

order_info = probe_summation_order(kernel)
for i in range(kernel.k + 1):
    for j in range(i):
        assert order_info[i, j] == 0

print("Step 2 passed.")

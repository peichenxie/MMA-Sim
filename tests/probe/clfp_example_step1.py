from mmasim_kernels.nv_ptx.turing import mma_kernels
from features.determinism import is_deterministic
from features.independence import is_independent

kernel = mma_kernels["m16n8k8.f16.f16.f16.f16"]

assert is_deterministic(kernel, trials=1000, runs=100)
assert is_independent(kernel, trials=100000)

print("Step 1 passed.")

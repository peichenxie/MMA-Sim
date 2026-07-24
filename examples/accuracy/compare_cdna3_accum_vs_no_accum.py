import torch
from mmasim.amd.sim import MFMA

torch.manual_seed(0)
MFMA_FP16 = MFMA("CDNA3", "f32_32x32x8_f16")

bsz = 10
A = (10 * torch.randn(bsz, 128, 128, device="cuda")).to(torch.float16)
B = (10 * torch.randn(bsz, 128, 128, device="cuda")).to(torch.float16)
C_zeros = torch.zeros(bsz, 128, 128, device="cuda", dtype=torch.float32)
D_MFMA_accum = torch.zeros(bsz, 128, 128, device="cuda", dtype=torch.float32)
D_MFMA_no_accum = torch.zeros(bsz, 128, 128, device="cuda", dtype=torch.float32)
D_real = A.double() @ B.double()
for t in range(bsz):
    for i in range(0, 128, 32):
        for j in range(0, 128, 32):
            for k in range(0, 128, 8):
                D_MFMA_accum[t, i : i + 32, j : j + 32] = MFMA_FP16(
                    A[t, i : i + 32, k : k + 8],
                    B[t, k : k + 8, j : j + 32],
                    D_MFMA_accum[t, i : i + 32, j : j + 32],
                )
                D_MFMA_no_accum[t, i : i + 32, j : j + 32] += MFMA_FP16(
                    A[t, i : i + 32, k : k + 8],
                    B[t, k : k + 8, j : j + 32],
                    C_zeros[t, i : i + 32, j : j + 32],
                )

delta_accum = D_MFMA_accum - D_real
delta_no_accum = D_MFMA_no_accum - D_real
print(f"delta_accum mean:", delta_accum.mean().item())
print(f"delta_no_accum mean:", delta_no_accum.mean().item())

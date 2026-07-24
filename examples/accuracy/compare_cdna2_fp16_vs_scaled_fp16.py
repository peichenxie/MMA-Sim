import torch
from mmasim.amd.sim import MFMA

torch.manual_seed(0)
MFMA_FP16 = MFMA("CDNA2", "f32_32x32x8f16")
MFMA_BF16 = MFMA("CDNA2", "f32_32x32x8bf16_1k")

bsz = 10
A = (0.001 * torch.randn(bsz, 128, 128, device="cuda")).to(torch.float16)
A_scaled = A * 1024
B = torch.randn(bsz, 128, 128, device="cuda").to(torch.float16)
D_FP16 = torch.zeros(bsz, 128, 128, device="cuda", dtype=torch.float32)
D_FP16_scaled = torch.zeros(bsz, 128, 128, device="cuda", dtype=torch.float32)
D_BF16 = torch.zeros(bsz, 128, 128, device="cuda", dtype=torch.float32)
D_real = A.double() @ B.double()
for t in range(bsz):
    for i in range(0, 128, 32):
        for j in range(0, 128, 32):
            for k in range(0, 128, 8):
                D_FP16[t, i : i + 32, j : j + 32] = MFMA_FP16(
                    A[t, i : i + 32, k : k + 8],
                    B[t, k : k + 8, j : j + 32],
                    D_FP16[t, i : i + 32, j : j + 32],
                )
                D_FP16_scaled[t, i : i + 32, j : j + 32] = MFMA_FP16(
                    A_scaled[t, i : i + 32, k : k + 8],
                    B[t, k : k + 8, j : j + 32],
                    D_FP16_scaled[t, i : i + 32, j : j + 32],
                )
                D_BF16[t, i : i + 32, j : j + 32] = MFMA_BF16(
                    A[t, i : i + 32, k : k + 8].bfloat16(),
                    B[t, k : k + 8, j : j + 32].bfloat16(),
                    D_BF16[t, i : i + 32, j : j + 32],
                )

print("Original MSE:", (D_real - D_FP16).square().mean().item())
print("Scaled MSE:", (D_real - D_FP16_scaled * 2.0**-10).square().mean().item())
print("BF16 MSE:", (D_real - D_BF16).square().mean().item())

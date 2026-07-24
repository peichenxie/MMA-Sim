import torch
from mmasim.nv_ptx.sim import MMA

HMMA_F32 = MMA("Blackwell", "m16n8k16.f32.f16.f16.f32")
HMMA_F16 = MMA("Blackwell", "m16n8k16.f16.f16.f16.f16")

bsz = 10
A = (10 * torch.randn(bsz, 128, 128, device="cuda")).to(torch.float16)
B = (10 * torch.randn(bsz, 128, 128, device="cuda")).to(torch.float16)
D_HMMA_F16 = torch.zeros(bsz, 128, 128, device="cuda", dtype=torch.float16)
D_HMMA_F32 = torch.zeros(bsz, 128, 128, device="cuda", dtype=torch.float32)
D_real = A.double() @ B.double()
for t in range(bsz):
    for i in range(0, 128, 16):
        for j in range(0, 128, 8):
            for k in range(0, 128, 16):
                D_HMMA_F32[t, i : i + 16, j : j + 8] = HMMA_F32(
                    A[t, i : i + 16, k : k + 16],
                    B[t, k : k + 16, j : j + 8],
                    D_HMMA_F32[t, i : i + 16, j : j + 8],
                )
                D_HMMA_F16[t, i : i + 16, j : j + 8] = HMMA_F16(
                    A[t, i : i + 16, k : k + 16],
                    B[t, k : k + 16, j : j + 8],
                    D_HMMA_F16[t, i : i + 16, j : j + 8],
                )

print("HMMA.F16 MSE:", (D_real - D_HMMA_F16).square().mean().item())
print("HMMA.F32 MSE:", (D_real - D_HMMA_F32).square().mean().item())
print(
    "HMMA.F32 + Convert to F16 MSE:",
    (D_real - D_HMMA_F32.half()).square().mean().item(),
)

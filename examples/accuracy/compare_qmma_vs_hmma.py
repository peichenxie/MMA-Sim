import torch
from mmasim.nv_ptx.sim import MMA

HMMA = MMA("Ada Lovelace", "m16n8k16.f32.f16.f16.f32")
QMMA = MMA("Ada Lovelace", "m16n8k16.f32.e5m2.e5m2.f32")

bsz = 10
A = (10 * torch.randn(bsz, 128, 128, device="cuda")).to(torch.float8_e5m2)
B = (10 * torch.randn(bsz, 128, 128, device="cuda")).to(torch.float8_e5m2)
D_HMMA = torch.zeros(bsz, 128, 128, device="cuda", dtype=torch.float32)
D_QMMA = torch.zeros(bsz, 128, 128, device="cuda", dtype=torch.float32)
D_real = A.double() @ B.double()
for t in range(bsz):
    for i in range(0, 128, 16):
        for j in range(0, 128, 8):
            for k in range(0, 128, 16):
                D_QMMA[t, i : i + 16, j : j + 8] = QMMA(
                    A[t, i : i + 16, k : k + 16],
                    B[t, k : k + 16, j : j + 8],
                    D_QMMA[t, i : i + 16, j : j + 8],
                )
                D_HMMA[t, i : i + 16, j : j + 8] = HMMA(
                    A[t, i : i + 16, k : k + 16].half(),
                    B[t, k : k + 16, j : j + 8].half(),
                    D_HMMA[t, i : i + 16, j : j + 8],
                )

print("QMMA MSE:", (D_real - D_QMMA).square().mean().item())
print("HMMA MSE:", (D_real - D_HMMA).square().mean().item())

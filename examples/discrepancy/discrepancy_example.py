import torch
from mmasim.nv_ptx.sim import MMA, WGMMA, TCGen05MMA
from mmasim.amd.sim import MFMA

a = torch.tensor([-(2.0**13), -0.5, -0.25, -0.125])
b = torch.tensor([2.0**10, 1.0, 1.0, 1.0])
c = torch.tensor(2.0**23)

arch = "Volta"
instr = "m8n8k4.f32.f16.f16.f32"
print(arch, "FP16", MMA(arch, instr).dpa(a, b, c).item())

arch = "Turing"
instr = "m16n8k8.f32.f16.f16.f32"
print(arch, "FP16", MMA(arch, instr).dpa(a, b, c).item())

arch = "Ampere"
instr = "m8n8k4.f64.f64.f64.f64"
print(arch, "FP64", MMA(arch, instr).dpa(a, b, c).item())
instr = "m16n8k4.f32.tf32.tf32.f32"
print(arch, "TF32", MMA(arch, instr).dpa(a, b, c).item())
instr = "m16n8k8.f32.bf16.bf16.f32"
print(arch, "BF16", MMA(arch, instr).dpa(a, b, c).item())
instr = "m16n8k8.f32.f16.f16.f32"
print(arch, "FP16", MMA(arch, instr).dpa(a, b, c).item())

arch = "Ada Lovelace"
instr = "m8n8k4.f64.f64.f64.f64"
print(arch, "FP64", MMA(arch, instr).dpa(a, b, c).item())
instr = "m16n8k4.f32.tf32.tf32.f32"
print(arch, "TF32", MMA(arch, instr).dpa(a, b, c).item())
instr = "m16n8k8.f32.bf16.bf16.f32"
print(arch, "BF16", MMA(arch, instr).dpa(a, b, c).item())
instr = "m16n8k8.f32.f16.f16.f32"
print(arch, "FP16", MMA(arch, instr).dpa(a, b, c).item())
instr = "m16n8k16.f32.e5m2.e5m2.f32"
print(arch, "FP8", MMA(arch, instr).dpa(a, b, c).item())

arch = "Hopper"
instr = "m8n8k4.f64.f64.f64.f64"
print(arch, "FP64", MMA(arch, instr).dpa(a, b, c).item())
instr = "m16n8k8.f32.tf32.tf32.f32"
print(arch, "TF32", MMA(arch, instr).dpa(a, b, c).item())
instr = "m16n8k16.f32.bf16.bf16.f32"
print(arch, "BF16", MMA(arch, instr).dpa(a, b, c).item())
instr = "m16n8k16.f32.f16.f16.f32"
print(arch, "FP16", MMA(arch, instr).dpa(a, b, c).item())
instr = "m64n8k32.f32.e5m2.e5m2"
print(arch, "FP8", WGMMA(arch, instr).dpa(a, b, c).item())

arch = "Blackwell"
instr = "m8n8k4.f64.f64.f64.f64"
print(arch, "FP64", MMA(arch, instr).dpa(a, b, c).item())
instr = "m16n8k8.f32.tf32.tf32.f32"
print(arch, "TF32", MMA(arch, instr).dpa(a, b, c).item())
instr = "m16n8k16.f32.bf16.bf16.f32"
print(arch, "BF16", MMA(arch, instr).dpa(a, b, c).item())
instr = "m16n8k16.f32.f16.f16.f32"
print(arch, "FP16", MMA(arch, instr).dpa(a, b, c).item())
instr = "m64n8k32.f32.e5m2.e5m2"
print(arch, "FP8", TCGen05MMA(arch, instr).dpa(a, b, c).item())

arch = "RTX Blackwell"
instr = "m8n8k4.f64.f64.f64.f64"
print(arch, "FP64", MMA(arch, instr).dpa(a, b, c).item())
instr = "m16n8k4.f32.tf32.tf32.f32"
print(arch, "TF32", MMA(arch, instr).dpa(a, b, c).item())
instr = "m16n8k8.f32.bf16.bf16.f32"
print(arch, "BF16", MMA(arch, instr).dpa(a, b, c).item())
instr = "m16n8k8.f32.f16.f16.f32"
print(arch, "FP16", MMA(arch, instr).dpa(a, b, c).item())
instr = "m16n8k16.f32.e5m2.e5m2.f32"
print(arch, "FP8", MMA(arch, instr).dpa(a, b, c).item())

arch = "CDNA1"
instr = "f32_16x16x4f32"
print(arch, "FP32", MFMA(arch, instr).dpa(a, b, c).item())
instr = "f32_16x16x16f16"
print(arch, "FP16", MFMA(arch, instr).dpa(a, b, c).item())
instr = "f32_16x16x8bf16"
print(arch, "BF16", MFMA(arch, instr).dpa(a, b, c).item())

arch = "CDNA2"
instr = "f64_16x16x4f64"
print(arch, "FP64", MFMA(arch, instr).dpa(a, b, c).item())
instr = "f32_16x16x4f32"
print(arch, "FP32", MFMA(arch, instr).dpa(a, b, c).item())
instr = "f32_16x16x16f16"
print(arch, "FP16", MFMA(arch, instr).dpa(a.half(), b.half(), c).item())
instr = "f32_16x16x8bf16"
print(arch, "BF16", MFMA(arch, instr).dpa(a.bfloat16(), b.bfloat16(), c).item())
instr = "f32_16x16x16bf16_1k"
print(arch, "BF16_1k", MFMA(arch, instr).dpa(a.bfloat16(), b.bfloat16(), c).item())

arch = "CDNA3"
instr = "f64_16x16x4_f64"
print(arch, "FP64", MFMA(arch, instr).dpa(a, b, c).item())
instr = "f32_16x16x4_f32"
print(arch, "FP32", MFMA(arch, instr).dpa(a, b, c).item())
instr = "f32_16x16x8_xf32"
print(arch, "TF32", MFMA(arch, instr).dpa(a, b, c).item())
instr = "f32_16x16x16_f16"
print(arch, "FP16", MFMA(arch, instr).dpa(a, b, c).item())
instr = "f32_16x16x16_bf16"
print(arch, "BF16", MFMA(arch, instr).dpa(a, b, c).item())
instr = "f32_16x16x32_bf8_bf8"
print(arch, "FP8", MFMA(arch, instr).dpa(a, b, c).item())

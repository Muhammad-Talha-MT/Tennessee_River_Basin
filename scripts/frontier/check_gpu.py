"""Quick check that this PyTorch/ROCm install can run what the GNN needs (run it inside a GPU job):

  srun -N1 -n1 --gpus-per-node=8 python scripts/frontier/check_gpu.py

Checks: a GPU is visible; bf16 autocast LSTM forward/backward; a sparse CSR matrix product with a custom backward
(the routing and the lumped model's basin means); index_add_ in bf16 (the graph layers); activation checkpointing.
Prints a timing of a small LSTM so GPUs can be compared roughly with the L40S used so far.
"""
import time

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

print("torch", torch.__version__, "| hip", getattr(torch.version, "hip", None), "| cuda", torch.version.cuda)
assert torch.cuda.is_available(), "no GPU visible (run inside a GPU job, via srun)"
n = torch.cuda.device_count()
print("devices:", n, [torch.cuda.get_device_name(i) for i in range(n)][:2])
d = torch.device("cuda:0")

# bf16 LSTM, forward and backward, under autocast (as the per-reach trunk)
lstm = nn.LSTM(128, 128, batch_first=True).to(d)
x = torch.randn(4000, 144, 128, device=d, requires_grad=True)
torch.cuda.synchronize()
t = time.perf_counter()
for _ in range(3):
    with torch.autocast("cuda", torch.bfloat16):
        y, _ = lstm(x)
    y.float().square().mean().backward()
torch.cuda.synchronize()
print(f"bf16 LSTM 4000 reaches x 144 steps, 3 forward+backward: {time.perf_counter() - t:.2f} s")

# sparse CSR product with a custom backward through the transpose
class SparseProduct(torch.autograd.Function):
    @staticmethod
    def forward(ctx, a, at, x):
        ctx.at = at
        return a @ x

    @staticmethod
    def backward(ctx, grad):
        return None, None, ctx.at @ grad.float().contiguous()

m = 20000
idx = torch.randint(0, m, (2, 200000), device=d)
coo = torch.sparse_coo_tensor(idx, torch.rand(200000, device=d), (m, m)).coalesce()
a, at = coo.to_sparse_csr(), coo.t().coalesce().to_sparse_csr()
v = torch.randn(m, 128, device=d, requires_grad=True)
out = SparseProduct.apply(a, at, v)
out.sum().backward()
ref = (coo.to_dense() @ v.detach())
assert torch.allclose(out, ref, atol=1e-3, rtol=1e-3), "sparse product differs from dense"
assert torch.isfinite(v.grad).all()
print("sparse CSR product and backward: ok")

# bf16 index_add_ (graph layers) and checkpointing
h = torch.randn(10000, 40, 32, device=d, dtype=torch.bfloat16)
src = torch.randint(0, 10000, (9000,), device=d)
tgt = torch.randint(0, 10000, (9000,), device=d)
agg = torch.zeros_like(h).index_add_(0, tgt, h[src])
assert torch.isfinite(agg.float()).all()
f = lambda z: torch.nn.functional.silu(z) * 2
z = torch.randn(1000, 64, device=d, requires_grad=True)
checkpoint(f, z, use_reentrant=False).sum().backward()
print("index_add_ in bf16 and checkpointing: ok")
print("ALL OK")

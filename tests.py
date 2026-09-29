"""
tests.py — python tests.py. no download, no GPU, a few seconds.
if these pass, the numbers in results/ aren't coming from a bug.
"""
import torch
import torch.nn.functional as F

from quant import (QuantLinear, W8A8Linear, dequantize, fake_quant_act, pack_int4,
                   quantize, unpack_int4)

torch.manual_seed(0)
fails = 0


def check(name, cond, why=""):
    global fails
    print(f"{'PASS' if cond else 'FAIL'}  {name}" + ("" if cond else f"  -> {why}"))
    fails += not cond


W = torch.randn(96, 128) * 0.05

# rounding error can't be more than half a step
for gran in ("tensor", "channel", "group"):
    for sym in (True, False):
        q, s, z = quantize(W, 8, gran, 64, sym)
        err = (dequantize(q, s, z, W.shape) - W).abs()
        step = s.expand(-1, q.numel() // s.shape[0]).reshape(W.shape)
        check(f"int8 {gran:7s} sym={sym!s:5} error <= step/2",
              bool((err <= step / 2 + 1e-6).all()), f"max err {err.max():.2e}")

# finer granularity -> smaller error (the whole point of groups)
def mse(**kw):
    q, s, z = quantize(W, 4, **kw)
    return (dequantize(q, s, z, W.shape) - W).pow(2).mean().item()
W[3, 7] = 1.0  # plant an outlier
e_t, e_c, e_g = mse(granularity="tensor"), mse(granularity="channel"), mse(granularity="group", group_size=32)
check("int4: tensor > channel > group error", e_t > e_c > e_g, f"{e_t:.2e} {e_c:.2e} {e_g:.2e}")

# packing is lossless and halves the bytes
for signed, lo, hi, dt in ((True, -8, 8, torch.int8), (False, 0, 16, torch.uint8)):
    q = torch.randint(lo, hi, (64, 32)).to(dt)
    p = pack_int4(q)
    check(f"pack/unpack int4 signed={signed}", torch.equal(unpack_int4(p, q.shape, signed), q))
    check(f"packed int4 is half an int8 (signed={signed})", p.numel() == q.numel() // 2)

# QuantLinear really is linear with the dequantized weight
b = torch.randn(96) * 0.01
x = torch.randn(5, 128)
for bits, gran, sym in ((8, "channel", True), (4, "group", True), (4, "group", False), (3, "group", False)):
    ql = QuantLinear(W, b, bits, gran, 64, sym)
    ref = F.linear(x, ql.weight(), b.half().float())
    check(f"QuantLinear W{bits} {gran} sym={sym} forward", torch.allclose(ql(x), ref, atol=1e-5))
    rel = (ql(x) - F.linear(x, W, b)).norm() / F.linear(x, W, b).norm()
    check(f"QuantLinear W{bits} close to fp (rel err {rel:.3f})", rel < {8: 0.02, 4: 0.25, 3: 0.5}[bits])

check("int4 layer ~half the bytes of int8",
      QuantLinear(W, b, 4, "channel").nbytes() < 0.6 * QuantLinear(W, b, 8, "channel").nbytes())

# the outlier story in miniature: one huge activation wrecks per-token int8,
# splitting it out fixes it
x = torch.randn(4, 128) * 0.5
x[:, 17] = 40.0
exact = F.linear(x, W, b)
naive = W8A8Linear(W, b)(x)
split = W8A8Linear(W, b, outlier_threshold=6.0)(x)
e_naive = (naive - exact).norm() / exact.norm()
e_split = (split - exact).norm() / exact.norm()
check(f"outlier split beats naive W8A8 ({e_split:.4f} < {e_naive:.4f})", e_split < e_naive / 3)
check("fake_quant_act keeps shape", fake_quant_act(x).shape == x.shape)

# surgery on a (tiny, random) GPT-2: conv1d -> QuantLinear keeps the outputs
from transformers import GPT2Config, GPT2LMHeadModel
from quant import quantize_gpt2
import copy
g = GPT2LMHeadModel(GPT2Config(n_layer=2, n_embd=64, n_head=4, vocab_size=100, n_positions=32)).eval()
ids = torch.randint(0, 100, (1, 16))
with torch.no_grad():
    ref = g(ids).logits
    gq = copy.deepcopy(g)
    before, after = quantize_gpt2(gq, lambda w, b: QuantLinear(w, b, 8, "channel"))
    out = gq(ids).logits
check("GPT-2 surgery keeps logits (W8)", torch.allclose(out, ref, atol=2e-2), f"{(out - ref).abs().max():.3e}")
check("GPT-2 surgery: W8 ~half the fp16 bytes", 0.45 < after / before < 0.6, f"{after / before:.2f}")

print(f"\n{'all good' if not fails else f'{fails} failing'}")
raise SystemExit(1 if fails else 0)

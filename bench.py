"""
bench.py — quantize GPT-2 small a bunch of different ways, measure what each
one costs in perplexity and what it saves in memory.

    python bench.py                 # full WikiText-2 test set (GPU recommended)
    python bench.py --windows 24    # first 24 x 1024 tokens, fine on a laptop CPU

writes results/results.md, results/results.json, results/ppl_vs_bits.png
"""
import argparse
import copy
import json
import math
import os
import time

import torch
from datasets import load_dataset
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

from quant import QuantLinear, W8A8Linear, quantize_gpt2

# name, how to build each layer. the order is the story of the README:
# int8 is free, int4 needs groups (and a zero-point), activations are where
# the real trouble is. the baseline computes in fp32 but its size is counted
# as fp16, since that's how you'd actually ship it.
CONFIGS = [
    ("fp16 baseline",         None),
    ("W8 per-tensor",         lambda w, b: QuantLinear(w, b, 8, "tensor")),
    ("W8 per-channel",        lambda w, b: QuantLinear(w, b, 8, "channel")),
    ("W4 per-channel",        lambda w, b: QuantLinear(w, b, 4, "channel")),
    ("W4 group128",           lambda w, b: QuantLinear(w, b, 4, "group", 128)),
    ("W4 group64",            lambda w, b: QuantLinear(w, b, 4, "group", 64)),
    ("W4 group64 + zero-pt",  lambda w, b: QuantLinear(w, b, 4, "group", 64, sym=False)),
    ("W4 group32 + zero-pt",  lambda w, b: QuantLinear(w, b, 4, "group", 32, sym=False)),
    ("W8A8 naive",            lambda w, b: W8A8Linear(w, b)),
    ("W8A8 + outlier split",  lambda w, b: W8A8Linear(w, b, outlier_threshold=6.0)),
]


def load_windows(tok, n_windows, ctx=1024):
    data = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
    ids = tok("\n\n".join(data["text"]), return_tensors="pt").input_ids[0]
    n = len(ids) // ctx
    if n_windows:
        n = min(n, n_windows)
    # non-overlapping 1024-token windows. strided eval would give slightly
    # lower ppl for everyone; what matters here is the DIFFERENCE between rows.
    return ids[: n * ctx].view(n, ctx)


@torch.no_grad()
def perplexity(model, windows, device):
    model.eval()
    nll, count = 0.0, 0
    for w in windows:
        x = w.unsqueeze(0).to(device)
        loss = model(x, labels=x).loss   # HF shifts the labels internally
        nll += loss.item() * (x.numel() - 1)
        count += x.numel() - 1
    return math.exp(nll / count)


def other_bytes(model):
    """everything I don't quantize: embeddings (tied with lm_head), layernorms.
    counted in fp16."""
    t = model.transformer
    n = t.wte.weight.numel() + t.wpe.weight.numel()
    n += sum(p.numel() for m in model.modules() if isinstance(m, torch.nn.LayerNorm)
             for p in m.parameters())
    return n * 2


def plot(rows, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    base = rows[0]["ppl"]
    pts = [r for r in rows if not r["name"].startswith("W8A8")]
    plt.figure(figsize=(6.4, 4))
    plt.axhline(base, color="gray", ls="--", lw=1, label=f"fp16 = {base:.2f}")
    for r in pts[1:]:
        plt.scatter(r["bits_per_weight"], r["ppl"], s=40, zorder=3)
        plt.annotate(r["name"], (r["bits_per_weight"], r["ppl"]), fontsize=7,
                     xytext=(5, 3), textcoords="offset points")
    plt.xlabel("effective bits per weight (scales included)")
    plt.ylabel("WikiText-2 perplexity (lower = better)")
    plt.title("GPT-2 small: memory vs quality")
    plt.grid(alpha=0.3)
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(path, dpi=140)
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--windows", type=int, default=None, help="limit eval to N x 1024 tokens")
    ap.add_argument("--bandwidth", type=float, default=100.0,
                    help="GB/s, for the theoretical decode-speed column")
    ap.add_argument("--out", default="results")
    a = ap.parse_args()

    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(a.out, exist_ok=True)
    tok = GPT2TokenizerFast.from_pretrained("openai-community/gpt2")
    base = GPT2LMHeadModel.from_pretrained("openai-community/gpt2").float()
    windows = load_windows(tok, a.windows)
    print(f"eval on {windows.numel():,} WikiText-2 tokens, device {device}\n")

    rest = other_bytes(base)
    n_weights = sum(b.attn.c_attn.weight.numel() + b.attn.c_proj.weight.numel()
                    + b.mlp.c_fc.weight.numel() + b.mlp.c_proj.weight.numel()
                    for b in base.transformer.h)
    rows = []
    for name, make in CONFIGS:
        t = time.time()
        model = copy.deepcopy(base)
        if make is None:
            before = after = n_weights * 2 + sum(
                b.attn.c_attn.bias.numel() + b.attn.c_proj.bias.numel()
                + b.mlp.c_fc.bias.numel() + b.mlp.c_proj.bias.numel()
                for b in base.transformer.h) * 2
        else:
            before, after = quantize_gpt2(model, make)
        ppl = perplexity(model.to(device), windows, device)
        total = after + rest
        row = {
            "name": name,
            "ppl": ppl,
            "bits_per_weight": after * 8 / n_weights,
            "block_MB": after / 1e6,
            "total_MB": total / 1e6,
            # decoding is memory-bound: every token reads every weight once.
            # so an upper bound on speed is bandwidth / model size.
            "tok_per_s_bound": a.bandwidth * 1e9 / total,
        }
        if name.startswith("W8A8 +"):
            fr = [f for m in model.modules() if isinstance(m, W8A8Linear) for f in m.outlier_frac]
            row["outlier_cols_pct"] = 100 * sum(fr) / len(fr)
        rows.append(row)
        print(f"{name:24s} ppl {ppl:8.2f} | {row['bits_per_weight']:5.2f} bits/w | "
              f"blocks {row['block_MB']:6.1f} MB | {time.time() - t:5.0f}s")
        del model

    base_ppl = rows[0]["ppl"]
    lines = [
        f"GPT-2 small (124M), WikiText-2 test, {windows.numel():,} tokens, context 1024.\n",
        "| config | perplexity | Δ vs fp16 | bits / weight | block weights | whole model "
        f"| decode bound @ {a.bandwidth:.0f} GB/s |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        d = (r["ppl"] / base_ppl - 1) * 100
        lines.append(f"| {r['name']} | {r['ppl']:.2f} | {d:+.1f}% | {r['bits_per_weight']:.2f} "
                     f"| {r['block_MB']:.1f} MB | {r['total_MB']:.1f} MB "
                     f"| {r['tok_per_s_bound']:.0f} tok/s |")
    for r in rows:
        if "outlier_cols_pct" in r:
            lines.append(f"\nW8A8 + outlier split: on average {r['outlier_cols_pct']:.2f}% of input "
                         f"dimensions per layer went through the fp16 path.")
    md = "\n".join(lines) + "\n"
    print("\n" + md)
    open(os.path.join(a.out, "results.md"), "w").write(md)
    json.dump(rows, open(os.path.join(a.out, "results.json"), "w"), indent=2)
    plot(rows, os.path.join(a.out, "ppl_vs_bits.png"))


if __name__ == "__main__":
    main()

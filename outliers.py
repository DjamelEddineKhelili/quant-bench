"""
outliers.py — go look at the activations going INTO every linear layer of
GPT-2 and find the "massive" dimensions. this is the reason naive W8A8 hurts.

    python outliers.py --windows 8

prints the worst layers + dims, saves results/outliers.png
"""
import argparse
import os
from collections import Counter

import torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

from bench import load_windows

THRESHOLD = 6.0   # the LLM.int8() paper's cutoff


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--windows", type=int, default=8)
    ap.add_argument("--out", default="results")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    tok = GPT2TokenizerFast.from_pretrained("openai-community/gpt2")
    model = GPT2LMHeadModel.from_pretrained("openai-community/gpt2").eval()
    windows = load_windows(tok, a.windows)

    # forward hooks: record max |x| per input channel for every linear layer
    stats, hooks = {}, []
    for i, block in enumerate(model.transformer.h):
        for tag, mod in [("attn.c_attn", block.attn.c_attn), ("attn.c_proj", block.attn.c_proj),
                         ("mlp.c_fc", block.mlp.c_fc), ("mlp.c_proj", block.mlp.c_proj)]:
            key = f"h{i}.{tag}"

            def hook(m, inp, out, key=key):
                mx = inp[0].abs().reshape(-1, inp[0].shape[-1]).amax(dim=0)
                stats[key] = torch.maximum(stats[key], mx) if key in stats else mx
            hooks.append(mod.register_forward_hook(hook))

    for w in windows:
        model(w.unsqueeze(0))
    for h in hooks:
        h.remove()

    print(f"input dims with max |x| > {THRESHOLD}, per layer:\n")
    dims = Counter()
    for key, mx in stats.items():
        hot = (mx > THRESHOLD).nonzero().flatten().tolist()
        if key.endswith(("c_attn", "c_fc")):   # these two read the (layernormed) residual stream,
            dims.update(hot)                      # so their dims mean the same thing across layers
        typical = mx.median().item()
        print(f"  {key:16s} {len(hot):4d} / {mx.numel()} dims | median max|x| {typical:5.2f} "
              f"| worst {mx.max().item():7.1f} (dim {mx.argmax().item()})")
    print("\nresidual-stream dims that are outliers in many layers (c_attn / c_fc inputs):")
    for d, n in dims.most_common(6):
        print(f"  dim {d:4d} shows up in {n} of 24 layers")

    # the picture: one qkv input, per-channel max |x|. flat grass + a few towers.
    key = max((k for k in stats if k.endswith("c_attn")), key=lambda k: stats[k].max())
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    mx = stats[key].numpy()
    plt.figure(figsize=(7, 3))
    plt.bar(range(len(mx)), mx, width=1.0)
    plt.axhline(THRESHOLD, color="red", ls="--", lw=1, label=f"|x| = {THRESHOLD}")
    plt.yscale("log")
    plt.xlabel("hidden dimension")
    plt.ylabel("max |activation|")
    plt.title(f"GPT-2 {key} input: a few dimensions dominate")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(a.out, "outliers.png"), dpi=140)
    print(f"\nsaved {a.out}/outliers.png ({key})")


if __name__ == "__main__":
    main()

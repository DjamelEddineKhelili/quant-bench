# quant-bench

![tests](https://github.com/DjamelEddineKhelili/quant-bench/actions/workflows/tests.yml/badge.svg)

**Weight quantization for GPT-2, written from scratch.** INT8 / INT4, per-tensor,
per-channel, per-group, with and without zero-point, real 4-bit packing, and a
small reproduction of the LLM.int8() outlier trick. No bitsandbytes, no
auto-gptq: `torch.round`, some reshapes and a benchmark.

Why I built it: when an LLM generates a token, it has to read **every single
weight** from memory. The math is cheap; moving the bytes isn't. So decoding
is memory-bound, and the most direct lever is to make each weight smaller.
I wanted to see for myself how far you can push that before the model
starts talking nonsense.

![perplexity vs bits](results/ppl_vs_bits.png)

## Results

GPT-2 small (124M), WikiText-2 test, 16,384 tokens (non-overlapping 1024-token windows, CPU run).
Only the linear layers inside the transformer blocks are quantized ("block weights").

| config | perplexity | Δ vs fp16 | bits / weight | block weights | whole model | decode bound @ 100 GB/s |
|---|---:|---:|---:|---:|---:|---:|
| fp16 baseline | 30.81 | — | 16.02 | 170.0 MB | 248.9 MB | 402 tok/s |
| W8 per-tensor | 35.20 | +14.2% | 8.02 | 85.1 MB | 163.9 MB | 610 tok/s |
| **W8 per-channel** | **30.80** | **±0.0%** | 8.03 | 85.3 MB | 164.1 MB | 609 tok/s |
| W4 per-channel | 48.12 | +56.2% | 4.03 | 42.8 MB | 121.6 MB | 822 tok/s |
| W4 group128 | 37.07 | +20.3% | 4.14 | 44.0 MB | 122.8 MB | 814 tok/s |
| W4 group64 | 35.90 | +16.5% | 4.27 | 45.3 MB | 124.1 MB | 806 tok/s |
| **W4 group64 + zero-pt** | **32.95** | **+6.9%** | 4.39 | 46.6 MB | 125.5 MB | 797 tok/s |
| W4 group32 + zero-pt | 32.41 | +5.2% | 4.77 | 50.6 MB | 129.4 MB | 773 tok/s |
| W8A8 naive | 30.99 | +0.6% | 8.03 | 85.3 MB | 164.1 MB | 609 tok/s |
| W8A8 + outlier split | 30.87 | +0.2% | 8.03 | 85.3 MB | 164.1 MB | 609 tok/s |

"bits / weight" includes the scales and zero-points, so it's the honest number, not the marketing one.
"decode bound" = memory bandwidth ÷ model size, the best case for a memory-bound decoder.

## What I learned

**1. INT8 is free, if you pick the right granularity.** Per-channel INT8 lands on
the exact same perplexity as fp16 (30.80 vs 30.81). Per-*tensor* INT8 costs 14%:
one scale for the whole matrix means the single biggest weight decides the step
size for everyone, and the small weights get rounded into mush.

**2. INT4 needs groups AND a zero-point.** One scale per row at 4 bits is
+56%. Groups of 64 bring it down to +16.5%. The biggest single win was
switching to asymmetric (min/max + zero-point) quantization: **+6.9%**.
A group of 64 weights is rarely centered on 0, and with symmetric absmax
a good chunk of the 16 levels sit on a side of 0 where there are hardly any weights.
Final: block weights go from **170 MB to 46.6 MB (3.6x smaller)**.

**3. The whole model only gets 2x smaller, and that's a scale thing.** I leave
the embeddings in fp16, and in GPT-2 small they're 38.6M of the 124M params
(the 50k vocab × 768). At 7B+ they're a rounding error, so the full 4x shows
up there. Good reminder that "4-bit model" means different things at
different sizes.

**4. The outliers are real, and they're always the same few dimensions.**
`python outliers.py` hooks every linear layer and records the max |activation|
per input dimension. Most sit around 1. A handful go above 6, and they're the
*same* dimensions layer after layer: dim 138 shows up in 10 of the 24
residual-stream inputs, dim 266 in 9, 480 and 64 in 7. The inputs to the last MLP
projection peak at 43.8.

![outliers](results/outliers.png)

That's what breaks per-token INT8 activations: one 40 in a row full of 1s
sets the scale, and everything else rounds to 0. The LLM.int8() fix sends the
outlier columns through an fp16 matmul and everything else through int8. Here it
routes **0.23% of the dimensions** and cuts the W8A8 damage from +0.6% to +0.2%.
Honest take: at 124M the effect is small. The paper shows it becomes
catastrophic past ~6.7B params, which I can't run on my machine. The mechanism is
already visible here though.

## Honest limits

- **It's not faster (yet).** `QuantLinear` stores packed ints and dequantizes
  to float in plain PyTorch before every matmul, so on CPU it's actually
  *slower* than fp32. The speed win needs a fused kernel that dequantizes
  inside the matmul (what Marlin / ExLlama do). The memory win is real; the
  speed column is the theoretical bound.
- Activations are fake-quantized (quantize → dequantize → float matmul):
  numerically what an int8 kernel sees, without writing the kernel.
- The W8A8 + split layer keeps an fp16 copy of W for the outlier columns and
  doesn't count it in the size. A real implementation would store only those
  columns.
- 16k eval tokens because it ran on CPU. `python bench.py` without
  `--windows` runs the full test set. Absolute perplexity is higher than the
  GPT-2 paper's because of non-overlapping windows on raw text. The *differences*
  between rows are what matter.

## Run it

```bash
pip install -r requirements.txt
python tests.py                 # math sanity checks, no download, seconds
python bench.py --windows 16    # the table + plot above (~20 min on CPU)
python bench.py                 # full WikiText-2 test set (GPU)
python outliers.py              # the activation outlier analysis
```

## Files

| file | lines | what |
|---|---:|---|
| `quant.py` | ~190 | quantize / dequantize, int4 packing, `QuantLinear`, `W8A8Linear`, GPT-2 surgery |
| `bench.py` | ~170 | perplexity + memory for every config, writes `results/` |
| `outliers.py` | ~85 | forward hooks, finds the massive activation dims |
| `tests.py` | ~90 | error bounds, pack/unpack, layer equivalence, outlier toy case |

## Next

- GPTQ: quantize column by column and push each rounding error onto the
  columns that aren't quantized yet (second-order, uses the Hessian of the layer)
- AWQ: scale the salient channels up before quantizing
- a Triton kernel for the fused int4 dequant-matmul, to turn the memory win into actual tokens/s

## References

- Dettmers et al., *LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale* (2022)
- Frantar et al., *GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers* (2022)
- Lin et al., *AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration* (2023)
- Xiao et al., *SmoothQuant* (2022)

---
CRTnoise · MIT

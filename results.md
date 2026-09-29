GPT-2 small (124M), WikiText-2 test, 16,384 tokens, context 1024.

| config | perplexity | Δ vs fp16 | bits / weight | block weights | whole model | decode bound @ 100 GB/s |
|---|---:|---:|---:|---:|---:|---:|
| fp16 baseline | 30.81 | +0.0% | 16.02 | 170.0 MB | 248.9 MB | 402 tok/s |
| W8 per-tensor | 35.20 | +14.2% | 8.02 | 85.1 MB | 163.9 MB | 610 tok/s |
| W8 per-channel | 30.80 | -0.0% | 8.03 | 85.3 MB | 164.1 MB | 609 tok/s |
| W4 per-channel | 48.12 | +56.2% | 4.03 | 42.8 MB | 121.6 MB | 822 tok/s |
| W4 group128 | 37.07 | +20.3% | 4.14 | 44.0 MB | 122.8 MB | 814 tok/s |
| W4 group64 | 35.90 | +16.5% | 4.27 | 45.3 MB | 124.1 MB | 806 tok/s |
| W4 group64 + zero-pt | 32.95 | +6.9% | 4.39 | 46.6 MB | 125.5 MB | 797 tok/s |
| W4 group32 + zero-pt | 32.41 | +5.2% | 4.77 | 50.6 MB | 129.4 MB | 773 tok/s |
| W8A8 naive | 30.99 | +0.6% | 8.03 | 85.3 MB | 164.1 MB | 609 tok/s |
| W8A8 + outlier split | 30.87 | +0.2% | 8.03 | 85.3 MB | 164.1 MB | 609 tok/s |

W8A8 + outlier split: on average 0.23% of input dimensions per layer went through the fp16 path.

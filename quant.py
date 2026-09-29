"""
quant.py — weight quantization from scratch. no bitsandbytes, no auto-gptq,
just torch.round and some bookkeeping.

the one-line idea: a weight matrix is a pile of floats that mostly live in a
small range, so store each one as a small integer + one shared float
("scale") per tensor / row / group of 64. int8 = 4x smaller than fp32,
int4 = 8x smaller.

why anyone cares: when an LLM generates one token, it has to read EVERY
weight from memory once. the math is cheap, the memory traffic isn't.
fewer bytes per weight = fewer bytes to move = faster tokens.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------- core
def quantize(w, bits=8, granularity="channel", group_size=64, sym=True):
    """
    w: (out, in) float weight
    granularity:
      "tensor"  one scale for the whole matrix (one outlier ruins everyone)
      "channel" one scale per output row
      "group"   one scale per row per block of `group_size` inputs
    sym=True : q in [-2^(b-1), 2^(b-1)-1], w ≈ q * scale
    sym=False: q in [0, 2^b - 1],          w ≈ (q - zero) * scale
               (uses the full range when a group isn't centered on 0.
                matters a lot at 4 bits, almost nothing at 8)
    returns q (int8 tensor, shaped like w), scale, zero (None if sym)
    """
    out_f, in_f = w.shape
    if granularity == "tensor":
        g = w.reshape(1, -1)
    elif granularity == "channel":
        g = w
    elif granularity == "group":
        assert in_f % group_size == 0, f"{in_f} not divisible by group {group_size}"
        g = w.reshape(-1, group_size)
    else:
        raise ValueError(granularity)

    if sym:
        qmax = 2 ** (bits - 1) - 1
        scale = g.abs().amax(dim=1, keepdim=True).clamp(min=1e-8) / qmax
        q = torch.round(g / scale).clamp(-qmax - 1, qmax)
        zero = None
    else:
        qmax = 2 ** bits - 1
        lo, hi = g.amin(dim=1, keepdim=True), g.amax(dim=1, keepdim=True)
        scale = (hi - lo).clamp(min=1e-8) / qmax
        zero = torch.round(-lo / scale)
        q = torch.round(g / scale + zero).clamp(0, qmax)
    return q.to(torch.int8 if sym else torch.uint8).reshape(out_f, in_f), scale, zero


def dequantize(q, scale, zero, shape):
    """inverse of quantize. same grouping trick: reshape so every row of the
    view shares one scale, multiply, reshape back."""
    g = q.reshape(scale.shape[0], -1).float()
    if zero is not None:
        g = g - zero
    return (g * scale.float()).reshape(shape)


# ---------------------------------------------------------------- int4 packing
# torch has no int4 dtype, so without this "4-bit" would still eat 8 bits in RAM.
# two nibbles per byte: low nibble = even index, high nibble = odd index.
def pack_int4(q):
    """q: int8 in [-8, 7] or uint8 in [0, 15], even number of elements -> uint8 half the size"""
    u = q.flatten().to(torch.int16)
    if q.dtype == torch.int8:
        u = u + 8   # shift signed [-8, 7] to [0, 15]
    u = u.to(torch.uint8)
    return u[0::2] | (u[1::2] << 4)


def unpack_int4(packed, shape, signed):
    lo = packed & 0x0F
    hi = packed >> 4
    u = torch.stack([lo, hi], dim=1).flatten().to(torch.int16)
    if signed:
        return (u - 8).to(torch.int8).reshape(shape)
    return u.to(torch.uint8).reshape(shape)


# ---------------------------------------------------------------- layers
class QuantLinear(nn.Module):
    """drop-in replacement for nn.Linear that STORES the weight quantized
    and dequantizes it on the fly in forward ("weight-only" quantization).
    activations stay in float, so the only error is in the weights."""

    def __init__(self, weight, bias, bits=8, granularity="channel", group_size=64, sym=True):
        super().__init__()
        self.shape = tuple(weight.shape)
        self.bits, self.sym = bits, sym
        q, scale, zero = quantize(weight.float(), bits, granularity, group_size, sym)
        if bits <= 4:
            q = pack_int4(q)   # (3-bit would still take a full nibble here, so I don't bench it)
        self.register_buffer("q", q)
        # scales in fp16, like everyone does. they're a rounding error in the budget.
        self.register_buffer("scale", scale.half())
        self.register_buffer("zero", None if zero is None else zero.to(torch.uint8))
        self.register_buffer("bias", None if bias is None else bias.detach().half())

    def weight(self):
        q = unpack_int4(self.q, self.shape, self.sym) if self.bits <= 4 else self.q
        zero = None if self.zero is None else self.zero.float()
        return dequantize(q, self.scale, zero, self.shape)

    def forward(self, x):
        b = None if self.bias is None else self.bias.to(x.dtype)
        return F.linear(x, self.weight().to(x.dtype), b)

    def nbytes(self):
        """what this layer really costs in memory, scales and zeros included"""
        return sum(t.numel() * t.element_size()
                   for t in (self.q, self.scale, self.zero, self.bias) if t is not None)


def fake_quant_act(x, bits=8):
    """per-token absmax quantization of the activations (each row of x gets
    its own scale). simulated: quantize -> dequantize, then a float matmul.
    numerically what an int8 kernel would see, without writing the kernel."""
    qmax = 2 ** (bits - 1) - 1
    scale = x.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8) / qmax
    return torch.round(x / scale).clamp(-qmax - 1, qmax) * scale


class W8A8Linear(QuantLinear):
    """weights AND activations in int8, the thing you need for real int8
    matmuls. the trap: transformers have a handful of hidden dimensions where
    activations are huge (|x| > 6 while the rest sits around 0.5). one of those
    in a row and the per-token scale explodes, everything else rounds to 0.

    outlier_threshold=None -> naive W8A8, eats the error.
    outlier_threshold=6.0  -> LLM.int8() trick (Dettmers et al. 2022): columns
    with an outlier go through a float matmul, the other 99.x% through int8."""

    def __init__(self, weight, bias, outlier_threshold=None):
        super().__init__(weight, bias, bits=8, granularity="channel", sym=True)
        self.threshold = outlier_threshold
        # the decomposition needs the fp16 columns of W for the outlier dims.
        # a real kernel would store only those columns; I keep the full fp16
        # copy for simplicity and DON'T count it in nbytes (see README).
        if outlier_threshold is not None:
            self.register_buffer("w_fp", weight.detach().half())
        self.outlier_frac = []

    def forward(self, x):
        w = self.weight().to(x.dtype)
        b = None if self.bias is None else self.bias.to(x.dtype)
        if self.threshold is None:
            return F.linear(fake_quant_act(x), w, b)
        flat = x.reshape(-1, x.shape[-1])
        cols = (flat.abs() > self.threshold).any(dim=0)   # input dims with an outlier somewhere
        self.outlier_frac.append(cols.float().mean().item())
        x_int = x.masked_fill(cols, 0.0)                  # int8 path: everyone but the outliers
        x_fp = x * cols                                   # float path: only the outliers
        return (F.linear(fake_quant_act(x_int), w, b)
                + F.linear(x_fp, self.w_fp.to(x.dtype)))


# ---------------------------------------------------------------- GPT-2 surgery
def conv1d_to_linear_weight(conv):
    """HF's GPT-2 uses a Conv1D that is secretly a linear layer with the
    weight stored transposed: (in, out) instead of (out, in)."""
    return conv.weight.detach().t().contiguous(), conv.bias.detach()


def quantize_gpt2(model, make_layer):
    """swap every linear layer inside the transformer blocks.
    embeddings + lm_head (tied) stay fp — standard for weight-only schemes,
    and they're a lookup table, not a matmul, on the input side.
    returns (fp16 bytes before, bytes after) for the swapped layers."""
    before = after = 0
    for block in model.transformer.h:
        for parent, name in [(block.attn, "c_attn"), (block.attn, "c_proj"),
                             (block.mlp, "c_fc"), (block.mlp, "c_proj")]:
            conv = getattr(parent, name)
            w, b = conv1d_to_linear_weight(conv)
            new = make_layer(w, b)
            before += (w.numel() + b.numel()) * 2   # fp16 baseline
            after += new.nbytes()
            setattr(parent, name, new)
    return before, after

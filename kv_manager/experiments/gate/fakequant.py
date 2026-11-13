"""Quantize -> dequantize without bit packing, for bit widths kvm.quant cannot pack yet (int3).

Same arithmetic as kvm.quant.quantize / dequantize: per-channel min/max over a block's tokens for
keys, per-token min/max over head_dim for values, fp16 scale = (max - min) / (2^bits - 1) clamped at
1e-5, fp16 zero-point = min, codes rounded and clamped to [0, 2^bits - 1]. Only the packing step
is skipped, so the dequantized values are what a packed 3-bit format would reload. Bytes for the
gate come from kvm.quant.block_bytes(bits, ...), which is exact for any bits dividing the block.

For 2 and 4 bits this gives bit-identical results to kvm.quant.roundtrip_blocks (checked in
tests/test_gate_uniform.py); the gate still calls kvm.quant for those widths.
"""

import torch

from kvm.quant import KEY_REDUCE_DIM, VALUE_REDUCE_DIM, roundtrip_blocks


def fake_quant(x: torch.Tensor, bits: int, reduce_dim: int) -> torch.Tensor:
    """quantize() followed by dequantize(), minus packing. Returns float32."""
    xf = x.float()
    mn = xf.amin(dim=reduce_dim, keepdim=True)
    mx = xf.amax(dim=reduce_dim, keepdim=True)
    levels = (1 << bits) - 1
    scale = ((mx - mn) / levels).clamp(min=1e-5).half()
    zero = mn.half()
    q = ((xf - zero.float()) / scale.float()).round().clamp(0, levels).to(torch.uint8)
    return q.float() * scale.float() + zero.float()


def roundtrip_blocks_unpacked(x: torch.Tensor, bits: int, block_tokens: int, kind: str) -> torch.Tensor:
    """Same contract as kvm.quant.roundtrip_blocks (partial tail block left unchanged)."""
    reduce_dim = {"key": KEY_REDUCE_DIM, "value": VALUE_REDUCE_DIM}[kind]
    n_tok, d = x.shape[-2], x.shape[-1]
    full = (n_tok // block_tokens) * block_tokens
    out = x.clone()
    if full:
        xb = x[..., :full, :].reshape(*x.shape[:-2], full // block_tokens, block_tokens, d)
        deq = fake_quant(xb, bits, reduce_dim).to(x.dtype)
        out[..., :full, :] = deq.reshape(*x.shape[:-2], full, d)
    return out


def roundtrip(x: torch.Tensor, bits: int, block_tokens: int, kind: str) -> torch.Tensor:
    """kvm.quant's packed format where it exists (2, 4, 8 bits), unpacked arithmetic otherwise."""
    if bits in (2, 4, 8):
        return roundtrip_blocks(x, bits, block_tokens, kind)
    return roundtrip_blocks_unpacked(x, bits, block_tokens, kind)

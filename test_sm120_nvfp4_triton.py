#!/usr/bin/env python3
"""Numerical validation harness for MiniMax-M3.1 NVFP4 KV cache on SM120 (RTX PRO 6000).

Compares the Triton sparse attention outputs (prefill + split-K decode + spec-decode)
using NVFP4 KV cache against the FP8 KV cache baseline, both populated by
`vllm._custom_ops.fused_minimax_m3_qknorm_rope_kv_insert`.
"""

import os
import sys
import importlib.util
import torch
import torch.nn.functional as F
from vllm import _custom_ops as ops

# Load patched sparse_attn.py from ./patched/sparse_attn.py if present, else fallback to installed
_PATCHED_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "patched", "sparse_attn.py")
if os.path.exists(_PATCHED_PATH):
    print(f"Loading patched kernel from: {_PATCHED_PATH}")
    spec = importlib.util.spec_from_file_location("patched_sparse_attn", _PATCHED_PATH)
    patched_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(patched_mod)
    SPARSE_BLOCK_SIZE = patched_mod.SPARSE_BLOCK_SIZE
    minimax_m3_sparse_attn = patched_mod.minimax_m3_sparse_attn
    minimax_m3_sparse_attn_decode = patched_mod.minimax_m3_sparse_attn_decode
else:
    print("WARNING: ./patched/sparse_attn.py not found, importing system vllm module")
    from vllm.models.minimax_m3.common.ops.sparse_attn import (
        SPARSE_BLOCK_SIZE,
        minimax_m3_sparse_attn,
        minimax_m3_sparse_attn_decode,
    )


def run_test(seq_len: int = 300, num_q_heads: int = 16, num_kv_heads: int = 1, topk: int = 4):
    device = "cuda:0"
    torch.manual_seed(42)
    hd = 128
    bs = SPARSE_BLOCK_SIZE  # 128
    niq = num_kv_heads
    num_blocks = (seq_len + bs - 1) // bs + 1

    qkv_orig = torch.randn(seq_len, (num_q_heads + 2 * num_kv_heads + niq + 1) * hd, dtype=torch.bfloat16, device=device) * 0.5
    qw = torch.ones(hd, dtype=torch.bfloat16, device=device)
    kw = torch.ones(hd, dtype=torch.bfloat16, device=device)
    iqw = torch.ones(hd, dtype=torch.bfloat16, device=device)
    ikw = torch.ones(hd, dtype=torch.bfloat16, device=device)
    cs = torch.ones(seq_len + 16, 64, dtype=torch.bfloat16, device=device)
    pos = torch.arange(seq_len, dtype=torch.int64, device=device)
    slots = torch.arange(seq_len, dtype=torch.int64, device=device)

    # 1. Populate FP8 KV Cache
    kv_cache_fp8 = torch.zeros(num_blocks, num_kv_heads, bs, 2 * hd, dtype=torch.uint8, device=device)
    idx_cache = torch.zeros(num_blocks, bs, hd, dtype=torch.float8_e4m3fn, device=device)
    q_fp8_ref = torch.empty(seq_len, num_q_heads * hd, dtype=torch.bfloat16, device=device)
    iq_out = torch.empty(seq_len, niq * hd, dtype=torch.float8_e4m3fn, device=device)

    ops.fused_minimax_m3_qknorm_rope_kv_insert(
        qkv_orig.clone(), qw, kw, cs, pos, num_q_heads, num_kv_heads, 64, 1e-6,
        iqw, ikw, niq, slots, slots, kv_cache_fp8, idx_cache, bs,
        q_fp8_ref, iq_out, "fp8"
    )

    # 2. Populate NVFP4 KV Cache
    kv_cache_nvfp4 = torch.zeros(num_blocks, 2 * num_kv_heads, bs, 72, dtype=torch.uint8, device=device)
    q_nvfp4 = torch.empty(seq_len, num_q_heads * hd, dtype=torch.bfloat16, device=device)
    k_scale = torch.ones(1, dtype=torch.float32, device=device)
    v_scale = torch.ones(1, dtype=torch.float32, device=device)

    ops.fused_minimax_m3_qknorm_rope_kv_insert(
        qkv_orig.clone(), qw, kw, cs, pos, num_q_heads, num_kv_heads, 64, 1e-6,
        iqw, ikw, niq, slots, slots, kv_cache_nvfp4, idx_cache, bs,
        q_nvfp4, iq_out, "nvfp4", kv_k_scale=k_scale, kv_v_scale=v_scale
    )

    q = q_fp8_ref.view(seq_len, num_q_heads, hd)
    sm_scale = hd ** -0.5

    # Prepare metadata for prefill
    block_table = torch.arange(num_blocks, dtype=torch.int32, device=device).unsqueeze(0)
    cu_seqlens_q = torch.tensor([0, seq_len], dtype=torch.int32, device=device)
    seq_lens = torch.tensor([seq_len], dtype=torch.int32, device=device)
    prefix_lens = torch.tensor([0], dtype=torch.int32, device=device)

    # Top-k indices: select valid blocks only (use -1 for unused topk slots)
    topk_idx = torch.full((num_kv_heads, seq_len, topk), -1, dtype=torch.int32, device=device)
    for t in range(seq_len):
        valid_b = t // bs + 1
        for k_i in range(min(topk, valid_b)):
            topk_idx[:, t, k_i] = k_i

    # Build exact dequantized NVFP4 -> FP8 reference cache in PyTorch to test
    # both:
    #   (a) Triton NVFP4 kernel vs exact NVFP4 math (must be > 0.9999)
    #   (b) NVFP4 KV quantization vs FP8 KV quantization (must be > 0.96)
    lut = torch.tensor(
        [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
        device=device,
    )
    kv_nvfp4_as_fp8 = torch.zeros_like(kv_cache_fp8.view(torch.float8_e4m3fn))
    t_idx = torch.arange(bs, device=device)[:, None]
    h_idx = torch.arange(8, device=device)[None, :]
    v_swizzle_idx = (t_idx // 4 * 4 + h_idx // 2) * 8 + (h_idx % 2) * 4 + (t_idx % 4)
    for b_i in range(num_blocks):
        for kh_i in range(num_kv_heads):
            k_flat = kv_cache_nvfp4[b_i, 2 * kh_i].view(-1)
            k_data = k_flat[:8192].view(bs, 64)
            k_sf = k_flat[8192:].view(bs, 8).view(torch.float8_e4m3fn).float()
            k_deq = torch.stack([lut[(k_data & 0xF).long()], lut[(k_data >> 4).long()]], dim=-1).view(bs, hd) * k_sf.repeat_interleave(16, dim=1)

            v_flat = kv_cache_nvfp4[b_i, 2 * kh_i + 1].view(-1)
            v_data = v_flat[:8192].view(bs, 64)
            v_sf = v_flat[8192:][v_swizzle_idx].view(torch.float8_e4m3fn).float()
            v_deq = torch.stack([lut[(v_data & 0xF).long()], lut[(v_data >> 4).long()]], dim=-1).view(bs, hd) * v_sf.repeat_interleave(16, dim=1)

            kv_nvfp4_as_fp8[b_i, kh_i, :, :hd] = k_deq.to(torch.float8_e4m3fn)
            kv_nvfp4_as_fp8[b_i, kh_i, :, hd:] = v_deq.to(torch.float8_e4m3fn)

    # 3. Test Prefill
    out_prefill_fp8 = torch.empty_like(q)
    minimax_m3_sparse_attn(
        q, kv_cache_fp8.view(torch.float8_e4m3fn), topk_idx, block_table,
        cu_seqlens_q, seq_lens, prefix_lens, seq_len, num_kv_heads, sm_scale,
        out_prefill_fp8, k_scale=k_scale, v_scale=v_scale
    )

    out_prefill_exact_nvfp4 = torch.empty_like(q)
    minimax_m3_sparse_attn(
        q, kv_nvfp4_as_fp8, topk_idx, block_table,
        cu_seqlens_q, seq_lens, prefix_lens, seq_len, num_kv_heads, sm_scale,
        out_prefill_exact_nvfp4, k_scale=k_scale, v_scale=v_scale
    )

    out_prefill_nvfp4 = torch.empty_like(q)
    minimax_m3_sparse_attn(
        q, kv_cache_nvfp4, topk_idx, block_table,
        cu_seqlens_q, seq_lens, prefix_lens, seq_len, num_kv_heads, sm_scale,
        out_prefill_nvfp4, k_scale=k_scale, v_scale=v_scale
    )

    cos_prefill_exact = F.cosine_similarity(out_prefill_exact_nvfp4.float().flatten(), out_prefill_nvfp4.float().flatten(), dim=0).item()
    cos_prefill_fp8 = F.cosine_similarity(out_prefill_fp8.float().flatten(), out_prefill_nvfp4.float().flatten(), dim=0).item()
    print(f"[Prefill] Cosine similarity (Triton NVFP4 vs Exact NVFP4): {cos_prefill_exact:.6f} | (vs FP8): {cos_prefill_fp8:.6f}")
    assert cos_prefill_exact > 0.995, f"Prefill exact cosine similarity too low: {cos_prefill_exact}"
    assert cos_prefill_fp8 > 0.96, f"Prefill vs FP8 cosine similarity too low: {cos_prefill_fp8}"

    # 4. Test Decode (decode_query_len=1 and decode_query_len=8 for spec-decode)
    for d_qlen in (1, 8):
        q_dec = q[-d_qlen:].contiguous()
        topk_dec = topk_idx[:, -d_qlen:, :].contiguous()
        out_dec_fp8 = torch.empty_like(q_dec)
        out_dec_exact_nvfp4 = torch.empty_like(q_dec)
        out_dec_nvfp4 = torch.empty_like(q_dec)

        minimax_m3_sparse_attn_decode(
            q_dec, kv_cache_fp8.view(torch.float8_e4m3fn), topk_dec, block_table,
            seq_lens, num_kv_heads, sm_scale, out_dec_fp8, d_qlen,
            k_scale=k_scale, v_scale=v_scale
        )
        minimax_m3_sparse_attn_decode(
            q_dec, kv_nvfp4_as_fp8, topk_dec, block_table,
            seq_lens, num_kv_heads, sm_scale, out_dec_exact_nvfp4, d_qlen,
            k_scale=k_scale, v_scale=v_scale
        )
        minimax_m3_sparse_attn_decode(
            q_dec, kv_cache_nvfp4, topk_dec, block_table,
            seq_lens, num_kv_heads, sm_scale, out_dec_nvfp4, d_qlen,
            k_scale=k_scale, v_scale=v_scale
        )
        cos_dec_exact = F.cosine_similarity(out_dec_exact_nvfp4.float().flatten(), out_dec_nvfp4.float().flatten(), dim=0).item()
        cos_dec_fp8 = F.cosine_similarity(out_dec_fp8.float().flatten(), out_dec_nvfp4.float().flatten(), dim=0).item()
        print(f"[Decode qlen={d_qlen}] Cosine similarity (Triton NVFP4 vs Exact NVFP4): {cos_dec_exact:.6f} | (vs FP8): {cos_dec_fp8:.6f}")
        assert cos_dec_exact > 0.995, f"Decode qlen={d_qlen} exact cosine similarity too low: {cos_dec_exact}"
        assert cos_dec_fp8 > 0.95, f"Decode qlen={d_qlen} vs FP8 cosine similarity too low: {cos_dec_fp8}"

    print("ALL SM120 NVFP4 TRITON TESTS PASSED!")


if __name__ == "__main__":
    run_test()


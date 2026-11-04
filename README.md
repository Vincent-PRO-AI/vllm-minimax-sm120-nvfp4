# MiniMax-M3.1-Flash — SM120 Blackwell (`sm_120`) NVFP4 KV Cache Triton Backend

Hardware-accelerated Triton sparse-attention KV cache reader enabling `--kv-cache-dtype nvfp4` for **MiniMax-M3.1-Flash** on NVIDIA **SM120 Blackwell** GPUs (RTX PRO 6000 Blackwell Server Edition 96 GB, RTX 5090), where `tcgen05` and TMEM (`SM100` MSA backend) are not present.

- **Base Container Image**: `vllm/vllm-openai:minimax-m31-flash`
- **Verified Image Digest**: `docker.io/vllm/vllm-openai@sha256:86f582dae25ea92802353727164ae4eb03b37b18baf9119b805ffaec2a7ef07c`

---

## 1. Architecture & Page Layout

- **KV Writer (`fused_minimax_m3_qknorm_rope_kv_insert`)**: Compiled for `SM80+` in `vllm/_custom_ops.py`. Writes `nvfp4` pages natively on `sm_120`.
- **Page Shape**: `[num_blocks, 2 * num_kv_heads, 128, 72]` (`uint8`), `kv_head_major_slots=True`, `KVCacheLayout.LBHNC`:
  - Slot `2 * h`: **K** slot (`128 tokens × 72 bytes = 9216 bytes`)
  - Slot `2 * h + 1`: **V** slot (`128 tokens × 72 bytes = 9216 bytes`)
  - Bytes `[0 .. 8191]` (`128 × 64` contiguous bytes, row stride `64`): packed `E2M1` nibbles (`packed & 0xF` = even dim `2d`, `packed >> 4` = odd dim `2d+1`).
  - Bytes `[8192 .. 9215]` (`1024` bytes): `float8_e4m3fn` 16-element block scales (`8` scales per token for `head_dim=128`):
    - **K scales** (`V_SWIZZLE=False`): linear `8192 + t * 8 + (d // 16)`
    - **V scales** (`V_SWIZZLE=True`): 4-token per 32-byte swizzle `8192 + (t // 4 * 4 + h_grp // 2) * 8 + (h_grp % 2) * 4 + (t % 4)`
- **Hardware SIMD PTX Dequantization (`_e2m1x2_scale_to_f32x2_ptx`)**:
  Uses native Blackwell SM100/SM120 PTX instructions:
  - `cvt.rn.f16x2.e2m1x2` (unpacks 2 `E2M1` nibbles to `half2` in 1 instruction)
  - `cvt.rn.f16x2.e4m3x2` (converts duplicated `E4M3` block scale to `half2`)
  - `mul.rn.f16x2` (SIMD `half2` product before widening to `fp32`)

---

## 2. Controlled A/B Benchmark (`4× NVIDIA RTX PRO 6000 Blackwell SE 96 GB`, `sm_120`)

Strict A/B on the same pod and node (`TP=4`, `EP=4`, `block-size 128`, `gmu 0.95`, `max-model-len 524288`, `max-num-seqs 256`, `max-num-batched-tokens 32768`, DSpark MTP 7 tokens in `fp8`, `--attention-config '{"indexer_kv_dtype":"fp8"}'`), where only `--kv-cache-dtype` (`fp8` vs `nvfp4`) changes:

### A. End-to-End Capacity, Throughput & TTFT (`vllm bench serve`)
| Metric | FP8 KV Cache | NVFP4 KV Cache | Delta |
| :--- | :--- | :--- | :--- |
| **Main KV Bytes / Token / Head** | `256 B` | `144 B` | **-43.75%** |
| **KV Pool Capacity** | `803,712 tokens` (`1.53×` @ 512k) | **`1,110,400 tokens` (`2.12×` @ 512k)** | **+38.2%** |
| **C1 (512/128) Output Throughput** | `84.0 tok/s` | `82.0 tok/s` | `-2.4%` |
| **C8 (512/128) Output Throughput** | `223.3 tok/s` | `221.4 tok/s` | `-0.9%` |
| **C16 (512/128) Output Throughput** | `372.2 tok/s` | `349.7 tok/s` | `-6.0%` |
| **Median ITL @ C16** | `60.5 ms` | `64.2 ms` | `+6.1%` |
| **Prefill 32k TTFT** | `8.36 s` | **`6.15 s`** | **-26.4% (faster)** |
| **Prefill 128k TTFT** | `32.62 s` | **`25.60 s`** | **-21.5% (faster)** |

> **Why Long-Context Prefill is 21–26% Faster**: Sparse attention is memory-bandwidth bound on KV reads at long context. Reducing the main KV footprint from `256 B/tok` to `144 B/tok` (`-43.75%` memory traffic) outweighs the E2M1 dequantization cost as sequence length grows, while short decode pays only `2–6%`.

### B. Topology Constraints (`TP=4` Verified, `TP=8` Excluded Upstream)
| Topology | Status | Reason |
| :--- | :--- | :--- |
| **TP=4 (`rtx_pro_6000_4x`)** | **Verified** | Each rank holds `16` Q heads over `1` KV head, satisfying the NVFP4 attention kernel constraint. |
| **TP=8 + `nvfp4`** | **Excluded (Upstream Gate)** | Refused by vLLM upstream: `each rank holds 8 query heads over 1 KV head(s), and the NVFP4 attention kernels need 16 per KV head`. Use `--kv-cache-dtype fp8` for `TP=8`. |
| **TP=4 + DP=2 (8 GPUs)** | **Not Recommended** | Boots cleanly, but `128k` prefill TTFT degrades to `45.9 s` (vs `25.6 s` in pure `TP=4`) with lower `C16` throughput (`324.8 tok/s`). |

### C. Numerical Precision (`test_sm120_nvfp4_triton.py`)
| Kernel Path | Cosine vs Exact PyTorch NVFP4 | Cosine vs FP8 KV Cache |
| :--- | :--- | :--- |
| **Prefill (`_gqa_sparse_fwd_kernel`)** | **0.997738** (`0.9999987` in FP32) | **0.966431** |
| **Decode `qlen=1` (`_gqa_sparse_decode_kernel`)** | **0.997195** | **0.956308** |
| **Decode `qlen=8` (MTP / Spec-Decode)** | **0.995720** | **0.959066** |

### D. Kernel Microbenchmark (`seq_len=16384`, `topk=64` blocks = 8192 attended tokens)
| Kernel | FP8 Baseline | NVFP4 Software ALU | NVFP4 SIMD PTX (`cvt.rn.f16x2.e2m1x2`) | PTX Speedup vs ALU |
| :--- | :--- | :--- | :--- | :--- |
| **Prefill (`q=2048`)** | `3.822 ms` | `16.547 ms` | **`11.238 ms`** | **1.47×** (`-32.1%` latency) |
| **Decode (`B=16, q=1`)** | `0.063 ms` | `0.218 ms` | **`0.154 ms`** | **1.41×** (`-29.4%` latency) |

---

## 3. Repository Structure

- [`patched/sparse_attention.py`](patched/sparse_attention.py): Backend wiring (`MiniMaxM3SparseBackend` & `MiniMaxM3SparseTritonImpl` support for `kv_cache_dtype="nvfp4"`, `LBHNC` layout, `2 * num_kv_heads` slots of `72` bytes).
- [`patched/sparse_attn.py`](patched/sparse_attn.py): Triton sparse prefill & split-K decode kernels with `_e2m1x2_scale_to_f32x2_ptx` and `_load_nvfp4_tile`.
- [`patched/PATCH.diff`](patched/PATCH.diff): Byte-exact unified diff verified in CI against `orig/`.
- [`recipes/minimax-m3.1-flash-rtx_pro_6000_4x.yaml`](recipes/minimax-m3.1-flash-rtx_pro_6000_4x.yaml): Verified `rtx_pro_6000_4x` recipe configuration.
- [`test_sm120_nvfp4_triton.py`](test_sm120_nvfp4_triton.py): Standalone numerical & shape verification harness.

---

## 4. How to Apply in vLLM Container (`vllm/vllm-openai:minimax-m31-flash`)

```bash
cp patched/sparse_attention.py \
   /usr/local/lib/python3.12/dist-packages/vllm/models/minimax_m3/common/sparse_attention.py
cp patched/sparse_attn.py \
   /usr/local/lib/python3.12/dist-packages/vllm/models/minimax_m3/common/ops/sparse_attn.py
```

Then pass `--kv-cache-dtype nvfp4 --attention-config '{"indexer_kv_dtype":"fp8"}'` to `vllm serve`.

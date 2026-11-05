# MiniMax-M3.1-Flash — SM120 Blackwell (`sm_120`) NVFP4 KV Cache Triton Backend

Triton sparse-attention KV cache reader enabling `--kv-cache-dtype nvfp4` for **MiniMax-M3.1-Flash** on NVIDIA **SM120 Blackwell** GPUs (RTX PRO 6000 Blackwell Server Edition 96 GB, RTX 5090), where `tcgen05` and TMEM (`SM100` MSA backend) are not present.

- **Base container image**: `vllm/vllm-openai:minimax-m31-flash` (vLLM `v0.1.dev23341+g8744dd2fb`)
- **Verified image digest**: `docker.io/vllm/vllm-openai@sha256:86f582dae25ea92802353727164ae4eb03b37b18baf9119b805ffaec2a7ef07c`

---

## 1. Architecture & page layout

- **KV writer (`fused_minimax_m3_qknorm_rope_kv_insert`)**: already compiled for `SM80+` in `vllm/_custom_ops.py`; it writes `nvfp4` pages natively on `sm_120`. **This patch adds no CUDA code** — only the Triton reader and the backend wiring.
- **Page shape**: `[num_blocks, 2 * num_kv_heads, 128, 72]` (`uint8`), `kv_head_major_slots=True`, `KVCacheLayout.LBHNC`:
  - Slot `2 * h`: **K** slot (`128 tokens × 72 bytes = 9216 bytes`)
  - Slot `2 * h + 1`: **V** slot (`128 tokens × 72 bytes = 9216 bytes`)
  - Bytes `[0 .. 8191]` (`128 × 64` contiguous bytes, row stride `64`): packed `E2M1` nibbles (`packed & 0xF` = even dim `2d`, `packed >> 4` = odd dim `2d+1`).
  - Bytes `[8192 .. 9215]` (`1024` bytes): `float8_e4m3fn` 16-element block scales (`8` scales per token for `head_dim=128`):
    - **K scales** (`V_SWIZZLE=False`): linear `8192 + t * 8 + (d // 16)`
    - **V scales** (`V_SWIZZLE=True`): 4-token per 32-byte swizzle `8192 + (t // 4 * 4 + h_grp // 2) * 8 + (h_grp % 2) * 4 + (t % 4)`
- **Hardware SIMD PTX dequantization (`_e2m1x2_scale_to_f32x2_ptx`)**:
  - `cvt.rn.f16x2.e2m1x2` (unpacks 2 `E2M1` nibbles to `half2` in one instruction)
  - `cvt.rn.f16x2.e4m3x2` (converts duplicated `E4M3` block scale to `half2`)
  - `mul.rn.f16x2` (SIMD `half2` product before widening to `fp32`)

---

## 2. Controlled A/B benchmark (`4× RTX PRO 6000 Blackwell SE 96 GB`, `sm_120`)

Same pod, same node, same flags (`TP=4`, `EP=4`, `block-size 128`, `gmu 0.95`, `max-model-len 524288`, `max-num-seqs 256`, `max-num-batched-tokens 32768`, DSpark MTP 7 tokens in `fp8`, `--attention-config '{"indexer_kv_dtype":"fp8"}'`); **only `--kv-cache-dtype` changes**. Tool: `vllm bench serve`, random dataset, `seed 1000+input_len`, `--ignore-eos`, one measurement per point, **LMCache disabled** (see §3).

| Metric | FP8 KV cache | NVFP4 KV cache | Delta |
| :--- | :--- | :--- | :--- |
| **Main KV bytes / token / head** | `256 B` | `144 B` | **-43.75%** |
| **KV pool capacity** | `803,712 tokens` (`1.53×` @ 512k) | **`1,110,528 tokens` (`2.12×` @ 512k)** | **+38.2%** |
| C1 (512/128) output | `79.8 tok/s` | `78.1 tok/s` | `-2.1%` |
| C8 (512/128) output | `221.5 tok/s` | **`230.2 tok/s`** | `+3.9%` |
| C16 (512/128) output | `344.0 tok/s` | **`367.3 tok/s`** | `+6.8%` |
| Median ITL @ C8 / C16 | `54.5 / 67.2 ms` | **`52.8 / 62.7 ms`** | `-3.1% / -6.7%` |
| 8k in / 64 out — output | `36.4 tok/s` | **`41.2 tok/s`** | **+13.2%** |
| 8k in — TTFT | `5.96 s` | **`3.93 s`** | **-34.1%** |
| Prefill 32k — TTFT | `6.05 s` | **`5.45 s`** | `-10.0%` |
| Prefill 128k — TTFT | `24.79 s` | **`22.06 s`** | `-11.0%` |

NVFP4 wins everywhere except C1 (-2.1%), and the gain grows with context: cutting the KV footprint from `256 B` to `144 B` per token reduces the memory traffic that dominates sparse attention, which outweighs the E2M1 dequantization cost.

> **Correction vs an earlier revision of this README.** It reported `-6.0%` at C16 and `-21…-26%` on prefill, measured with LMCache enabled. That A/B was not symmetric: LMCache stores FP8 pages but fails on the 72-byte NVFP4 pages, so only the NVFP4 leg paid the failure cost (and the engine could die — §3). With LMCache disabled on both legs, the numbers above are the comparable ones.

### 2.1 Topology: `TP=4` verified, `TP=8` measurable

| Topology | Status | Numbers |
| :--- | :--- | :--- |
| **TP=4 + EP=4** (`rtx_pro_6000_4x`) | **Verified** | `1,110,528` tokens pool, C16 `367.3 tok/s`, 128k TTFT `22.06 s` |
| **TP=8 + EP=8** | **Boots; needs a one-line gate move upstream** | `2,900,992` tokens pool (`5.53×` @ 512k, **2.77 concurrent 1M-token requests**), C16 `376.5 tok/s` (+2.5%), 8k `31.0 tok/s`, 128k TTFT `30.88 s` (+40%) |
| **TP=4 + DP=2** (8 GPUs, single engine) | Not recommended | C16 `324.8 tok/s`, 128k TTFT `45.9 s` (vs `22.06 s` at TP4) |

`TP=8` is currently refused by a **model-level** guard in `vllm/model_executor/models/config.py` (`MiniMaxM3SparseConfig.verify_and_update_config`), whose own comment states it exists for the SM100 MSA CUTLASS Q8KV4 kernel (16 query heads per KV head per rank). The Triton path does not carry that constraint: the numerical harness passes with `gqa_group_size=8` (§2.2). Relaxing it belongs in the MSA backend selector, not in the model config — a separate change, measured here with a canary-only patch.

`TP=8` buys capacity, not speed: twice the pool for twice the GPUs, but +40% TTFT on long prefill and only +2.5% at C16. For throughput on 8 GPUs, two `TP=4` replicas behind a router is the better shape (hypothesis, not measured here).

### 2.2 Numerical precision (`test_sm120_nvfp4_triton.py`)

Cosine similarity against an exact PyTorch NVFP4 dequantization:

| Kernel path | `gqa_group_size=16` (TP=4) | `gqa_group_size=8` (TP=8) |
| :--- | :--- | :--- |
| **Prefill** (`_gqa_sparse_fwd_kernel`) | **0.997738** (`0.9999987` in FP32) | **0.997692** |
| **Decode `qlen=1`** (`_gqa_sparse_decode_kernel`) | **0.997195** | **0.996711** |
| **Decode `qlen=8`** (MTP / spec-decode) | **0.995720** | **0.997168** |

### 2.3 Kernel microbenchmark (`seq_len=16384`, `topk=64` blocks = 8192 attended tokens)

| Kernel | FP8 baseline | NVFP4 software ALU | NVFP4 SIMD PTX (`cvt.rn.f16x2.e2m1x2`) | PTX speedup vs ALU |
| :--- | :--- | :--- | :--- | :--- |
| **Prefill (`q=2048`)** | `3.822 ms` | `16.547 ms` | **`11.238 ms`** | **1.47×** (`-32.1%` latency) |
| **Decode (`B=16, q=1`)** | `0.063 ms` | `0.218 ms` | **`0.154 ms`** | **1.41×** (`-29.4%` latency) |

---

## 3. Production caveat: LMCache cannot store NVFP4 pages

With the `LMCacheMPConnector` (`--kv-transfer-config`), the D2H store path cannot copy the 72-byte NVFP4 pages:

```
lmcache/v1/multiprocess/modules/lmcache_driven_transfer.py:781 store_with_chunk_mask
  → object_group_transfer.py:770 lmcache_memcpy_async_d2h
  → torch_ops/mem_kernels.py:616  RuntimeError: cudaMemcpy failed with error code 1
```

Measured threshold (`--chunk-size 1024`): `0` errors at 512-token prompts (no store attempted), `15` errors at 2048-token prompts. Under concurrency the failure escalates and kills the engine:

```
(EngineCore) ERROR core.py:1545
  scheduler.py:3354 _update_from_kv_xfer_finished
    assert req_id in self.requests   →  AssertionError  →  EngineDeadError
```

Consequences: no prefix caching, and possible engine death. Options: disable LMCache for NVFP4 profiles, fix the alignment in LMCache, or harden the KV connector in vLLM (an assertion on a late completion event is a robustness defect independent of NVFP4).

---

## 4. Repository structure

- [`patched/sparse_attention.py`](patched/sparse_attention.py): backend wiring (`MiniMaxM3SparseBackend` & `MiniMaxM3SparseTritonImpl` support for `kv_cache_dtype="nvfp4"`, `LBHNC` layout, `2 * num_kv_heads` slots of `72` bytes).
- [`patched/sparse_attn.py`](patched/sparse_attn.py): Triton sparse prefill & split-K decode kernels with `_e2m1x2_scale_to_f32x2_ptx` and `_load_nvfp4_tile`.
- [`patched/PATCH.diff`](patched/PATCH.diff): unified diff against the two upstream files; CI applies it to the pinned vLLM revision and compares the result byte-for-byte with `patched/`.
- [`recipes/minimax-m3.1-flash-rtx_pro_6000_4x.yaml`](recipes/minimax-m3.1-flash-rtx_pro_6000_4x.yaml): verified `rtx_pro_6000_4x` recipe configuration.
- [`test_sm120_nvfp4_triton.py`](test_sm120_nvfp4_triton.py): standalone numerical and shape harness (loads `./patched/sparse_attn.py`, ~50 MB of VRAM).

### Validated artifacts

| File | SHA-256 (LF, no BOM) |
| :--- | :--- |
| `patched/sparse_attn.py` | `30aea01a924750de9876d4260773734b2fce5a33816e517f9a8dd353aec44d4f` |
| `patched/sparse_attention.py` | `62a5abb02777aa46fbe2acc5a2276c95babcc6fe831663b820496f335715edc2` |
| `patched/PATCH.diff` | `6fe5cbe3eaee009473b87ca3621b47906ace7fc5796d187b4bb2bb605928940b` |
| `test_sm120_nvfp4_triton.py` | `9a99c48e4d7ae6ba4ddddcdb0dfeef6038f50ed360cf25c3845e8fd4af857a5c` |

---

## 5. How to apply in the vLLM container

```bash
cp patched/sparse_attention.py \
   /usr/local/lib/python3.12/dist-packages/vllm/models/minimax_m3/common/sparse_attention.py
cp patched/sparse_attn.py \
   /usr/local/lib/python3.12/dist-packages/vllm/models/minimax_m3/common/ops/sparse_attn.py

# fail fast if the container is not the verified build
sha256sum /usr/local/lib/python3.12/dist-packages/vllm/models/minimax_m3/common/ops/sparse_attn.py
```

Then pass `--kv-cache-dtype nvfp4 --attention-config '{"indexer_kv_dtype":"fp8"}'` to `vllm serve`.

> These files are LF-only and tied to a specific vLLM build: this is a source patch, not a configuration knob. Re-apply and re-run the harness after any vLLM upgrade.

# Training Decisions — Coding / Security SLM

Living document for architecture and training choices.
Goal: match or beat similar-parameter open models by following current lab practice.

Corpus: `Aniket200325/coder-pretrain-60gb` (~59 GB raw text, finalized).  
Post-pretrain path: **SFT** for PR review / security / automation agents (separate dataset; not this corpus alone).

**Compute envelope (planning):** ~**$1,000 GCP credits** (primary multi-GPU) + ~**$750 CloudRift** (overflow / cheap 4090 hours) + Colab for bring-up + ~**$25 Runpod** emergency. Prefer **multi A100 or H100 on GCP** when quota/availability allows.

---

## 1. Tokenizer (DECIDED)

### Choice
- **Reuse** the **Qwen2.5-Coder** tokenizer (byte-level BPE / BBPE).
- **Do not** train a custom tokenizer on the 60 GB corpus (labs trained theirs on far more data).
- **Do not** use DeepSeek’s tokenizer, even though the model architecture will follow DeepSeek.

### Why
- Large proven BBPE vocab (~**151,646** tokens) matches current lab practice (DeepSeek ~128K, Llama 3 ~128K, Qwen ~152K).
- Coding-oriented merges + built-in **FIM** and repo/file special tokens.
- Same tokenizer family used from Qwen2.5-Coder **0.5B → 32B**.
- Tokenizer and architecture are independent: Qwen tokenizer + DeepSeek MLA/MoE is a valid mix.

### Hub / load
```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(
    "Qwen/Qwen2.5-Coder-7B",  # any Qwen2.5-Coder size; same vocab family
    trust_remote_code=True,
)
```
Verify Apache-2.0 (or current card license) before commercial redistribution of the tokenizer files.

### Embedding + LM head (locked to tokenizer)
- `vocab_size` / embedding matrix / LM head **must match Qwen’s vocab** (~151,646), **not** DeepSeek’s 129,280.
- **Tie input embeddings and LM head** (≤ ~3B active class), same practice as Qwen2.5-Coder small sizes, so the large vocab does not dominate the parameter budget.
- Untied embeddings only if we later target larger dense sizes where Qwen also unties.

### Special tokens
- Use **Qwen’s** special-token set and IDs (FIM, eos/pad, chat/control tokens as present in the tokenizer).
- Plan pretrain/SFT formatting around Qwen FIM tokens when we add fill-in-the-middle:
  - `<|fim_prefix|>`, `<|fim_middle|>`, `<|fim_suffix|>`, `<|fim_pad|>`
  - `<|repo_name|>`, `<|file_sep|>` (if doing repo-level packing later)
- Architecture remains DeepSeek-style MLA + RoPE + MoE; only token *IDs* come from Qwen.

### BOS / EOS / PAD
- **Do not** copy DeepSeek `bos_token_id` / `eos_token_id` from their configs.
- Use whatever IDs **`tokenizer.bos_token_id`**, **`tokenizer.eos_token_id`**, **`tokenizer.pad_token_id`** (and related) define for Qwen2.5-Coder.
- Config JSON for our model must be generated from the loaded Qwen tokenizer, not from a DeepSeek config template.

### Retune note (when training starts)
DeepSeek’s token-batch / LR schedules assume *their* tokenizer lengths. With Qwen’s tokenizer, the same raw text yields different token counts — retune **tokens per batch**, warmup, and context length in **tokens**, not characters.

### Explicitly rejected (for this project)
| Option | Reason |
| --- | --- |
| Custom BBPE trained only on our 60 GB | Weaker merges than lab tokenizers |
| DeepSeek-V3/V4 tokenizer | Less coding/FIM-oriented than Qwen2.5-Coder for our mix |
| StarCoder2 ~49K / DeepSeek-Coder-V1 ~32K | Behind current large-vocab lab practice |
| WordPiece / Unigram-as-default | Not what decoder coding SLMs ship |

### Status
**LOCKED** — 2026-07-11.

---

## 2. Architecture — MLA + RoPE + KV cache (DECIDED)

### Choice
- Use **DeepSeek-V3-style Multi-head Latent Attention (MLA)** with **decoupled RoPE** for the full model lifetime.
- **Do not** switch attention type between phases (no “GQA in Phase A, MLA later”).
- **Do not** adopt DeepSeek-**V4** hybrid attention (CSA/HCA + sliding window) for v1 — that stack targets ~1M context and is much harder to implement/debug.
- **Product target context:** **128K** tokens for coding/security (multi-file / small–medium repos). Achieved via a **staged continue-pretrain ladder**, not by pretraining natively at 128K from step 0.
- Pair long context with **agentic file tools / retrieval** for huge monorepos; 128K alone is not “entire arbitrary codebase.”
- **256K / 300K** is **deferred** (optional later CPT if more compute appears); not a v1 product requirement.

### Why MLA + decoupled RoPE (V3, not V4)
- Lab-proven on DeepSeek-V2/V3; strong quality vs much smaller KV than full MHA.
- Decoupled RoPE keeps rotary on a small Q/K slice so low-rank KV compression stays compatible with RoPE.
- Fits coding/security workloads that need long windows without jumping to V4’s million-token machinery on day one.

### Exact MLA shape (locked for ~3B-active MoE)
| Knob | Value |
| --- | --- |
| `d_model` | **2048** |
| `n_layers` | **28** |
| `n_heads` (MLA) | **16** |
| `kv_lora_rank` | **512** |
| `q_lora_rank` | **0** (no Q compression in v1 — fewer moving parts) |
| `qk_rope_head_dim` | **64** |
| `qk_nope_head_dim` | **128** |
| `v_head_dim` | **128** |
| RMSNorm on compressed latents | **Yes** (V3-style) |
| RoPE θ (Phase A) | **500_000** (Qwen-like; easier 32K/128K extension) |
| YaRN / NTK-style scale | Applied in **Phase B** CPT only |

### Long-context training ladder (not automatic)
Long context does **not** emerge from short pretrain alone. Same MLA+RoPE weights are continued with higher `max_seq_len` and RoPE scaling:

| Phase | Context | Role |
| --- | --- | --- |
| **A — Main pretrain** | **8K** (4K if budget-tight) | Learn code/language; **MLA + decoupled RoPE already active** |
| **B1 — CPT** | **32K** | First long-context adaptation + RoPE scale |
| **B2 — CPT** | **128K** | **Product target** (repo-scale window) |
| **C — (optional)** | ≤ **128K** | Long-context SFT / agent finetune so the model *uses* the window |
| **Later (deferred)** | 256K–300K | Only if extra compute + clear product need |

Within a phase, batches may mix shorter sequences; capability still requires enough training at (or packed toward) the phase max length.

### KV cache (MLA) — how it works across phases
**Inference / decode cache (what “KV cache” usually means):**
- Per layer, cache mainly compressed latent **`c_KV`** (≈ `kv_lora_rank` per token) plus the small **RoPE key** slice — not full per-head K/V.
- Memory grows with sequence length \(L\), but much more slowly than MHA: roughly \(\propto L \times d_c\) (+ small RoPE dim).
- **Cache format is fixed by MLA** for the whole project (Phase A → B → C → serve).
- **Cache capacity** (`max_seq_len` allocation) grows when we raise the target context (up to **128K** for v1 product).
- **RoPE scale / θ** are updated in B/C so later positions are meaningful; weights continue from the previous checkpoint.

**Training:**
- Teacher-forced pretrain/CPT usually does **not** keep a persistent decode KV cache across steps; each step runs attention over the current sequence (plus kernels / checkpointing / parallelism as needed).
- Raising context in B increases activation memory and attention FLOPs; it does **not** “carry forward” a Phase-A cache into a 128K cache.

```text
Architecture (fixed for all phases):  MLA + decoupled RoPE
Phase A:   train at 8K
Phase B:   continue same arch at 32K → 128K (+ RoPE scale)
Inference: allocate MLA KV up to 128K (v1 product)
```

### Explicitly rejected (for v1)
| Option | Reason |
| --- | --- |
| Native pretrain at 128K from step 0 | Prohibitively expensive / sample-inefficient at ~3B active |
| Product target 256K–300K for v1 | Scaled down to **128K** to fit compute; 256K deferred |
| DeepSeek-V4 CSA/HCA (+ mHC) as day-1 attention | Built for ~1M agents; high implementation risk |
| Plain GQA instead of MLA | Simpler, but drops the main DeepSeek KV efficiency lever |
| Apply RoPE to full compressed K without decoupling | Breaks MLA↔RoPE compatibility |
| Rely on context alone for huge monorepos | Still need tools / indexing beyond 128K |

### Status
**LOCKED** (structure + numeric MLA shape + **128K** product ladder + KV policy) — 2026-07-11.

---

## 3. MoE (DECIDED)

### Choice
- Use **DeepSeekMoE** (shared + fine-grained routed experts), scaled like **DeepSeek-V2-Lite / Coder-V2-Lite**, **not** DeepSeek-V3’s 256-expert cluster config.
- **Target size class:** **~3B active parameters** (aim at the **high end of Lite**, not denser 5B-active), with **~10–16B total parameters** on disk (capacity via experts; compare to dense models by **active** FLOPs).
- Prefer **slightly fewer active params under 3B** over overshooting — e.g. **~2.4–2.8B active** is acceptable if needed to fit memory/batch; do **not** push toward 5B active on this corpus/budget.

### Exact MoE shape (locked)
| Knob | Value |
| --- | --- |
| Shared experts | **2** (always on) |
| Routed experts | **64** |
| Top‑k (routed) | **6** per token |
| Dense FFN intermediate (for would-be dense / first layers) | **11008** (SwiGLU) |
| Expert FFN intermediate | **1408** (~½–⅔ of dense half-width accounting; tune ±128 if needed to hit ~2.4–2.8B active) |
| Dense vs MoE layers | First **1** layer **dense FFN**; layers **2–28** **MoE** |
| Tied embeddings | **Yes** (Qwen vocab) |

Note: MoE uses **experts**, not attention “heads.” MLA head counts stay in §2.

### Routing & load balancing (locked)
- Affinity: start with either **softmax** over experts or **V3-style sigmoid + normalize** among the selected top‑k (implementation may pick one; both are acceptable).
- **v1 training:** use an **auxiliary load-balancing loss** (`α ≈ 0.01`) so experts stay utilized on our relatively small corpus.
- **Also support / plan for bias-based balancing** (V3-style per-expert bias on selection scores, optionally aux-loss-light or aux-loss-free later) once the aux-loss baseline is stable.
- Gating weights for combining expert outputs should follow the paper pattern we implement (biased selection must not silently corrupt gate magnitudes).
- Router logits: init near **0**; keep router compute in **FP32**.

### Why this shape (not V3 256 / top‑8)
- V2-Lite is the published DeepSeek MoE config closest to **~2.4B active**; we scale width/depth to sit in the **~2.4–3.0B active** band.
- **64 routed experts** fit a ~3B-active train; **256** risks many undertrained experts on a ~60 GB pretrain corpus.
- Shared experts carry common syntax/code priors; routed experts specialize (languages, security patterns, etc.).

### Explicitly rejected (for v1)
| Option | Reason |
| --- | --- |
| V3-like **256 routed / top‑8** | Cluster-scale; overkill; expert starvation risk on our data |
| Mixtral-only (no shared experts) without DeepSeek shared+routed design | Diverges from chosen lab recipe |
| All-MoE from layer 0 | Labs keep early dense FFNs; we follow that |
| Dense-only as the primary architecture | Acceptable as a later ablation, not the main line |
| ~5B active MoE on this budget/corpus | Undertrains; prefer **≤3B active** + more tokens |

### Status
**LOCKED** (MoE topology + **~3B active** target + numeric expert sizes + balancing) — 2026-07-11.

---

## 4. Mixed precision & training systems (DECIDED)

Phased stack: **v1 = robust BF16 baseline** (always on); **v2 = DeepSeek-style FP8 GEMM path** once v1 is stable and Hopper-class (or equivalent) FP8 Tensor Cores are available (**H100 on GCP** enables v2 experiments; A100 stays BF16).

### 4.1 Mixed precision — F_prop / grads / updates
| Mode | Policy |
| --- | --- |
| **v1 (default)** | **BF16** for almost all compute and activations (or **TF32** matmul paths on Ampere if BF16 Tensor Cores are unavailable). Keep **FP32** for loss, LayerNorm/RMSNorm, and **MoE router** (and other numerically sensitive ops) when needed for stability. |
| **v2 (FP8 path)** | Run heavy **Linear / MoE expert GEMMs** in FP8 once a BF16 baseline looks healthy **and** H100 (or better) is available; keep master weights / sensitive ops in higher precision as required by the implementation. |

Optimizer-state dtype and exact F_prop vs D_grad vs W_grad splits for FP8 follow the chosen framework (e.g. Transformer Engine / custom kernels); do not invent a custom split until BF16 MoE+MLA training is green.

### 4.2 Fine-grained quantization (FP8 / v2)
- Copy **DeepSeek-V3 fine-grained grouping**:
  - Activations: **1×128** tiles (per-token × 128 channels)
  - Weights: **128×128** blocks
- Used with the FP8 path to control outliers; **not required** for pure BF16 v1.

### 4.3 Accumulation precision
- Prefer GEMM kernels that support **FP32 accumulation** (promoted / CUDA-core-side accum where applicable), especially for FP8 MMA.
- Avoid “fast FP8-only Tensor Core accum” for pretraining unless validated as lossless enough.

### 4.4 Online quantization (FP8 / v2)
- Use **online** fine-grained scales: compute amax/scale **per tile/block each step**.
- Prefer online over delayed/history tensor-wise scaling (DeepSeek practice).

### 4.5 FP8 format — mantissa over exponent (FP8 / v2)
- Default **E4M3** for weights and activations in GEMMs, with **per-block / per-tile scales** to cover dynamic range.
- Introduce E5M2 only where range truly requires it (e.g. some gradients), if the stack supports hybrid FP8.

### 4.6 Flash-Attention
- **On from day one** (Phase A and all later phases).
- Use FlashAttention-2/3 (or vendor equivalent) with an **MLA-compatible** attention implementation.
- Independent of FP8; part of the v1 baseline.

### 4.7 Gradient checkpointing
- **On from day one** to cut activation memory (MoE + long context).
- Prefer selective checkpointing of expensive blocks (attention / MoE) when tunable; otherwise full-block checkpointing is acceptable for v1.

### 4.8 Sequence parallelism
- **Important for 32K–128K CPT**; enables fitting long sequences across devices.
- **Auto-enable when multi-GPU is detected** (training entrypoint should turn SP on if `world_size > 1` and the parallel plan allows it).
- **Single-GPU:** design the trainer so SP is a no-op / disabled cleanly; rely on Flash-Attention + gradient checkpointing (+ microbatching / shorter packs as needed). Document that full **128K** on one GPU may still be memory-bound without multi-GPU SP.
- Do **not** require DeepSeek DualPipe for v1; FSDP/ZeRO (+ expert parallel later if needed) is enough at Lite scale.

### Hardware note
- Full FP8 fine-grained training assumes **Hopper (H100)-class** (or better) Tensor Cores and a mature FP8 stack.
- On **A100 / RTX 4090**: stay on **v1 BF16/TF32 + Flash-Attention + checkpointing (+ SP when multi-GPU)**.
- **V100:** avoid for main train (poor BF16 path).

### Explicitly rejected / deferred
| Item | Stance |
| --- | --- |
| Day-1 full DeepSeek FP8 for all tensors | Deferred to **v2** after BF16 baseline (prefer H100) |
| Delayed tensor-wise FP8 scaling as primary | Prefer **online fine-grained** |
| DualPipe / custom EP all-to-all as v1 requirement | Optional later; not blocking |
| Training without Flash-Attention or checkpointing | Rejected |

### Status
**LOCKED** (v1 systems + v2 FP8 policy) — 2026-07-11.

---

## 5. Optimizer, LR, batch tokens, MTP (DECIDED)

Scaled for Lite-like MoE (~2.4–3.0B active) and our ~60 GB corpus. Follow DeepSeek-V2-Lite / V3 where transferable; do not copy V3’s multi-trillion / 15K-sequence batch sizes literally.

### 5.1 Optimizer
- **AdamW** with β₁=**0.9**, β₂=**0.95**, **weight_decay=0.1**, gradient clip norm **1.0** (same family as DeepSeek-V2-Lite / V3).
- Prefer fused AdamW when available.
- **Muon** (DeepSeek-V4) is optional **v2** experiment after AdamW is stable — not required for v1.

### 5.2 Learning rate
| Phase | Policy |
| --- | --- |
| **A — Main pretrain** | Linear warmup **2000 steps** → peak **3e-4** → **cosine decay to 10% of peak** (**3e-5**) |
| **B/C — Long-context CPT** | Peak ≈ **0.1–0.3×** Phase-A peak (e.g. **3e-5 – 1e-4**) so extension does not wreck the base |

Rationale: peak sits between Lite (**4.2e-4**) and V3 (**2.2e-4**). Cosine is the locked default; Lite-style step decay (×0.316 at ~80% / ~90% tokens) remains an acceptable alternative if we want stricter Lite parity later.

### 5.3 Batch tokens
- Optimize for **global tokens per optimizer step**, not sequence-count alone.
- **Phase A (@ 8K):** target **0.5M–2M** global tokens/step; with multi A100/H100 prefer toward **1M–2M**. Achieve via microbatch × devices × **gradient accumulation**.
- Optional early **batch ramp**: start near **0.5M** and climb to the target over the first ~1–5% of tokens (V3-style; Lite used constant batch — ramp is optional).
- **Phase B (32K / 128K):** keep tokens/step in a similar ballpark when possible; if memory-bound at 128K, reduce tokens/step rather than OOM.
- **Corpus reality:** ~60 GB raw ≈ on the order of **~15–25B tokens** per epoch (tokenizer-dependent). Phase A is **multi-epoch**; watch val loss for overfitting.

### 5.4 Phase A token budget (locked)
| Item | Value |
| --- | --- |
| **Primary budget** | **40B tokens** seen |
| **Hard cap** | **60B** only if val loss still falling and SFT compute reserve intact |
| **Default epochs** | ~**2** on the ~15–25B unique token estimate |
| **Early stop** | Val loss flat for ~**3B** tokens, or rising (overfit) → stop and move to SFT |

Lab-style 100B–500B+ remains aspirational if more compute appears; **do not** block shipping Phase A on that band.

### 5.5 Multi-Token Prediction (MTP)
- **Enable MTP** during pretrain (DeepSeek-V3-style).
- Depth **D = 1** (one extra token via sequential MTP module); do not start with D>1.
- Share embedding + LM head with the main model where the architecture allows.
- Loss weight **λ = 0.3** for most of Phase A; **λ = 0.1** for the final ~20–30% of Phase A.
- CPT: keep MTP with **λ = 0.1** (or briefly disable if unstable).
- Serving: MTP modules may be dropped for standard decode, or retained later for speculative decoding.

### Explicitly rejected / deferred
| Item | Stance |
| --- | --- |
| Plain Adam (no weight decay) | Rejected |
| Copying V3’s 12M–61M tokens/step as a hard requirement | Rejected — scale to available GPUs |
| MTP depth D>1 for v1 | Deferred |
| Muon as day-1 optimizer | Deferred to v2 |
| Phase A 100B–500B as a hard requirement | Deferred — **40B (cap 60B)** locked for this budget |

### Status
**LOCKED** (optimizer / LR / batch-token targets / Phase A **40B** / MTP) — 2026-07-11.

---

## 6. Hardware & parallel plan (DECIDED)

### Priority order
1. **GCP — multi A100 or H100** (primary): fastest path for MoE + 8K Phase A and later 32K/128K CPT. Use Spot/preemptible when credits allow and checkpointing is solid.
2. **CloudRift RTX 4090** (overflow / cheap token hours): BF16 Phase A continuation if GCP quota/credits are tight.
3. **Colab A100 / RTX 6000**: bring-up, unit tests, short smokes; checkpoint to Hub often (session kills).
4. **Runpod (~$25)**: emergency debug only — not Phase A.
5. **CloudRift V100**: **rejected** for main train (BF16/stack mismatch).

### Preferred GCP layouts
| Job | Preferred | Parallelism |
| --- | --- | --- |
| Phase A (8K) | **2–8× A100 40/80GB** or **2–8× H100** if available | **FSDP2 / ZeRO-3** for MoE params+Adam; data parallel across GPUs; optional EP later |
| Phase B1 (32K) | **4–8× A100 80GB** or H100 | FSDP2 + **sequence parallel** |
| Phase B2 (128K) | **4–8× A100 80GB / H100** | FSDP2 + **SP** required; reduce microbatch / tokens-per-step if needed |
| SFT | **1–2× A100** | FSDP or single-GPU LoRA/full FT as fits |

Always verify **credits cover the GPU SKU + region** before launching long jobs.

### Budget split (guidance)
| Bucket | Approx $ | Notes |
| --- | --- | --- |
| Phase A | **~$1,100–1,300** | Prefer GCP multi-GPU for wall-clock; CloudRift for cheap overflow |
| Phase B (32K→128K) | **~$150–250** | Short CPT; stop if val/long-context probes fail |
| SFT | **~$250–350** | Keep reserved — this is the agent-quality lever |
| Debug / buffer | **~$100–150** | OOM retries, disks, egress |

### Checkpointing / preemption
- Save to **GCS and/or Hub** every **15–30 min** (Spot) or every **1–2k steps** (on-demand).
- Resume must restore **model + optimizer + data cursor** (see §8).

### Status
**LOCKED** — 2026-07-11.

---

## 7. Training stack (DECIDED)

| Layer | Choice |
| --- | --- |
| Framework | **PyTorch 2.4+**, CUDA 12.x |
| Parallelism | **FSDP2** (preferred) or Accelerate FSDP; ZeRO-3-style param/optim sharding for MoE |
| Attention | **FlashAttention-2/3** with MLA-compatible implementation |
| Precision | **BF16** compute; **FP32** loss / RMSNorm / MoE router |
| Checkpointing | Activation checkpoint on attn + MoE blocks |
| Logging | W&B or TensorBoard |
| Artifacts | Private Hub model repo + GCS for large ckpts |
| FP8 / TE | Optional **v2 on H100 only** after BF16 is green |

**Not required for v1:** Megatron DualPipe, custom EP all-to-all (add only if single-node FSDP cannot fit).

### Status
**LOCKED** — 2026-07-11.

---

## 8. Data pipeline for training (DECIDED)

1. **Source:** stream `Aniket200325/coder-pretrain-60gb` train shards (prefer offline **tokenized+packed** cache to keep GPUs busy).
2. **Tokenizer:** Qwen2.5-Coder; count everything in **tokens**.
3. **Packing:** concat documents to **8192** for Phase A; insert EOS between docs; Phase B raises pack length to **32K** then **128K**.
4. **FIM:** ~**40%** of **code** packs use Qwen FIM tokens (SPM/PSM); non-code = plain next-token.
5. **Mix:** keep finalized corpus proportions in v1 (no aggressive rebalance).
6. **Shuffle:** shuffled shard order + epoch seed; large shuffle buffer if online.
7. **Val:** fixed val split from finalize (~17k docs); tokenize once; never train on it.
8. **Resume cursor:** persist `(epoch, shard_id, row_offset, global_step)` in every checkpoint.
9. **Decontam:** rely on finalize removals; still keep eval prompts out of any later SFT scrapes.

### Status
**LOCKED** — 2026-07-11.

---

## 9. Init & stability knobs (DECIDED)

| Knob | Value |
| --- | --- |
| Weight init | Truncated normal σ ≈ **0.02**; depth-scale residual outs `1/√(2L)` |
| Embeddings | Normal σ **0.02**; **tied** LM head |
| RoPE θ | **500_000** in Phase A; YaRN/NTK scale in Phase B |
| Grad clip | **1.0** |
| AdamW | β **0.9 / 0.95**, wd **0.1**, eps **1e-8** |
| Dropout | **0** during pretrain |
| MTP λ | **0.3** → **0.1** for last 20–30% of Phase A |
| MoE aux LB | **α ≈ 0.01**; bias balancing after aux baseline is stable |
| Router init | Near **0**; FP32 router |
| NaN policy | Skip bad step + log; halt if **≥3** NaNs in **100** steps |

### Status
**LOCKED** — 2026-07-11.

---

## 10. Eval & stop criteria (DECIDED)

### During Phase A
| Signal | Cadence | Action |
| --- | --- | --- |
| Train loss (smoothed) | continuous | sanity |
| Val loss | every **~0.5–1B** tokens | primary early-stop signal |
| Smoke generations | each val | syntax/plausibility check |
| Tokens / $ / hour | continuous | kill bad parallel configs early |

**Stop Phase A when any of:**
- Val loss best with **no improvement for ~3B tokens**, or
- Val loss **rising** while train loss falls (overfit), or
- Hit **40B** tokens (or **60B** cap), or
- Phase A spend hits the reserved budget ceiling.

### Quality probes (end of Phase A + after SFT)
| Suite | Role |
| --- | --- |
| HumanEval (+ pass@1) | coding sanity |
| MBPP | coding sanity |
| Small private **diff / vuln review** set (20–50) | product-relevant smoke (expand in SFT) |

Do **not** burn GPU hours on heavy eval every step during pretrain.

### “Good enough to leave Phase A”
- Val loss clearly below early plateau
- Completions syntactically plausible in major languages
- No NaNs for the last ~5B tokens  
→ move compute to **SFT** (and only then short Phase B long-context CPT if budget remains).

### Status
**LOCKED** — 2026-07-11.

---

## Decision log

| Date | Topic | Decision |
| --- | --- | --- |
| 2026-07-11 | Tokenizer | Reuse Qwen2.5-Coder BBPE (~152K); tie embeddings on small models; Qwen special/bos/eos IDs; DeepSeek MLA/MoE/precision for the model+recipe |
| 2026-07-11 | MLA + RoPE + KV | DeepSeek-V3 MLA + decoupled RoPE for all phases; skip V4 attention for v1; numeric shape `d_model=2048`, 28 layers, 16 heads, `kv_lora_rank=512`, `q_lora_rank=0`, rope/nope/v dims 64/128/128; **product context 128K** via CPT ladder 8K→32K→128K; 256K deferred |
| 2026-07-11 | MoE | DeepSeekMoE Lite-like: **~2.4–3.0B active** / ~10–16B total; 2 shared + 64 routed, top‑6; first **1** dense FFN; expert intermediate **1408**; dense intermediate **11008**; aux LB α≈0.01 + bias balancing later |
| 2026-07-11 | Mixed precision / systems | v1: BF16/TF32 + FP32 for loss/norm/router; Flash-Attention + grad checkpointing from day one; SP auto on multi-GPU. v2 FP8 on **H100** after BF16 green |
| 2026-07-11 | Optimizer / LR / batch / MTP | AdamW (0.9/0.95/wd0.1, clip1.0); LR warmup 2K → 3e-4 cosine to 3e-5; CPT 0.1–0.3× peak; 0.5M–2M tokens/step; Phase A **40B (cap 60B)**; MTP D=1, λ=0.3→0.1 |
| 2026-07-11 | Hardware | Primary: **multi A100/H100 on GCP**; overflow CloudRift 4090; Colab bring-up; reject V100 for main train; reserve $ for SFT |
| 2026-07-11 | Stack / data / init / eval | PyTorch + FSDP2 + FA2; Hub stream + packed FIM~40%; init σ0.02 + depth-scale; val every ~1B; early-stop then SFT |

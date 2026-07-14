# Fine-Tune Decisions — Qwen3-4B Domain CPT + SFT

Living document for fine-tuning choices.
Goal: specialize a strong open 4B model for coding/security automation and PR review, deployable on Jetson Orin Nano 8GB and/or a small cloud machine.

Corpus (Phase 1): `Aniket200325/coder-pretrain-60gb` (~59 GB raw text; Phase 1 trains a **~5B-token** stratified/streaming slice under Colab budget, not necessarily a full epoch).  
Phase 1.5 (optional): security / curated-repo CPT if Phase 1 under-delivers.  
Phase 2: instruction / PR-review SFT (security + issue/PR data preferred here by default).

Scratch (from-scratch) work lives in [`../scratch/`](../scratch/). This folder is the fine-tune path.

---

## 1. Base model (DECIDED)

### Choice
- **Primary:** [`unsloth/Qwen3-4B-Base`](https://huggingface.co/unsloth/Qwen3-4B-Base) (fast Colab mirror) / [`Qwen/Qwen3-4B-Base`](https://huggingface.co/Qwen/Qwen3-4B-Base) fallback
- **Fallback (Orin / quality):** `Qwen/Qwen2.5-Coder-7B` or `Qwen/Qwen2.5-Coder-3B` if Qwen3 deploy or throughput disappoints

### Why Base, not instruct
| Checkpoint | Role | Fit for our plan |
| --- | --- | --- |
| **`Qwen3-4B-Base`** | Dense text foundation; Unsloth-native; packing + FA2 path | **Phase 1 corpus CPT → Phase 2 instruction SFT** |
| **`Qwen3-4B` (instruct)** | Post-trained chat / reasoning | Skip as CPT start — would fight existing SFT/RL priors |

Our pipeline is **domain continued pretraining on the 60 GB corpus, then instruction fine-tuning**. That matches **Base**. Coding specialization comes from CPT on `coder-pretrain-60gb`, not from a Coder-branded base.

### Why not Qwen3.5-4B-Base (superseded 2026-07-13)
| Issue | Impact |
| --- | --- |
| Checkpoint is **VLM-shaped** (`Qwen3_5ForConditionalGeneration` + `vision_config`) | Unsloth **skips sample packing** even after vision strip |
| Hybrid **linear attention** needs extra CUDA kernels | Colab install fragile; without them tok/s collapses to ~few k |
| Measured ~6k tok/s without packing | ~5B tokens infeasible on ~45 Colab hours |

### Product / deploy constraints (locked intent)
- Use cases: automation agents, PR review (bugs / issues / vulnerabilities).
- Must be runnable on **Jetson Orin Nano 8GB** and/or a **small cloud** box.
- Inference plan: **quantize** for edge (Q4/Q5 or INT8); do not assume full BF16 weights + long KV fit in 8GB.
- Qwen3-4B is a standard dense transformer — broader Orin / llama.cpp support than Qwen3.5 hybrid.

### Explicitly rejected (for v1 start)
| Option | Reason |
| --- | --- |
| `Qwen3.5-4B-Base` for Phase 1 CPT on Unsloth | Packing blocked + hybrid-attn kernel dependency (see above) |
| Start CPT from instruct checkpoints | Wrong stage for corpus continue-pretrain |
| Jump to 7B+ as primary Jetson target | Tight/unreliable on 8GB with long PR context |
| From-scratch SLM as the main line | Moved to `scratch/`; fine-tune path is primary product path |

### Status
**LOCKED** — 2026-07-13 (switched from Qwen3.5 → Qwen3).

---

## 2. Corpus sufficiency & data roadmap (DECIDED)

### Choice
- **Phase 1 uses the existing ~60 GB corpus as the source:** `Aniket200325/coder-pretrain-60gb`.
- **Do not block** Phase 1 on collecting a larger general crawl.
- **High-signal domain data** (security writeups, issue↔PR / review threads, curated repo-level packs) is deferred to **Phase 1.5 (optional CPT mix)** and/or **Phase 2 (SFT)**, chosen **after** we see Phase 1 metrics.

### Why 60 GB is enough as the source pool
| Goal | Verdict |
| --- | --- |
| From-scratch parity with lab coders | Not this path |
| Domain CPT on Qwen3-4B-Base | **Yes** — a multi-billion-token LoRA CPT pass is useful |
| Full corpus epoch on Colab credits alone | **No** — target a **~5B-token** slice instead |
| PR/security reviewer after Phase 1 alone | **No** — needs instruction/task data later |

Approximate mix (finalized): code ~30 GB, FineWeb-Edu ~15 GB, docs ~10 GB, math ~3 GB, wiki ~2 GB, with dedup/decontam. Strength is general coding + technical text; gap is PR/vuln/review-specific signal — that gap is addressed post–Phase 1, not by more StarCoder-like bulk.

### Token estimate (corpus vs Phase 1 compute)
| Item | Estimate |
| --- | --- |
| Raw text (full corpus) | ~59 GB |
| Tokens / full epoch (tokenizer-dependent) | **~15–25B** |
| **Phase 1 compute target (Colab-first)** | **~5B tokens** stretch goal (stratified / streaming sample of the train split) |
| Floor if throughput is weak | **~2–3B tokens** still counts as a useful CPT pass |

Measure real **tok/s** in the first Colab hour and re-forecast: `tokens ≈ tok/s × 3600 × remaining_hours`. Do **not** assume a full 60 GB epoch on Colab credits alone.

### Post–Phase 1 data (gated on performance)
Add only if Phase 1 val/smoke shows weak domain lift or Phase 2 needs raw material:

| Data type | Prefer stage | Role |
| --- | --- | --- |
| Security writeups / vuln↔patch pairs | **1.5 CPT and/or 2 SFT** | Security priors |
| Issue ↔ PR / review comment threads | **2 SFT** (primary); light 1.5 if useful | PR-review task format |
| Curated repo-level packs | **1.5 CPT** (long-context) or SFT with multi-file prompts | Multi-file reasoning |

**Default:** skip Phase 1.5 if Phase 1 looks healthy → go straight to **Phase 2 SFT** and put security/PR data there.

### Explicitly rejected
| Option | Reason |
| --- | --- |
| Delay Phase 1 for another 100+ GB general GitHub | Low ROI vs starting CPT |
| Multi-epoch hammer on the same 60 GB as the default | Overfit / forgetting risk; Colab budget can’t finish a full epoch anyway |
| Expect Phase 1 alone to teach structured PR findings | Task skill → Phase 2 |
| Assume full ~15–25B epoch on ~300 Colab credits | Not feasible; target **~5B** with LoRA optimizations |

### Status
**LOCKED** — 2026-07-12.

---

## 3. Phase 1 method & budget (DECIDED) — Colab / LoRA revision

### Choice
- **Method:** **LoRA** continued pretrain on `Qwen3-4B-Base` (primary). **Not** full FT for v1 Phase 1.
- **Stack class:** **Unsloth-class** (or equivalent fused/fast LoRA trainer) + BF16 + FlashAttention2/xformers where available; minimize padding waste via packing.
- **Seq length:** **2048–4096** (Colab A100 40GB default **2048** + large batch for tok/s; trial **4096** only if measured tok/s wins).
- **Token target:** **~5B tokens** stretch on Colab; accept **~2–3B** if measured throughput can’t hit 5B.
- **Data:** stream/sample from full `coder-pretrain-60gb` train split (shuffle + seed); no need to materialize a separate 5B-token export first.
- **QLoRA:** only if BF16 LoRA OOMs on the assigned GPU; prefer BF16 LoRA on A100 40GB.
- **Full FT:** deferred until more multi-GPU / non-Colab budget exists.

### Why LoRA now (supersedes earlier full-FT preference)
Full FT for ~15–25B tokens needs weeks on one A100 and far more than ~300 Colab credits. LoRA + shorter context + fast kernels is what makes a **~5B-token** Phase 1 conceivable on Colab A100 40GB.

### LoRA knobs (Phase 1 defaults)
| Knob | Value |
| --- | --- |
| Rank `r` | **64** (try **128** if VRAM allows and loss plateaus) |
| Alpha | **`2r`** (e.g. 128 when r=64) |
| Targets | All attention + MLP linears (Unsloth/PEFT defaults for the arch); leave embeds/LM head frozen unless ablations say otherwise |
| Dropout | **0** for CPT |
| Precision | **BF16** LoRA |
| Peak LR | Higher than full FT CPT is OK for adapters — start **~1e-4** (tune 5e-5–2e-4); cosine + short warmup |
| Packing | **On** — concat docs to `max_seq_len` with EOS; minimize pad |
| FIM | **Off by default** for speed; optional later ≤20% on code |

### Colab A100 40GB throughput defaults (Phase 1 script / notebook)
| Knob | Value | Notes |
| --- | --- | --- |
| `max_seq_len` | **2048** | Prefer over 4096 when maximizing tokens/credit |
| `per_device_train_batch_size` | **48** (try **56+** if peak &lt; ~30GB) | Fill toward **35–38GB** peak on 40GB A100 |
| `gradient_accumulation_steps` | **1** | Prefer real batch over fake accum for throughput |
| Attention kernels | **FlashAttention2 / xformers** (optional but preferred) | Dense Qwen3 full-attn path |
| Packing | **On** (text tokenizer / CausalLM) | Confirm `Sample packing is ACTIVE` (script aborts if skipped) |
| Vision strip | **Off** by default | Only for VLM-shaped ckpts (`--strip_vision`) |
| Logs | every **5** steps + **EARLY_PROJECTION** (~10 min) | Scale batch/seq from `tok/s` + `peak` VRAM |

### Throughput / success criteria
- Early (~10 min) and first-hour goal: push toward **~25–35k tok/s** sustained if possible (needed to approach 5B in ~40–50 useful hours).
- If projection under 5B over remaining credits → raise batch (while peak VRAM < ~38GB), keep seq 2048, or lock a **2–3B** target and stop cleanly.
- Prefer **A100 40GB** over 80GB unless 40GB OOMs after packing+high batch (80GB costs ~+23% credits; only worth it if tok/s rises more than that).

### Phase 2 / 1.5 (intent only — details TBD)
- **Phase 2:** instruction / PR SFT — continue with **LoRA** (merge or stack adapters as needed).
- **Phase 1.5:** optional short CPT on security / curated-repo packs if Phase 1 under-delivers.

### Status
**LOCKED** — 2026-07-12 (LoRA / 2K–4K / ~5B Colab-first).

---

## 4. Phase 1 hardware, Colab sessions & resume (DECIDED)

### Hardware preference (Colab compute units)
| GPU | Typical units/hour | Role |
| --- | --- | --- |
| **A100 40GB** | **~5–7** | **Primary** — more hours per credit → more tokens |
| A100 80GB | ~10–15 | **Avoid for token-max** — fewer hours for the same 300 credits |
| L4 / weaker | varies | Accept only if A100 40GB unavailable; expect lower tok/s |

Assume ~**300 Colab credits** → roughly **~43–60 h** on 40GB before waste; plan **~35–50 useful train hours** after setup/disconnects.

### Colab ~12 h session limit — resume is mandatory
Colab sessions can end around **~12 hours**. Phase 1 **must** be multi-session:

| Requirement | Policy |
| --- | --- |
| Checkpoint frequency | At least every **30–60 minutes** wall time **and** every **N steps** (e.g. 200–500) |
| What to save | LoRA adapter weights + optimizer state + **trainer state** (global step, tokens_seen, epoch, data cursor / sample offset, RNG, best val) |
| Where to save | **Hugging Face Hub** private repo and/or Google Drive — **not** only `/content` (ephemeral) |
| `LATEST` pointer | File or Hub revision tag pointing at the newest good ckpt |
| Resume | On session start: load `LATEST` → continue until token target or credits run out |
| Preemption / disconnect | Treat as normal; never rely on a single 12 h run finishing 5B |
| End-of-session | Push a final ckpt **before** expected cutoff (~11 h mark) |

### Anti-waste (keep GPU busy)
- Cache model + tokenizer on Drive/Hub after first download.
- Prefer streaming Hub dataset with resilient resume cursor over re-downloading the full 60 GB each session.
- Avoid huge blocking Hub uploads every step; periodic adapter-only pushes are enough.
- Log tok/s, tokens_seen, and ETA each session.

### Overflow compute (optional later)
If Colab cannot reach ~5B: continue the **same LoRA run** on CloudRift 4090 / GCP using the same Hub checkpoints (same resume format).

### Train stack pins (LOCKED)
| Layer | Choice |
| --- | --- |
| Trainer stack | **Unsloth** `FastLanguageModel` + TRL `SFTTrainer` / `SFTConfig` |
| Transformers | **≥4.51** (Qwen3 support; v5 also fine) |
| Model load | BF16 / `load_in_16bit=True`; **`load_in_4bit=False`** |
| QLoRA | **Rejected** for Phase 1 (quality / Unsloth guidance) |
| Entry script | [`train_phase1.py`](train_phase1.py) + [`phase1_colab.ipynb`](phase1_colab.ipynb) |

### Still TBD (later sections)
- Phase 2 SFT schema and sources
- Orin export / quant path
- Eval suite beyond val loss + smoke completions

### Status
**LOCKED** — 2026-07-12.

---

## Decision log

| Date | Topic | Decision |
| --- | --- | --- |
| 2026-07-12 | Base model | `Qwen/Qwen3.5-4B-Base`; instruct rejected as CPT start; Qwen2.5-Coder-3B as deploy fallback — **superseded 2026-07-13** |
| 2026-07-12 | Corpus / roadmap | 60 GB as source pool; security / issue-PR / curated-repo → Phase 1.5 and/or 2 after Phase 1 results |
| 2026-07-12 | Phase 1 method (v1) | Full FT CPT, 1 epoch — **superseded same day** |
| 2026-07-12 | Phase 1 method (v2) | **LoRA** CPT, seq **2K–4K**, Unsloth-class stack, target **~5B tokens** on Colab; full FT deferred |
| 2026-07-12 | Colab hardware / resume | Prefer **A100 40GB**; avoid 80GB for token-max; **mandatory multi-session resume** (≤~12 h sessions) via Hub/Drive ckpts |
| 2026-07-12 | Train stack | Unsloth + TRL packing CPT; **no QLoRA**; see `train_phase1.py` |
| 2026-07-13 | Packing / throughput | Attempted Qwen3.5 vision strip; Unsloth still skipped packing → fatal guard |
| 2026-07-13 | Base model (v2) | Switch to **`Qwen3-4B-Base`**; FA2/xformers path; packing on; batch **48** |
| 2026-07-14 | Model weights load | Revert to **Unsloth `FastLanguageModel.from_pretrained`** (same path that worked for Qwen3.5); purge incomplete Hub cache on one retry |

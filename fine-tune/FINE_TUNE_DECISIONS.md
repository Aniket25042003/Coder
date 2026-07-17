# Fine-Tune Decisions — Qwen2.5-Coder-7B Domain CPT + SFT

Living document for fine-tuning choices.
Goal: specialize a strong open **text-only** coder (~7B) for coding/security automation and PR review, deployable on Jetson Orin Nano 8GB and/or a small cloud machine.

Corpus (Phase 1): `Aniket200325/coder-pretrain-60gb` (~59 GB raw text; Phase 1 trains a **~5B-token** stratified/streaming slice under Colab budget, not necessarily a full epoch).  
Phase 1.5 (optional): security / curated-repo CPT if Phase 1 under-delivers.  
Phase 2: instruction / PR-review SFT (security + issue/PR data preferred here by default).

Scratch (from-scratch) work lives in [`../scratch/`](../scratch/). This folder is the fine-tune path.

---

## 1. Base model (DECIDED)

### Choice
- **Primary:** [`unsloth/Qwen2.5-Coder-7B`](https://huggingface.co/unsloth/Qwen2.5-Coder-7B) (**Base**, not Instruct)
- **Not Phase 1:** `unsloth/Qwen2.5-Coder-7B-Instruct` — reserved for later comparison; Phase 2 SFT starts from Base CPT adapters
- **Rejected lineage (2026-07):**
  - Qwen3.5 / multimodal *ForConditionalGeneration* — packing skipped (VLM)
  - Gemma 4 E4B — packing skipped (`vision-language model detected`) despite text tokenizer
  - Qwen3-4B / Qwen2.5-Coder-3B earlier Colab attempts — CDN / incomplete-cache download failures (retry recipe now locked below)

### Why Qwen2.5-Coder-7B Base
| Checkpoint | Role | Fit for our plan |
| --- | --- | --- |
| **`Qwen2.5-Coder-7B`** | Dense **text-only** Causal LM; coder-pretrained | **Phase 1 CPT → Phase 2 SFT**; Unsloth packing expected ACTIVE |
| **`…-Instruct`** | Chat / tools | Skip as CPT start |

### Packing constraint (locked)
| Rule | Rationale |
| --- | --- |
| Load with **`FastLanguageModel`** (not `FastVisionModel`) | Text Causal LM path |
| Do **not** use `UnslothVisionDataCollator` | Vision SFT path skips packing |
| No chat template in Phase 1 | CPT = full-token LM loss on `text` field |
| Abort if `trainer.args.packing` is False | No silent no-packing CPT |
| Prefer models **without** `vision_config` / not `*ForConditionalGeneration` | Unsloth VLM packing guard |

### Product / deploy constraints (locked intent)
- Use cases: automation agents, PR review (bugs / issues / vulnerabilities).
- Must be runnable on **Jetson Orin Nano 8GB** and/or a **small cloud** box.
- Inference plan: **quantize** for edge (Q4/Q5 or INT8); do not assume full BF16 weights + long KV fit in 8GB.

### Explicitly rejected (for v1 Phase 1)
| Option | Reason |
| --- | --- |
| Start CPT from Instruct | Wrong stage for corpus continue-pretrain |
| QLoRA for Phase 1 | Quality lock: **BF16 LoRA only**; Phase 2 may revisit |
| Quiet fallback if packing inactive | Burns Colab credits with bad tok/s |
| Multimodal bases (Gemma 4 E4B, Qwen3.5, VL families) | Packing blocked by Unsloth VLM guard |
| From-scratch SLM as the main line | Moved to `scratch/`; fine-tune path is primary product path |

### Status
**LOCKED** — 2026-07-14 (**Qwen2.5-Coder-7B Base**).

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
| Domain CPT on Qwen2.5-Coder-7B | **Yes** — a multi-billion-token LoRA CPT pass is useful |
| Full corpus epoch on Colab credits alone | **No** — target a **~5B-token** slice instead |
| PR/security reviewer after Phase 1 alone | **No** — needs instruction/task data later |

Approximate mix (finalized): code ~30 GB, FineWeb-Edu ~15 GB, docs ~10 GB, math ~3 GB, wiki ~2 GB, with dedup/decontam.

### Token estimate (corpus vs Phase 1 compute)
| Item | Estimate |
| --- | --- |
| Raw text (full corpus) | ~59 GB |
| Tokens / full epoch (tokenizer-dependent) | **~15–25B** |
| **Phase 1 compute target (Colab-first)** | **~5B tokens** stretch goal |
| Floor if throughput is weak | **~2–3B tokens** still counts as a useful CPT pass |

Measure real **tok/s** in the first Colab hour and re-forecast: `tokens ≈ tok/s × 3600 × remaining_hours`.

### Status
**LOCKED** — 2026-07-12.

---

## 3. Phase 1 method & budget (DECIDED)

### Choice
- **Method:** **BF16 LoRA** continued pretrain on `unsloth/Qwen2.5-Coder-7B`. **Not** full FT; **not** QLoRA for Phase 1.
- **Loader:** Unsloth **`FastLanguageModel`** + TRL `SFTTrainer` / `SFTConfig`.
- **Seq length:** **2048** default (trial **4096** only if measured tok/s wins).
- **Token target:** **~5B** stretch; accept **~2–3B** if throughput cannot hit 5B.
- **Data:** stream from `coder-pretrain-60gb` train split.

### LoRA knobs (Phase 1 defaults)
| Knob | Value |
| --- | --- |
| Rank `r` | **64** |
| Alpha | **128** (`2r`) |
| Target modules | `q/k/v/o_proj`, `gate/up/down_proj` |
| Dropout | **0** |
| Precision | **BF16** LoRA (`load_in_4bit=False`) |
| Peak LR | **~1e-4**; cosine + warmup |
| Packing | **Required** — abort if inactive |

### Colab A100 40GB throughput defaults
| Knob | Value | Notes |
| --- | --- | --- |
| `max_seq_len` | **2048** | Prefer over 4096 for tokens/credit |
| `per_device_train_batch_size` | **110** full (A100 80GB) / **2** smoke | Cut on OOM; peak ~32GB at batch 32 → room to fill 80GB |
| `gradient_accumulation_steps` | **1** | Prefer real batch over accum |
| Packing | **On** | Fatal if skipped |
| Wall-clock stop | **11.5 h** | `should_training_stop` + `should_save` before Colab ~12h kill |
| Logs | every **5** steps + **EARLY_PROJECTION** (~10 min) | Scale batch from tok/s + peak VRAM |

### Status
**LOCKED** — 2026-07-14 (Qwen2.5-Coder-7B / BF16 LoRA / packing-required).

---

## 4. Phase 1 hardware, Colab sessions & resume (DECIDED)

### Hardware
| GPU | Role |
| --- | --- |
| **A100 80GB** | **Primary** for 7B (batch ~110; wall-clock stop 11.5h) |
| A100 40GB | Fallback with lower batch (~8–12) |

### Resume (mandatory; ~12 h Colab sessions)
| Requirement | Policy |
| --- | --- |
| Checkpoint | Every **30 min** wall + every **N** steps |
| What to save | LoRA adapters + trainer state + `phase1_state.json` (`tokens_seen`, step, tok/s) |
| Where | Hub private repo and/or Drive — not only `/content` |
| `LATEST` | Pointer to newest good ckpt |
| Resume | Drive `LATEST` → local `LATEST` → newest `checkpoint-*` |

### Train stack pins (LOCKED)
| Layer | Choice |
| --- | --- |
| Trainer | Unsloth **`FastLanguageModel`** + TRL `SFTTrainer` / `SFTConfig` |
| Install | Unsloth Colab `--no-deps` + matching `xformers` for torch minor |
| Download | `UNSLOTH_STABLE_DOWNLOADS=1`, `UNSLOTH_DISABLE_STATISTICS=1`, `HF_HUB_DISABLE_XET=1`, HF `snapshot_download` → local_dir |
| Model load | BF16; **`load_in_4bit=False`** |
| QLoRA | **Rejected** for Phase 1 |
| Entry | Sole Colab: [`qwen25_coder_7b_phase1_cpt_colab.ipynb`](qwen25_coder_7b_phase1_cpt_colab.ipynb) |

### Status
**LOCKED** — 2026-07-14.

---

## Decision log

| Date | Topic | Decision |
| --- | --- | --- |
| 2026-07-12 | Base model | `Qwen/Qwen3.5-4B-Base` — **superseded** |
| 2026-07-12 | Corpus / roadmap | 60 GB source; ~5B Colab token target; multi-session Hub/Drive resume |
| 2026-07-12 | Phase 1 method | **LoRA** CPT, seq 2K–4K, Unsloth-class, no QLoRA |
| 2026-07-13 | Base model (v2) | `Qwen3-4B-Base` — **superseded** (CDN incomplete cache) |
| 2026-07-14 | Base model (v3) | `Qwen2.5-Coder-3B` — **superseded** |
| 2026-07-14 | Base model (v4) | `unsloth/gemma-4-E4B` — **superseded** (VLM packing skip) |
| 2026-07-14 | Base model (v5) | **`unsloth/Qwen2.5-Coder-7B` Base**; BF16 LoRA; packing-required; `FastLanguageModel` Colab notebook |

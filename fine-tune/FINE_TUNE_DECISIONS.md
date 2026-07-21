# Fine-Tune Decisions — Qwen2.5-Coder-7B Domain CPT + SFT

Living document for fine-tuning choices.
Goal: specialize a strong open **text-only** coder (~7B) for coding/security automation and PR review, deployable on Jetson Orin Nano 8GB and/or a small cloud machine.

Corpus (Phase 1): `Aniket200325/coder-pretrain-60gb` (~59 GB raw text).  
**Phase 1 cutover (2026-07):** stop CPT near **~400M tokens** (product pivot to SFT; further CPT only if SFT under-delivers).  
Phase 1.5 (optional): security / curated-repo CPT if Phase 2 under-delivers.  
Phase 2: instruction / PR-review / patch / implement **QLoRA SFT** on a **merged CPT domain base** (~140K mix; see §7).

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
| **Actual Phase 1 cutover (Colab credits / product)** | **~400M tokens** — then Phase 2 SFT; optional more CPT later |

Measure real **tok/s** in the first Colab hour and re-forecast: `tokens ≈ tok/s × 3600 × remaining_hours`.

### Status
**LOCKED** — 2026-07-12; **cutover revised** 2026-07-19 (~400M → SFT).

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

## 5. Phase 1 → Phase 2 handoff (DECIDED)

### CPT cutover
| Item | Choice |
| --- | --- |
| Stop near | **~400M** tokens (current final CPT session targets this) |
| Keep on Drive/Hub | Last good **`checkpoint-*`** (trainer resume) **and** adapter **`final/`** |
| Do **not** use `final/` as `trainer.train(resume_from_checkpoint=...)` | Adapter-only; use `checkpoint-*` for CPT resume |
| Optional more CPT | Only if Phase 2 SFT under-delivers on code fluency (Phase 1.5) |

### Merge before SFT (**required**)
| Item | Choice |
| --- | --- |
| Pattern | **Merge CPT LoRA into Base**, then train a **fresh** SFT adapter |
| Base | `unsloth/Qwen2.5-Coder-7B` |
| Adapter source | Drive `…/coder-qwen25-coder-7b-phase1-lora/final` (or equivalent Hub adapters) |
| Output (Drive) | `/content/drive/MyDrive/coder-qwen25-coder-7b-cpt-merged` |
| Output (Hub) | Private **`Aniket200325/coder-qwen25-coder-7b-cpt-merged`** |
| Script | [`merge_cpt_lora_colab.py`](merge_cpt_lora_colab.py) (Colab) |
| Do **not** | Continue the same CPT LoRA into SFT as the default path |

### Status
**LOCKED** — 2026-07-19; Hub id locked 2026-07-20.

---

## 6. Phase 2 SFT method (DECIDED)

### Choice
- **Init:** merged CPT domain base **`Aniket200325/coder-qwen25-coder-7b-cpt-merged`** (Drive mirror OK). Not stock Instruct as the main line.
- **Method:** **fresh QLoRA** SFT (`load_in_4bit=True`) for Colab cost / VRAM headroom. BF16 LoRA allowed as a later quality A/B.
- **Product tasks:** (1) GitHub / PR **review**, (2) **patch** from review feedback, (3) **implement** code from instructions — needs both reasoning and coding ability.
- **Template:** Qwen2.5 chat template **on** (unlike Phase 1). Attach Instruct **ChatML** tokenizer/template only — not Instruct weights.
- **Loss:** assistant-only / completion masking via Unsloth `train_on_responses_only` (Qwen markers `<|im_start|>user\n` / `<|im_start|>assistant\n`).
- **Stock Instruct:** reserved for optional baseline compare, not the primary Phase 2 start.
- **Entry:** Colab **notebook** [`qwen25_coder_7b_phase2_sft_colab.ipynb`](qwen25_coder_7b_phase2_sft_colab.ipynb) (same style as Phase 1 CPT notebook) — not a standalone train script as the primary path.
- **Adapter Hub:** **`Aniket200325/coder-qwen25-coder-7b-sft-qlora-v1`** (private). Drive checkpoints required regardless.
- **Checkpoints / resume:** **60 min** wall-clock timed saves (`CKPT_MINUTES=60` full; smoke `5`) → mirror to Drive + write `LATEST` on `checkpoint-*` (not `final/`); `RESUME="auto"` for full runs (Phase 1 pattern).
- **Final merge:** optional **Cell 22** only after the last train session (`RUN_FINAL_MERGE=False` by default) → Drive/Hub SFT-merged BF16.

### SFT knobs (LOCKED — 2026-07-20; notebook defaults 2026-07-21)
| Knob | Value | Notes |
| --- | --- | --- |
| Precision | **QLoRA 4-bit** | Fresh adapters; do not reuse CPT adapter weights |
| LoRA `r` / alpha | **32 / 64** start | Raise to 64/128 if underfitting |
| LR | **~2e-5** | Lower than CPT `1e-4`; cosine + short warmup |
| Seq | **2048** default; trial **4096** if diffs need it | Mix already filtered at 2048 |
| Packing | **Off** (`packing=False`) | Chat + `train_on_responses_only` is fragile with packing (mask bugs / cross-sample bleed). Throughput via batch + grad accum instead. Unlike Phase 1 (packing required) |
| Batch (smoke → tune) | smoke `2`; full start **`4`** (tune toward 4–8) | Fine-tune after smoke VRAM/tok/s |
| Grad accum | smoke `4`; full start **`8`** (effective **32**) | Target effective batch ~32–64 |
| Epochs | smoke `MAX_STEPS=30`; full **1** epoch first | Watch train/eval loss |
| Data | Hub `Aniket200325/coder-sft-mix-v1` (`train`) | See **§7**; ~136K after cleanup |
| Eval set | `eval_securecodepairs` split | Tiny / security-only — early signal, not sole quality gate |
| Eval / step-save | smoke every **10** / **15**; full every **200** / **500** steps | Plus timed Drive ckpt every **60 min** |
| Smoke gate | Labels not all `-100`; packing confirmed **off**; one generate with `add_generation_prompt=True` | Before full run |

### Explicitly deferred
| Item | Status |
| --- | --- |
| SFT Colab **training** notebook | **Done** — [`qwen25_coder_7b_phase2_sft_colab.ipynb`](qwen25_coder_7b_phase2_sft_colab.ipynb) |
| Exact `eval_steps` / `save_steps` / N | **Locked in notebook:** full `EVAL_STEPS=200`, `SAVE_STEPS=500`; timed `CKPT_MINUTES=60` |
| SFT adapter Hub publish id | **`Aniket200325/coder-qwen25-coder-7b-sft-qlora-v1`** |
| Optional final SFT→BF16 merge | Notebook **Cell 22** (`RUN_FINAL_MERGE`); Drive `coder-qwen25-coder-7b-sft-merged` |
| Jetson quant export of SFT adapters | After Phase 2 train |

### Status
**LOCKED** (method + train knobs) — 2026-07-20; **data mix + preprocess locked** — 2026-07-20 (§7); **training notebook + adapter Hub id + 1h ckpt/auto-resume locked** — 2026-07-21.

---

## 7. Phase 2 SFT datasets & preprocessing (DECIDED)

### Product framing (drives every choice below)
The model must: **review** GitHub diffs, **patch** code from review feedback, and **implement** new code. So the mix balances (a) general coding instruct, (b) review↔fix pairs from real PRs, (c) security reasoning + secure rewrites. Prefer sources that already contain natural reasoning in the assistant turn; do **not** invent fake CoT tags.

### Script & publish
| Item | Choice |
| --- | --- |
| Script | [`preprocess_sft_mix_colab.py`](preprocess_sft_mix_colab.py) (Colab) |
| Runtime | Colab; full `load_dataset` download (not streaming) |
| Drive out | `/content/drive/MyDrive/coder-sft-mix-v1/` → `sft_mix_v1.parquet`, `sft_eval_securecodepairs.parquet`, `manifest.json`, `samples.jsonl` |
| Hub | Private dataset `--hub_dataset_id` with splits `train` + `eval_securecodepairs` (required unless `--skip_hub` for smoke) |
| Length-filter tokenizer | **`Qwen/Qwen2.5-Coder-7B-Instruct` tokenizer only** (ChatML). **Not** Instruct model weights — Phase-2 weights stay Base → CPT-merge → SFT |

### Mix (target ~140K rows)
| Bucket | Source | Count | Share | Role |
| --- | --- | --- | --- | --- |
| **A. Code instruct** | [`nvidia/OpenCodeInstruct`](https://huggingface.co/datasets/nvidia/OpenCodeInstruct) | **75K** (sampled) | ~53% | Implement / write code; chat format + coding ability |
| **B. Review / patch** | [`ronantakizawa/github-codereview`](https://huggingface.co/datasets/ronantakizawa/github-codereview) | **50K** (sampled) | ~36% | PR review comments + apply patches |
| **C. Security** | `cve-sft-v5` + `SecureCodePairs` + CyberNative DPO | **~15.1K** (all usable) | ~11% | Vuln explain / remediate / secure rewrite |
| | [`auren-research/cve-sft-v5`](https://huggingface.co/datasets/auren-research/cve-sft-v5) | **10,000** | | Structured CVE reasoning |
| | [`ismailtasdelen/SecureCodePairs`](https://huggingface.co/datasets/ismailtasdelen/SecureCodePairs) | **~470** code (train split; **2** variants) | | High-trust vuln↔secure pairs |
| | [`CyberNative/Code_Vulnerability_Security_DPO`](https://huggingface.co/datasets/CyberNative/Code_Vulnerability_Security_DPO) | **4,656** | | Extra secure-rewrite volume (`chosen` only) |

Do **not** pad to an arbitrary 150K with lower-quality rows after cleanup. Hold out SecureCodePairs `validation` / `test` / `benchmark` for eval (not SFT).

### Design locks (product-aligned)

| Decision | Choice | Why |
| --- | --- | --- |
| Review task split (within positives) | **~70% `code_review`** (diff → comment) · **~30% `code_review_fix`** (before + comment → after); no row duplicated into both | Primary product is review; patching is second skill |
| Negatives in review bucket | **~15%** of the 50K (`No issues found.`) | Teaches when *not* to nitpick |
| Review quality filter | Try `quality_score >= 0.75` first; fall back to **0.6** if pool too small; drop empty fields; positives `comment_type != "none"` | Signal over volume |
| Review upsample | 2× weight on `bug` / `security` / `performance` vs other types | Matches reviewer usefulness |
| OpenCodeInstruct sample | Cascade `average_test_score >= 0.8` → `>= 0.5` → rest; ~50/50 `generic`/`algorithmic`; fixed-seed → 75K | Verified-ish solutions |
| System prompts | **One family per `task`** (exact strings below) | Consistent inference behavior |
| Reasoning style | Source-native structure only; **no** synthetic `<think>` wrappers | Avoid fake CoT |
| CyberNative use | `question` → `chosen` only; drop empty / identical-to-`rejected` | Breadth only, not security oracle |
| SecureCodePairs | Train only for mix; **2** variants (review + rewrite); hold out val/test/benchmark | High trust, tiny |
| Target schema | Conversational `messages`; **do not** pre-render ChatML | TRL/Unsloth + assistant-only loss |
| Min length | user ≥ **32** chars; assistant ≥ **16** (accept-phrase negatives exempt) | Drop junk |
| Seq filter | Drop if ChatML token length `> max_seq_length` (default **2048**); prefer drop over truncate | Clean gradients |
| Dedup | MD5 of normalized user+assistant; keep first | Cross-source dupes |

### Exact system prompts (locked in script)
| `task` | System string |
| --- | --- |
| `code_instruct` | `You are a helpful coding assistant. Implement correct, clear solutions.` |
| `code_review` | `You are a senior GitHub code reviewer. Find bugs, risks, and security issues. Be concise. If the code is fine, say so.` |
| `code_review_fix` | `You apply GitHub review feedback and produce the corrected code.` |
| `security` | `You are a defensive security assistant. Explain the weakness and provide a secure fix or remediation.` |

### Canonical row schema
```json
{
  "messages": [
    {"role": "system", "content": "<task system>"},
    {"role": "user", "content": "<prompt>"},
    {"role": "assistant", "content": "<target>"}
  ],
  "source": "opencodeinstruct|github-codereview|cve-sft-v5|securecodepairs|cybernative-dpo",
  "task": "code_instruct|code_review|code_review_fix|security"
}
```

### Per-source mapping (preprocess)

**A. OpenCodeInstruct → `code_instruct`**
- `user` = `input`; `assistant` = `output`.

**B. github-codereview**
- **`code_review`:** user = language + file path + fenced `diff_context` (+ optional `before_code`); assistant = `reviewer_comment` or `No issues found.` for negatives.
- **`code_review_fix`:** user = `before_code` + reviewer comment; assistant = `after_code` only.

**C. Security → `security`**
- **CVE-SFT:** CVE metadata instruction → structured markdown (plain explanation, deep dive, attack scenario, remediation, code example).
- **SecureCodePairs:** (1) review vuln → root_cause/attack/fix + secure_code; (2) rewrite securely → secure_code + short fix.
- **CyberNative:** `question` → `chosen`.

### Global cleanup order
1. Sanitize (NUL strip, newline normalize, trim); drop empty user/assistant.
2. Min-length filter.
3. Dedup.
4. ChatML seq filter via Instruct tokenizer.
5. Fixed-seed shuffle; write Drive artifacts + Hub push.

### Explicitly rejected (v1 SFT data)
| Option | Reason |
| --- | --- |
| Full OpenCodeInstruct 5M | Overkill after CPT; dilutes review/security |
| Agentic SWE trajectory dumps as main mix | Scaffold-specific; defer until tool format is fixed |
| Pre-baked ChatML strings as the dataset | Breaks TRL masking / template portability |
| Invented CoT tags on all rows | Fake reasoning; prefer source-native structure |
| Training on SecureCodePairs benchmark split | Keep for eval |
| Treating CyberNative as high-trust security label | Community quality concerns on `chosen` |
| Merging Instruct **weights** with CPT LoRA | Different base; LoRA is Base-relative — template/tokenizer only |

### Status
**LOCKED** — 2026-07-20 (script added same day).

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
| 2026-07-19 | Phase 1 cutover | Stop CPT near **~400M tokens**; pivot to SFT |
| 2026-07-19 | Phase 2 init | **Merge CPT LoRA → domain base**; **fresh QLoRA** for SFT (not continue CPT adapters) |
| 2026-07-20 | Phase 2 data mix | ~140K: OpenCodeInstruct 75K + github-codereview 50K + security ~15.1K (CVE + SecureCodePairs + CyberNative) |
| 2026-07-20 | Phase 2 preprocess | Conversational `messages`; 70/30 review vs fix; 15% review negatives; task-level systems; source-native reasoning; no fake CoT |
| 2026-07-20 | Preprocess script | [`preprocess_sft_mix_colab.py`](preprocess_sft_mix_colab.py); Drive + private Hub; Instruct tokenizer for length filter only |
| 2026-07-20 | Phase 2 train form | Colab **notebook** like Phase 1 |
| 2026-07-20 | Phase 2 train knobs | QLoRA; batch 4–8 + effective ~32–64; eval every N steps; init `Aniket200325/coder-qwen25-coder-7b-cpt-merged` |
| 2026-07-20 | Phase 2 packing | **Off** for SFT (chat + assistant-only loss); unlike Phase 1 CPT |
| 2026-07-21 | Phase 2 notebook | [`qwen25_coder_7b_phase2_sft_colab.ipynb`](qwen25_coder_7b_phase2_sft_colab.ipynb); Hub adapter `…/coder-qwen25-coder-7b-sft-qlora-v1`; **60 min** Drive ckpt + `RESUME=auto`; optional Cell 22 final BF16 merge |

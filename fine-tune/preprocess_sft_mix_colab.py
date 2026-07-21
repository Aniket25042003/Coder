#!/usr/bin/env python3
"""
Build Phase-2 SFT conversational mix (Colab).

Downloads the five locked HF datasets, filters/samples (~140K), maps to
Qwen chat `messages`, cleans, writes Google Drive artifacts, and pushes a
private Hub dataset.

Tokenizer note
--------------
Length filtering uses `Qwen/Qwen2.5-Coder-7B-Instruct` **tokenizer only**
(ChatML template). Phase-1/2 **model weights** stay on the Base → CPT-merge
lineage — do NOT merge Instruct weights with CPT adapters.

Typical Colab usage:
  1. Runtime → any GPU/CPU (preprocess is disk/CPU-bound; T4 is fine).
  2. Mount Drive; set HF_TOKEN (Colab userdata or env).
  3. !pip install -q datasets transformers pyarrow huggingface_hub pandas numpy
  4. !python preprocess_sft_mix_colab.py \\
       --hub_dataset_id YOUR_USER/coder-sft-mix-v1

  Smoke (fast path):
  !python preprocess_sft_mix_colab.py --smoke_n 200 --skip_hub \\
       --out_dir /content/drive/MyDrive/coder-sft-mix-v1-smoke

OpenCodeInstruct sampling uses select_columns + pandas/numpy (never walks
all 5M rows in pure Python). Seq filter uses a cheap char prefilter, then
tokenizes only borderline examples.

See fine-tune/FINE_TUNE_DECISIONS.md §7.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import random
import re
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Locked constants (FINE_TUNE_DECISIONS.md §7)
# ---------------------------------------------------------------------------

SYSTEM_PROMPTS: dict[str, str] = {
    "code_instruct": (
        "You are a helpful coding assistant. Implement correct, clear solutions."
    ),
    "code_review": (
        "You are a senior GitHub code reviewer. Find bugs, risks, and security "
        "issues. Be concise. If the code is fine, say so."
    ),
    "code_review_fix": (
        "You apply GitHub review feedback and produce the corrected code."
    ),
    "security": (
        "You are a defensive security assistant. Explain the weakness and "
        "provide a secure fix or remediation."
    ),
}

ACCEPT_PHRASE = "No issues found."
MIN_USER_CHARS = 32
MIN_ASSISTANT_CHARS = 16
REVIEW_HIGH_TYPES = frozenset({"bug", "security", "performance"})
TOKENIZER_ID = "Qwen/Qwen2.5-Coder-7B-Instruct"  # template only, not weights

SOURCE_OPENCODE = "opencodeinstruct"
SOURCE_REVIEW = "github-codereview"
SOURCE_CVE = "cve-sft-v5"
SOURCE_SCP = "securecodepairs"
SOURCE_CYBER = "cybernative-dpo"


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Build Phase-2 SFT mix (Drive + private Hub)"
    )
    p.add_argument(
        "--out_dir",
        default="/content/drive/MyDrive/coder-sft-mix-v1",
        help="Google Drive (or local) output directory",
    )
    p.add_argument(
        "--hub_dataset_id",
        default="",
        help="Private Hub dataset id (required unless --skip_hub)",
    )
    p.add_argument(
        "--hf_token",
        default="",
        help="HF token; else HF_TOKEN / HUGGING_FACE_HUB_TOKEN / Colab userdata",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max_seq_length", type=int, default=2048)
    p.add_argument("--opencode_n", type=int, default=75_000)
    p.add_argument("--review_n", type=int, default=50_000)
    p.add_argument(
        "--tokenizer_id",
        default=TOKENIZER_ID,
        help="Tokenizer for ChatML length filter only (not model weights)",
    )
    p.add_argument(
        "--skip_hub",
        action="store_true",
        help="Skip Hub push (debug / smoke only)",
    )
    p.add_argument(
        "--smoke_n",
        type=int,
        default=0,
        help="If >0, cap each source early for end-to-end dry run",
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="Allow writing into an existing out_dir (overwrite files)",
    )
    p.add_argument(
        "--samples_n",
        type=int,
        default=50,
        help="Random rows to write into samples.jsonl",
    )
    p.add_argument(
        "--push_only",
        action="store_true",
        help="Skip rebuild; load parquets from --out_dir and push to Hub only",
    )
    return p.parse_args()


def _resolve_token(cli_token: str) -> str:
    token = (cli_token or "").strip()
    if token:
        return token
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN") or ""
    if token:
        return token
    try:
        from google.colab import userdata  # type: ignore

        token = userdata.get("HF_TOKEN") or ""
    except Exception:
        token = ""
    return token


def _maybe_mount_drive(out_dir: Path) -> None:
    if "/content/drive" not in str(out_dir):
        return
    if Path("/content/drive/MyDrive").is_dir():
        print("Drive already mounted.", flush=True)
        return
    try:
        from google.colab import drive  # type: ignore

        drive.mount("/content/drive")
    except Exception as e:
        raise SystemExit(
            f"Need Google Drive for paths under /content/drive but mount failed: {e}"
        )


def _sanitize_text(text: Any) -> str:
    if text is None:
        return ""
    s = str(text)
    s = s.replace("\x00", "")
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    s = s.strip()
    return s


def _norm_for_hash(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def _make_row(system: str, user: str, assistant: str, source: str, task: str) -> dict[str, Any]:
    return {
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
            {"role": "assistant", "content": assistant},
        ],
        "source": source,
        "task": task,
    }


def _user_assistant(row: dict[str, Any]) -> tuple[str, str]:
    msgs = row["messages"]
    user = next(m["content"] for m in msgs if m["role"] == "user")
    assistant = next(m["content"] for m in msgs if m["role"] == "assistant")
    return user, assistant


def _cap_dataset(ds: Any, smoke_n: int, label: str) -> Any:
    if smoke_n <= 0 or len(ds) <= smoke_n:
        return ds
    print(f"  smoke: capping {label} {len(ds)} → {smoke_n}", flush=True)
    return ds.select(range(smoke_n))


def _weighted_sample_indices_np(
    pool: "Any",
    k: int,
    weights: "Any",
    rng: "Any",
) -> "Any":
    """Sample k unique indices from pool using weights (numpy)."""
    import numpy as np

    pool = np.asarray(pool, dtype=np.int64)
    if k <= 0 or len(pool) == 0:
        return np.array([], dtype=np.int64)
    if k >= len(pool):
        out = pool.copy()
        rng.shuffle(out)
        return out
    w = np.asarray(weights, dtype=np.float64)
    w = np.maximum(w, 1e-12)
    w = w / w.sum()
    return rng.choice(pool, size=k, replace=False, p=w)


def _cascade_sample_domain(
    scores: "Any",
    dom_l: "Any",
    domain: str,
    k: int,
    rng: "Any",
    taken: "Any",
) -> "Any":
    """Prefer high test scores, then mid, then low — within one domain."""
    import numpy as np

    if k <= 0:
        return np.array([], dtype=np.int64)
    picked: list[Any] = []
    need = k
    tiers = (
        scores >= 0.8,
        (scores >= 0.5) & (scores < 0.8),
        scores < 0.5,
    )
    for tier in tiers:
        if need <= 0:
            break
        mask = tier & (dom_l == domain) & (~taken)
        idx = np.flatnonzero(mask)
        if idx.size == 0:
            continue
        take_n = min(need, int(idx.size))
        chosen = rng.choice(idx, size=take_n, replace=False)
        taken[chosen] = True
        picked.append(chosen)
        need -= take_n
    if not picked:
        return np.array([], dtype=np.int64)
    return np.concatenate(picked)


# ---------------------------------------------------------------------------
# OpenCodeInstruct
# ---------------------------------------------------------------------------


def build_opencode(ds: Any, n: int, seed: int, smoke_n: int) -> list[dict[str, Any]]:
    """Sample 75K via pandas/numpy on score+domain only; map only selected rows."""
    import numpy as np
    import pandas as pd

    print("=== OpenCodeInstruct select + map ===", flush=True)
    target_n = min(n, smoke_n) if smoke_n else n
    if smoke_n:
        # Keep a larger preload so cascade/domain stratify still works in smoke.
        preload = min(len(ds), max(smoke_n * 50, target_n * 10, 5_000))
        ds = _cap_dataset(ds, preload, "OpenCodeInstruct-preload")

    print(f"  raw rows: {len(ds)}", flush=True)
    print("  loading average_test_score + domain (select_columns → pandas)...", flush=True)
    meta = ds.select_columns(["average_test_score", "domain"]).to_pandas()
    scores = pd.to_numeric(meta["average_test_score"], errors="coerce").fillna(-1.0).to_numpy(
        dtype=np.float64
    )
    dom_l = meta["domain"].astype(str).str.lower().to_numpy()
    del meta
    gc.collect()

    n_high = int((scores >= 0.8).sum())
    n_mid = int(((scores >= 0.5) & (scores < 0.8)).sum())
    n_low = int((scores < 0.5).sum())
    print(
        f"  score tiers: high>=0.8:{n_high} mid>=0.5:{n_mid} low:{n_low}",
        flush=True,
    )

    rng = np.random.default_rng(seed)
    taken = np.zeros(len(scores), dtype=bool)
    half = target_n // 2
    other_half = target_n - half

    print("  cascading sample by domain (numpy)...", flush=True)
    picked_g = _cascade_sample_domain(scores, dom_l, "generic", half, rng, taken)
    picked_a = _cascade_sample_domain(scores, dom_l, "algorithmic", other_half, rng, taken)

    need = target_n - int(picked_g.size) - int(picked_a.size)
    if need > 0:
        # Backfill from any remaining rows, still preferring higher scores.
        print(f"  backfill shortfall={need} from remaining tiers...", flush=True)
        leftover_parts: list[Any] = []
        for tier in (
            scores >= 0.8,
            (scores >= 0.5) & (scores < 0.8),
            scores < 0.5,
        ):
            if need <= 0:
                break
            idx = np.flatnonzero(tier & (~taken))
            if idx.size == 0:
                continue
            take_n = min(need, int(idx.size))
            chosen = rng.choice(idx, size=take_n, replace=False)
            taken[chosen] = True
            leftover_parts.append(chosen)
            need -= take_n
        if leftover_parts:
            backfill = np.concatenate(leftover_parts)
            if picked_g.size < half:
                picked_g = np.concatenate([picked_g, backfill])
            else:
                picked_a = np.concatenate([picked_a, backfill])

    picked = np.concatenate([picked_g, picked_a]) if picked_a.size or picked_g.size else picked_g
    if picked.size > target_n:
        picked = rng.choice(picked, size=target_n, replace=False)
    rng.shuffle(picked)

    n_g = int((dom_l[picked] == "generic").sum()) if picked.size else 0
    n_a = int((dom_l[picked] == "algorithmic").sum()) if picked.size else 0
    print(f"  sampled: {picked.size} (generic≈{n_g}, algorithmic≈{n_a})", flush=True)

    # Load text only for selected indices (sorted for Arrow efficiency).
    print("  mapping selected rows (input/output only)...", flush=True)
    order = np.argsort(picked)
    picked_sorted = picked[order]
    subset = ds.select(picked_sorted.tolist()).select_columns(["input", "output"])
    rows: list[dict[str, Any]] = []
    skipped_empty = 0
    for ex in subset:
        user = _sanitize_text(ex.get("input"))
        assistant = _sanitize_text(ex.get("output"))
        if not user or not assistant:
            skipped_empty += 1
            continue
        rows.append(
            _make_row(
                SYSTEM_PROMPTS["code_instruct"],
                user,
                assistant,
                SOURCE_OPENCODE,
                "code_instruct",
            )
        )
    print(f"  mapped: {len(rows)} (skipped empty: {skipped_empty})", flush=True)
    del scores, dom_l, taken, subset
    gc.collect()
    return rows


# ---------------------------------------------------------------------------
# github-codereview
# ---------------------------------------------------------------------------


def _format_review_user(ex: dict[str, Any]) -> str:
    lang = _sanitize_text(ex.get("language") or ex.get("repo_language") or "unknown")
    path = _sanitize_text(ex.get("file_path") or "")
    diff = _sanitize_text(ex.get("diff_context") or "")
    before = _sanitize_text(ex.get("before_code") or "")
    parts = [
        f"Language: {lang}",
        f"File: {path}" if path else "File: (unknown)",
        "",
        "Diff hunk:",
        "```",
        diff,
        "```",
    ]
    if before:
        parts.extend(["", "Code context (before):", "```", before, "```"])
    parts.extend(["", "Write a concise inline review comment for this change."])
    return "\n".join(parts)


def _format_fix_user(ex: dict[str, Any]) -> str:
    before = _sanitize_text(ex.get("before_code") or "")
    comment = _sanitize_text(ex.get("reviewer_comment") or "")
    lang = _sanitize_text(ex.get("language") or "unknown")
    path = _sanitize_text(ex.get("file_path") or "")
    return "\n".join(
        [
            f"Language: {lang}",
            f"File: {path}" if path else "File: (unknown)",
            "",
            "Original code:",
            "```",
            before,
            "```",
            "",
            "Reviewer feedback:",
            comment,
            "",
            "Apply the feedback and output the corrected code only.",
        ]
    )


def build_codereview(ds: Any, n: int, seed: int, smoke_n: int) -> list[dict[str, Any]]:
    """Vectorized quality pooling via pandas on light columns; map only samples."""
    import numpy as np
    import pandas as pd

    print("=== github-codereview select + map ===", flush=True)
    target_n = min(n, smoke_n) if smoke_n else n
    if smoke_n:
        ds = _cap_dataset(ds, max(smoke_n * 30, 5_000), "codereview-preload")
    print(f"  raw rows: {len(ds)}", flush=True)

    n_pos_target = int(round(target_n * 0.85))
    n_neg_target = target_n - n_pos_target
    n_review_target = int(round(n_pos_target * 0.70))
    n_fix_target = n_pos_target - n_review_target
    # Oversample candidates so empty-field skips during map still fill quotas.
    pos_cand = min(len(ds), max(n_pos_target * 3, n_pos_target + 5_000))
    neg_cand = min(len(ds), max(n_neg_target * 3, n_neg_target + 2_000))

    print("  loading light meta (quality/is_negative/comment_type) → pandas...", flush=True)
    meta = ds.select_columns(
        ["quality_score", "is_negative", "comment_type"]
    ).to_pandas()
    q = pd.to_numeric(meta["quality_score"], errors="coerce").fillna(-1.0)
    is_neg = meta["is_negative"].fillna(False).astype(bool)
    ctype = meta["comment_type"].astype(str).str.lower()
    del meta
    gc.collect()

    def pool_at(threshold: float) -> tuple[Any, Any]:
        pos_mask = (~is_neg) & (q >= threshold) & (ctype != "none")
        neg_mask = is_neg & (q >= threshold)
        return np.flatnonzero(pos_mask.to_numpy()), np.flatnonzero(neg_mask.to_numpy())

    quality_used = 0.75
    pos_idx, neg_idx = pool_at(0.75)
    if len(pos_idx) < n_pos_target or len(neg_idx) < n_neg_target:
        print(
            f"  quality>=0.75 too small (pos={len(pos_idx)}, neg={len(neg_idx)}); "
            "falling back to >=0.6",
            flush=True,
        )
        quality_used = 0.6
        pos_idx, neg_idx = pool_at(0.6)

    print(
        f"  pools @>={quality_used}: positives={len(pos_idx)} negatives={len(neg_idx)} "
        f"(need pos={n_pos_target} neg={n_neg_target})",
        flush=True,
    )

    rng = np.random.default_rng(seed + 1)
    ctype_arr = ctype.to_numpy()
    pos_ctypes = ctype_arr[pos_idx]
    weights = np.where(np.isin(pos_ctypes, list(REVIEW_HIGH_TYPES)), 2.0, 1.0)
    chosen_pos_ids = _weighted_sample_indices_np(
        pos_idx, min(pos_cand, len(pos_idx)), weights, rng
    )
    rng.shuffle(chosen_pos_ids)

    rng_neg = np.random.default_rng(seed + 2)
    neg_take = min(neg_cand, len(neg_idx))
    if neg_take <= 0:
        chosen_neg = np.array([], dtype=np.int64)
    elif neg_take >= len(neg_idx):
        chosen_neg = neg_idx.copy()
        rng_neg.shuffle(chosen_neg)
    else:
        chosen_neg = rng_neg.choice(neg_idx, size=neg_take, replace=False)

    print(
        f"  candidates: pos={len(chosen_pos_ids)} neg={len(chosen_neg)} "
        f"(will keep review={n_review_target} fix={n_fix_target} neg={n_neg_target})",
        flush=True,
    )

    print("  mapping selected review/fix/neg rows...", flush=True)
    review_rows: list[dict[str, Any]] = []
    fix_rows: list[dict[str, Any]] = []
    # Split candidates: first portion for review, rest for fix (no duplicate ids).
    split_at = int(round(len(chosen_pos_ids) * 0.70))
    review_cand = chosen_pos_ids[:split_at]
    fix_cand = chosen_pos_ids[split_at:]

    for i in review_cand.tolist():
        if len(review_rows) >= n_review_target:
            break
        ex = ds[int(i)]
        if not (
            _sanitize_text(ex.get("reviewer_comment"))
            and _sanitize_text(ex.get("diff_context"))
            and _sanitize_text(ex.get("before_code"))
        ):
            continue
        review_rows.append(
            _make_row(
                SYSTEM_PROMPTS["code_review"],
                _format_review_user(ex),
                _sanitize_text(ex.get("reviewer_comment")),
                SOURCE_REVIEW,
                "code_review",
            )
        )

    for i in fix_cand.tolist():
        if len(fix_rows) >= n_fix_target:
            break
        ex = ds[int(i)]
        if not (
            _sanitize_text(ex.get("reviewer_comment"))
            and _sanitize_text(ex.get("before_code"))
            and _sanitize_text(ex.get("after_code"))
        ):
            continue
        fix_rows.append(
            _make_row(
                SYSTEM_PROMPTS["code_review_fix"],
                _format_fix_user(ex),
                _sanitize_text(ex.get("after_code")),
                SOURCE_REVIEW,
                "code_review_fix",
            )
        )

    neg_rows: list[dict[str, Any]] = []
    for i in chosen_neg.tolist():
        if len(neg_rows) >= n_neg_target:
            break
        ex = ds[int(i)]
        if not (
            _sanitize_text(ex.get("diff_context")) or _sanitize_text(ex.get("before_code"))
        ):
            continue
        neg_rows.append(
            _make_row(
                SYSTEM_PROMPTS["code_review"],
                _format_review_user(ex),
                ACCEPT_PHRASE,
                SOURCE_REVIEW,
                "code_review",
            )
        )

    rows = review_rows + fix_rows + neg_rows
    print(
        f"  mapped: {len(rows)} "
        f"(review={len(review_rows)} fix={len(fix_rows)} neg={len(neg_rows)})",
        flush=True,
    )
    return rows


# ---------------------------------------------------------------------------
# Security sources
# ---------------------------------------------------------------------------


def build_cve(ds: Any, smoke_n: int) -> list[dict[str, Any]]:
    print("=== cve-sft-v5 map ===", flush=True)
    ds = _cap_dataset(ds, smoke_n if smoke_n else 0, "cve")
    rows: list[dict[str, Any]] = []
    for ex in ds:
        cve_id = _sanitize_text(ex.get("cve_id"))
        software = _sanitize_text(ex.get("affected_software"))
        cvss = ex.get("cvss_score")
        cwe = _sanitize_text(ex.get("cwe_id"))
        vector = _sanitize_text(ex.get("cvss_vector"))
        user = (
            f"Analyze the following vulnerability: {cve_id}\n"
            f"Affected software: {software}\n"
            f"CVSS Score: {cvss} | CWE: {cwe}\n"
            f"CVSS Vector: {vector}"
        )
        assistant = "\n\n".join(
            [
                f"## Plain Explanation\n{_sanitize_text(ex.get('plain_explanation'))}",
                f"## Technical Deep Dive\n{_sanitize_text(ex.get('technical_deep_dive'))}",
                f"## Attack Scenario\n{_sanitize_text(ex.get('attack_scenario'))}",
                f"## Remediation\n{_sanitize_text(ex.get('remediation'))}",
                f"## Code Example\n{_sanitize_text(ex.get('vulnerable_code_example'))}",
            ]
        )
        if not cve_id or not _sanitize_text(ex.get("plain_explanation")):
            continue
        rows.append(
            _make_row(
                SYSTEM_PROMPTS["security"],
                user,
                assistant,
                SOURCE_CVE,
                "security",
            )
        )
    print(f"  mapped: {len(rows)}", flush=True)
    return rows


def _scp_review_user(ex: dict[str, Any]) -> str:
    lang = _sanitize_text(ex.get("language") or "unknown")
    title = _sanitize_text(ex.get("title") or "")
    desc = _sanitize_text(ex.get("description") or "")
    code = _sanitize_text(ex.get("vulnerable_code") or "")
    return "\n".join(
        [
            f"Language: {lang}",
            f"Title: {title}" if title else "",
            f"Description: {desc}" if desc else "",
            "",
            "Review this code for security vulnerabilities:",
            "```",
            code,
            "```",
            "",
            "Explain the issue and provide a secure version.",
        ]
    ).strip()


def _scp_review_assistant(ex: dict[str, Any]) -> str:
    parts = [
        f"Root cause: {_sanitize_text(ex.get('root_cause'))}",
        f"Attack: {_sanitize_text(ex.get('attack'))}",
        f"Fix guidance: {_sanitize_text(ex.get('fix'))}",
        "",
        "Secure code:",
        "```",
        _sanitize_text(ex.get("secure_code")),
        "```",
    ]
    return "\n".join(parts)


def _scp_rewrite_user(ex: dict[str, Any]) -> str:
    lang = _sanitize_text(ex.get("language") or "unknown")
    code = _sanitize_text(ex.get("vulnerable_code") or "")
    return "\n".join(
        [
            f"Language: {lang}",
            "",
            "Rewrite the following code securely:",
            "```",
            code,
            "```",
            "",
            "Output the secure implementation and a short fix note.",
        ]
    )


def _scp_rewrite_assistant(ex: dict[str, Any]) -> str:
    fix = _sanitize_text(ex.get("fix") or ex.get("guideline") or "")
    return "\n".join(
        [
            "```",
            _sanitize_text(ex.get("secure_code")),
            "```",
            "",
            f"Fix: {fix}" if fix else "",
        ]
    ).strip()


def build_securecodepairs_train(ds: Any, smoke_n: int) -> list[dict[str, Any]]:
    print("=== SecureCodePairs train (2 variants) ===", flush=True)
    ds = _cap_dataset(ds, smoke_n if smoke_n else 0, "scp-train")
    rows: list[dict[str, Any]] = []
    for ex in ds:
        if not _sanitize_text(ex.get("vulnerable_code")) or not _sanitize_text(
            ex.get("secure_code")
        ):
            continue
        rows.append(
            _make_row(
                SYSTEM_PROMPTS["security"],
                _scp_review_user(ex),
                _scp_review_assistant(ex),
                SOURCE_SCP,
                "security",
            )
        )
        rows.append(
            _make_row(
                SYSTEM_PROMPTS["security"],
                _scp_rewrite_user(ex),
                _scp_rewrite_assistant(ex),
                SOURCE_SCP,
                "security",
            )
        )
    print(f"  mapped: {len(rows)} (2× train rows)", flush=True)
    return rows


def build_securecodepairs_eval(splits: dict[str, Any], smoke_n: int) -> list[dict[str, Any]]:
    print("=== SecureCodePairs eval holdout ===", flush=True)
    rows: list[dict[str, Any]] = []
    for name, ds in splits.items():
        if ds is None:
            continue
        capped = _cap_dataset(ds, smoke_n if smoke_n else 0, f"scp-{name}")
        for ex in capped:
            if not _sanitize_text(ex.get("vulnerable_code")) or not _sanitize_text(
                ex.get("secure_code")
            ):
                continue
            # Single canonical eval form (review variant)
            row = _make_row(
                SYSTEM_PROMPTS["security"],
                _scp_review_user(ex),
                _scp_review_assistant(ex),
                SOURCE_SCP,
                "security",
            )
            row["eval_split"] = name
            rows.append(row)
    print(f"  eval mapped: {len(rows)}", flush=True)
    return rows


def build_cybernative(ds: Any, smoke_n: int) -> list[dict[str, Any]]:
    print("=== CyberNative DPO → SFT (chosen only) ===", flush=True)
    ds = _cap_dataset(ds, smoke_n if smoke_n else 0, "cybernative")
    rows: list[dict[str, Any]] = []
    dropped_identical = 0
    for ex in ds:
        question = _sanitize_text(ex.get("question"))
        chosen = _sanitize_text(ex.get("chosen"))
        rejected = _sanitize_text(ex.get("rejected"))
        if not question or not chosen:
            continue
        if rejected and _norm_for_hash(chosen) == _norm_for_hash(rejected):
            dropped_identical += 1
            continue
        rows.append(
            _make_row(
                SYSTEM_PROMPTS["security"],
                question,
                chosen,
                SOURCE_CYBER,
                "security",
            )
        )
    print(f"  mapped: {len(rows)} (dropped identical chosen/rejected: {dropped_identical})", flush=True)
    return rows


# ---------------------------------------------------------------------------
# Global cleanup
# ---------------------------------------------------------------------------


def cleanup_rows(
    rows: list[dict[str, Any]],
    tokenizer: Any,
    max_seq_length: int,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Sanitize → min-length → dedup → fast char prefilter → tokenize borderline."""
    print("=== Global cleanup ===", flush=True)
    stats = {
        "input": len(rows),
        "dropped_empty": 0,
        "dropped_min_length": 0,
        "dropped_dedup": 0,
        "dropped_seq_char": 0,
        "dropped_seq_tok": 0,
        "seq_tokenized": 0,
        "kept": 0,
    }
    cleaned: list[dict[str, Any]] = []
    for row in rows:
        msgs = row["messages"]
        for m in msgs:
            m["content"] = _sanitize_text(m["content"])
        user, assistant = _user_assistant(row)
        if not user or not assistant:
            stats["dropped_empty"] += 1
            continue
        is_accept = assistant == ACCEPT_PHRASE
        if len(user) < MIN_USER_CHARS:
            stats["dropped_min_length"] += 1
            continue
        if not is_accept and len(assistant) < MIN_ASSISTANT_CHARS:
            stats["dropped_min_length"] += 1
            continue
        cleaned.append(row)
    print(f"  after sanitize/min-len: {len(cleaned)}", flush=True)

    seen: set[str] = set()
    deduped: list[dict[str, Any]] = []
    for row in cleaned:
        user, assistant = _user_assistant(row)
        key = hashlib.md5(
            f"{_norm_for_hash(user)}\n{_norm_for_hash(assistant)}".encode("utf-8")
        ).hexdigest()
        if key in seen:
            stats["dropped_dedup"] += 1
            continue
        seen.add(key)
        deduped.append(row)
    print(f"  after dedup: {len(deduped)}", flush=True)

    # Char prefilter: ~chars/4 ≈ tokens for code-heavy text (conservative).
    # Keep if estimate << limit; drop if estimate >> limit; else tokenize.
    chars_keep = max_seq_length * 3  # clearly under
    chars_drop = max_seq_length * 6  # clearly over
    kept: list[dict[str, Any]] = []
    borderline: list[dict[str, Any]] = []

    for row in deduped:
        user, assistant = _user_assistant(row)
        sys_c = row["messages"][0]["content"]
        n_chars = len(sys_c) + len(user) + len(assistant) + 64  # ChatML overhead
        if n_chars > chars_drop:
            stats["dropped_seq_char"] += 1
            continue
        if n_chars <= chars_keep:
            kept.append(row)
        else:
            borderline.append(row)

    print(
        f"  char prefilter: kept={len(kept)} borderline={len(borderline)} "
        f"dropped={stats['dropped_seq_char']}",
        flush=True,
    )

    for i, row in enumerate(borderline):
        if i and i % 2_000 == 0:
            print(f"  seq tokenize progress: {i}/{len(borderline)}", flush=True)
        stats["seq_tokenized"] += 1
        try:
            ids = tokenizer.apply_chat_template(
                row["messages"],
                tokenize=True,
                add_generation_prompt=False,
            )
            n_tok = len(ids)
        except Exception:
            user, assistant = _user_assistant(row)
            sys_c = row["messages"][0]["content"]
            n_tok = max(1, (len(sys_c) + len(user) + len(assistant)) // 3)
        if n_tok > max_seq_length:
            stats["dropped_seq_tok"] += 1
            continue
        kept.append(row)

    stats["kept"] = len(kept)
    print(f"  cleanup stats: {stats}", flush=True)
    return kept, stats


def _count_tags(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_source = Counter(r["source"] for r in rows)
    by_task = Counter(r["task"] for r in rows)
    return {"by_source": dict(by_source), "by_task": dict(by_task), "total": len(rows)}


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------


def write_artifacts(
    out_dir: Path,
    train_rows: list[dict[str, Any]],
    eval_rows: list[dict[str, Any]],
    manifest: dict[str, Any],
    samples_n: int,
    seed: int,
) -> None:
    from datasets import Dataset

    out_dir.mkdir(parents=True, exist_ok=True)
    train_path = out_dir / "sft_mix_v1.parquet"
    eval_path = out_dir / "sft_eval_securecodepairs.parquet"
    manifest_path = out_dir / "manifest.json"
    samples_path = out_dir / "samples.jsonl"

    print(f"Writing {train_path} ({len(train_rows)} rows)...", flush=True)
    Dataset.from_list(train_rows).to_parquet(str(train_path))

    print(f"Writing {eval_path} ({len(eval_rows)} rows)...", flush=True)
    Dataset.from_list(eval_rows).to_parquet(str(eval_path))

    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"Wrote {manifest_path}", flush=True)

    rng = random.Random(seed + 99)
    sample_idx = list(range(len(train_rows)))
    rng.shuffle(sample_idx)
    sample_idx = sample_idx[: min(samples_n, len(train_rows))]
    with samples_path.open("w", encoding="utf-8") as f:
        for i in sample_idx:
            f.write(json.dumps(train_rows[i], ensure_ascii=False) + "\n")
    print(f"Wrote {samples_path} ({len(sample_idx)} samples)", flush=True)


def push_hub(
    hub_dataset_id: str,
    train_rows: list[dict[str, Any]],
    eval_rows: list[dict[str, Any]],
    token: str,
) -> None:
    from datasets import Dataset, DatasetDict

    print(f"Pushing private dataset → {hub_dataset_id} ...", flush=True)
    # DatasetDict requires identical features across splits — pad eval_split on train.
    train_clean = [
        {
            "messages": r["messages"],
            "source": r["source"],
            "task": r["task"],
            "eval_split": "",
        }
        for r in train_rows
    ]
    eval_clean = [
        {
            "messages": r["messages"],
            "source": r["source"],
            "task": r["task"],
            "eval_split": r.get("eval_split", ""),
        }
        for r in eval_rows
    ]
    train_ds = Dataset.from_list(train_clean)
    if eval_clean:
        eval_ds = Dataset.from_list(eval_clean)
    else:
        eval_ds = train_ds.select([])
    dsd = DatasetDict({"train": train_ds, "eval_securecodepairs": eval_ds})
    dsd.push_to_hub(hub_dataset_id, private=True, token=token)
    print("Hub push done.", flush=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    for k in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE"):
        os.environ.pop(k, None)

    args = _parse_args()
    out_dir = Path(args.out_dir).expanduser().resolve()
    token = _resolve_token(args.hf_token)
    if token:
        os.environ["HF_TOKEN"] = token
        os.environ["HUGGING_FACE_HUB_TOKEN"] = token

    if not args.skip_hub:
        if not args.hub_dataset_id.strip():
            raise SystemExit("--hub_dataset_id is required unless --skip_hub")
        if not token:
            raise SystemExit("HF token required for Hub push (or pass --skip_hub)")

    _maybe_mount_drive(out_dir)
    if (
        not args.push_only
        and out_dir.exists()
        and any(out_dir.iterdir())
        and not args.force
    ):
        raise SystemExit(
            f"out_dir already exists and is non-empty: {out_dir}\n"
            "Pass --force to overwrite, or choose a new --out_dir."
        )
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Out:       ", out_dir, flush=True)
    print("Hub:       ", args.hub_dataset_id or "(skipped)", flush=True)
    print("Seed:      ", args.seed, flush=True)
    print("Max seq:   ", args.max_seq_length, flush=True)
    print("Tokenizer: ", args.tokenizer_id, "(length filter / ChatML only)", flush=True)
    print("Smoke n:   ", args.smoke_n, flush=True)
    if not token:
        print(
            "WARNING: no HF_TOKEN — downloads are unauthenticated (slower / rate-limited). "
            "Set Colab secret HF_TOKEN or pass --hf_token.",
            flush=True,
        )

    if args.push_only:
        from datasets import Dataset, DatasetDict

        train_path = out_dir / "sft_mix_v1.parquet"
        eval_path = out_dir / "sft_eval_securecodepairs.parquet"
        if not train_path.is_file():
            raise SystemExit(f"--push_only requires {train_path}")
        print(f"Loading train from {train_path} ...", flush=True)
        train_ds = Dataset.from_parquet(str(train_path))
        if "eval_split" not in train_ds.column_names:
            train_ds = train_ds.add_column("eval_split", [""] * len(train_ds))
        else:
            # Normalize nulls to empty string
            train_ds = train_ds.map(
                lambda x: {"eval_split": x.get("eval_split") or ""},
                desc="normalize train eval_split",
            )

        if eval_path.is_file():
            print(f"Loading eval from {eval_path} ...", flush=True)
            eval_ds = Dataset.from_parquet(str(eval_path))
            if "eval_split" not in eval_ds.column_names:
                eval_ds = eval_ds.add_column("eval_split", [""] * len(eval_ds))
            # Keep only shared columns in a stable order
            cols = ["messages", "source", "task", "eval_split"]
            train_ds = train_ds.select_columns(cols)
            eval_ds = eval_ds.select_columns(cols)
        else:
            train_ds = train_ds.select_columns(
                ["messages", "source", "task", "eval_split"]
            )
            eval_ds = train_ds.select([])

        print(
            f"Push-only: train={len(train_ds)} eval={len(eval_ds)}",
            flush=True,
        )
        dsd = DatasetDict({"train": train_ds, "eval_securecodepairs": eval_ds})
        dsd.push_to_hub(args.hub_dataset_id.strip(), private=True, token=token)
        print("Hub push done.", flush=True)
        print("\nDone (push_only).", flush=True)
        return

    from datasets import load_dataset
    from transformers import AutoTokenizer

    train_rows: list[dict[str, Any]] = []

    # Process OpenCode first, then free it — keeps Colab RAM under control.
    print("\n=== Download + sample OpenCodeInstruct ===", flush=True)
    opencode = load_dataset(
        "nvidia/OpenCodeInstruct", split="train", token=token or None
    )
    print(f"  OpenCodeInstruct: {len(opencode)}", flush=True)
    train_rows.extend(build_opencode(opencode, args.opencode_n, args.seed, args.smoke_n))
    del opencode
    gc.collect()

    print("\n=== Download + sample github-codereview ===", flush=True)
    review = load_dataset(
        "ronantakizawa/github-codereview", split="train", token=token or None
    )
    print(f"  github-codereview train: {len(review)}", flush=True)
    train_rows.extend(build_codereview(review, args.review_n, args.seed, args.smoke_n))
    del review
    gc.collect()

    print("\n=== Download + map security sets ===", flush=True)
    cve = load_dataset("auren-research/cve-sft-v5", split="train", token=token or None)
    print(f"  cve-sft-v5: {len(cve)}", flush=True)
    train_rows.extend(build_cve(cve, args.smoke_n))
    del cve
    gc.collect()

    scp_all = load_dataset("ismailtasdelen/SecureCodePairs", token=token or None)
    if hasattr(scp_all, "keys"):
        scp_train = scp_all["train"]
        eval_splits = {
            k: scp_all[k]
            for k in ("validation", "test", "benchmark")
            if k in scp_all
        }
    else:
        scp_train = scp_all
        eval_splits = {}
    print(
        f"  SecureCodePairs train: {len(scp_train)} eval_splits={list(eval_splits)}",
        flush=True,
    )
    train_rows.extend(build_securecodepairs_train(scp_train, args.smoke_n))
    eval_rows = build_securecodepairs_eval(eval_splits, args.smoke_n)
    del scp_all, scp_train, eval_splits
    gc.collect()

    cyber = load_dataset(
        "CyberNative/Code_Vulnerability_Security_DPO",
        split="train",
        token=token or None,
    )
    print(f"  CyberNative DPO: {len(cyber)}", flush=True)
    train_rows.extend(build_cybernative(cyber, args.smoke_n))
    del cyber
    gc.collect()

    print(f"\nMapped train rows (pre-cleanup): {len(train_rows)}", flush=True)

    print(f"\nLoading tokenizer for length filter: {args.tokenizer_id}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_id,
        trust_remote_code=True,
        token=token or None,
    )
    if tokenizer.chat_template is None:
        raise SystemExit(
            f"Tokenizer {args.tokenizer_id} has no chat_template; "
            "cannot length-filter conversational rows."
        )

    train_rows, clean_stats = cleanup_rows(train_rows, tokenizer, args.max_seq_length)
    eval_rows, eval_clean_stats = cleanup_rows(eval_rows, tokenizer, args.max_seq_length)

    rng = random.Random(args.seed)
    rng.shuffle(train_rows)

    counts = _count_tags(train_rows)
    print(f"\nFinal train counts: {counts}", flush=True)

    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "seed": args.seed,
        "max_seq_length": args.max_seq_length,
        "tokenizer_id_length_filter": args.tokenizer_id,
        "tokenizer_note": (
            "Instruct tokenizer used for ChatML length filter only; "
            "Phase-2 model weights stay on Base→CPT-merge lineage."
        ),
        "opencode_n_target": args.opencode_n,
        "review_n_target": args.review_n,
        "smoke_n": args.smoke_n,
        "min_user_chars": MIN_USER_CHARS,
        "min_assistant_chars": MIN_ASSISTANT_CHARS,
        "system_prompts": SYSTEM_PROMPTS,
        "cleanup_train": clean_stats,
        "cleanup_eval": eval_clean_stats,
        "final_train": counts,
        "final_eval_total": len(eval_rows),
        "hub_dataset_id": args.hub_dataset_id or None,
        "out_dir": str(out_dir),
        "sources": [
            "nvidia/OpenCodeInstruct",
            "ronantakizawa/github-codereview",
            "auren-research/cve-sft-v5",
            "ismailtasdelen/SecureCodePairs",
            "CyberNative/Code_Vulnerability_Security_DPO",
        ],
    }

    write_artifacts(
        out_dir, train_rows, eval_rows, manifest, args.samples_n, args.seed
    )

    if not args.skip_hub:
        push_hub(args.hub_dataset_id.strip(), train_rows, eval_rows, token)

    print("\nDone.", flush=True)
    print(f"  train → {out_dir / 'sft_mix_v1.parquet'}", flush=True)
    print(f"  eval  → {out_dir / 'sft_eval_securecodepairs.parquet'}", flush=True)
    print(f"  meta  → {out_dir / 'manifest.json'}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        sys.exit(130)

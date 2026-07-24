#!/usr/bin/env python3
"""
Build Phase-3 CoT Reasoning Conversational Mix (~25K samples).

Extracts, cleans, normalizes, and packages Chain-of-Thought (CoT) reasoning traces
from 7 upstream datasets into Qwen2.5 <think> ChatML format.

Data Sources:
  Code & Algorithmic Reasoning (~15K target):
    1. open-r1/codeforces-cots           (2,500 target)
    2. bespokelabs/Bespoke-Stratos-17k   (2,500 target)
    3. open-thoughts/OpenThoughts-114k   (5,000 target)
    4. Glint-Research/Fable-5-traces     (4,600 target)
    5. Roman1111111/gpt5.5-terminal      (131 target - all valid)

  Security & Vulnerability CoT (~10K target):
    6. samscrack/solidity-audit-cot      (5,100 target)
    7. trendmicro-ailab/Primus-Reasoning (4,900 target)

Held-out Eval Split:
  500 unique samples held out from the extracted pool before training mix creation.

Target Format:
  messages: [
    {"role": "system", "content": "..."},
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "<think>\\n...\\n</think>\\n\\n..."}
  ]

Typical Colab usage:
  1. Runtime -> CPU / T4 GPU
  2. Mount Drive; set HF_TOKEN
  3. !pip install -q datasets transformers pyarrow huggingface_hub pandas numpy
  4. !python preprocess_cot_mix_colab.py \
       --hub_dataset_id Aniket200325/coder-reasoning-cot-v1 \
       --max_seq_length 8192

  Smoke test:
  !python preprocess_cot_mix_colab.py --smoke_n 50 --skip_hub \
       --out_dir /content/drive/MyDrive/coder-reasoning-cot-v1-smoke
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
# Constants & System Prompts
# ---------------------------------------------------------------------------

SYSTEM_PROMPTS: dict[str, str] = {
    "code_reasoning": (
        "You are an expert coding assistant. Think through problems step-by-step "
        "in <think> tags before providing your solution."
    ),
    "security_reasoning": (
        "You are a security analyst. Analyze vulnerabilities step-by-step "
        "in <think> tags before providing your findings and remediation."
    ),
}

MIN_USER_CHARS = 24
MIN_THINK_CHARS = 40
MIN_ASSISTANT_CHARS = 16
TOKENIZER_ID = "Qwen/Qwen2.5-Coder-7B-Instruct"  # for ChatML length filtering only

SRC_CODEFORCES = "codeforces-cots"
SRC_STRATOS = "bespoke-stratos"
SRC_OPENTHOUGHTS = "openthoughts"
SRC_FABLE = "fable5-traces"
SRC_GPT55_TERM = "gpt55-terminal"
SRC_SOLIDITY = "solidity-audit-cot"
SRC_PRIMUS = "primus-reasoning"


# ---------------------------------------------------------------------------
# Utility & Helper Functions
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Build Phase-3 CoT Reasoning Dataset (Drive + private Hub)"
    )
    p.add_argument(
        "--out_dir",
        default="/content/drive/MyDrive/coder-reasoning-cot-v1",
        help="Google Drive (or local) output directory",
    )
    p.add_argument(
        "--hub_dataset_id",
        default="Aniket200325/coder-reasoning-cot-v1",
        help="Private Hub dataset id",
    )
    p.add_argument(
        "--hf_token",
        default="",
        help="HF token; else HF_TOKEN / HUGGING_FACE_HUB_TOKEN / Colab userdata",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max_seq_length", type=int, default=8192)
    p.add_argument("--eval_n", type=int, default=500, help="Held-out eval set size")
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


def _maybe_mount_drive(out_dir: Path) -> Path:
    if "/content/drive" not in str(out_dir):
        return out_dir
    if Path("/content/drive/MyDrive").is_dir():
        print("Drive already mounted.", flush=True)
        return out_dir
    try:
        from google.colab import drive  # type: ignore
        drive.mount("/content/drive")
        return out_dir
    except Exception as e:
        fallback = Path("/kaggle/working/coder-reasoning-cot-v1" if Path("/kaggle/working").is_dir() else "./coder-reasoning-cot-v1").resolve()
        print(f"Notice: Google Drive unavailable ({e}). Using local fallback output dir: {fallback}", flush=True)
        return fallback


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


def _format_think_assistant(think_content: str, solution_content: str) -> str:
    """Format thought and solution into standard Qwen <think> format."""
    think = _sanitize_text(think_content)
    solution = _sanitize_text(solution_content)
    
    # Strip any existing <think> tags inside reasoning string
    think = re.sub(r"</?think>", "", think).strip()
    solution = re.sub(r"</?think>", "", solution).strip()
    
    return f"<think>\n{think}\n</think>\n\n{solution}"


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


def _cap_dataset(ds: Any, smoke_n: int, label: str) -> Any:
    if smoke_n <= 0 or len(ds) <= smoke_n:
        return ds
    print(f"  smoke: capping {label} {len(ds)} -> {smoke_n}", flush=True)
    return ds.select(range(smoke_n))


# ---------------------------------------------------------------------------
# Source Builders
# ---------------------------------------------------------------------------

def build_codeforces(token: str, target_n: int, seed: int, smoke_n: int) -> list[dict[str, Any]]:
    """Source 1: open-r1/codeforces-cots (solutions_py or solutions split)."""
    from datasets import load_dataset
    print("=== Extracting open-r1/codeforces-cots ===", flush=True)
    
    ds = None
    for config_name in ("solutions_py", "solutions_decontaminated", "solutions", "default"):
        try:
            ds = load_dataset("open-r1/codeforces-cots", config_name, split="train", token=token or None)
            print(f"  Loaded config '{config_name}': {len(ds)} rows", flush=True)
            break
        except Exception:
            continue
            
    if ds is None:
        try:
            ds = load_dataset("open-r1/codeforces-cots", split="train", token=token or None)
            print(f"  Loaded default split: {len(ds)} rows", flush=True)
        except Exception as e:
            print(f"  WARNING: Failed to load open-r1/codeforces-cots ({e}); skipping.", flush=True)
            return []

    ds = _cap_dataset(ds, smoke_n if smoke_n else 0, "codeforces")
    rows: list[dict[str, Any]] = []

    for ex in ds:
        user, assistant = "", ""
        if "messages" in ex and isinstance(ex["messages"], list):
            for m in ex["messages"]:
                r = m.get("role") or m.get("from")
                c = _sanitize_text(m.get("content") or m.get("value"))
                if r in ("user", "human"):
                    user = c
                elif r in ("assistant", "gpt"):
                    assistant = c
        elif "problem" in ex and ("solution" in ex or "completion" in ex):
            user = _sanitize_text(ex.get("problem"))
            think = _sanitize_text(ex.get("reasoning") or ex.get("cot") or "")
            sol = _sanitize_text(ex.get("solution") or ex.get("completion") or "")
            assistant = _format_think_assistant(think, sol) if think else sol

        if not user or not assistant:
            continue

        # Enforce <think> format if not already present
        if "<think>" not in assistant:
            if "reasoning" in ex and ex["reasoning"]:
                assistant = _format_think_assistant(ex["reasoning"], assistant)

        rows.append(_make_row(SYSTEM_PROMPTS["code_reasoning"], user, assistant, SRC_CODEFORCES, "code_reasoning"))

    if smoke_n:
        target_n = min(target_n, smoke_n)
    if len(rows) > target_n:
        rng = random.Random(seed)
        rows = rng.sample(rows, target_n)
    print(f"  mapped: {len(rows)}", flush=True)
    return rows


def build_bespoke_stratos(token: str, target_n: int, seed: int, smoke_n: int) -> list[dict[str, Any]]:
    """Source 2: bespokelabs/Bespoke-Stratos-17k."""
    from datasets import load_dataset
    print("=== Extracting bespokelabs/Bespoke-Stratos-17k ===", flush=True)
    
    try:
        ds = load_dataset("bespokelabs/Bespoke-Stratos-17k", split="train", token=token or None)
    except Exception as e:
        print(f"  WARNING: Failed to load Bespoke-Stratos-17k ({e}); skipping.", flush=True)
        return []

    print(f"  raw rows: {len(ds)}", flush=True)
    ds = _cap_dataset(ds, smoke_n if smoke_n else 0, "stratos")
    rows: list[dict[str, Any]] = []

    for ex in ds:
        user, assistant = "", ""
        convs = ex.get("conversations") or ex.get("messages") or []
        for m in convs:
            r = m.get("role") or m.get("from")
            c = _sanitize_text(m.get("content") or m.get("value"))
            if r in ("user", "human"):
                user = c
            elif r in ("assistant", "gpt"):
                assistant = c

        if not user or not assistant:
            continue

        # Format check for <think> tags
        if "<think>" not in assistant and "</think>" not in assistant:
            # Check if there is a 'Thought' / 'Solution' pattern in Stratos
            if "Thought:" in assistant and "Solution:" in assistant:
                parts = assistant.split("Solution:", 1)
                think_part = parts[0].replace("Thought:", "").strip()
                sol_part = parts[1].strip()
                assistant = _format_think_assistant(think_part, sol_part)
            else:
                lines = assistant.split("\n\n")
                if len(lines) >= 2:
                    think_part = "\n\n".join(lines[:-1])
                    sol_part = lines[-1]
                    assistant = _format_think_assistant(think_part, sol_part)

        rows.append(_make_row(SYSTEM_PROMPTS["code_reasoning"], user, assistant, SRC_STRATOS, "code_reasoning"))

    if smoke_n:
        target_n = min(target_n, smoke_n)
    if len(rows) > target_n:
        rng = random.Random(seed + 1)
        rows = rng.sample(rows, target_n)
    print(f"  mapped: {len(rows)}", flush=True)
    return rows


def build_openthoughts(token: str, target_n: int, seed: int, smoke_n: int) -> list[dict[str, Any]]:
    """Source 3: open-thoughts/OpenThoughts-114k."""
    from datasets import load_dataset
    print("=== Extracting open-thoughts/OpenThoughts-114k ===", flush=True)

    ds = None
    for config_name in ("metadata", "default"):
        try:
            ds = load_dataset("open-thoughts/OpenThoughts-114k", config_name, split="train", token=token or None)
            print(f"  Loaded config '{config_name}': {len(ds)} rows", flush=True)
            break
        except Exception:
            continue

    if ds is None:
        try:
            ds = load_dataset("open-thoughts/OpenThoughts-114k", split="train", token=token or None)
        except Exception as e:
            print(f"  WARNING: Failed to load OpenThoughts-114k ({e}); skipping.", flush=True)
            return []

    ds = _cap_dataset(ds, smoke_n if smoke_n else 0, "openthoughts")
    rows: list[dict[str, Any]] = []

    for ex in ds:
        user, assistant = "", ""
        if "problem" in ex and ("deepseek_reasoning" in ex or "reasoning" in ex):
            user = _sanitize_text(ex.get("problem"))
            reasoning = _sanitize_text(ex.get("deepseek_reasoning") or ex.get("reasoning"))
            sol = _sanitize_text(ex.get("deepseek_solution") or ex.get("solution") or ex.get("ground_truth_solution"))
            if reasoning and sol:
                assistant = _format_think_assistant(reasoning, sol)
        elif "conversations" in ex:
            convs = ex["conversations"]
            for m in convs:
                r = m.get("role") or m.get("from")
                c = _sanitize_text(m.get("content") or m.get("value"))
                if r in ("user", "human"):
                    user = c
                elif r in ("assistant", "gpt"):
                    assistant = c

        if not user or not assistant:
            continue

        rows.append(_make_row(SYSTEM_PROMPTS["code_reasoning"], user, assistant, SRC_OPENTHOUGHTS, "code_reasoning"))

    if smoke_n:
        target_n = min(target_n, smoke_n)
    if len(rows) > target_n:
        rng = random.Random(seed + 2)
        rows = rng.sample(rows, target_n)
    print(f"  mapped: {len(rows)}", flush=True)
    return rows


def build_fable5(token: str, target_n: int, seed: int, smoke_n: int) -> list[dict[str, Any]]:
    """Source 4: Glint-Research/Fable-5-traces."""
    from datasets import load_dataset
    print("=== Extracting Glint-Research/Fable-5-traces ===", flush=True)
    
    try:
        ds = load_dataset("Glint-Research/Fable-5-traces", split="train", token=token or None)
    except Exception as e:
        print(f"  WARNING: Could not load Glint-Research/Fable-5-traces ({e}); skipping.", flush=True)
        return []

    ds = _cap_dataset(ds, smoke_n if smoke_n else 0, "fable5")
    rows: list[dict[str, Any]] = []

    for ex in ds:
        out_type = _sanitize_text(ex.get("output_type"))
        if out_type and out_type not in ("text", "chat", "response", ""):
            continue

        context = _sanitize_text(ex.get("context") or ex.get("user") or ex.get("prompt"))
        cot = _sanitize_text(ex.get("cot") or ex.get("reasoning") or ex.get("thinking"))
        out = _sanitize_text(ex.get("output") or ex.get("completion") or ex.get("response"))

        if not context or not out:
            continue

        if cot:
            assistant = _format_think_assistant(cot, out)
        else:
            assistant = out

        rows.append(_make_row(SYSTEM_PROMPTS["code_reasoning"], context, assistant, SRC_FABLE, "code_reasoning"))

    if smoke_n:
        target_n = min(target_n, smoke_n)
    if len(rows) > target_n:
        rng = random.Random(seed + 3)
        rows = rng.sample(rows, target_n)
    print(f"  mapped: {len(rows)}", flush=True)
    return rows


def build_gpt55_terminal(token: str, target_n: int, seed: int, smoke_n: int) -> list[dict[str, Any]]:
    """Source 5: Roman1111111/gpt5.5-terminal (~131 target)."""
    from datasets import load_dataset
    print("=== Extracting Roman1111111/gpt5.5-terminal ===", flush=True)
    
    try:
        ds = load_dataset("Roman1111111/gpt5.5-terminal", split="train", token=token or None)
    except Exception as e:
        print(f"  WARNING: Could not load Roman1111111/gpt5.5-terminal ({e}); skipping.", flush=True)
        return []

    ds = _cap_dataset(ds, smoke_n if smoke_n else 0, "gpt55_terminal")
    rows: list[dict[str, Any]] = []

    for ex in ds:
        user = _sanitize_text(ex.get("task") or ex.get("instruction") or ex.get("user") or ex.get("prompt"))
        reasoning = _sanitize_text(ex.get("reasoning") or ex.get("analysis") or ex.get("cot") or "")
        solution = _sanitize_text(ex.get("solution") or ex.get("commands") or ex.get("output") or ex.get("completion"))

        if not user or not solution:
            continue

        assistant = _format_think_assistant(reasoning, solution) if reasoning else solution
        rows.append(_make_row(SYSTEM_PROMPTS["code_reasoning"], user, assistant, SRC_GPT55_TERM, "code_reasoning"))

    print(f"  mapped: {len(rows)}", flush=True)
    return rows


def build_solidity_audit(token: str, target_n: int, seed: int, smoke_n: int) -> list[dict[str, Any]]:
    """Source 6: samscrack/solidity-audit-cot (5,100 target)."""
    from datasets import load_dataset
    print("=== Extracting samscrack/solidity-audit-cot ===", flush=True)
    
    try:
        ds = load_dataset("samscrack/solidity-audit-cot", split="train", token=token or None)
    except Exception as e:
        print(f"  WARNING: Could not load samscrack/solidity-audit-cot ({e}); skipping.", flush=True)
        return []

    ds = _cap_dataset(ds, smoke_n if smoke_n else 0, "solidity_audit")
    rows: list[dict[str, Any]] = []

    for ex in ds:
        instr = _sanitize_text(ex.get("input_instruction"))
        code = _sanitize_text(ex.get("contract_code") or ex.get("vulnerable_code") or ex.get("code"))
        r_steps = ex.get("reasoning_steps") or ex.get("reasoning") or ex.get("cot")
        finding = _sanitize_text(ex.get("finding") or ex.get("verdict") or ex.get("output") or ex.get("audit_result"))

        if not code:
            continue

        user_parts = ["Audit the following Solidity contract for security vulnerabilities and logic flaws:"]
        if instr:
            user_parts.append(f"Specification: {instr}")
        user_parts.extend(["```solidity", code, "```", "", "Explain your audit reasoning step-by-step and provide remediation guidance."])
        user = "\n".join(user_parts)

        if isinstance(r_steps, list):
            reasoning_str = "\n\n".join(_sanitize_text(s) for s in r_steps if s)
        else:
            reasoning_str = _sanitize_text(r_steps)

        if not finding:
            finding = "Audit completed. See step-by-step reasoning for vulnerability analysis and fixes."

        assistant = _format_think_assistant(reasoning_str, finding) if reasoning_str else finding
        rows.append(_make_row(SYSTEM_PROMPTS["security_reasoning"], user, assistant, SRC_SOLIDITY, "security_reasoning"))

    if smoke_n:
        target_n = min(target_n, smoke_n)
    if len(rows) > target_n:
        rng = random.Random(seed + 4)
        rows = rng.sample(rows, target_n)
    print(f"  mapped: {len(rows)}", flush=True)
    return rows


def build_primus(token: str, target_n: int, seed: int, smoke_n: int) -> list[dict[str, Any]]:
    """Source 7: trendmicro-ailab/Primus-Reasoning (4,900 target)."""
    from datasets import load_dataset
    print("=== Extracting trendmicro-ailab/Primus-Reasoning ===", flush=True)

    try:
        ds = load_dataset("trendmicro-ailab/Primus-Reasoning", split="train", token=token or None)
    except Exception as e:
        print(f"  WARNING: Could not load Primus-Reasoning ({e}); skipping.", flush=True)
        return []

    ds = _cap_dataset(ds, smoke_n if smoke_n else 0, "primus")
    rows: list[dict[str, Any]] = []

    for ex in ds:
        user, assistant = "", ""
        if "conversations" in ex or "messages" in ex:
            convs = ex.get("conversations") or ex.get("messages") or []
            for m in convs:
                r = m.get("role") or m.get("from")
                c = _sanitize_text(m.get("content") or m.get("value"))
                if r in ("user", "human"):
                    user = c
                elif r in ("assistant", "gpt"):
                    assistant = c

        if not user or not assistant:
            continue

        if "<|reserved_special_token_0|>" in assistant:
            assistant = assistant.replace("<|reserved_special_token_0|>", "<think>\n")
            assistant = assistant.replace("<|reserved_special_token_1|>", "\n</think>\n\n")

        if "<think>" not in assistant:
            lines = assistant.split("\n\n")
            if len(lines) >= 2:
                assistant = _format_think_assistant("\n\n".join(lines[:-1]), lines[-1])

        rows.append(_make_row(SYSTEM_PROMPTS["security_reasoning"], user, assistant, SRC_PRIMUS, "security_reasoning"))

    if smoke_n:
        target_n = min(target_n, smoke_n)
    if len(rows) > target_n:
        rng = random.Random(seed + 5)
        rows = rng.sample(rows, target_n)
    print(f"  mapped: {len(rows)}", flush=True)
    return rows


# ---------------------------------------------------------------------------
# Preprocessing, Cleaning & Deduplication
# ---------------------------------------------------------------------------

def cleanup_cot_rows(
    rows: list[dict[str, Any]],
    tokenizer: Any,
    max_seq_length: int,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Sanitize -> validate <think> tags -> min-length -> dedup -> char prefilter -> tokenize length gate."""
    print("=== Global CoT Cleanup & Verification ===", flush=True)
    stats = {
        "input": len(rows),
        "dropped_empty": 0,
        "dropped_invalid_think": 0,
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
        
        user = msgs[1]["content"]
        assistant = msgs[2]["content"]

        if not user or not assistant:
            stats["dropped_empty"] += 1
            continue

        if len(user) < MIN_USER_CHARS:
            stats["dropped_min_length"] += 1
            continue

        # Check <think> tag structure
        if "<think>" not in assistant or "</think>" not in assistant:
            stats["dropped_invalid_think"] += 1
            continue

        try:
            think_content = assistant.split("<think>")[1].split("</think>")[0].strip()
            answer_content = assistant.split("</think>")[1].strip()
        except IndexError:
            stats["dropped_invalid_think"] += 1
            continue

        if len(think_content) < MIN_THINK_CHARS or len(answer_content) < MIN_ASSISTANT_CHARS:
            stats["dropped_min_length"] += 1
            continue

        # Re-format normalized ChatML assistant text
        msgs[2]["content"] = f"<think>\n{think_content}\n</think>\n\n{answer_content}"
        cleaned.append(row)

    print(f"  after sanitize & think-structure validation: {len(cleaned)}", flush=True)

    # Deduplication
    seen: set[str] = set()
    deduped: list[dict[str, Any]] = []
    for row in cleaned:
        user = row["messages"][1]["content"]
        assistant = row["messages"][2]["content"]
        key = hashlib.md5(f"{_norm_for_hash(user)}\n{_norm_for_hash(assistant)}".encode("utf-8")).hexdigest()
        if key in seen:
            stats["dropped_dedup"] += 1
            continue
        seen.add(key)
        deduped.append(row)

    print(f"  after deduplication: {len(deduped)}", flush=True)

    # Length filter using ChatML Tokenizer
    chars_keep = max_seq_length * 2.5
    chars_drop = max_seq_length * 5.0
    kept: list[dict[str, Any]] = []
    borderline: list[dict[str, Any]] = []

    for row in deduped:
        user = row["messages"][1]["content"]
        assistant = row["messages"][2]["content"]
        sys_c = row["messages"][0]["content"]
        n_chars = len(sys_c) + len(user) + len(assistant) + 64
        if n_chars > chars_drop:
            stats["dropped_seq_char"] += 1
            continue
        if n_chars <= chars_keep:
            kept.append(row)
        else:
            borderline.append(row)

    print(
        f"  char prefilter: kept={len(kept)} borderline={len(borderline)} dropped={stats['dropped_seq_char']}",
        flush=True,
    )

    for i, row in enumerate(borderline):
        if i and i % 1000 == 0:
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
            user = row["messages"][1]["content"]
            assistant = row["messages"][2]["content"]
            n_tok = max(1, (len(user) + len(assistant)) // 3)
            
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
# Artifact Storage & Hub Upload
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
    train_path = out_dir / "cot_mix_v1.parquet"
    eval_path = out_dir / "cot_eval_heldout.parquet"
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

    print(f"Pushing private dataset -> {hub_dataset_id} ...", flush=True)
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
            "eval_split": "heldout_eval",
        }
        for r in eval_rows
    ]
    train_ds = Dataset.from_list(train_clean)
    eval_ds = Dataset.from_list(eval_clean) if eval_clean else train_ds.select([])
    
    dsd = DatasetDict({"train": train_ds, "eval_heldout": eval_ds})
    dsd.push_to_hub(hub_dataset_id, private=True, token=token)
    print("Hub push completed successfully.", flush=True)


# ---------------------------------------------------------------------------
# Main Orchestrator
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

    out_dir = _maybe_mount_drive(out_dir)
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

    print("Out Dir:     ", out_dir, flush=True)
    print("Hub ID:      ", args.hub_dataset_id or "(skipped)", flush=True)
    print("Seed:        ", args.seed, flush=True)
    print("Max Seq Len: ", args.max_seq_length, flush=True)
    print("Eval Size:   ", args.eval_n, "samples held out", flush=True)
    print("Tokenizer:   ", args.tokenizer_id, flush=True)
    print("Smoke n:     ", args.smoke_n, flush=True)

    if args.push_only:
        from datasets import Dataset, DatasetDict
        train_path = out_dir / "cot_mix_v1.parquet"
        eval_path = out_dir / "cot_eval_heldout.parquet"
        if not train_path.is_file():
            raise SystemExit(f"--push_only requires {train_path}")
        train_ds = Dataset.from_parquet(str(train_path))
        eval_ds = Dataset.from_parquet(str(eval_path)) if eval_path.is_file() else train_ds.select([])
        dsd = DatasetDict({"train": train_ds, "eval_heldout": eval_ds})
        dsd.push_to_hub(args.hub_dataset_id.strip(), private=True, token=token)
        print("Push-only completed.", flush=True)
        return

    from transformers import AutoTokenizer

    raw_pool: list[dict[str, Any]] = []

    print("\n=== Extracting Code & Algorithmic Reasoning Sources (~15K Target) ===", flush=True)
    raw_pool.extend(build_codeforces(token, 2500, args.seed, args.smoke_n))
    gc.collect()

    raw_pool.extend(build_bespoke_stratos(token, 2500, args.seed, args.smoke_n))
    gc.collect()

    raw_pool.extend(build_openthoughts(token, 5000, args.seed, args.smoke_n))
    gc.collect()

    raw_pool.extend(build_fable5(token, 4600, args.seed, args.smoke_n))
    gc.collect()

    raw_pool.extend(build_gpt55_terminal(token, 131, args.seed, args.smoke_n))
    gc.collect()

    print("\n=== Extracting Security & Vulnerability Sources (~10K Target) ===", flush=True)
    raw_pool.extend(build_solidity_audit(token, 5100, args.seed, args.smoke_n))
    gc.collect()

    raw_pool.extend(build_primus(token, 4900, args.seed, args.smoke_n))
    gc.collect()

    print(f"\nTotal raw extracted pool: {len(raw_pool)} rows", flush=True)

    print(f"\nLoading ChatML Tokenizer for length filtering: {args.tokenizer_id}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_id,
        trust_remote_code=True,
        token=token or None,
    )

    cleaned_pool, clean_stats = cleanup_cot_rows(raw_pool, tokenizer, args.max_seq_length)

    # Shuffle clean pool
    rng = random.Random(args.seed)
    rng.shuffle(cleaned_pool)

    # Separate 500 held-out evaluation samples
    eval_target = min(args.eval_n, max(10, len(cleaned_pool) // 20)) if args.smoke_n else args.eval_n
    eval_rows = cleaned_pool[:eval_target]
    train_rows = cleaned_pool[eval_target:]

    for r in eval_rows:
        r["eval_split"] = "heldout_eval"

    print(f"\nFinal Split Allocation:")
    print(f"  Train Mix: {len(train_rows)} rows")
    print(f"  Held-out Eval: {len(eval_rows)} rows")

    train_counts = _count_tags(train_rows)
    eval_counts = _count_tags(eval_rows)

    print(f"\nTrain Distribution: {train_counts}", flush=True)
    print(f"Eval Distribution: {eval_counts}", flush=True)

    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "seed": args.seed,
        "max_seq_length": args.max_seq_length,
        "tokenizer_id_length_filter": args.tokenizer_id,
        "smoke_n": args.smoke_n,
        "cleanup_stats": clean_stats,
        "final_train_counts": train_counts,
        "final_eval_counts": eval_counts,
        "hub_dataset_id": args.hub_dataset_id or None,
        "out_dir": str(out_dir),
        "sources": [
            "open-r1/codeforces-cots",
            "bespokelabs/Bespoke-Stratos-17k",
            "open-thoughts/OpenThoughts-114k",
            "Glint-Research/Fable-5-traces",
            "Roman1111111/gpt5.5-terminal",
            "samscrack/solidity-audit-cot",
            "trendmicro-ailab/Primus-Reasoning",
        ],
    }

    write_artifacts(out_dir, train_rows, eval_rows, manifest, args.samples_n, args.seed)

    if not args.skip_hub:
        push_hub(args.hub_dataset_id.strip(), train_rows, eval_rows, token)

    print("\nProcessing finished successfully.")
    print(f"  Train Parquet -> {out_dir / 'cot_mix_v1.parquet'}", flush=True)
    print(f"  Eval Parquet  -> {out_dir / 'cot_eval_heldout.parquet'}", flush=True)
    print(f"  Manifest      -> {out_dir / 'manifest.json'}", flush=True)
    print(f"  Samples       -> {out_dir / 'samples.jsonl'}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        sys.exit(130)

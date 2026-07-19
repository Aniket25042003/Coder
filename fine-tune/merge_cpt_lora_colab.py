#!/usr/bin/env python3
"""
Merge Phase-1 CPT LoRA adapters into Qwen2.5-Coder-7B Base (Colab).

Produces a standalone BF16 domain base for Phase-2 QLoRA SFT.
Does NOT continue CPT training — adapters from `final/` only.

Typical Colab usage:
  1. Runtime → A100 (40GB is enough for BF16 merge of 7B).
  2. Mount Drive; set HF_TOKEN if Hub push enabled.
  3. !pip install ... (same Unsloth Colab pins as Phase-1 notebook) OR run after Phase-1 install cells.
  4. !python merge_cpt_lora_colab.py
     # or override:
     !python merge_cpt_lora_colab.py \\
       --adapter_dir /content/drive/MyDrive/coder-qwen25-coder-7b-phase1-lora/final \\
       --out_dir /content/drive/MyDrive/coder-qwen25-coder-7b-cpt-merged \\
       --hub_model_id YOUR_USER/coder-qwen25-coder-7b-cpt-merged

See fine-tune/FINE_TUNE_DECISIONS.md §5–6.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Merge CPT LoRA into Qwen2.5-Coder-7B Base")
    p.add_argument(
        "--base_model",
        default="unsloth/Qwen2.5-Coder-7B",
        help="HF id or local snapshot of the Base model",
    )
    p.add_argument(
        "--adapter_dir",
        default="/content/drive/MyDrive/coder-qwen25-coder-7b-phase1-lora/final",
        help="Directory with adapter_config.json + adapter_model.safetensors (Phase-1 final/)",
    )
    p.add_argument(
        "--out_dir",
        default="/content/drive/MyDrive/coder-qwen25-coder-7b-cpt-merged",
        help="Where to write merged BF16 weights + tokenizer",
    )
    p.add_argument(
        "--max_seq_length",
        type=int,
        default=2048,
        help="Passed to FastLanguageModel.from_pretrained (load-time only)",
    )
    p.add_argument(
        "--hub_model_id",
        default="",
        help="Optional private Hub repo to push merged model (empty = Drive only)",
    )
    p.add_argument(
        "--hf_token",
        default="",
        help="HF token; else HF_TOKEN / HUGGING_FACE_HUB_TOKEN / Colab userdata",
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="Delete out_dir if it already exists",
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


def _require_adapter_dir(adapter_dir: Path) -> None:
    if not adapter_dir.is_dir():
        raise SystemExit(f"Adapter dir missing: {adapter_dir}")
    cfg = adapter_dir / "adapter_config.json"
    weights = list(adapter_dir.glob("adapter_model*.safetensors")) + list(
        adapter_dir.glob("adapter_model.bin")
    )
    if not cfg.is_file():
        raise SystemExit(f"Missing adapter_config.json in {adapter_dir}")
    if not weights:
        raise SystemExit(
            f"Missing adapter weights in {adapter_dir} "
            "(expected adapter_model.safetensors or .bin)"
        )
    # Soft check: final/ should not be used alone as trainer resume
    trainer_bits = ("optimizer.pt", "trainer_state.json", "scheduler.pt")
    if any((adapter_dir / n).exists() for n in trainer_bits):
        print(
            "Note: adapter_dir looks like a full trainer checkpoint; "
            "merge still uses PEFT adapters only.",
            flush=True,
        )


def _maybe_mount_drive(adapter_dir: Path, out_dir: Path) -> None:
    needs_drive = "/content/drive" in str(adapter_dir) or "/content/drive" in str(out_dir)
    if not needs_drive:
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


def main() -> None:
    # Env before Unsloth import (same lesson as Phase-1 notebook)
    os.environ.setdefault("UNSLOTH_STABLE_DOWNLOADS", "1")
    os.environ.setdefault("UNSLOTH_DISABLE_STATISTICS", "1")
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    for k in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE"):
        os.environ.pop(k, None)

    args = _parse_args()
    adapter_dir = Path(args.adapter_dir).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    token = _resolve_token(args.hf_token)
    if token:
        os.environ["HF_TOKEN"] = token
        os.environ["HUGGING_FACE_HUB_TOKEN"] = token

    _maybe_mount_drive(adapter_dir, out_dir)
    _require_adapter_dir(adapter_dir)

    if out_dir.exists():
        if not args.force:
            raise SystemExit(
                f"out_dir already exists: {out_dir}\n"
                "Pass --force to delete and rewrite, or choose a new --out_dir."
            )
        print(f"Removing existing out_dir: {out_dir}", flush=True)
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Base:   ", args.base_model, flush=True)
    print("Adapter:", adapter_dir, flush=True)
    print("Out:    ", out_dir, flush=True)

    import torch
    from peft import PeftModel
    from unsloth import FastLanguageModel

    if not torch.cuda.is_available():
        raise SystemExit("CUDA GPU required for BF16 merge of 7B on Colab.")

    print("Loading base (BF16, not 4-bit — required for a clean merge)...", flush=True)
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.base_model,
        max_seq_length=args.max_seq_length,
        dtype=None,  # bf16 on A100
        load_in_4bit=False,
        token=token or None,
    )

    print("Attaching CPT LoRA adapters...", flush=True)
    model = PeftModel.from_pretrained(model, str(adapter_dir))
    print("Merging + unloading adapters...", flush=True)
    model = model.merge_and_unload()

    # Prefer Unsloth merged saver when available (safe sharding / dtype)
    print("Saving merged BF16 weights...", flush=True)
    saved = False
    if hasattr(model, "save_pretrained_merged"):
        try:
            model.save_pretrained_merged(
                str(out_dir),
                tokenizer,
                save_method="merged_16bit",
            )
            saved = True
            print("Saved via Unsloth save_pretrained_merged(merged_16bit).", flush=True)
        except Exception as e:
            print(f"Unsloth merged save failed ({e}); falling back to HF save.", flush=True)

    if not saved:
        model.save_pretrained(str(out_dir), safe_serialization=True)
        tokenizer.save_pretrained(str(out_dir))
        print("Saved via model.save_pretrained + tokenizer.", flush=True)

    meta = {
        "base_model": args.base_model,
        "adapter_dir": str(adapter_dir),
        "out_dir": str(out_dir),
        "phase": "cpt_merged_domain_base",
        "notes": "Phase-2 QLoRA SFT should load this dir (or Hub mirror) as model_name; do not reuse CPT adapter weights.",
    }
    phase_state = adapter_dir / "phase1_state.json"
    if phase_state.is_file():
        try:
            meta["phase1_state"] = json.loads(phase_state.read_text())
        except Exception as e:
            meta["phase1_state_error"] = str(e)
    (out_dir / "merge_meta.json").write_text(json.dumps(meta, indent=2))

    # Sanity: should look like a full model, not an adapter-only folder
    has_cfg = (out_dir / "config.json").is_file()
    weight_files = list(out_dir.glob("*.safetensors")) + list(out_dir.glob("pytorch_model*.bin"))
    large = [p for p in weight_files if p.stat().st_size > 50_000_000]
    if not has_cfg or not large:
        raise SystemExit(
            f"Merge output looks incomplete under {out_dir} "
            f"(config.json={has_cfg}, large_weight_files={len(large)})."
        )
    gb = sum(p.stat().st_size for p in out_dir.rglob("*") if p.is_file()) / 1e9
    print(f"Merge complete ({gb:.1f} GB) → {out_dir}", flush=True)

    if args.hub_model_id:
        if not token:
            raise SystemExit("--hub_model_id set but no HF token available")
        print(f"Pushing merged model → {args.hub_model_id} (private)...", flush=True)
        model.push_to_hub(args.hub_model_id, private=True, token=token)
        tokenizer.push_to_hub(args.hub_model_id, private=True, token=token)
        print("Hub push done.", flush=True)

    print("Next: Phase-2 QLoRA SFT from this merged base (fresh adapters).", flush=True)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        sys.exit(130)

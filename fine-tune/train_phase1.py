#!/usr/bin/env python3
"""
Phase 1 LoRA continued pretrain for Qwen3.5-4B-Base (Unsloth + TRL).

Locked defaults: fine-tune/FINE_TUNE_DECISIONS.md
  - Model: unsloth/Qwen3.5-4B-Base (BF16 LoRA, no QLoRA)
  - Data: Aniket200325/coder-pretrain-60gb (streaming, packed)
  - Seq: 2048 default (try 4096 only if tok/s wins), token budget ~5B
  - Colab: multi-session resume via local/Hub/Drive LATEST

Smoke test (~10 min on Colab A100):
  python train_phase1.py \\
      --token_budget 1000000 --max_steps 50 \\
      --per_device_train_batch_size 2 --gradient_accumulation_steps 1 \\
      --max_seq_len 2048 --output_dir ./ckpts/phase1-smoke \\
      --save_steps 25 --logging_steps 5

Full Phase 1 (multi-session; resume after each ~12h Colab kill):
  PYTHONUNBUFFERED=1 python -u train_phase1.py \\
      --token_budget 5000000000 --max_seq_len 2048 \\
      --per_device_train_batch_size 16 --gradient_accumulation_steps 1 \\
      --output_dir ./ckpts/phase1-lora \\
      --hub_model_id YOUR_USER/coder-qwen35-4b-phase1-lora \\
      --resume auto --ckpt_minutes 30 --logging_steps 5

First ~10–15 min: watch tok/s + VRAM (GB). Aim ~35–38 GB used.
  - If VRAM < 30 GB → raise --per_device_train_batch_size (24, then 32)
  - If OOM → lower batch (12 → 8) or keep seq 2048
  - Optional: trial --max_seq_len 4096 --per_device_train_batch_size 8; keep whichever tok/s wins
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
import time
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from transformers import TrainerCallback, TrainerControl, TrainerState, TrainingArguments

LOG = logging.getLogger("phase1")


# =============================================================================
# Config
# =============================================================================


@dataclass
class Phase1Config:
    model_name: str = "unsloth/Qwen3.5-4B-Base"
    dataset: str = "Aniket200325/coder-pretrain-60gb"
    data_dir: Optional[str] = None
    output_dir: str = "./ckpts/phase1-lora"
    hub_model_id: Optional[str] = None
    drive_ckpt_dir: Optional[str] = None
    resume: str = "auto"  # "auto" | "none" | path

    max_seq_len: int = 2048
    token_budget: int = 5_000_000_000
    max_steps: Optional[int] = None  # override; else derived from budget

    per_device_train_batch_size: int = 16
    gradient_accumulation_steps: int = 1
    learning_rate: float = 1e-4
    warmup_steps: int = 100
    weight_decay: float = 0.01
    logging_steps: int = 5
    save_steps: int = 250
    ckpt_minutes: float = 30.0
    seed: int = 42

    lora_r: int = 64
    lora_alpha: int = 128
    lora_dropout: float = 0.0

    remaining_colab_hours: float = 45.0  # for tok/s projection
    smoke_every_saves: int = 4
    push_to_hub: bool = True
    project_after_minutes: float = 10.0  # early projection for batch/seq tuning


# =============================================================================
# State / resume helpers
# =============================================================================


@dataclass
class Phase1State:
    tokens_seen: int = 0
    global_step: int = 0
    tok_per_s: float = 0.0
    max_seq_len: int = 2048
    lora_r: int = 64
    model_name: str = ""
    best_loss: float = float("inf")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Phase1State":
        return cls(**{k: d[k] for k in cls.__dataclass_fields__ if k in d})  # type: ignore[misc]


def setup_logging() -> None:
    # Unbuffered-friendly logging for Colab / pipes
    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    root = logging.getLogger()
    root.handlers.clear()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
        force=True,
        stream=sys.stdout,
    )
    try:
        from transformers.utils import logging as hf_logging

        hf_logging.set_verbosity_info()
        hf_logging.enable_default_handler()
        hf_logging.enable_explicit_format()
    except Exception:
        pass


def write_latest(output_dir: Path, ckpt_path: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "LATEST").write_text(str(ckpt_path.resolve()))


def read_latest(output_dir: Path) -> Optional[Path]:
    latest = output_dir / "LATEST"
    if not latest.exists():
        return None
    p = Path(latest.read_text().strip())
    return p if p.exists() else None


def save_phase1_state(ckpt_dir: Path, state: Phase1State, cfg: Phase1Config) -> None:
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    (ckpt_dir / "phase1_state.json").write_text(json.dumps(state.to_dict(), indent=2))
    (ckpt_dir / "phase1_config.json").write_text(json.dumps(asdict(cfg), indent=2, default=str))


def load_phase1_state(ckpt_dir: Path) -> Optional[Phase1State]:
    f = ckpt_dir / "phase1_state.json"
    if not f.exists():
        return None
    return Phase1State.from_dict(json.loads(f.read_text()))


def resolve_resume_path(cfg: Phase1Config) -> Optional[str]:
    if cfg.resume in ("none", "", "False", "false"):
        return None
    if cfg.resume != "auto":
        p = Path(cfg.resume)
        if not p.exists():
            raise FileNotFoundError(f"--resume path not found: {p}")
        return str(p)

    # Prefer Drive LATEST, then local output_dir LATEST
    candidates: List[Path] = []
    if cfg.drive_ckpt_dir:
        candidates.append(Path(cfg.drive_ckpt_dir))
    candidates.append(Path(cfg.output_dir))
    for root in candidates:
        latest = read_latest(root)
        if latest is not None:
            LOG.info("Resume auto -> %s", latest)
            return str(latest)
        # Also accept HF Trainer checkpoint-* folders
        ckpts = sorted(root.glob("checkpoint-*"), key=lambda p: p.stat().st_mtime)
        if ckpts:
            LOG.info("Resume auto -> %s", ckpts[-1])
            return str(ckpts[-1])
    LOG.info("Resume auto: no checkpoint found; starting fresh")
    return None


def mirror_to_drive(src: Path, drive_dir: Optional[str]) -> None:
    if not drive_dir:
        return
    dest_root = Path(drive_dir)
    dest_root.mkdir(parents=True, exist_ok=True)
    dest = dest_root / src.name
    # Lightweight: copy phase1_state + adapter files if present
    import shutil

    if dest.exists():
        shutil.rmtree(dest)
    shutil.copytree(src, dest, dirs_exist_ok=True)
    write_latest(dest_root, dest)
    LOG.info("Mirrored checkpoint to Drive: %s", dest)


# =============================================================================
# Callbacks: token budget, timed ckpt, tok/s, first-hour projection
# =============================================================================


class TokenBudgetCallback(TrainerCallback):
    def __init__(self, cfg: Phase1Config, state: Phase1State) -> None:
        self.cfg = cfg
        self.phase_state = state
        self._tokens_at_step_start = state.tokens_seen
        self._t0 = time.time()
        self._last_log_t = self._t0
        self._last_tokens = state.tokens_seen
        self._early_projected = False
        self._hour_projected = False
        self._last_timed_save = self._t0
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

    def _tokens_per_step(self, args: TrainingArguments) -> int:
        world = max(int(os.environ.get("WORLD_SIZE", "1")), 1)
        return (
            args.per_device_train_batch_size
            * args.gradient_accumulation_steps
            * self.cfg.max_seq_len
            * world
        )

    def _vram_gb(self) -> tuple[float, float]:
        if not torch.cuda.is_available():
            return 0.0, 0.0
        alloc = torch.cuda.memory_allocated() / (1024**3)
        peak = torch.cuda.max_memory_allocated() / (1024**3)
        return alloc, peak

    def _log_projection(self, label: str, now: float) -> None:
        tok_s = self.phase_state.tok_per_s or (
            (self.phase_state.tokens_seen - self._tokens_at_step_start) / max(now - self._t0, 1.0)
        )
        elapsed_min = (now - self._t0) / 60.0
        proj = tok_s * 3600.0 * self.cfg.remaining_colab_hours
        alloc_gb, peak_gb = self._vram_gb()
        LOG.info(
            "%s: elapsed=%.1fmin tok/s≈%.0f → ~%.2fB tokens over %.0f remaining Colab hours "
            "(target %.2fB) | VRAM alloc=%.1fGB peak=%.1fGB | "
            "Tune: if peak<30GB raise batch; if OOM cut batch or keep seq=2048; "
            "compare seq=4096 only if tok/s wins.",
            label,
            elapsed_min,
            tok_s,
            proj / 1e9,
            self.cfg.remaining_colab_hours,
            self.cfg.token_budget / 1e9,
            alloc_gb,
            peak_gb,
        )
        if proj < 0.6 * self.cfg.token_budget:
            LOG.warning(
                "Projection < 60%% of token_budget. Raise --per_device_train_batch_size "
                "(try 24/32) before burning more credits; keep --max_seq_len 2048 unless 4096 is faster."
            )

    def on_step_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs: Any,
    ) -> TrainerControl:
        tps = self._tokens_per_step(args)
        self.phase_state.tokens_seen += tps
        self.phase_state.global_step = state.global_step

        now = time.time()
        dt = now - self._last_log_t
        if dt >= 30.0 or state.global_step % max(args.logging_steps, 1) == 0:
            delta_tok = self.phase_state.tokens_seen - self._last_tokens
            tok_s = delta_tok / max(dt, 1e-6)
            self.phase_state.tok_per_s = tok_s
            remain = max(self.cfg.token_budget - self.phase_state.tokens_seen, 0)
            eta_h = (remain / max(tok_s, 1.0)) / 3600.0
            alloc_gb, peak_gb = self._vram_gb()
            loss_s = "n/a"
            if state.log_history:
                loss = state.log_history[-1].get("loss")
                if isinstance(loss, (int, float)):
                    loss_s = f"{loss:.4f}"
            LOG.info(
                "step=%s tokens_seen=%s tok/s=%.0f eta_budget=%.1fh loss=%s "
                "VRAM=%.1fGB peak=%.1fGB batch=%s seq=%s accum=%s",
                state.global_step,
                f"{self.phase_state.tokens_seen:,}",
                tok_s,
                eta_h,
                loss_s,
                alloc_gb,
                peak_gb,
                args.per_device_train_batch_size,
                self.cfg.max_seq_len,
                args.gradient_accumulation_steps,
            )
            sys.stdout.flush()
            self._last_log_t = now
            self._last_tokens = self.phase_state.tokens_seen

        # Early projection (~10 min) for batch/seq decisions without waiting an hour
        if (
            not self._early_projected
            and (now - self._t0) >= self.cfg.project_after_minutes * 60.0
        ):
            self._early_projected = True
            self._log_projection("EARLY_PROJECTION", now)

        if not self._hour_projected and (now - self._t0) >= 3600.0:
            self._hour_projected = True
            self._log_projection("FIRST_HOUR_PROJECTION", now)

        # Timed checkpoint request
        if (now - self._last_timed_save) >= self.cfg.ckpt_minutes * 60.0:
            control.should_save = True
            self._last_timed_save = now
            LOG.info("Timed checkpoint triggered (every %.0f min)", self.cfg.ckpt_minutes)

        if self.phase_state.tokens_seen >= self.cfg.token_budget:
            LOG.info(
                "Token budget reached (%s >= %s). Stopping.",
                f"{self.phase_state.tokens_seen:,}",
                f"{self.cfg.token_budget:,}",
            )
            control.should_training_stop = True
            control.should_save = True

        return control

    def on_save(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs: Any,
    ) -> TrainerControl:
        ckpt = Path(args.output_dir) / f"checkpoint-{state.global_step}"
        if ckpt.exists():
            save_phase1_state(ckpt, self.phase_state, self.cfg)
            write_latest(Path(args.output_dir), ckpt)
            mirror_to_drive(ckpt, self.cfg.drive_ckpt_dir)
        if state.log_history:
            loss = state.log_history[-1].get("loss")
            if isinstance(loss, (int, float)) and loss < self.phase_state.best_loss:
                self.phase_state.best_loss = float(loss)
        return control


class OOMHint:
    HINT = (
        "CUDA OOM: cut --per_device_train_batch_size, set --max_seq_len 2048, "
        "or raise --gradient_accumulation_steps while lowering batch."
    )


# =============================================================================
# Data
# =============================================================================


def load_text_dataset(cfg: Phase1Config):
    from datasets import load_dataset

    last_err: Optional[Exception] = None
    for attempt in range(5):
        try:
            if cfg.data_dir:
                paths = list(Path(cfg.data_dir).rglob("*.parquet"))
                paths = [p for p in paths if "val" not in p.as_posix().lower()]
                if not paths:
                    raise FileNotFoundError(f"No train parquet under {cfg.data_dir}")
                ds = load_dataset(
                    "parquet",
                    data_files=[str(p) for p in paths],
                    split="train",
                    streaming=True,
                )
            else:
                ds = load_dataset(cfg.dataset, split="train", streaming=True)
            ds = ds.shuffle(seed=cfg.seed, buffer_size=10_000)

            def _keep(example: Dict[str, Any]) -> Dict[str, Any]:
                text = example.get("text") or ""
                return {"text": text if isinstance(text, str) else ""}

            # Streaming datasets may not expose column_names until taken
            try:
                cols = list(ds.column_names or [])
            except Exception:
                cols = []
            drop = [c for c in cols if c != "text"]
            ds = ds.map(_keep, remove_columns=drop) if drop else ds.map(_keep)
            ds = ds.filter(lambda ex: bool(ex["text"] and ex["text"].strip()))
            return ds
        except Exception as e:  # noqa: BLE001
            last_err = e
            LOG.warning("Dataset load attempt %s failed: %s", attempt + 1, e)
            time.sleep(2 ** attempt)
    raise RuntimeError(f"Failed to load dataset: {last_err}")


def load_val_dataset(cfg: Phase1Config):
    from datasets import load_dataset

    if cfg.data_dir:
        paths = [p for p in Path(cfg.data_dir).rglob("*.parquet") if "val" in p.as_posix().lower()]
        if not paths:
            return None
        ds = load_dataset("parquet", data_files=[str(p) for p in paths], split="train", streaming=True)
    else:
        for split in ("validation", "val"):
            try:
                ds = load_dataset(cfg.dataset, split=split, streaming=True)
                break
            except Exception:
                ds = None
        if ds is None:
            LOG.warning("No val split found; skipping eval_dataset")
            return None

    def _keep(example: Dict[str, Any]) -> Dict[str, Any]:
        text = example.get("text") or ""
        return {"text": text if isinstance(text, str) else ""}

    ds = ds.map(_keep)
    ds = ds.filter(lambda ex: bool(ex["text"] and ex["text"].strip()))
    return ds


# =============================================================================
# Model
# =============================================================================


def load_model_and_tokenizer(cfg: Phase1Config):
    try:
        from unsloth import FastLanguageModel
    except ImportError as e:
        raise SystemExit(
            "Unsloth is required. Install per https://unsloth.ai/docs/get-started/install "
            "or see requirements-phase1.txt / phase1_colab.ipynb"
        ) from e

    model_name = cfg.model_name
    LOG.info("Loading %s (BF16 LoRA, seq=%s)", model_name, cfg.max_seq_len)
    try:
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_name,
            max_seq_length=cfg.max_seq_len,
            load_in_4bit=False,
            load_in_16bit=True,
            full_finetuning=False,
        )
    except Exception as e:
        if model_name.startswith("unsloth/"):
            alt = "Qwen/Qwen3.5-4B-Base"
            LOG.warning("Failed loading %s (%s); falling back to %s", model_name, e, alt)
            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name=alt,
                max_seq_length=cfg.max_seq_len,
                load_in_4bit=False,
                load_in_16bit=True,
                full_finetuning=False,
            )
            cfg.model_name = alt
        else:
            raise

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = FastLanguageModel.get_peft_model(
        model,
        r=cfg.lora_r,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=cfg.seed,
        max_seq_length=cfg.max_seq_len,
    )
    return model, tokenizer


# =============================================================================
# Train
# =============================================================================


def estimate_max_steps(cfg: Phase1Config) -> int:
    world = max(int(os.environ.get("WORLD_SIZE", "1")), 1)
    tokens_per_step = (
        cfg.per_device_train_batch_size
        * cfg.gradient_accumulation_steps
        * cfg.max_seq_len
        * world
    )
    steps = int(math.ceil(cfg.token_budget / max(tokens_per_step, 1)))
    # Headroom: packing may yield slightly different effective tokens
    return max(steps + 100, 100)


def build_trainer(cfg: Phase1Config, model, tokenizer, train_ds, val_ds, phase_state: Phase1State):
    from trl import SFTConfig, SFTTrainer

    max_steps = cfg.max_steps if cfg.max_steps is not None else estimate_max_steps(cfg)
    LOG.info("max_steps=%s (token_budget=%s)", max_steps, f"{cfg.token_budget:,}")

    out = Path(cfg.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    sft_kwargs: Dict[str, Any] = dict(
        output_dir=str(out),
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        learning_rate=cfg.learning_rate,
        lr_scheduler_type="cosine",
        warmup_steps=cfg.warmup_steps,
        weight_decay=cfg.weight_decay,
        logging_steps=cfg.logging_steps,
        logging_first_step=True,
        log_level="info",
        disable_tqdm=False,
        save_steps=cfg.save_steps,
        save_total_limit=3,
        max_steps=max_steps,
        bf16=True,
        optim="adamw_torch_fused" if torch.cuda.is_available() else "adamw_torch",
        seed=cfg.seed,
        report_to="none",
        remove_unused_columns=False,
        dataloader_num_workers=2,
        max_seq_length=cfg.max_seq_len,
        packing=True,
        dataset_text_field="text",
    )
    # Hub push (adapters)
    if cfg.hub_model_id and cfg.push_to_hub and os.environ.get("HF_TOKEN"):
        sft_kwargs.update(
            push_to_hub=True,
            hub_model_id=cfg.hub_model_id,
            hub_strategy="every_save",
            hub_private_repo=True,
        )

    try:
        args = SFTConfig(**sft_kwargs)
    except TypeError:
        # Older TRL / transformers: drop optional fields
        for k in ("packing", "dataset_text_field", "max_seq_length", "logging_first_step", "log_level"):
            sft_kwargs.pop(k, None)
        args = SFTConfig(**sft_kwargs)

    callbacks = [TokenBudgetCallback(cfg, phase_state)]

    trainer_kwargs: Dict[str, Any] = dict(
        model=model,
        args=args,
        train_dataset=train_ds,
        processing_class=tokenizer,
        callbacks=callbacks,
    )
    if val_ds is not None:
        trainer_kwargs["eval_dataset"] = val_ds

    try:
        trainer = SFTTrainer(**trainer_kwargs)
    except TypeError:
        trainer_kwargs.pop("processing_class", None)
        trainer_kwargs["tokenizer"] = tokenizer
        # packing via dataset map fallback handled by TRL defaults
        trainer = SFTTrainer(**trainer_kwargs)

    # Ensure packing flags when using older signature
    if hasattr(trainer, "args") and not getattr(trainer.args, "packing", False):
        LOG.warning(
            "SFTTrainer packing flag may be inactive on this TRL version; "
            "upgrade trl if pad waste is high."
        )
    return trainer


def train(cfg: Phase1Config) -> None:
    setup_logging()
    LOG.info("Phase 1 config: %s", json.dumps(asdict(cfg), indent=2, default=str))

    if cfg.max_seq_len not in (2048, 4096):
        LOG.warning("max_seq_len=%s (decisions prefer 2048 or 4096)", cfg.max_seq_len)

    resume_path = resolve_resume_path(cfg)
    phase_state = Phase1State(
        max_seq_len=cfg.max_seq_len,
        lora_r=cfg.lora_r,
        model_name=cfg.model_name,
    )
    if resume_path:
        loaded = load_phase1_state(Path(resume_path))
        if loaded:
            phase_state = loaded
            LOG.info(
                "Restored phase1_state tokens_seen=%s step=%s",
                f"{phase_state.tokens_seen:,}",
                phase_state.global_step,
            )
        if phase_state.tokens_seen >= cfg.token_budget:
            LOG.info("Token budget already met; nothing to do.")
            return

    try:
        model, tokenizer = load_model_and_tokenizer(cfg)
        train_ds = load_text_dataset(cfg)
        val_ds = load_val_dataset(cfg)
        trainer = build_trainer(cfg, model, tokenizer, train_ds, val_ds, phase_state)
    except torch.cuda.OutOfMemoryError:
        LOG.error(OOMHint.HINT)
        raise SystemExit(2)

    try:
        trainer.train(resume_from_checkpoint=resume_path)
    except torch.cuda.OutOfMemoryError:
        LOG.error(OOMHint.HINT)
        raise SystemExit(2)
    except Exception:
        LOG.error("Training failed:\n%s", traceback.format_exc())
        # Best-effort save
        try:
            trainer.save_model()
            out = Path(cfg.output_dir) / f"checkpoint-{trainer.state.global_step}"
            out.mkdir(parents=True, exist_ok=True)
            save_phase1_state(out, phase_state, cfg)
            write_latest(Path(cfg.output_dir), out)
            mirror_to_drive(out, cfg.drive_ckpt_dir)
        except Exception as e:  # noqa: BLE001
            LOG.warning("Emergency save failed: %s", e)
        raise

    # Final save
    final_dir = Path(cfg.output_dir) / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))
    save_phase1_state(final_dir, phase_state, cfg)
    write_latest(Path(cfg.output_dir), final_dir)
    mirror_to_drive(final_dir, cfg.drive_ckpt_dir)
    if cfg.hub_model_id and os.environ.get("HF_TOKEN"):
        try:
            trainer.model.push_to_hub(cfg.hub_model_id, private=True, token=os.environ["HF_TOKEN"])
            tokenizer.push_to_hub(cfg.hub_model_id, private=True, token=os.environ["HF_TOKEN"])
        except Exception as e:  # noqa: BLE001
            LOG.warning("Final Hub push failed: %s", e)

    LOG.info(
        "Done. tokens_seen=%s tok/s≈%.0f output=%s",
        f"{phase_state.tokens_seen:,}",
        phase_state.tok_per_s,
        cfg.output_dir,
    )


# =============================================================================
# CLI
# =============================================================================


def parse_args(argv: Optional[List[str]] = None) -> Phase1Config:
    p = argparse.ArgumentParser(description="Phase 1 Unsloth LoRA CPT for Qwen3.5-4B-Base")
    p.add_argument("--model", type=str, default=Phase1Config.model_name)
    p.add_argument("--dataset", type=str, default=Phase1Config.dataset)
    p.add_argument("--data_dir", type=str, default=None)
    p.add_argument("--output_dir", type=str, default=Phase1Config.output_dir)
    p.add_argument("--hub_model_id", type=str, default=None)
    p.add_argument("--drive_ckpt_dir", type=str, default=None)
    p.add_argument("--resume", type=str, default="auto")
    p.add_argument("--max_seq_len", type=int, default=2048, choices=[2048, 4096, 1024, 8192])
    p.add_argument("--token_budget", type=int, default=5_000_000_000)
    p.add_argument("--max_steps", type=int, default=None)
    p.add_argument("--per_device_train_batch_size", type=int, default=16)
    p.add_argument("--gradient_accumulation_steps", type=int, default=1)
    p.add_argument("--learning_rate", type=float, default=1e-4)
    p.add_argument("--warmup_steps", type=int, default=100)
    p.add_argument("--logging_steps", type=int, default=5)
    p.add_argument("--save_steps", type=int, default=250)
    p.add_argument("--ckpt_minutes", type=float, default=30.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--lora_r", type=int, default=64)
    p.add_argument("--lora_alpha", type=int, default=128)
    p.add_argument("--remaining_colab_hours", type=float, default=45.0)
    p.add_argument("--project_after_minutes", type=float, default=10.0)
    p.add_argument("--no_push_to_hub", action="store_true")
    args = p.parse_args(argv)

    return Phase1Config(
        model_name=args.model,
        dataset=args.dataset,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        hub_model_id=args.hub_model_id,
        drive_ckpt_dir=args.drive_ckpt_dir,
        resume=args.resume,
        max_seq_len=args.max_seq_len,
        token_budget=args.token_budget,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        ckpt_minutes=args.ckpt_minutes,
        seed=args.seed,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        remaining_colab_hours=args.remaining_colab_hours,
        project_after_minutes=args.project_after_minutes,
        push_to_hub=not args.no_push_to_hub,
    )


def main(argv: Optional[List[str]] = None) -> None:
    cfg = parse_args(argv)
    train(cfg)


if __name__ == "__main__":
    main()

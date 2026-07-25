import sys
import os
from pathlib import Path

# Add site-packages paths to sys.path to guarantee unsloth discovery across subprocesses
for p in [
    "/usr/local/lib/python3.12/dist-packages",
    "/usr/local/lib/python3.10/dist-packages",
    "/opt/conda/lib/python3.10/site-packages",
    "/opt/conda/lib/python3.12/site-packages",
    str(Path.home() / ".local/lib/python3.12/site-packages"),
    str(Path.home() / ".local/lib/python3.10/site-packages"),
]:
    if Path(p).is_dir() and p not in sys.path:
        sys.path.insert(0, p)

#!/usr/bin/env python3
"""
Phase 3 CoT Reasoning Fine-Tuning — Qwen2.5-Coder-7B SFT-merged
Supports both single-GPU and Multi-GPU DDP via torchrun:
    torchrun --nproc_per_node=2 fine-tune/train_phase3_cot.py
"""

import os
import re
import sys
import time
import math
import json
import shutil
import argparse
from pathlib import Path
from dataclasses import dataclass, asdict

# Environment setup BEFORE imports
os.environ.setdefault("UNSLOTH_STABLE_DOWNLOADS", "1")
os.environ.setdefault("UNSLOTH_DISABLE_STATISTICS", "1")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("HF_XET_HIGH_PERFORMANCE", "1")
for k in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE"):
    os.environ.pop(k, None)
os.environ.setdefault("PYTHONUNBUFFERED", "1")

import torch
from datasets import load_dataset
from transformers import TrainerCallback, AutoTokenizer
from huggingface_hub import HfApi, snapshot_download
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template, train_on_responses_only
from trl import SFTConfig, SFTTrainer

def parse_args():
    p = argparse.ArgumentParser(description="Phase 3 CoT Fine-Tuning")
    p.add_argument("--smoke", action="store_true", help="Run 30-step smoke test")
    p.add_argument("--batch", type=int, default=10, help="Per-device train batch size")
    p.add_argument("--accum", type=int, default=6, help="Gradient accumulation steps")
    p.add_argument("--lr", type=float, default=2.5e-5, help="Learning rate")
    p.add_argument("--epochs", type=int, default=2, help="Number of train epochs")
    p.add_argument("--max_seq_len", type=int, default=4096, help="Max sequence length")
    p.add_argument("--resume", type=str, default="auto", help="Resume mode: auto, none, or path")
    p.add_argument("--model_hub", type=str, default="Aniket200325/coder-qwen25-coder-7b-sft-qlora-v1-merged")
    p.add_argument("--dataset_id", type=str, default="Aniket200325/coder-reasoning-cot-v1")
    p.add_argument("--hub_adapter_id", type=str, default="Aniket200325/coder-qwen25-coder-7b-cot-qlora-v1")
    p.add_argument("--hf_token", type=str, default="", help="Hugging Face API token")
    return p.parse_args()

def main():
    args_cli = parse_args()
    
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    is_main = local_rank == 0

    hf_token = args_cli.hf_token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN") or ""
    if not hf_token:
        try:
            from kaggle_secrets import UserSecretsClient
            hf_token = UserSecretsClient().get_secret("HF_TOKEN") or ""
        except Exception:
            pass
    if not hf_token:
        try:
            from google.colab import userdata
            hf_token = userdata.get("HF_TOKEN") or ""
        except Exception:
            pass
    assert hf_token, "HF_TOKEN required for downloading base model and syncing adapter to Hub"
    os.environ["HF_TOKEN"] = hf_token
    os.environ["HUGGING_FACE_HUB_TOKEN"] = hf_token

    is_kaggle = "KAGGLE_KERNEL_RUN_TYPE" in os.environ or Path("/kaggle/working").is_dir()
    if is_kaggle:
        out_dir = Path("/kaggle/working/ckpts/qwen25-coder-7b-phase3-cot-smoke" if args_cli.smoke else "/kaggle/working/ckpts/qwen25-coder-7b-phase3-cot")
    else:
        out_dir = Path("./ckpts/qwen25-coder-7b-phase3-cot-smoke" if args_cli.smoke else "./ckpts/qwen25-coder-7b-phase3-cot")
    out_dir.mkdir(parents=True, exist_ok=True)

    if is_main:
        print(f"=== Phase 3 CoT Fine-Tuning (Rank {local_rank}/{world_size}) ===")
        print(f"Base Model:    {args_cli.model_hub}")
        print(f"Dataset:       {args_cli.dataset_id}")
        print(f"Max Seq Len:   {args_cli.max_seq_len}")
        print(f"Batch/Device:  {args_cli.batch} | Accum: {args_cli.accum} | Effective Batch: {args_cli.batch * args_cli.accum * world_size}")
        print(f"Output Dir:    {out_dir}")

    # Prefetch base model
    model_cache = Path("/content/models") if Path("/content").is_dir() else Path("/kaggle/working/models") if is_kaggle else Path("./models")
    model_cache = model_cache / args_cli.model_hub.replace("/", "--")
    
    if is_main and not (model_cache / "config.json").is_file():
        model_cache.parent.mkdir(parents=True, exist_ok=True)
        print(f"Prefetching base model {args_cli.model_hub} → {model_cache}...")
        snapshot_download(
            repo_id=args_cli.model_hub,
            local_dir=str(model_cache),
            token=hf_token,
            max_workers=8,
        )

    if world_size > 1 and torch.cuda.is_available():
        if not torch.distributed.is_initialized():
            torch.distributed.init_process_group(
                backend="nccl" if torch.cuda.is_available() else "gloo",
                init_method="env://",
            )
        torch.cuda.set_device(local_rank)
        torch.distributed.barrier()

    # Load Model & Tokenizer
    load_kwargs = dict(
        model_name=str(model_cache if (model_cache / "config.json").is_file() else args_cli.model_hub),
        max_seq_length=args_cli.max_seq_len,
        load_in_4bit=True,
        token=hf_token,
    )
    if world_size == 1 and torch.cuda.device_count() > 1:
        load_kwargs["device_map"] = "auto"
        load_kwargs["max_memory"] = {i: "13GB" for i in range(torch.cuda.device_count())}

    model, tokenizer = FastLanguageModel.from_pretrained(**load_kwargs)
    tokenizer = get_chat_template(tokenizer, chat_template="qwen-2.5")

    # Target modules LoRA setup
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=42,
    )

    # Load Dataset
    raw_ds = load_dataset(args_cli.dataset_id, token=hf_token)
    train_raw = raw_ds["train"]
    eval_raw = raw_ds["eval_heldout"]

    if args_cli.smoke:
        train_raw = train_raw.select(range(min(256, len(train_raw))))
        eval_raw = eval_raw.select(range(min(32, len(eval_raw))))

    def to_text(example):
        try:
            text = tokenizer.apply_chat_template(
                example["messages"],
                tokenize=False,
                add_generation_prompt=False,
            )
        except Exception:
            text = ""
        return {"text": text if isinstance(text, str) else ""}

    train_ds = train_raw.map(to_text, num_proc=2 if len(train_raw) > 64 else 1, remove_columns=train_raw.column_names)
    eval_ds = eval_raw.map(to_text, num_proc=1, remove_columns=eval_raw.column_names)
    train_ds = train_ds.filter(lambda ex: bool(ex["text"] and ex["text"].strip()))
    eval_ds = eval_ds.filter(lambda ex: bool(ex["text"] and ex["text"].strip()))

    def sync_checkpoint_fn(src: Path):
        if not is_main:
            return
        readme = src / "README.md"
        if readme.is_file():
            lines = readme.read_text().splitlines()
            new_lines = [
                f"base_model: {args_cli.model_hub}" if line.startswith("base_model:") else line
                for line in lines
            ]
            readme.write_text("\n".join(new_lines))

        if args_cli.hub_adapter_id and hf_token:
            try:
                api = HfApi(token=hf_token)
                api.create_repo(repo_id=args_cli.hub_adapter_id, private=True, exist_ok=True)
                print(f"Syncing {src.name} → HF Hub ({args_cli.hub_adapter_id})...", flush=True)
                api.upload_folder(
                    folder_path=str(src),
                    path_in_repo=src.name,
                    repo_id=args_cli.hub_adapter_id,
                    repo_type="model",
                )
                print(f"Synced {src.name} → HF Hub ✓", flush=True)
            except Exception as e:
                print(f"Notice: HF Hub sync failed for {src.name}: {e}", flush=True)

    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    use_fp16 = torch.cuda.is_available() and not use_bf16

    class Phase3SyncCallback(TrainerCallback):
        def on_save(self, args, state, control, **kwargs):
            ckpt = Path(args.output_dir) / f"checkpoint-{state.global_step}"
            if ckpt.exists():
                sync_checkpoint_fn(ckpt)

    sft_kwargs = dict(
        output_dir=str(out_dir),
        per_device_train_batch_size=args_cli.batch,
        gradient_accumulation_steps=args_cli.accum,
        learning_rate=args_cli.lr,
        lr_scheduler_type="cosine",
        weight_decay=0.01,
        logging_steps=1 if args_cli.smoke else 5,
        save_steps=10 if args_cli.smoke else 50,
        save_total_limit=3,
        bf16=use_bf16,
        fp16=use_fp16,
        optim="paged_adamw_8bit" if torch.cuda.is_available() else "adamw_torch",
        seed=42,
        report_to="none",
        remove_unused_columns=False,
        dataset_num_proc=2,
        packing=False,
        dataset_text_field="text",
        max_seq_length=None,
        eval_strategy="steps",
        eval_steps=10 if args_cli.smoke else 50,
        eval_accumulation_steps=2,
        load_best_model_at_end=False,
        do_eval=True,
    )
    if args_cli.smoke:
        sft_kwargs["max_steps"] = 30
        sft_kwargs["warmup_steps"] = 5
    else:
        sft_kwargs["num_train_epochs"] = args_cli.epochs
        sft_kwargs["warmup_ratio"] = 0.03
        sft_kwargs["max_steps"] = -1

    if world_size > 1:
        sft_kwargs["ddp_find_unused_parameters"] = False

    def build_sft_config(kwargs):
        kwargs = dict(kwargs)
        if not kwargs.get("packing", False):
            kwargs["max_seq_length"] = None
            kwargs["max_length"] = None
        try:
            return SFTConfig(**kwargs)
        except TypeError:
            pass
        if "evaluation_strategy" in kwargs:
            kwargs["eval_strategy"] = kwargs.pop("evaluation_strategy")
            try:
                return SFTConfig(**kwargs)
            except TypeError:
                pass
        for k in (
            "packing", "logging_first_step",
            "max_length", "max_seq_length", "do_eval", "hub_private_repo",
            "evaluation_strategy",
        ):
            kwargs.pop(k, None)
        return SFTConfig(**kwargs)

    args = build_sft_config(sft_kwargs)

    _BaseSFTTrainer = SFTTrainer

    class _NoPaddingFreeSFTTrainer(_BaseSFTTrainer):
        @property
        def padding_free(self):
            return False
        @padding_free.setter
        def padding_free(self, value):
            pass

    trainer_kwargs = dict(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=tokenizer,
        callbacks=[Phase3SyncCallback()],
    )
    try:
        trainer = _NoPaddingFreeSFTTrainer(**trainer_kwargs)
    except TypeError:
        trainer_kwargs.pop("processing_class", None)
        trainer_kwargs["tokenizer"] = tokenizer
        trainer = _NoPaddingFreeSFTTrainer(**trainer_kwargs)

    trainer = train_on_responses_only(
        trainer,
        instruction_part="<|im_start|>user\n",
        response_part="<|im_start|>assistant\n",
    )

    if is_main:
        print("Starting training...")
    trainer.train()

    if is_main:
        print("Training complete! Saving final model...")
        final_dir = out_dir / "final"
        trainer.save_model(str(final_dir))
        sync_checkpoint_fn(final_dir)

if __name__ == "__main__":
    main()

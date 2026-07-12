#!/usr/bin/env python3
"""
Local finalize + sanity-check for coder-pretrain dataset (Jetson / Linux).

Uses DuckDB for phases 1-3 (partition → per-bucket shuffle → train/val merge)
with low RAM via disk spill. Keeps the same global bucket shuffle semantics as
the notebook (zlib.crc32(id) % K hash buckets, deterministic per-bucket order).

Quick start
-----------
  export HF_TOKEN=hf_...
  pip install -r requirements-finalize.txt
  export HF_HUB_ENABLE_HF_TRANSFER=1

  # Full pipeline: mirror → duckdb finalize → upload → sanity-check
  python finalize_local.py finalize --repo-id YOUR_USER/coder-pretrain-60gb

  # Sanity-check only (reads from Hub after upload)
  python finalize_local.py sanity-check --repo-id YOUR_USER/coder-pretrain-60gb

Disk layout (default scratch: ~/scratch/coder-pretrain)
  data_mirror/     ~20-25 GB  (deleted after phase 1)
  shuffle/         ~20-25 GB  (deleted after phase 2)
  pending/         ~20-25 GB  (deleted after phase 3)
  final_staging/   ~20-25 GB  (deleted after upload)

Peak disk ~50-60 GB at a time — fits a 256 GB SSD with headroom.
Peak RAM ~5 GB DuckDB limit + OS — fits 8 GB Jetson Orin Nano.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
import zlib
from pathlib import Path
from typing import Any

import duckdb
import orjson
from huggingface_hub import (
    CommitOperationAdd,
    HfApi,
    create_commit,
    hf_hub_download,
    snapshot_download,
)

# ---------------------------------------------------------------------------
log = logging.getLogger("finalize-local")

GB = 1024**3
SOURCES = ["code", "web", "math", "wiki", "docs"]

DEFAULT_REPO = "Aniket200325/coder-pretrain-60gb"
DEFAULT_SCRATCH = os.path.expanduser("~/scratch/coder-pretrain")


# --------------------------------------------------------------------------- UDFs
def crc32_bucket(doc_id: str, k: int) -> int:
    """Match notebook: zlib.crc32(id.encode()) % k."""
    return zlib.crc32(doc_id.encode()) % int(k)


def shuffle_order(doc_id: str, bucket: int, seed: int) -> int:
    """Deterministic per-bucket shuffle key (seed + bucket, like Random(seed+b))."""
    return zlib.crc32(f"{doc_id}:{int(seed) + int(bucket)}".encode()) & 0xFFFFFFFF


def _make_is_validation(seed: int, val_fraction: float):
    scale = 1_000_000
    threshold = int(val_fraction * scale)

    def is_validation(doc_id: str) -> bool:
        return (zlib.crc32(f"{doc_id}:val:{seed}".encode()) & 0xFFFFFFFF) % scale < threshold

    return is_validation


# --------------------------------------------------------------------------- config
def build_config(args: argparse.Namespace) -> dict[str, Any]:
    token = args.hf_token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not token:
        sys.exit("Set HF_TOKEN env var or pass --hf-token.")

    scratch = os.path.abspath(args.scratch_dir)
    return {
        "repo_id": args.repo_id,
        "private": args.private,
        "hf_token": token,
        "scratch_dir": scratch,
        "seed": args.seed,
        "shuffle_buckets": args.shuffle_buckets,
        "val_fraction": args.val_fraction,
        "final_shard_bytes": args.final_shard_bytes,
        "upload_batch_size": args.upload_batch_size,
        "duckdb_threads": args.threads,
        "duckdb_memory_limit": args.memory_limit,
        "mirror_subdir": "data_mirror",
        "skip_upload": args.skip_upload,
        "skip_mirror": args.skip_mirror,
    }


# --------------------------------------------------------------------------- duckdb
def open_duckdb(cfg: dict[str, Any]) -> duckdb.DuckDBPyConnection:
    tmp = os.path.join(cfg["scratch_dir"], "duckdb_tmp")
    os.makedirs(tmp, exist_ok=True)
    con = duckdb.connect(database=":memory:")
    con.execute(f"SET threads={cfg['duckdb_threads']}")
    con.execute(f"SET memory_limit='{cfg['duckdb_memory_limit']}'")
    con.execute(f"SET temp_directory='{tmp}'")
    con.execute("SET max_temp_directory_size='200GB'")
    con.execute("SET preserve_insertion_order=false")

    con.create_function("crc32_bucket", crc32_bucket, ["VARCHAR", "INTEGER"], "INTEGER")
    con.create_function("shuffle_order", shuffle_order, ["VARCHAR", "INTEGER", "INTEGER"], "UBIGINT")
    con.create_function(
        "is_validation",
        _make_is_validation(cfg["seed"], cfg["val_fraction"]),
        ["VARCHAR"],
        "BOOLEAN",
    )
    return con


# --------------------------------------------------------------------------- hub I/O
def _is_empty_hub_commit(exc: Exception) -> bool:
    msg = str(exc).lower()
    return "empty commit" in msg or "no files have been modified" in msg


def upload_file_with_retry(api: HfApi, cfg: dict, local_path: str, repo_path: str, retries: int = 5) -> None:
    delay = 5
    for attempt in range(1, retries + 1):
        try:
            api.upload_file(
                path_or_fileobj=local_path,
                path_in_repo=repo_path,
                repo_id=cfg["repo_id"],
                repo_type="dataset",
                token=cfg["hf_token"],
            )
            return
        except Exception as exc:
            if _is_empty_hub_commit(exc):
                return
            if attempt == retries:
                raise
            log.warning("Upload %s failed (%s/%s): %s — retry in %ss", repo_path, attempt, retries, exc, delay)
            time.sleep(delay)
            delay = min(delay * 2, 120)


def upload_batch_with_retry(cfg: dict, pairs: list[tuple[str, str]], message: str, retries: int = 5) -> None:
    if not pairs:
        return
    delay = 5
    for attempt in range(1, retries + 1):
        try:
            ops = [CommitOperationAdd(path_in_repo=repo, path_or_fileobj=local) for local, repo in pairs]
            create_commit(
                repo_id=cfg["repo_id"],
                repo_type="dataset",
                operations=ops,
                commit_message=message,
                token=cfg["hf_token"],
            )
            return
        except Exception as exc:
            if _is_empty_hub_commit(exc):
                return
            if attempt == retries:
                raise
            log.warning("Batch upload failed (%s/%s): %s — retry in %ss", attempt, retries, exc, delay)
            time.sleep(delay)
            delay = min(delay * 2, 120)


def load_finalize_state(cfg: dict) -> dict:
    default = {
        "mirror_complete": False,
        "shuffle_complete": False,
        "buckets_done": [],
        "uploaded_train_parts": [],
        "uploaded_val_parts": [],
        "train_rows": 0,
        "train_bytes": 0,
        "val_rows": 0,
        "val_bytes": 0,
        "done": False,
        "engine": "duckdb-local",
    }
    local = Path(cfg["scratch_dir"]) / "manifest" / "finalize.json"
    if local.is_file():
        with open(local, "rb") as fh:
            state = json.loads(fh.read())
        default.update(state)
        return default
    try:
        p = hf_hub_download(
            repo_id=cfg["repo_id"],
            repo_type="dataset",
            filename="manifest/finalize.json",
            token=cfg["hf_token"],
            force_download=True,
        )
        with open(p, "rb") as fh:
            state = json.loads(fh.read())
        default.update(state)
        return default
    except Exception:
        return default


def save_finalize_state(cfg: dict, state: dict) -> None:
    manifest_dir = Path(cfg["scratch_dir"]) / "manifest"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    local = manifest_dir / "finalize.json"
    local.write_bytes(orjson.dumps(state))
    upload_file_with_retry(HfApi(token=cfg["hf_token"]), cfg, str(local), "manifest/finalize.json")


# --------------------------------------------------------------------------- paths
def _sql_str(path: str | Path) -> str:
    """Escape a filesystem path for embedding as a DuckDB SQL string literal."""
    return str(path).replace("'", "''")


def scratch_paths(cfg: dict) -> dict[str, Path]:
    root = Path(cfg["scratch_dir"])
    return {
        "root": root,
        "mirror": root / cfg["mirror_subdir"],
        "shuffle": root / "shuffle",
        "pending_train": root / "pending" / "train",
        "pending_val": root / "pending" / "validation",
        "staging_train": root / "final_staging" / "train",
        "staging_val": root / "final_staging" / "validation",
        "removed": root / "dedup" / "removed_ids.parquet",
    }


# --------------------------------------------------------------------------- phases
def download_removed_ids(cfg: dict, paths: dict[str, Path]) -> int:
    paths["removed"].parent.mkdir(parents=True, exist_ok=True)
    if paths["removed"].is_file():
        log.info("removed_ids already cached at %s", paths["removed"])
    else:
        log.info("Downloading dedup/removed_ids.parquet ...")
        p = hf_hub_download(
            repo_id=cfg["repo_id"],
            repo_type="dataset",
            filename="dedup/removed_ids.parquet",
            token=cfg["hf_token"],
        )
        paths["removed"].write_bytes(Path(p).read_bytes())
    n = duckdb.connect().execute(
        "SELECT count(*) FROM read_parquet(?)", [str(paths["removed"])]
    ).fetchone()[0]
    log.info("Removed-id list: %s documents", n)
    return int(n)


def mirror_data(cfg: dict, paths: dict[str, Path], state: dict) -> None:
    if cfg["skip_mirror"]:
        log.info("Skipping mirror (--skip-mirror). Expect parquet under %s/data/", paths["mirror"])
        return

    k = cfg["shuffle_buckets"]
    pending_ok = any(paths["pending_train"].glob("bucket_*.parquet"))
    staging_ok = any(paths["staging_train"].glob("*.parquet"))
    shuffle_ok = _shuffle_has_buckets(paths["shuffle"], k)

    # Mirror is deleted after phase 1 — do not re-download if later phases already have data.
    if (len(state.get("buckets_done", [])) >= k and pending_ok) or shuffle_ok or staging_ok:
        log.info(
            "Skipping mirror — later phases already have local data "
            "(pending=%s, shuffle=%s, staging=%s).",
            pending_ok,
            shuffle_ok,
            staging_ok,
        )
        # Drop any partial re-download so it does not waste disk.
        if paths["mirror"].exists() and not shuffle_ok:
            import shutil
            partial = list(paths["mirror"].rglob("*.parquet"))
            if partial and len(state.get("buckets_done", [])) >= k and pending_ok:
                log.info("Removing partial/unused data_mirror to free disk...")
                shutil.rmtree(paths["mirror"], ignore_errors=True)
        return

    mirror = paths["mirror"]
    if state.get("mirror_complete") and any(mirror.rglob("*.parquet")):
        log.info("Mirror already present at %s — skipping download.", mirror)
        return

    log.info("Mirroring data/** from Hub to %s ...", mirror)
    mirror.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=cfg["repo_id"],
        repo_type="dataset",
        allow_patterns=["data/**"],
        local_dir=str(mirror),
        token=cfg["hf_token"],
    )
    state["mirror_complete"] = True
    save_finalize_state(cfg, state)
    log.info("Mirror complete.")


def _data_glob(paths: dict[str, Path]) -> str:
    return str(paths["mirror"] / "data" / "*" / "*.parquet")


def phase1_partition(cfg: dict, con: duckdb.DuckDBPyConnection, paths: dict[str, Path], state: dict) -> None:
    k = cfg["shuffle_buckets"]
    buckets_done = state.get("buckets_done", [])
    pending_ok = any(paths["pending_train"].glob("bucket_*.parquet"))

    # After phase 2, shuffle/ and mirror/ are deleted — skip if pending already exists.
    if len(buckets_done) >= k and pending_ok:
        log.info("Phase 1 skip — phase 2 pending already on disk (%s buckets done).", len(buckets_done))
        return
    if state.get("shuffle_complete") and _shuffle_has_buckets(paths["shuffle"], k):
        log.info("Phase 1 skip — shuffle partitions already on disk.")
        return

    data_glob = _data_glob(paths)
    if not any(Path(paths["mirror"]).rglob("data/**/*.parquet")):
        raise FileNotFoundError(f"No mirrored parquet under {paths['mirror']}/data/ — run mirror first.")

    shuffle = paths["shuffle"]
    if shuffle.exists():
        import shutil
        shutil.rmtree(shuffle)
    shuffle.mkdir(parents=True, exist_ok=True)

    removed_path = str(paths["removed"])
    has_removed = paths["removed"].is_file() and duckdb.connect().execute(
        "SELECT count(*) FROM read_parquet(?)", [removed_path]
    ).fetchone()[0] > 0

    log.info("Phase 1 — DuckDB hash-partition into %s buckets (this is one streaming pass)...", k)
    t0 = time.time()

    # DuckDB COPY TO requires a string literal path (not a bound ? parameter).
    out = _sql_str(shuffle)
    data_lit = _sql_str(data_glob)
    removed_lit = _sql_str(removed_path)

    if has_removed:
        con.execute(
            f"""
            COPY (
                SELECT id, text, source, meta,
                       crc32_bucket(id, {int(k)}) AS bucket
                FROM read_parquet('{data_lit}', union_by_name=true)
                WHERE id NOT IN (SELECT id FROM read_parquet('{removed_lit}'))
            )
            TO '{out}' (FORMAT PARQUET, PARTITION_BY (bucket), COMPRESSION ZSTD, OVERWRITE_OR_IGNORE)
            """
        )
    else:
        con.execute(
            f"""
            COPY (
                SELECT id, text, source, meta,
                       crc32_bucket(id, {int(k)}) AS bucket
                FROM read_parquet('{data_lit}', union_by_name=true)
            )
            TO '{out}' (FORMAT PARQUET, PARTITION_BY (bucket), COMPRESSION ZSTD, OVERWRITE_OR_IGNORE)
            """
        )

    elapsed = time.time() - t0
    state["shuffle_complete"] = True
    save_finalize_state(cfg, state)
    log.info("Phase 1 done in %.1f min. Shuffle dir: %s", elapsed / 60, shuffle)

    # Free mirror disk.
    import shutil
    if paths["mirror"].exists():
        log.info("Removing mirror to free disk...")
        shutil.rmtree(paths["mirror"], ignore_errors=True)


def _shuffle_has_buckets(shuffle_dir: Path, k: int) -> bool:
    if not shuffle_dir.is_dir():
        return False
    found = sum(1 for i in range(k) if any((shuffle_dir / f"bucket={i}").glob("*.parquet")))
    return found >= k


def _bucket_glob(shuffle_dir: Path, bucket: int) -> str:
    return str(shuffle_dir / f"bucket={bucket}" / "*.parquet")


def phase2_shuffle_split(cfg: dict, con: duckdb.DuckDBPyConnection, paths: dict[str, Path], state: dict) -> dict:
    k = cfg["shuffle_buckets"]
    seed = cfg["seed"]
    buckets_done = set(state.get("buckets_done", []))

    paths["pending_train"].mkdir(parents=True, exist_ok=True)
    paths["pending_val"].mkdir(parents=True, exist_ok=True)

    # Hub manifest from a different machine may list buckets_done without local files.
    if buckets_done and not any(paths["pending_train"].glob("bucket_*.parquet")):
        log.warning(
            "Manifest has buckets_done=%s but no local pending/ — resetting bucket progress for this scratch.",
            len(buckets_done),
        )
        buckets_done = set()
        state["buckets_done"] = []

    per_source: dict[str, dict[str, int]] = {
        k: dict(v) for k, v in state.get("per_source_final", {}).items()
    }
    train_rows = int(state.get("train_rows", 0))
    val_rows = int(state.get("val_rows", 0))
    train_bytes = int(state.get("train_bytes", 0))
    val_bytes = int(state.get("val_bytes", 0))

    todo = [b for b in range(k) if b not in buckets_done]
    if not todo:
        log.info("Phase 2 skip — all %s buckets already processed.", k)
        return state.get("per_source_final", {})

    log.info("Phase 2 — shuffle + train/val split for %s buckets...", len(todo))
    t0 = time.time()

    for i, b in enumerate(todo):
        bucket_glob = _bucket_glob(paths["shuffle"], b)
        if not any(paths["shuffle"].glob(f"bucket={b}/*.parquet")):
            log.warning("Bucket %s empty — skipping.", b)
            buckets_done.add(b)
            continue

        train_out = str(paths["pending_train"] / f"bucket_{b:03d}.parquet")
        val_out = str(paths["pending_val"] / f"bucket_{b:03d}.parquet")
        bucket_lit = _sql_str(bucket_glob)
        train_lit = _sql_str(train_out)
        val_lit = _sql_str(val_out)

        # Per-bucket shuffle (ORDER BY shuffle_order) then split.
        # COPY TO path must be a SQL string literal (not a bound ?).
        con.execute(
            f"""
            COPY (
                SELECT id, text, source, meta
                FROM read_parquet('{bucket_lit}', union_by_name=true)
                WHERE NOT is_validation(id)
                ORDER BY shuffle_order(id, {int(b)}, {int(seed)})
            )
            TO '{train_lit}' (FORMAT PARQUET, COMPRESSION ZSTD, OVERWRITE_OR_IGNORE)
            """
        )
        con.execute(
            f"""
            COPY (
                SELECT id, text, source, meta
                FROM read_parquet('{bucket_lit}', union_by_name=true)
                WHERE is_validation(id)
                ORDER BY shuffle_order(id, {int(b)}, {int(seed)})
            )
            TO '{val_lit}' (FORMAT PARQUET, COMPRESSION ZSTD, OVERWRITE_OR_IGNORE)
            """
        )

        # Stats for this bucket.
        row = con.execute(
            """
            SELECT
                sum(CASE WHEN split = 'train' THEN 1 ELSE 0 END),
                sum(CASE WHEN split = 'train' THEN length(text) ELSE 0 END),
                sum(CASE WHEN split = 'val' THEN 1 ELSE 0 END),
                sum(CASE WHEN split = 'val' THEN length(text) ELSE 0 END)
            FROM (
                SELECT 'train' AS split, text FROM read_parquet(?)
                UNION ALL
                SELECT 'val', text FROM read_parquet(?)
            )
            """,
            [train_out, val_out],
        ).fetchone()
        tr, tb, vr, vb = (int(x or 0) for x in row)
        train_rows += tr
        val_rows += vr
        train_bytes += tb
        val_bytes += vb

        for src, cnt, byts in con.execute(
            """
            SELECT source, count(*), sum(length(text))
            FROM read_parquet(?, union_by_name=true)
            GROUP BY source
            """,
            [train_out],
        ).fetchall():
            cur = per_source.setdefault(src, {"rows": 0, "bytes": 0})
            cur["rows"] += int(cnt)
            cur["bytes"] += int(byts or 0)
        for src, cnt, byts in con.execute(
            """
            SELECT source, count(*), sum(length(text))
            FROM read_parquet(?, union_by_name=true)
            GROUP BY source
            """,
            [val_out],
        ).fetchall():
            cur = per_source.setdefault(src, {"rows": 0, "bytes": 0})
            cur["rows"] += int(cnt)
            cur["bytes"] += int(byts or 0)

        buckets_done.add(b)
        state["buckets_done"] = sorted(buckets_done)
        state["per_source_final"] = per_source
        state["train_rows"] = train_rows
        state["val_rows"] = val_rows
        state["train_bytes"] = train_bytes
        state["val_bytes"] = val_bytes
        # Checkpoint to Hub every 8 buckets (not every bucket — Hub commits are slow).
        if (i + 1) % 8 == 0 or (i + 1) == len(todo):
            save_finalize_state(cfg, state)
            log.info("Phase 2 progress: %s/%s buckets (%.1f min elapsed)", i + 1, len(todo), (time.time() - t0) / 60)

    log.info("Phase 2 done in %.1f min.", (time.time() - t0) / 60)

    import shutil
    if paths["shuffle"].exists():
        log.info("Removing shuffle dir to free disk...")
        shutil.rmtree(paths["shuffle"], ignore_errors=True)

    return per_source


def _table_text_bytes(table) -> int:
    import pyarrow.compute as pc

    if table.num_rows == 0:
        return 0
    return int(pc.sum(pc.utf8_length(table["text"])).as_py() or 0)


def _merge_pending_to_shards(
    pending_dir: Path,
    staging_dir: Path,
    target_bytes: int,
    batch_rows: int = 2_000,
) -> list[Path]:
    """Stream pending bucket_*.parquet into ~target_bytes final shards (low RAM)."""
    import gc

    import pyarrow as pa
    import pyarrow.parquet as pq

    staging_dir.mkdir(parents=True, exist_ok=True)
    # Clear any partial previous attempt.
    for old in staging_dir.glob("*.parquet"):
        old.unlink()

    files = sorted(pending_dir.glob("bucket_*.parquet"))
    if not files:
        return []

    accum: list = []
    accum_bytes = 0
    part_idx = 0
    staged: list[Path] = []
    cols = ["id", "text", "source", "meta"]

    def flush() -> None:
        nonlocal accum, accum_bytes, part_idx
        if not accum:
            return
        out_tbl = pa.concat_tables(accum) if len(accum) > 1 else accum[0]
        out_path = staging_dir / f"part-{part_idx:05d}.parquet"
        pq.write_table(out_tbl, out_path, compression="zstd", compression_level=9)
        staged.append(out_path)
        part_idx += 1
        del out_tbl
        accum = []
        accum_bytes = 0
        gc.collect()

    def feed(tbl) -> None:
        nonlocal accum_bytes
        nbytes = _table_text_bytes(tbl)
        if nbytes == 0 and tbl.num_rows == 0:
            return
        if nbytes > target_bytes and tbl.num_rows > 1:
            # Oversized batch: slice into smaller pieces.
            mid = tbl.num_rows // 2
            feed(tbl.slice(0, mid))
            feed(tbl.slice(mid))
            return
        if accum and accum_bytes + nbytes > target_bytes:
            flush()
        accum.append(tbl)
        accum_bytes += nbytes
        if accum_bytes >= target_bytes:
            flush()

    for i, path in enumerate(files):
        try:
            pf = pq.ParquetFile(path)
            if pf.metadata is None or pf.metadata.num_rows == 0:
                continue
            for batch in pf.iter_batches(batch_size=batch_rows, columns=cols):
                feed(pa.Table.from_batches([batch]))
        except Exception as exc:
            log.warning("Skipping unreadable pending file %s: %s", path.name, exc)
        if (i + 1) % 8 == 0 or (i + 1) == len(files):
            log.info("  merge progress: %s/%s pending files → %s shards so far", i + 1, len(files), part_idx)
        gc.collect()

    flush()
    return staged


def phase3_merge(cfg: dict, con: duckdb.DuckDBPyConnection, paths: dict[str, Path]) -> tuple[list[Path], list[Path]]:
    """Merge pending buckets into ~384 MB shards via streaming PyArrow (fits 8 GB RAM)."""
    target = cfg["final_shard_bytes"]
    paths["staging_train"].mkdir(parents=True, exist_ok=True)
    paths["staging_val"].mkdir(parents=True, exist_ok=True)

    pending_train = list(paths["pending_train"].glob("bucket_*.parquet"))
    pending_val = list(paths["pending_val"].glob("bucket_*.parquet"))
    train_parts = sorted(paths["staging_train"].glob("*.parquet"))
    val_parts = sorted(paths["staging_val"].glob("*.parquet"))

    # Resume: staging complete and pending already cleaned up.
    if train_parts and not pending_train:
        log.info("Phase 3 skip — staging shards already present (%s train, %s val).", len(train_parts), len(val_parts))
        return train_parts, val_parts

    if not pending_train:
        raise RuntimeError("No pending train buckets — phase 2 incomplete.")

    log.info(
        "Phase 3 — streaming merge of %s train + %s val pending files into ~%d MB shards...",
        len(pending_train),
        len(pending_val),
        target // (1024 * 1024),
    )
    t0 = time.time()

    # Drop DuckDB temp pressure before the PyArrow merge.
    try:
        con.execute("SET threads=1")
        con.execute("SET memory_limit='1GB'")
    except Exception:
        pass

    log.info("Phase 3 train merge...")
    train_parts = _merge_pending_to_shards(paths["pending_train"], paths["staging_train"], target)
    log.info("Phase 3 validation merge...")
    val_parts = _merge_pending_to_shards(paths["pending_val"], paths["staging_val"], target)

    log.info(
        "Phase 3 done in %.1f min — %s train + %s val shards.",
        (time.time() - t0) / 60,
        len(train_parts),
        len(val_parts),
    )

    import shutil
    pending_root = paths["pending_train"].parent
    if pending_root.exists():
        log.info("Removing pending dir to free disk...")
        shutil.rmtree(pending_root, ignore_errors=True)

    return train_parts, val_parts


def phase4_upload(
    cfg: dict,
    train_parts: list[Path],
    val_parts: list[Path],
    state: dict,
) -> tuple[list[str], list[str]]:
    if cfg["skip_upload"]:
        log.info("Skipping Hub upload (--skip-upload). Shards at final_staging/.")
        return [], []

    uploaded_train = set(state.get("uploaded_train_parts", []))
    uploaded_val = set(state.get("uploaded_val_parts", []))
    batch_size = cfg["upload_batch_size"]

    def _upload_split(parts: list[Path], split: str, already: set[str]) -> list[str]:
        prefix = f"final/{split}/"
        pending = []
        for p in parts:
            repo = prefix + p.name
            if repo in already:
                continue
            pending.append((str(p), repo))
        if not pending:
            return sorted(already)

        batches = [pending[i : i + batch_size] for i in range(0, len(pending), batch_size)]
        log.info("Uploading %s %s shards in %s batch commit(s)...", len(pending), split, len(batches))
        new_uploaded = list(already)
        for i, batch in enumerate(batches):
            upload_batch_with_retry(cfg, batch, f"DuckDB finalize {split} batch {i + 1}/{len(batches)}")
            new_uploaded.extend(repo for _, repo in batch)
            key = "uploaded_train_parts" if split == "train" else "uploaded_val_parts"
            state[key] = sorted(new_uploaded)
            save_finalize_state(cfg, state)
        return sorted(new_uploaded)

    ut = _upload_split(train_parts, "train", uploaded_train)
    uv = _upload_split(val_parts, "validation", uploaded_val)
    return ut, uv


def _load_collected_stats(cfg: dict) -> dict[str, dict[str, int]]:
    per: dict[str, dict[str, int]] = {}
    for source in SOURCES:
        try:
            p = hf_hub_download(
                repo_id=cfg["repo_id"],
                repo_type="dataset",
                filename=f"manifest/{source}.json",
                token=cfg["hf_token"],
            )
            st = json.loads(Path(p).read_bytes())
            if st.get("sub"):
                rows = sum(u.get("rows_written", 0) for u in st["sub"].values())
                byts = sum(u.get("bytes_written", 0) for u in st["sub"].values())
            else:
                rows = st.get("rows_written", 0)
                byts = st.get("bytes_written", 0)
            per[source] = {"rows": rows, "bytes": byts}
        except Exception:
            per[source] = {"rows": 0, "bytes": 0}
    return per


def _write_dataset_card(stats: dict) -> str:
    fin = stats["final"]

    def fmt_gb(n: int) -> str:
        return f"{n / GB:.2f} GB"

    rows = [
        "| Source | Collected rows | Collected size | Final rows | Final size |",
        "| --- | --- | --- | --- | --- |",
    ]
    for source in SOURCES:
        col = stats["per_source_collected"].get(source, {"rows": 0, "bytes": 0})
        fn = stats["per_source_final"].get(source, {"rows": 0, "bytes": 0})
        rows.append(
            f"| {source} | {col['rows']:,} | {fmt_gb(col['bytes'])} | "
            f"{fn['rows']:,} | {fmt_gb(fn['bytes'])} |"
        )
    table = "\n".join(rows)
    return f"""---
license: other
language:
- en
size_categories:
- 10B<n<100B
task_categories:
- text-generation
tags:
- code
- pretraining
configs:
- config_name: default
  data_files:
  - split: train
    path: final/train/*.parquet
  - split: validation
    path: final/validation/*.parquet
---

# Coding LLM Pretraining Corpus

Built with `build_pretrain_dataset.ipynb` + `finalize_local.py` (DuckDB).

## Composition

{table}

**Total final corpus:** {fmt_gb(fin['train_bytes'] + fin['val_bytes'])} of raw text
({fin['train_rows'] + fin['val_rows']:,} documents) across
{fin['train_shards']} train shards and {fin['val_shards']} validation shards.
{stats['removed_ids']:,} documents were removed by dedup/decontamination.
"""


def publish_metadata(cfg: dict, state: dict, per_source_final: dict, removed_count: int, ut: list[str], uv: list[str]) -> None:
    stats = {
        "final": {
            "train_rows": state["train_rows"],
            "train_bytes": state["train_bytes"],
            "train_shards": len(ut) or len(list(Path(cfg["scratch_dir"]).glob("final_staging/train/*.parquet"))),
            "val_rows": state["val_rows"],
            "val_bytes": state["val_bytes"],
            "val_shards": len(uv) or len(list(Path(cfg["scratch_dir"]).glob("final_staging/validation/*.parquet"))),
        },
        "per_source_final": per_source_final,
        "per_source_collected": _load_collected_stats(cfg),
        "removed_ids": removed_count,
    }
    scratch = Path(cfg["scratch_dir"])
    stats_path = scratch / "stats.json"
    stats_path.write_bytes(orjson.dumps(stats, option=orjson.OPT_INDENT_2))
    card_path = scratch / "README.md"
    card_path.write_text(_write_dataset_card(stats), encoding="utf-8")

    if not cfg["skip_upload"]:
        upload_file_with_retry(HfApi(token=cfg["hf_token"]), cfg, str(stats_path), "stats.json")
        upload_file_with_retry(HfApi(token=cfg["hf_token"]), cfg, str(card_path), "README.md")

    log.info("=== stats.json ===")
    print(json.dumps(stats, indent=2))


def cmd_finalize(cfg: dict) -> None:
    os.makedirs(cfg["scratch_dir"], exist_ok=True)
    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")

    api = HfApi(token=cfg["hf_token"])
    api.create_repo(repo_id=cfg["repo_id"], repo_type="dataset", private=cfg["private"], exist_ok=True)

    paths = scratch_paths(cfg)
    paths["root"].mkdir(parents=True, exist_ok=True)

    state = load_finalize_state(cfg)
    if state.get("done"):
        log.info("Already complete per manifest/finalize.json — skipping.")
        return

    con = open_duckdb(cfg)
    removed_count = download_removed_ids(cfg, paths)
    mirror_data(cfg, paths, state)
    phase1_partition(cfg, con, paths, state)
    per_source_final = phase2_shuffle_split(cfg, con, paths, state)
    train_parts, val_parts = phase3_merge(cfg, con, paths)

    # Recompute exact totals from staged shards (file-by-file to stay under 8 GB).
    def _sum_staged(parts: list[Path]) -> tuple[int, int]:
        import pyarrow.parquet as pq

        rows = byts = 0
        for p in parts:
            t = pq.read_table(p, columns=["text"])
            rows += t.num_rows
            byts += _table_text_bytes(t)
            del t
        return rows, byts

    state["train_rows"], state["train_bytes"] = _sum_staged(train_parts)
    state["val_rows"], state["val_bytes"] = _sum_staged(val_parts)
    save_finalize_state(cfg, state)

    ut, uv = phase4_upload(cfg, train_parts, val_parts, state)
    publish_metadata(cfg, state, per_source_final, removed_count, ut, uv)

    import shutil
    staging = paths["staging_train"].parent
    if staging.exists() and not cfg["skip_upload"]:
        shutil.rmtree(staging, ignore_errors=True)

    state["done"] = True
    save_finalize_state(cfg, state)

    total_gb = (state["train_bytes"] + state["val_bytes"]) / GB
    log.info(
        "=== Finalize complete: %s train + %s val = %.2f GB ===",
        f"{state['train_rows']:,}",
        f"{state['val_rows']:,}",
        total_gb,
    )
    log.info("Dataset: https://huggingface.co/datasets/%s", cfg["repo_id"])


def cmd_sanity_check(cfg: dict, n_samples: int, local: bool) -> None:
    """Print stats + sample documents from Hub (or local staging)."""
    print("=== stats.json ===")
    try:
        if local:
            stats_path = Path(cfg["scratch_dir"]) / "stats.json"
            stats = json.loads(stats_path.read_bytes())
        else:
            sp = hf_hub_download(
                repo_id=cfg["repo_id"],
                repo_type="dataset",
                filename="stats.json",
                token=cfg["hf_token"],
                force_download=True,
            )
            stats = json.loads(Path(sp).read_bytes())
        print(json.dumps(stats, indent=2))
        fin = stats["final"]
        total_gb = (fin["train_bytes"] + fin["val_bytes"]) / GB
        print(f"\nTotal final corpus: {total_gb:.2f} GB ({fin['train_rows'] + fin['val_rows']:,} docs)")

        print("\n=== Per-source budget check ===")
        for source in SOURCES:
            col = stats.get("per_source_collected", {}).get(source, {})
            fn = stats.get("per_source_final", {}).get(source, {})
            col_gb = col.get("bytes", 0) / GB
            fn_gb = fn.get("bytes", 0) / GB
            print(f"  {source:6s}  collected={col_gb:6.2f} GB  final={fn_gb:6.2f} GB  rows={fn.get('rows', 0):,}")
    except Exception as exc:
        log.warning("Could not read stats.json: %s", exc)

    print(f"\n=== {n_samples} sample train documents ===")
    try:
        if local:
            from datasets import load_dataset
            train_glob = str(Path(cfg["scratch_dir"]) / "final_staging" / "train" / "*.parquet")
            ds = load_dataset("parquet", data_files={"train": train_glob}, split="train", streaming=True)
        else:
            from datasets import load_dataset
            ds = load_dataset(cfg["repo_id"], split="train", streaming=True, token=cfg["hf_token"])

        for i, row in enumerate(ds):
            if i >= n_samples:
                break
            preview = row["text"][:500].replace("\n", " ")
            print(f"\n[{i}] source={row['source']} id={row['id'][:12]} meta={row['meta'][:120]}")
            print(f"    {preview}")
    except Exception as exc:
        log.warning("Could not stream samples: %s", exc)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="DuckDB finalize for coder-pretrain dataset (Jetson / Linux).")
    p.add_argument("--repo-id", default=DEFAULT_REPO)
    p.add_argument("--scratch-dir", default=DEFAULT_SCRATCH)
    p.add_argument("--hf-token", default=None)
    p.add_argument("--private", action="store_true", default=True)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--shuffle-buckets", type=int, default=64)
    p.add_argument("--val-fraction", type=float, default=0.001)
    p.add_argument("--final-shard-bytes", type=int, default=384 * 1024 * 1024)
    p.add_argument("--upload-batch-size", type=int, default=16)
    p.add_argument("--threads", type=int, default=4, help="DuckDB worker threads (4 is safe on Jetson Orin).")
    p.add_argument("--memory-limit", default="5GB", help="DuckDB RAM cap (use 5GB on 8GB Jetson).")
    p.add_argument("--skip-upload", action="store_true", help="Build local shards only; no Hub push.")
    p.add_argument("--skip-mirror", action="store_true", help="Use existing data_mirror/ under scratch.")

    sub = p.add_subparsers(dest="command", required=True)

    sub.add_parser("finalize", help="Run mirror + DuckDB phases 1-3 + upload + metadata.")

    sc = sub.add_parser("sanity-check", help="Print stats.json and sample train documents.")
    sc.add_argument("--samples", type=int, default=3)
    sc.add_argument("--local", action="store_true", help="Read from local scratch instead of Hub.")

    all_p = sub.add_parser("all", help="finalize then sanity-check.")
    all_p.add_argument("--samples", type=int, default=3)

    return p


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%H:%M:%S",
    )
    args = build_parser().parse_args()
    cfg = build_config(args)

    if args.command == "finalize":
        cmd_finalize(cfg)
    elif args.command == "sanity-check":
        cmd_sanity_check(cfg, args.samples, args.local)
    elif args.command == "all":
        cmd_finalize(cfg)
        cmd_sanity_check(cfg, args.samples, local=cfg["skip_upload"])


if __name__ == "__main__":
    main()

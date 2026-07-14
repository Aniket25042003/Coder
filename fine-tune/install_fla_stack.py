#!/usr/bin/env python3
"""Legacy: Qwen3.5 hybrid FLA deps (flash-linear-attention + causal-conv1d).

Phase 1 now uses Qwen3-4B-Base (dense full attention) — do NOT run this for that path.
Prefer xformers / flash-attn via phase1_colab.ipynb instead.

Colab / recent torch often has no matching causal-conv1d PyPI wheel, so pip
tries a source build and fails. This helper installs a Dao-AILab prebuilt wheel
(wheel-first, nearest torch version), then falls back to --no-build-isolation.
"""

from __future__ import annotations

import importlib
import subprocess
import sys
import urllib.error
import urllib.request
from typing import List, Optional, Sequence, Tuple


CAUSAL_CONV1D_RELEASE_TAG = "v1.6.2.post1"
CAUSAL_CONV1D_PKG_VER = "1.6.2.post1"
CAUSAL_CONV1D_BASE = (
    "https://github.com/Dao-AILab/causal-conv1d/releases/download"
)


def _pip(args: Sequence[str]) -> int:
    return subprocess.call([sys.executable, "-m", "pip", "install", *args])


def _importable(name: str) -> bool:
    importlib.invalidate_caches()
    try:
        __import__(name)
        return True
    except Exception:
        return False


def _torch_env() -> Tuple[str, str, str, str]:
    import torch

    py = f"cp{sys.version_info.major}{sys.version_info.minor}"
    torch_mm = ".".join(torch.__version__.split("+")[0].split(".")[:2])
    # cu128 / cu124 / cu121 all use cu12 wheels from Dao-AILab
    cuda = "12"
    if torch.version.cuda:
        cuda = str(torch.version.cuda).split(".")[0]
    abi = "TRUE" if bool(torch._C._GLIBCXX_USE_CXX11_ABI) else "FALSE"
    return py, torch_mm, cuda, abi


def _url_ok(url: str) -> bool:
    try:
        req = urllib.request.Request(url, method="HEAD")
        with urllib.request.urlopen(req, timeout=20) as resp:
            return 200 <= getattr(resp, "status", 200) < 400
    except Exception:
        try:
            with urllib.request.urlopen(url, timeout=20) as resp:
                return 200 <= getattr(resp, "status", 200) < 400
        except Exception:
            return False


def _candidate_torch_versions(torch_mm: str) -> List[str]:
    known = ["2.11", "2.10", "2.9", "2.8", "2.7", "2.6"]
    ordered = [torch_mm] + [v for v in known if v != torch_mm]
    # Prefer same-or-lower majors for ABI safety after exact miss
    out: List[str] = []
    for v in ordered:
        if v not in out:
            out.append(v)
    return out


def install_flash_linear_attention() -> bool:
    if _importable("fla") or _importable("flash_linear_attention"):
        print("flash-linear-attention already importable")
        return True
    print("Installing flash-linear-attention[cuda] ...")
    code = _pip(["-q", "flash-linear-attention[cuda]"])
    if code != 0:
        code = _pip(["-q", "flash-linear-attention"])
    ok = _importable("fla") or _importable("flash_linear_attention")
    print("flash-linear-attention OK" if ok else "flash-linear-attention FAILED")
    return ok


def install_causal_conv1d() -> bool:
    if _importable("causal_conv1d"):
        print("causal_conv1d already importable")
        return True

    py, torch_mm, cuda, abi = _torch_env()
    print(
        f"causal-conv1d: looking for prebuilt wheel "
        f"(py={py} torch={torch_mm} cu{cuda} cxx11abi={abi})"
    )

    for tv in _candidate_torch_versions(torch_mm):
        for try_abi in (abi, "TRUE" if abi == "FALSE" else "FALSE"):
            fname = (
                f"causal_conv1d-{CAUSAL_CONV1D_PKG_VER}+cu{cuda}torch{tv}"
                f"cxx11abi{try_abi}-{py}-{py}-linux_x86_64.whl"
            )
            url = f"{CAUSAL_CONV1D_BASE}/{CAUSAL_CONV1D_RELEASE_TAG}/{fname}".replace(
                "+", "%2B"
            )
            if not _url_ok(url):
                continue
            print(f"Installing prebuilt: {fname}")
            code = _pip(["-q", url])
            if code == 0 and _importable("causal_conv1d"):
                print(f"causal_conv1d OK via wheel (torch{tv} abi{try_abi})")
                return True
            print(f"Wheel install/import failed for {fname}")

    print(
        "No working prebuilt wheel; trying source build with --no-build-isolation ..."
    )
    _pip(["-q", "ninja", "packaging", "wheel"])
    code = _pip(
        [
            f"causal-conv1d=={CAUSAL_CONV1D_PKG_VER}",
            "--no-build-isolation",
            "--no-cache-dir",
            "--force-reinstall",
            "--no-deps",
        ]
    )
    ok = code == 0 and _importable("causal_conv1d")
    print("causal_conv1d OK via source" if ok else "causal_conv1d FAILED")
    return ok


def main() -> int:
    fla_ok = install_flash_linear_attention()
    conv_ok = install_causal_conv1d()
    print(f"FLA stack: flash-linear-attention={fla_ok} causal_conv1d={conv_ok}")
    if fla_ok and conv_ok:
        print("FLA stack OK")
        return 0
    print(
        "WARNING: FLA incomplete — Qwen3.5 will use slow torch fallback. "
        "Fix causal-conv1d before a long Colab run."
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())

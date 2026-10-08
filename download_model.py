# -*- coding: utf-8 -*-
"""Download the model for demo_inference_text.py (this file and the demo are all you need to run it).

    python download_model.py                       # kev-0.6b, q4f16 (323 MB) from Hugging Face
    python download_model.py --profile kev-4b      # larger, better decisions
    python download_model.py --variant q4          # 4-bit MatMulNBits, fp32 elsewhere
    python download_model.py --list

Kev publishes 4-bit builds only, so there is no fp32 file to fetch or to compare against.
Plain HTTPS from the standard library: no git, no git-lfs, no huggingface_hub. Interrupted downloads resume.
"""

from __future__ import annotations

import argparse
import pathlib
import shutil
import sys
import time
import urllib.error
import urllib.request
from typing import Iterable, Optional

ROOT = pathlib.Path(__file__).resolve().parent

PROFILES = {
    "kev-0.6b": {
        "repo": "onnx-community/kev-0.6b-ONNX",
        "dir": "onnx",
        "note": "LoRA + pointer head on Qwen3-0.6B-Base. ~340 MB.",
    },
    "kev-4b": {
        "repo": "onnx-community/kev-4b-ONNX",
        "dir": "onnx-4b",
        "note": "LoRA + pointer head on Qwen3-4B-Base. Better decisions, slower.",
    },
}

#: Only 4-bit builds are published for Kev; there is no fp32/fp16 to fall back on.
VARIANTS = {
    "q4f16": "default: 4-bit weights in an fp16 graph",
    "q4": "4-bit MatMulNBits, fp32 elsewhere",
}

#: Everything the runtime reads, beside the graph itself.
SUPPORT_FILES = ["tokenizer.json", "tokenizer_config.json", "config.json",
                 "added_tokens.json", "special_tokens_map.json"]

import pathlib
import shutil
import sys
import time
import urllib.error
import urllib.request
from typing import Iterable, Optional

ENDPOINT = "https://huggingface.co/{repo}/resolve/{revision}/{path}"
CHUNK = 1 << 20  # 1 MiB
USER_AGENT = "kev-decide-onnx-cpu/0.1 (urllib)"


def _human(size: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024 or unit == "GB":
            return "%.1f %s" % (size, unit)
        size /= 1024
    return "%.1f GB" % size


def _remote_size(url: str) -> Optional[int]:
    request = urllib.request.Request(url, method="HEAD", headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            length = response.headers.get("Content-Length")
            return int(length) if length else None
    except (urllib.error.URLError, ValueError):
        return None


def fetch(url: str, dest: pathlib.Path, label: str = "") -> pathlib.Path:
    """Download ``url`` to ``dest``, resuming a partial file if one is there."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    partial = dest.with_suffix(dest.suffix + ".part")
    total = _remote_size(url)

    if dest.exists() and total is not None and dest.stat().st_size == total:
        print("  have  %-44s %s" % (label or dest.name, _human(total)))
        return dest

    have = partial.stat().st_size if partial.exists() else 0
    headers = {"User-Agent": USER_AGENT}
    if have and total is not None and have < total:
        headers["Range"] = "bytes=%d-" % have
    elif have:
        have = 0
        partial.unlink()

    request = urllib.request.Request(url, headers=headers)
    started = time.time()
    with urllib.request.urlopen(request, timeout=60) as response:
        if response.status not in (200, 206):
            raise RuntimeError("HTTP %s for %s" % (response.status, url))
        if response.status == 200:
            have = 0  # server ignored the Range; start over
        mode = "ab" if have else "wb"
        with open(partial, mode) as handle:
            done = have
            last = 0.0
            while True:
                block = response.read(CHUNK)
                if not block:
                    break
                handle.write(block)
                done += len(block)
                now = time.time()
                if total and (now - last > 0.5):
                    last = now
                    rate = done / max(now - started, 1e-6)
                    sys.stdout.write(
                        "\r  %-44s %5.1f%%  %s/s   " %
                        (label or dest.name, 100 * done / total, _human(rate))
                    )
                    sys.stdout.flush()

    if total is not None and partial.stat().st_size != total:
        raise RuntimeError(
            "%s: got %d bytes, expected %d" % (dest.name, partial.stat().st_size, total)
        )
    shutil.move(str(partial), str(dest))
    elapsed = time.time() - started
    sys.stdout.write("\r  got   %-44s %s in %.0f s\n"
                     % (label or dest.name, _human(dest.stat().st_size), elapsed))
    sys.stdout.flush()
    return dest


def fetch_files(
    repo: str,
    paths: Iterable[str],
    dest_dir: pathlib.Path,
    revision: str = "main",
    optional: Iterable[str] = (),
) -> None:
    """Fetch each path in ``paths`` from ``repo`` into ``dest_dir``.

    Paths in ``optional`` are skipped if the repository does not have them,
    which keeps one file list usable across model variants.
    """
    optional = set(optional)
    for path in paths:
        url = ENDPOINT.format(repo=repo, revision=revision, path=path)
        target = dest_dir / path
        try:
            fetch(url, target, label=path)
        except (urllib.error.HTTPError, RuntimeError) as exc:
            if path in optional:
                print("  skip  %-44s (not in this repository)" % path)
                continue
            raise SystemExit("failed to download %s: %s" % (url, exc)) from exc


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--list", action="store_true", help="show profiles and variants")
    parser.add_argument("--profile", default="kev-0.6b", choices=sorted(PROFILES))
    parser.add_argument("--variant", default="q4f16", choices=sorted(VARIANTS))
    args = parser.parse_args()

    if args.list:
        print("profiles:
")
        for name, spec in PROFILES.items():
            print("  --profile %-9s %s
%s%s" % (name, spec["repo"], " " * 22, spec["note"]))
        print("
variants:
")
        for name, note in VARIANTS.items():
            print("  --variant %-7s %s" % (name, note))
        print("
Kev publishes 4-bit builds only; there is no fp32 reference to compare against.")
        return 0

    spec = PROFILES[args.profile]
    target = ROOT / "models" / spec["dir"]
    stem = "onnx/model_%s.onnx" % args.variant

    print("%s (%s) -> %s" % (spec["repo"], args.variant, target), flush=True)
    fetch_files(spec["repo"], SUPPORT_FILES + [stem, stem + "_data"], target,
                optional=[stem + "_data"])

    print("
ready: %s" % target)
    print("run:    uv run demo_inference_text.py")
    print("verify: uv run verify/run_all.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

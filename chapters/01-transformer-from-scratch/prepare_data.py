"""Download + tokenize a small text dataset to a single binary file of uint16 token IDs.

Supported datasets:
    --dataset tinyshakespeare   (1 MB, included in this script via download)
    --dataset tinystories       (~2 GB streaming from HF)

Output layout:
    data/<dataset>/train.bin
    data/<dataset>/val.bin

Tokenizer: tiktoken `gpt2` BPE (50257 tokens) by default. Override with `--tokenizer` if
you want to use a HuggingFace tokenizer.
"""

from __future__ import annotations

import argparse
import os
import urllib.request
from pathlib import Path

import numpy as np
import tiktoken


TINY_SHAKESPEARE_URL = (
    "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
)


def fetch_tinyshakespeare(out_dir: Path) -> str:
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_path = out_dir / "input.txt"
    if not raw_path.exists():
        print(f"Downloading TinyShakespeare to {raw_path}")
        urllib.request.urlretrieve(TINY_SHAKESPEARE_URL, raw_path)
    return raw_path.read_text()


def tokenize_and_save(text: str, out_dir: Path, tokenizer_name: str = "gpt2") -> None:
    enc = tiktoken.get_encoding(tokenizer_name)
    ids = enc.encode_ordinary(text)
    n = len(ids)
    split = int(0.9 * n)
    train, val = np.array(ids[:split], dtype=np.uint16), np.array(ids[split:], dtype=np.uint16)
    (out_dir / "train.bin").write_bytes(train.tobytes())
    (out_dir / "val.bin").write_bytes(val.tobytes())
    print(f"Wrote {len(train):,} train tokens and {len(val):,} val tokens to {out_dir}")
    print(f"Vocab size in use: {enc.n_vocab}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=["tinyshakespeare"], default="tinyshakespeare")
    p.add_argument("--tokenizer", default="gpt2", help="tiktoken encoding name")
    p.add_argument("--out", default=None, help="output directory (default: data/<dataset>)")
    args = p.parse_args()

    out_dir = Path(args.out) if args.out else Path("data") / args.dataset
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.dataset == "tinyshakespeare":
        text = fetch_tinyshakespeare(out_dir)
        tokenize_and_save(text, out_dir, args.tokenizer)
    else:
        raise SystemExit(f"Unknown dataset {args.dataset}")


if __name__ == "__main__":
    main()

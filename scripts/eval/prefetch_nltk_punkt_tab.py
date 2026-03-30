#!/usr/bin/env python3
import argparse
import os
import sys

import nltk


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Download NLTK punkt_tab resource to a local directory for offline evaluation."
    )
    parser.add_argument(
        "--dst",
        default="/docker/l00625974/Rebuttal/huggingface_cache/nltk_data",
        help="Target directory to store NLTK data.",
    )
    args = parser.parse_args()

    os.makedirs(args.dst, exist_ok=True)

    ok = nltk.download("punkt_tab", download_dir=args.dst, quiet=False)
    if not ok:
        print("[ERROR] Failed to download punkt_tab.", file=sys.stderr)
        return 1

    # Validate resource can be resolved from the target dir
    nltk.data.path.insert(0, args.dst)
    try:
        found = nltk.data.find("tokenizers/punkt_tab")
    except LookupError:
        print("[ERROR] punkt_tab not found after download.", file=sys.stderr)
        return 2

    print(f"[OK] punkt_tab ready: {found}")
    print(f"[OK] NLTK data dir: {args.dst}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

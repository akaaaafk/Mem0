#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Download HotpotQA (distractor, validation) from HuggingFace to data/hotpotqa/hotpotqa_dev_distractor.json.
Usage (from A-mem root):
  python data/download_hotpotqa.py
"""
import os
import sys
import json

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_A_MEM_ROOT = os.path.dirname(_SCRIPT_DIR)
_OUT_DIR = os.path.join(_A_MEM_ROOT, "data", "hotpotqa")
_OUT_FILE = os.path.join(_OUT_DIR, "hotpotqa_dev_distractor.json")


def main():
    try:
        from datasets import load_dataset
    except ImportError:
        print("请先安装: pip install datasets")
        sys.exit(1)

    print("Downloading HotpotQA (distractor, validation) from HuggingFace...")
    ds = load_dataset("hotpot_qa", "distractor", split="validation")
    # Convert to list of dicts (HF rows may be Arrow-backed)
    items = [dict(row) for row in ds]
    print(f"  Loaded {len(items)} examples.")

    os.makedirs(_OUT_DIR, exist_ok=True)
    with open(_OUT_FILE, "w", encoding="utf-8") as f:
        json.dump(items, f, ensure_ascii=False, indent=2)
    print(f"Saved to {_OUT_FILE}")
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)

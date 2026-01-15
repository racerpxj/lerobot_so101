#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def update_policy_preprocessor(path: Path, new_tokenizer: str) -> bool:
    if not path.exists():
        return False
    data = json.loads(path.read_text(encoding="utf-8"))
    updated = False
    for step in data.get("steps", []):
        if isinstance(step, dict) and step.get("registry_name") == "tokenizer_processor":
            cfg = step.get("config", {})
            if "tokenizer_name" in cfg:
                cfg["tokenizer_name"] = new_tokenizer
                step["config"] = cfg
                updated = True
    if updated:
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    return updated


def main() -> None:
    parser = argparse.ArgumentParser(description="Update tokenizer_name in policy_preprocessor.json")
    parser.add_argument("--model", action="append", help="Model directory to update (repeatable)")
    parser.add_argument("--paligemma", help="New paligemma/tokenizer path")
    parser.add_argument("--root", action="append", help="Alias of --model")
    parser.add_argument("--tokenizer", help="Alias of --paligemma")
    args = parser.parse_args()

    model_roots = args.model or []
    if args.root:
        model_roots.extend(args.root)
    tokenizer_path = args.paligemma or args.tokenizer

    if not model_roots or not tokenizer_path:
        raise SystemExit("Usage: --model <dir> --paligemma <path> (both required)")

    updated_any = False
    for root in [Path(p) for p in model_roots]:
        preproc = root / "policy_preprocessor.json"
        if update_policy_preprocessor(preproc, tokenizer_path):
            print(f"updated: {preproc}")
            updated_any = True
        else:
            print(f"skipped: {preproc}")
    if not updated_any:
        raise SystemExit("No files updated.")


if __name__ == "__main__":
    main()

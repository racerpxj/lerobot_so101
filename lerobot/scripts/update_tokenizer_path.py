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
    parser.add_argument("--root", action="append", required=True, help="Model directory to update")
    parser.add_argument("--tokenizer", required=True, help="New tokenizer path")
    args = parser.parse_args()

    updated_any = False
    for root in [Path(p) for p in args.root]:
        preproc = root / "policy_preprocessor.json"
        if update_policy_preprocessor(preproc, args.tokenizer):
            print(f"updated: {preproc}")
            updated_any = True
        else:
            print(f"skipped: {preproc}")
    if not updated_any:
        raise SystemExit("No files updated.")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Dispatch lightweight candidate preparation unless its diagnostic PR exists."""

import argparse
import json
import re
import subprocess


def dispatch(inputs: dict[str, str], repo: str) -> bool:
    revision = inputs.get("ort_commit") or inputs.get("model_commit", "")
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("candidate revision must be a full lowercase commit SHA")
    if "ort_tag" in inputs:
        tag = inputs["ort_tag"]
        if not re.fullmatch(r"v\d+\.\d+\.\d+", tag):
            raise ValueError("candidate ORT tag must be a stable version")
        branch = f"deps/candidate-ort-{tag}-{revision}"
    else:
        model = inputs["model"]
        if model not in {"htdemucs", "htdemucs_ft"}:
            raise ValueError("unsupported candidate model")
        branch = f"deps/candidate-{model}-{revision}"
    # Closed candidates are also deliberate decisions; do not reopen them weekly.
    existing = subprocess.run(
        ["gh", "pr", "list", "--repo", repo, "--head", branch, "--state", "all",
         "--json", "number"], check=True, capture_output=True, text=True,
    )
    if json.loads(existing.stdout):
        print(f"candidate already recorded: {branch}")
        return False
    command = ["gh", "workflow", "run", "dep-candidate.yml", "--repo", repo,
               "-f", "build_native=false"]
    for name, value in inputs.items():
        command.extend(["-f", f"{name}={value}"])
    subprocess.run(command, check=True)
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--ort-tag")
    group.add_argument("--model", choices=["htdemucs", "htdemucs_ft"])
    parser.add_argument("--ort-commit")
    parser.add_argument("--model-commit")
    parser.add_argument("--repo", default="thedavidweng/openkara-models")
    args = parser.parse_args()
    inputs = ({"ort_tag": args.ort_tag, "ort_commit": args.ort_commit or ""}
              if args.ort_tag else {"model": args.model, "model_commit": args.model_commit or ""})
    dispatch(inputs, args.repo)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Dispatch lightweight candidate preparation unless its diagnostic PR exists."""

import argparse
import json
import re
import subprocess


def dispatch(inputs: dict[str, str], repo: str) -> bool:
    if "ort_tag" in inputs:
        tag = inputs["ort_tag"]
        revision = inputs.get("ort_commit", "")
        if not re.fullmatch(r"[0-9a-f]{40}", revision):
            raise ValueError("candidate revision must be a full lowercase commit SHA")
        if not re.fullmatch(r"v\d+\.\d+\.\d+", tag):
            raise ValueError("candidate ORT tag must be a stable version")
        branch = f"deps/candidate-ort-{tag}-{revision}"
    elif "plugin" in inputs:
        plugin = inputs["plugin"]
        version = inputs.get("plugin_version", "")
        if plugin not in {"webgpu"}:
            raise ValueError("unsupported candidate plugin")
        if not re.fullmatch(r"\d+\.\d+\.\d+", version):
            raise ValueError("candidate plugin version must be X.Y.Z")
        branch = f"deps/candidate-plugin-{plugin}-v{version}"
    else:
        model = inputs["model"]
        revision = inputs.get("model_commit", "")
        if not re.fullmatch(r"[0-9a-f]{40}", revision):
            raise ValueError("candidate revision must be a full lowercase commit SHA")
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
    group.add_argument("--plugin", choices=["webgpu"])
    parser.add_argument("--ort-commit")
    parser.add_argument("--model-commit")
    parser.add_argument("--plugin-version")
    parser.add_argument("--repo", default="thedavidweng/openkara-models")
    args = parser.parse_args()
    if args.ort_tag:
        inputs = {"ort_tag": args.ort_tag, "ort_commit": args.ort_commit or ""}
    elif args.plugin:
        inputs = {"plugin": args.plugin, "plugin_version": args.plugin_version or ""}
    else:
        inputs = {"model": args.model, "model_commit": args.model_commit or ""}
    dispatch(inputs, args.repo)


if __name__ == "__main__":
    main()

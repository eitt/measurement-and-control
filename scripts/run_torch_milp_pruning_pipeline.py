"""
CLI wrapper for the experimental PyTorch + MILP pruning CMAPSS pipeline.

This mirrors the existing torch wrapper so the pruning experiment can be run
with a single command.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import List, Tuple


def parse_wrapper_args() -> Tuple[argparse.Namespace, List[str]]:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--skip-install", action="store_true")
    parser.add_argument("--pip-quiet", action="store_true")
    return parser.parse_known_args()


def ensure_requirements_installed(requirements_path: Path, pip_quiet: bool = False) -> None:
    if not requirements_path.exists():
        raise FileNotFoundError(f"Requirements file not found: {requirements_path}")
    print(f"[bootstrap] Installing dependencies from: {requirements_path}")
    cmd = [sys.executable, "-m", "pip", "install", "-r", str(requirements_path)]
    if pip_quiet:
        cmd.append("--quiet")
    subprocess.check_call(cmd)
    print("[bootstrap] Dependency preflight complete.")


def main() -> None:
    wrapper_args, pipeline_args = parse_wrapper_args()
    repo_root = Path(__file__).resolve().parents[1]
    requirements_path = repo_root / "requirements.txt"
    if not wrapper_args.skip_install:
        ensure_requirements_installed(requirements_path, pip_quiet=wrapper_args.pip_quiet)
    sys.path.insert(0, str(repo_root))
    from src.measurement_control.torch_rul_pso_milp_pruning import main as pipeline_main

    sys.argv = [sys.argv[0], *pipeline_args]
    pipeline_main()


if __name__ == "__main__":
    main()

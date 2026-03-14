"""
CLI wrapper for the organized PyTorch CMAPSS pipeline.

This wrapper guarantees dependencies are installed from requirements.txt
before importing project modules, so one command can bootstrap and run.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import List, Tuple


def parse_wrapper_args() -> Tuple[argparse.Namespace, List[str]]:
    """
    Parse wrapper-only flags and forward unknown flags to the pipeline.
    """

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--skip-install",
        action="store_true",
        help="Skip automatic 'pip install -r requirements.txt' preflight.",
    )
    parser.add_argument(
        "--pip-quiet",
        action="store_true",
        help="Install requirements with reduced pip output.",
    )
    return parser.parse_known_args()


def ensure_requirements_installed(requirements_path: Path, pip_quiet: bool = False) -> None:
    """
    Install dependencies for the current Python interpreter.
    """

    if not requirements_path.exists():
        raise FileNotFoundError(f"Requirements file not found: {requirements_path}")

    print(f"[bootstrap] Installing dependencies from: {requirements_path}")
    pip_cmd = [
        sys.executable,
        "-m",
        "pip",
        "install",
        "-r",
        str(requirements_path),
    ]
    if pip_quiet:
        pip_cmd.append("--quiet")
    subprocess.check_call(pip_cmd)
    print("[bootstrap] Dependency preflight complete.")


def main() -> None:
    """
    Install dependencies (unless disabled) and execute the PyTorch pipeline.
    """

    wrapper_args, pipeline_args = parse_wrapper_args()
    repo_root = Path(__file__).resolve().parents[1]
    requirements_path = repo_root / "requirements.txt"

    if not wrapper_args.skip_install:
        ensure_requirements_installed(requirements_path, pip_quiet=wrapper_args.pip_quiet)

    sys.path.insert(0, str(repo_root))
    from src.measurement_control.torch_rul_pso import main as pipeline_main

    # Keep pipeline CLI behavior by forwarding all non-wrapper arguments.
    sys.argv = [sys.argv[0], *pipeline_args]
    pipeline_main()


if __name__ == "__main__":
    main()

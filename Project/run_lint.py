"""Run documentation generation and code formatting in one step."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
DOCS_OUTPUT = REPO_ROOT / "Project" / "docs"


def run_step(command: list[str], label: str) -> None:
    """Execute *command* and exit if it fails."""

    print(f"\n=== {label} ===")
    result = subprocess.run(command, cwd=REPO_ROOT, check=False)
    if result.returncode != 0:
        raise SystemExit(result.returncode)


def main() -> None:
    run_step(
        [
            sys.executable,
            "-m",
            "pdoc",
            "-o",
            str(DOCS_OUTPUT.relative_to(REPO_ROOT)),
            "Project.src.main",
        ],
        "Generating API documentation",
    )
    run_step(
        [
            sys.executable,
            "-m",
            "black",
            "Project",
        ],
        "Formatting code with Black",
    )
    print("\nAll steps completed successfully.")


if __name__ == "__main__":
    main()

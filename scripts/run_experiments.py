from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODELS = ("mlp", "tabm")

SYNTHETIC_CONDITIONS = (
    "synthetic-n500-noise0",
    "synthetic-n500-noise10",
    "synthetic-n500-noise30",
    "synthetic-n2000-noise0",
    "synthetic-n2000-noise10",
    "synthetic-n2000-noise30",
    "synthetic-n8000-noise0",
    "synthetic-n8000-noise10",
    "synthetic-n8000-noise30",
)


def run(command: list[str], cwd: Path | None = None) -> None:
    print("$", " ".join(str(x) for x in command))
    subprocess.run(command, cwd=cwd, check=True)


def model_command(args: argparse.Namespace) -> list[str]:
    if args.micromamba_prefix:
        return [
            "micromamba",
            "run",
            "-p",
            args.micromamba_prefix,
            "python",
        ]

    return [args.python]


def check_tabm_repository(paper_dir: Path) -> None:
    required = [
        paper_dir / "bin" / "model.py",
        paper_dir / "exp" / "mlp" / "california" / "0-evaluation",
        paper_dir / "exp" / "tabm" / "california" / "0-evaluation",
    ]

    missing = [p for p in required if not p.exists()]

    if missing:
        raise FileNotFoundError(
            "The supplied TabM paper directory is incomplete:\n"
            + "\n".join(str(p) for p in missing)
        )


def link_dataset(source: Path, destination: Path) -> None:
    if not source.exists():
        raise FileNotFoundError(f"Prepared dataset not found: {source}")

    if destination.is_symlink():
        destination.unlink()
    elif destination.exists():
        print(f"Dataset already exists: {destination}")
        return

    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.symlink_to(source.resolve(), target_is_directory=True)

    print(f"Linked dataset: {destination} -> {source.resolve()}")


def make_config(
    paper_dir: Path,
    model: str,
    dataset_id: str,
    seed: int,
) -> Path:

    source = (
        paper_dir
        / "exp"
        / model
        / "california"
        / "0-evaluation"
        / f"{seed}.toml"
    )

    if not source.exists():
        raise FileNotFoundError(f"Source config not found: {source}")

    target = (
        paper_dir
        / "exp"
        / "comp8240-reproduction"
        / model
        / dataset_id
        / "0-evaluation"
        / f"{seed}.toml"
    )

    target.parent.mkdir(parents=True, exist_ok=True)

    text = source.read_text(encoding="utf-8")

    if dataset_id != "california":
        replacement = f'path = "data/{dataset_id}"'

        text, count = re.subn(
            r'^path\s*=\s*"data/california"\s*$',
            replacement,
            text,
            count=1,
            flags=re.MULTILINE,
        )

        if count != 1:
            raise RuntimeError(
                f"Could not replace California dataset path in {source}"
            )

    target.write_text(text, encoding="utf-8")

    return target


def run_model(
    args: argparse.Namespace,
    paper_dir: Path,
    model: str,
    dataset_id: str,
    seed: int,
) -> None:

    config = make_config(
        paper_dir=paper_dir,
        model=model,
        dataset_id=dataset_id,
        seed=seed,
    )

    relative_config = config.relative_to(paper_dir)

    command = (
        model_command(args)
        + [
            "bin/model.py",
            str(relative_config),
            "--force",
        ]
    )

    run(command, cwd=paper_dir)


def run_california(
    args: argparse.Namespace,
    paper_dir: Path,
) -> None:

    print("\n=== California Housing replication ===")

    for model in MODELS:
        for seed in range(5):
            print(f"\n{model.upper()} | seed {seed}")
            run_model(
                args,
                paper_dir,
                model,
                "california",
                seed,
            )


def prepare_tetouan(paper_dir: Path) -> None:
    print("\n=== Preparing Tetouan data ===")

    run(
        [sys.executable, str(PROJECT_ROOT / "scripts" / "prepare_tetouan.py")],
        cwd=PROJECT_ROOT,
    )

    source = (
        PROJECT_ROOT
        / "data"
        / "processed"
        / "tetouan-zone1"
    )

    destination = paper_dir / "data" / "tetouan-zone1"

    link_dataset(source, destination)


def run_tetouan(
    args: argparse.Namespace,
    paper_dir: Path,
) -> None:

    prepare_tetouan(paper_dir)

    print("\n=== Tetouan three-seed evaluation ===")

    for model in MODELS:
        for seed in range(3):
            print(f"\n{model.upper()} | seed {seed}")
            run_model(
                args,
                paper_dir,
                model,
                "tetouan-zone1",
                seed,
            )


def prepare_synthetic(paper_dir: Path) -> None:
    print("\n=== Preparing synthetic data ===")

    run(
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "generate_synthetic_regression.py"),
        ],
        cwd=PROJECT_ROOT,
    )

    source_root = (
        PROJECT_ROOT
        / "data"
        / "processed"
        / "synthetic-regression"
    )

    for condition in SYNTHETIC_CONDITIONS:
        link_dataset(
            source_root / condition,
            paper_dir / "data" / condition,
        )


def run_synthetic(
    args: argparse.Namespace,
    paper_dir: Path,
) -> None:

    prepare_synthetic(paper_dir)

    print("\n=== Synthetic three-seed evaluation ===")

    for condition in SYNTHETIC_CONDITIONS:
        print(f"\n--- {condition} ---")

        for model in MODELS:
            for seed in range(3):
                print(f"{model.upper()} | seed {seed}")
                run_model(
                    args,
                    paper_dir,
                    model,
                    condition,
                    seed,
                )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Reproduce the COMP8240 TabM experiments."
    )

    parser.add_argument(
        "experiment",
        choices=["california", "tetouan", "synthetic", "all"],
        help="Experiment to run.",
    )

    parser.add_argument(
        "--tabm-paper-dir",
        required=True,
        help="Path to the official yandex-research/tabm paper directory.",
    )

    parser.add_argument(
        "--python",
        default="python",
        help="Python executable for the TabM environment.",
    )

    parser.add_argument(
        "--micromamba-prefix",
        default=None,
        help=(
            "Optional micromamba environment prefix. "
            "For the original Colab setup use /content/tabm-env."
        ),
    )

    args = parser.parse_args()

    paper_dir = Path(args.tabm_paper_dir).resolve()

    check_tabm_repository(paper_dir)

    if args.experiment in ("california", "all"):
        run_california(args, paper_dir)

    if args.experiment in ("tetouan", "all"):
        run_tetouan(args, paper_dir)

    if args.experiment in ("synthetic", "all"):
        run_synthetic(args, paper_dir)


if __name__ == "__main__":
    main()

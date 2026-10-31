"""Paired-seed common-split training, calibration, evaluation and comparisons."""

import argparse
import json
import subprocess
import sys
from pathlib import Path
import numpy as np
from .metrics import paired_permutation


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        required=True,
        help="One frozen prepared records/tokenizer split shared by all seeds",
    )
    parser.add_argument("--config", default="config.json")
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["full", "mobivqa", "vqa_lite", "efficientad", "blip2", "anomalygpt"],
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[11, 23, 37, 41, 53])
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--foundation-path")
    parser.add_argument("--feature-checkpoint")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if len(args.seeds) != len(set(args.seeds)) or not 1 <= len(args.seeds) <= 16:
        raise ValueError(
            "Declare 1..16 distinct paired seeds for exact permutation inference"
        )
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    results = {name: [] for name in args.methods}

    def run(arguments):
        subprocess.run(
            [sys.executable, "-m", "tiny_defect_vqa.cli", *map(str, arguments)],
            check=True,
        )

    for seed in args.seeds:
        for method in args.methods:
            root = output / method / str(seed)
            common = [
                "--input",
                args.input,
                "--size",
                args.size,
                "--batch-size",
                args.batch_size,
                "--device",
                args.device,
            ]
            options = []
            if method == "blip2" and args.foundation_path:
                options += ["--foundation-path", args.foundation_path]
            if method == "efficientad" and args.feature_checkpoint:
                options += ["--feature-checkpoint", args.feature_checkpoint]
            run(
                [
                    "train",
                    *common,
                    "--config",
                    args.config,
                    "--variant",
                    method,
                    "--seed",
                    seed,
                    "--epochs",
                    args.epochs,
                    "--output",
                    root,
                    *options,
                ]
            )
            run(
                [
                    "calibrate",
                    *common,
                    "--model",
                    root / "model.pt",
                    "--output",
                    root / "calibrated.pt",
                ]
            )
            run(
                [
                    "evaluate",
                    *common,
                    "--model",
                    root / "calibrated.pt",
                    "--output",
                    root / "evaluation",
                ]
            )
            metrics = json.loads(
                (root / "evaluation/metrics.json").read_text(encoding="utf-8")
            )
            results[method].append(dict(seed=seed, **metrics))
    summary = {
        "seeds": args.seeds,
        "input": str(Path(args.input).resolve()),
        "results": results,
        "paired": {},
    }
    reference = "full" if "full" in results else args.methods[0]
    for method in args.methods:
        if method == reference:
            continue
        summary["paired"][method] = {}
        for metric in ["accuracy", "macro_f1", "ece15", "risk_at_80pct_test_ranking"]:
            left = [row[metric] for row in results[reference]]
            right = [row[metric] for row in results[method]]
            summary["paired"][method][metric] = paired_permutation(left, right)
    # Holm across the method comparisons of each metric.
    for metric in ["accuracy", "macro_f1", "ece15", "risk_at_80pct_test_ranking"]:
        ordered = sorted(
            summary["paired"],
            key=lambda name: summary["paired"][name][metric]["pvalue"],
        )
        previous = 0.0
        for rank, name in enumerate(ordered):
            row = summary["paired"][name][metric]
            previous = max(previous, min(1.0, (len(ordered) - rank) * row["pvalue"]))
            row["holm_pvalue"] = previous
    (output / "comparison.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = list(results)
    values = [[row["accuracy"] for row in results[name]] for name in names]
    fig, ax = plt.subplots(figsize=(max(5, 0.9 * len(names)), 3.5))
    ax.bar(
        names,
        [np.mean(value) for value in values],
        yerr=[np.std(value, ddof=1) if len(value) > 1 else 0 for value in values],
        color="#4b8197",
    )
    ax.set(ylabel="Typed-answer accuracy", ylim=(0, 1))
    fig.tight_layout()
    fig.savefig(output / "comparison.png", dpi=160)
    plt.close(fig)
    print(
        json.dumps(
            {
                "paired_seeds": args.seeds,
                "methods": names,
                "comparison": str(output / "comparison.json"),
            }
        )
    )


if __name__ == "__main__":
    main()

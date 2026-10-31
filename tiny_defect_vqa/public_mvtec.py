"""Bounded real MVTec/GT-mask VQA check with actual pretrained visual weights."""

import argparse
import json
import sys
import urllib.request
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from . import cli
from .method import build_model

TEACHER_REVISION = "fcab5146f84ae17597044ad5ddf1656ccf805401"
TEACHER_URL = f"https://raw.githubusercontent.com/nelson1425/EfficientAD/{TEACHER_REVISION}/models/teacher_small.pth"


def call(arguments):
    previous = sys.argv
    try:
        sys.argv = ["tiny-real", *map(str, arguments)]
        cli.main()
    finally:
        sys.argv = previous


def prepare_manifest(source, output, source_url):
    """Keep original images/masks; create a declared supervised source partition."""
    source, output = Path(source).resolve(), Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    normal = sorted((source / "bottle/train/good").glob("*.png"))[:4]
    if len(normal) != 4:
        raise ValueError("Need four original normal bottle images")
    anomalous = []
    for kind in ["broken_large", "contamination"]:
        for mask in sorted((source / "bottle/ground_truth" / kind).glob("*_mask.png"))[
            :2
        ]:
            image = (
                source
                / "bottle/test"
                / kind
                / (mask.stem.removesuffix("_mask") + ".png")
            )
            if not image.exists():
                raise FileNotFoundError(image)
            anomalous.append(
                (
                    image,
                    mask,
                    "broken part" if kind == "broken_large" else "contamination",
                )
            )
    if len(anomalous) != 4:
        raise ValueError(
            "Need two original masks for each of broken_large and contamination"
        )
    rows = []
    for i, split in enumerate(["train", "selection", "calibration", "test"]):
        rows.append(
            dict(
                dataset="MVTec AD",
                category="bottle",
                id=normal[i].relative_to(source).as_posix(),
                image=str(normal[i]),
                mask=None,
                anomalous=False,
                split=split,
                morphology="unknown morphology",
            )
        )
        image, mask, morphology = anomalous[i]
        with Image.open(mask) as target:
            if not (np.asarray(target) > 0).any():
                raise ValueError("The supplied original anomaly mask is empty")
        rows.append(
            dict(
                dataset="MVTec AD",
                category="bottle",
                id=image.relative_to(source).as_posix(),
                image=str(image),
                mask=str(mask),
                anomalous=True,
                split=split,
                morphology=morphology,
            )
        )
    manifest = output / "sources.json"
    manifest.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    report = dict(
        dataset="MVTec AD bottle",
        primary="https://www.mvtec.com/research-teaching/datasets/mvtec-ad",
        supplied_source_url=source_url,
        original_ids=[row["id"] for row in rows],
        images=8,
        source_masks="four original pixel masks, not generated polygons",
        license="CC BY-NC-SA 4.0",
        morphology_mapping={
            "broken_large": "broken part",
            "contamination": "contamination",
        },
        partition="new supervised VQA partition: one normal and one anomaly per role, each source stays in one role; not the official unsupervised MVTec split or the missing manuscript manifest",
    )
    (output / "source_provenance.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    return manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        required=True,
        help="Original unpacked MVTec layout containing bottle/",
    )
    parser.add_argument(
        "--source-url",
        default="https://www.mvtec.com/research-teaching/datasets/mvtec-ad",
    )
    parser.add_argument("--output", default="runs/real_mvtec")
    parser.add_argument(
        "--teacher", help="Optional existing community EfficientAD teacher_small.pth"
    )
    parser.add_argument("--size", type=int, default=256)
    args = parser.parse_args()
    torch.set_num_threads(1)
    output = Path(args.output)
    manifest = prepare_manifest(args.source, output, args.source_url)
    prepared = output / "qa"
    call(["prepare", "--manifest", manifest, "--output", prepared, "--seed", 11])
    records = prepared / "records.json"
    teacher = Path(args.teacher) if args.teacher else output / "teacher_small.pth"
    if not teacher.exists():
        with urllib.request.urlopen(TEACHER_URL, timeout=90) as response:
            data = response.read()
        if len(data) != 10779715:
            raise IOError("The pinned public teacher checkpoint is truncated")
        teacher.write_bytes(data)
    teacher_state = torch.load(teacher, weights_only=True, map_location="cpu")
    torch.manual_seed(11)
    config = dict(
        width=128,
        rank=128,
        vocabulary=8192,
        variant="full",
        pretrained=True,
        size=args.size,
    )
    initial = build_model(config, initialize_pretrained=True)
    official = torch.load(
        Path.home() / ".cache/torch/hub/checkpoints/mobilenet_v3_small-047dcff4.pth",
        weights_only=True,
    )
    torch.testing.assert_close(
        initial.encoder.features[0][0].weight,
        official["features.0.0.weight"],
        atol=0,
        rtol=0,
    )
    initial_conv = initial.encoder.features[0][0].weight.detach().clone()
    del initial
    report = {
        "source": json.loads((output / "source_provenance.json").read_text()),
        "image_size": args.size,
        "epochs": 1,
        "teacher": (
            {
                "path": str(teacher),
                "identity": "locally supplied PDN teacher; source attribution is caller provided",
            }
            if args.teacher
            else {
                "url": TEACHER_URL,
                "revision": TEACHER_REVISION,
                "identity": "community EfficientAD reproduction checkpoint, not original author release",
            }
        ),
        "methods": {},
    }
    for method in ["full", "efficientad"]:
        path = output / method
        options = (
            ["--pretrained"] if method == "full" else ["--feature-checkpoint", teacher]
        )
        common = ["--input", records, "--size", args.size, "--batch-size", 1]
        call(
            [
                "train",
                *common,
                "--variant",
                method,
                "--epochs",
                1,
                "--seed",
                11,
                "--output",
                path,
                *options,
            ]
        )
        call(
            [
                "calibrate",
                *common,
                "--model",
                path / "model.pt",
                "--output",
                path / "calibrated.pt",
            ]
        )
        call(
            [
                "evaluate",
                *common,
                "--model",
                path / "calibrated.pt",
                "--output",
                path / "evaluation",
            ]
        )
        model, saved = cli.load(path / "calibrated.pt", "cpu")
        if method == "full":
            assert not torch.equal(model.encoder.features[0][0].weight, initial_conv)
        else:
            for key, value in teacher_state.items():
                assert torch.equal(value, model.teacher.state_dict()[key])
            assert all(not value.requires_grad for value in model.teacher.parameters())
        metrics = json.loads((path / "evaluation/metrics.json").read_text())
        assert np.isfinite(metrics["accuracy"])
        report["methods"][method] = dict(
            metrics=metrics,
            initialized_from_real_pretraining=True,
            frozen_teacher_unchanged=method == "efficientad",
            calibrated_checkpoint_loaded=True,
        )
        del model, saved
        print(
            f"{method}: real images/masks and pretrained weights train/calibrate/evaluate passed",
            flush=True,
        )
    report["scope"] = (
        "two source images per partition; bounded real-data execution, no paper performance inference"
    )
    (output / "validation.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()

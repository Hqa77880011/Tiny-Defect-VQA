"""Functional mechanism checks for every contextual baseline and shared CLI."""

import gc
import json
from pathlib import Path
import torch
from .smoke import main as core_smoke, command
from .dataset import Questions
from .method import build_model
from .baselines import typed
from . import cli


def main():
    torch.set_num_threads(1)
    torch.manual_seed(11)
    core_smoke()
    root = Path(__file__).resolve().parents[1] / "runs" / "smoke"
    records = root / "qa/records.json"
    dataset = Questions(records, "train", size=64)
    batch = next(iter(torch.utils.data.DataLoader(dataset, batch_size=10)))
    base = {
        "width": 32,
        "rank": 16,
        "vocabulary": 8192,
        "size": 64,
        "pretrained": False,
        "pdn_width": 8,
        "feature_channels": 16,
        "foundation_width": 32,
        "vision_layers": 1,
        "language_layers": 1,
        "memory": 64,
    }
    summaries = {}
    for name in ["mobivqa", "vqa_lite", "efficientad", "blip2", "anomalygpt"]:
        config = dict(base, variant=name)
        model = build_model(config)
        if hasattr(model, "fit_reference"):
            model.fit_reference(dataset, 5, "cpu")
        model.train()
        initial = {
            key: value.detach().clone()
            for key, value in model.named_parameters()
            if value.requires_grad
        }
        optimizer = torch.optim.AdamW(
            [value for value in model.parameters() if value.requires_grad], lr=0.001
        )
        # Two steps ensure both zero-initialized LoRA factors are exercised.
        for _ in range(2):
            optimizer.zero_grad()
            prediction = model(batch)
            if name == "vqa_lite":
                batch["teacher_logits"] = typed(torch.zeros(10, 32), batch["intent"])
            loss = model.objective(batch, prediction)
            assert torch.isfinite(loss)
            loss.backward()
            optimizer.step()
        required = {
            "mobivqa": [
                "cross.0.attention.in_proj_weight",
                "cross.1.attention.in_proj_weight",
                "answerability.weight",
                "exit.weight",
            ],
            "vqa_lite": [
                "moe.routing",
                "moe.attentions.0.in_proj_weight",
                "answer.weight",
            ],
            "efficientad": [
                "student.0.weight",
                "autoencoder.0.weight",
                "decoder.0.weight",
            ],
            "blip2": [
                "queries",
                "query_self.in_proj_weight",
                "query_cross.in_proj_weight",
                "projection.weight",
                "answer.weight",
            ],
            "anomalygpt": [
                "prompts",
                "prompt_learner.0.weight",
                "pixel.0.weight",
                "language.0.q.left",
                "language.0.q.right",
                "answer.weight",
            ],
        }[name]
        for key in required:
            value = dict(model.named_parameters())[key]
            assert (
                value.grad is not None
                and torch.isfinite(value.grad).all()
                and value.grad.abs().sum() > 0
            ), (name, key)
            assert not torch.equal(initial[key], value), (name, key)
        if name in {"blip2", "anomalygpt"}:
            assert all(value.grad is None for value in model.vision.parameters())
            assert all(value.grad is None for value in model.words.parameters())
        if name == "mobivqa":
            model.eval()
            model.first_threshold = 0.0
            model.exit_threshold = 0.0
            with torch.no_grad():
                model.exit.weight.zero_()
                model.exit.bias[:] = torch.tensor([20.0, -20.0])
                early = model(batch)
                model.exit_threshold = 2.0
                full = model(batch)
            assert (early["exit_layer"] == 1).all() and (full["exit_layer"] == 2).all()
            assert (
                early["processed_visual_tokens"] < full["processed_visual_tokens"]
            ).all()
        model.eval()
        checkpoint = root / (name + "_reload.pt")
        torch.save(model.state_dict(), checkpoint)
        restored = build_model(config)
        restored.load_state_dict(torch.load(checkpoint, weights_only=True))
        restored.eval()
        if name == "mobivqa":
            model.first_threshold, model.exit_threshold = (
                restored.first_threshold,
                restored.exit_threshold,
            )
        with torch.no_grad():
            torch.testing.assert_close(
                model(batch)["probability"],
                restored(batch)["probability"],
                atol=1e-7,
                rtol=1e-6,
            )
        summary = {
            "loss": float(loss.detach()),
            "parameters": sum(value.numel() for value in model.parameters()),
            "trainable": sum(
                value.numel() for value in model.parameters() if value.requires_grad
            ),
        }
        summaries[name] = summary
        del model, restored, optimizer, prediction, loss, initial
        gc.collect()
        cfg = root / (name + "_config.json")
        cfg.write_text(json.dumps(config), encoding="utf-8")
        output = root / "baselines" / name
        arguments = [
            "train",
            "--input",
            str(records),
            "--config",
            str(cfg),
            "--variant",
            name,
            "--output",
            str(output),
            "--epochs",
            "1",
            "--batch-size",
            "5",
            "--size",
            "64",
            "--seed",
            "11",
        ]
        if name == "vqa_lite":
            arguments += ["--teacher", str(root / "training/model.pt")]
        if name == "efficientad":
            arguments += ["--penalty-images", str(root)]
        command(cli.main, arguments)
        command(
            cli.main,
            [
                "calibrate",
                "--input",
                str(records),
                "--model",
                str(output / "model.pt"),
                "--output",
                str(output / "calibrated.pt"),
                "--size",
                "64",
                "--batch-size",
                "5",
            ],
        )
        command(
            cli.main,
            [
                "evaluate",
                "--input",
                str(records),
                "--model",
                str(output / "calibrated.pt"),
                "--output",
                str(output / "evaluation"),
                "--size",
                "64",
                "--batch-size",
                "5",
            ],
        )
        command(
            cli.main,
            [
                "plot",
                "--input",
                str(output / "evaluation/metrics.json"),
                "--output",
                str(output / "metrics.png"),
            ],
        )
    print(
        json.dumps(
            {
                "baseline_mechanisms_passed": summaries,
                "shared_training_calibration_evaluation": 5,
            }
        )
    )


if __name__ == "__main__":
    main()

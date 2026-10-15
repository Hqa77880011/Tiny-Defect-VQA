import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from .metrics import measure


def read(path):
    with np.load(path, allow_pickle=False) as data:
        return {k: data[k] for k in data.files}


def dump(path, data):
    Path(path).write_text(json.dumps(data, indent=2, allow_nan=False), encoding="utf-8")


def tensor_batch(data, indices, device):
    return {k: torch.as_tensor(v[indices], device=device) for k, v in data.items()
            if v.ndim and len(v) == len(data["x"]) and v.dtype.kind in "biuf"}


def main():
    from . import method
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ["demo", "prepare", "train", "evaluate", "plot", "pack", "calibrate"] + (["execute"] if hasattr(method, "execute") else []):
        p = sub.add_parser(name)
        p.add_argument("--output", required=True)
        p.add_argument("--input")
        p.add_argument("--config")
        p.add_argument("--set", action="append", default=[])
        p.add_argument("--variant", default="full")
        p.add_argument("--seed", type=int, default=13)
        p.add_argument("--epochs", type=int, default=100)
        p.add_argument("--batch-size", type=int, default=16)
        p.add_argument("--lr", type=float, default=0.001)
        p.add_argument("--device", default="cpu")
        p.add_argument("--model")
        p.add_argument("--init")
        p.add_argument("--images")
        p.add_argument("--targets")
        p.add_argument("--size", type=int, default=64)
        p.add_argument("--mask", action="store_true")
        p.add_argument("--count", type=int, default=96)
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text()) if args.config else {}
    for setting in args.set:
        key, value = setting.split("=", 1)
        config[key] = json.loads(value)
    config["variant"] = args.variant
    out = Path(args.output)
    if args.command in {"train", "evaluate", "plot", "prepare"}:
        out.mkdir(parents=True, exist_ok=True)
    else:
        out.parent.mkdir(parents=True, exist_ok=True)
    if args.command == "execute":
        method.execute(args)
    elif args.command == "demo":
        np.savez_compressed(out, **method.demo(args.seed, args.count, config))
    elif args.command == "pack":
        from PIL import Image
        paths = sorted(p for p in Path(args.images).iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".tif", ".bmp"})
        inputs, targets = [], []
        for path in paths:
            with Image.open(path) as image:
                inputs.append(np.asarray(image.convert("RGB").resize((args.size, args.size)), dtype=np.float32).transpose(2, 0, 1) / 255)
            target = Path(args.targets) / path.name
            with Image.open(target) as image:
                if args.mask:
                    targets.append((np.asarray(image.convert("L").resize((args.size, args.size), Image.Resampling.NEAREST)) > 127).astype(np.float32)[None])
                else:
                    targets.append(np.asarray(image.convert("RGB").resize((args.size, args.size)), dtype=np.float32).transpose(2, 0, 1) / 255)
        if not paths:
            raise ValueError("No image pairs found")
        np.savez_compressed(out, x=np.stack(inputs), y=np.stack(targets), group=np.arange(len(paths)))
    elif args.command == "prepare":
        data = read(args.input)
        n = len(data["x"])
        if "y" not in data or len(data["y"]) != n:
            raise ValueError("x and y must have matching sample counts")
        groups = data.get("group", np.arange(n))
        if "split" in data:
            assignment = data["split"]
            codes = [0, 1, 2, 3] if np.max(assignment) == 3 else [0, 1, 2]
            masks = [assignment == code for code in codes]
        else:
            unique = np.unique(groups)
            first, second, third = [int(f * len(unique)) for f in [.6, .75, .85]]
            masks = [np.isin(groups, selected) for selected in [unique[:first], unique[first:second], unique[second:third], unique[third:]]]
        for i, mask in enumerate(masks):
            if not mask.any():
                raise ValueError("Each train/validation/test partition must be nonempty")
            for other in masks[:i]:
                if np.intersect1d(groups[mask], groups[other]).size:
                    raise ValueError("A group crosses split boundaries")
            values = {k: v[mask] if v.ndim and len(v) == n else v for k, v in data.items() if k != "split"}
            names = ["train.npz", "validation.npz", "calibration.npz", "test.npz"] if len(masks) == 4 else ["train.npz", "validation.npz", "test.npz"]
            np.savez_compressed(out / names[i], **values)
    elif args.command == "train":
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        data = read(Path(args.input) / "train.npz")
        validation = read(Path(args.input) / "validation.npz")
        config.setdefault("input_shape", list(data["x"].shape[1:]))
        config.setdefault("classes", int(np.max(data.get("state", data["y"]))) + 1 if data["y"].ndim == 1 and data["y"].dtype.kind in "iu" else 2)
        network = method.Model(config).to(args.device)
        if args.init:
            initial = torch.load(args.init, map_location=args.device, weights_only=True)
            network.load_state_dict(initial["state"])
        elif hasattr(network, "prepare"):
            network.prepare(data)
        optimizer = torch.optim.AdamW(network.parameters(), lr=args.lr)
        history, best = [], float("inf")
        start = time.perf_counter()
        for epoch in range(args.epochs):
            network.train()
            train_loss = 0.
            for indices in np.array_split(np.random.permutation(len(data["x"])), max(1, int(np.ceil(len(data["x"]) / args.batch_size)))):
                batch = tensor_batch(data, indices, args.device)
                prediction = network(batch)
                loss = network.objective(batch, prediction)
                if not torch.isfinite(loss):
                    raise FloatingPointError("Nonfinite training objective")
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_loss += float(loss.detach()) * len(indices)
            network.eval()
            total = 0.
            for begin in range(0, len(validation["x"]), args.batch_size):
                indices = np.arange(begin, min(begin + args.batch_size, len(validation["x"])))
                batch = tensor_batch(validation, indices, args.device)
                prediction = network(batch)
                total += float(network.objective(batch, prediction).detach()) * len(indices)
            score = total / len(validation["x"])
            history.append({"epoch": epoch + 1, "training_loss": train_loss / len(data["x"]), "validation_loss": score})
            if score < best:
                best = score
                torch.save({"state": network.state_dict(), "config": config, "training_groups": np.unique(data.get("group", np.arange(len(data["x"])))).tolist(), "validation_groups": np.unique(validation.get("group", np.arange(len(validation["x"])))).tolist()}, out / "model.pt")
        dump(out / "history.json", history)
        dump(out / "training.json", {"seconds": time.perf_counter() - start, "best_validation_loss": best})
    elif args.command == "calibrate":
        from scipy.optimize import minimize_scalar
        from scipy.special import softmax
        from sklearn.metrics import log_loss
        state = torch.load(args.model, map_location=args.device, weights_only=True)
        network = method.Model(state["config"]).to(args.device)
        network.load_state_dict(state["state"])
        network.eval()
        data = read(args.input)
        if "group" not in data:
            raise ValueError("Calibration requires globally unique independent group IDs")
        excluded = state.get("training_groups", []) + state.get("validation_groups", [])
        if np.intersect1d(data["group"], excluded).size:
            raise ValueError("Calibration groups overlap fitting or checkpoint-selection groups")
        prediction = network(tensor_batch(data, np.arange(len(data["x"])), args.device))
        if "probability" in prediction and data["y"].ndim == 1:
            logits = np.log(np.clip(prediction["probability"].detach().cpu().numpy(), 1e-10, 1))
            result = minimize_scalar(lambda value: log_loss(data["y"], softmax(logits / np.exp(value), axis=1), labels=np.arange(logits.shape[1])), bounds=(-3, 3), method="bounded")
            state["classification_temperature"] = float(np.exp(result.x))
        else:
            if "lower" in prediction:
                lower, upper = prediction["lower"], prediction["upper"]
            elif "prediction" in prediction:
                lower = upper = prediction["prediction"]
            else:
                raise ValueError("Calibration needs class probabilities or numeric prediction bounds")
            truth = torch.as_tensor(data["y"], device=lower.device)
            score = torch.maximum(torch.maximum(lower-truth, truth-upper), torch.zeros_like(lower)).reshape(len(truth), -1).max(-1).values.detach().cpu().numpy()
            unit = [score[data["group"] == group].max() for group in np.unique(data["group"])]
            rank = int(np.ceil((len(unit)+1)*.9))
            state["interval_expansion"] = float(np.sort(unit)[rank-1]) if rank <= len(unit) else float("inf")
        torch.save(state, out)
    elif args.command == "evaluate":
        state = torch.load(args.model, map_location=args.device, weights_only=True)
        network = method.Model(state["config"]).to(args.device)
        network.load_state_dict(state["state"])
        network.eval()
        data = read(args.input)
        stored, elapsed = {}, []
        for begin in range(0, len(data["x"]), args.batch_size):
            indices = np.arange(begin, min(begin + args.batch_size, len(data["x"])))
            batch = tensor_batch(data, indices, args.device)
            if args.device.startswith("cuda"):
                torch.cuda.synchronize()
            start = time.perf_counter()
            prediction = network(batch)
            if "classification_temperature" in state and "probability" in prediction:
                prediction["probability"] = (prediction["probability"].clamp_min(1e-10).log() / state["classification_temperature"]).softmax(-1)
            if "interval_expansion" in state:
                lower = prediction.get("lower", prediction.get("prediction"))
                upper = prediction.get("upper", prediction.get("prediction"))
                prediction["lower"] = lower-state["interval_expansion"]
                prediction["upper"] = upper+state["interval_expansion"]
                if "bound_lower" in prediction:
                    prediction["lower"] = torch.maximum(prediction["lower"], prediction["bound_lower"])
                    prediction["upper"] = torch.minimum(prediction["upper"], prediction["bound_upper"])
            if args.device.startswith("cuda"):
                torch.cuda.synchronize()
            elapsed.append((time.perf_counter() - start) / len(indices))
            for key, value in prediction.items():
                if isinstance(value, torch.Tensor) and value.ndim and len(value) == len(indices):
                    stored.setdefault(key, []).append(value.detach().cpu().numpy())
        result = {key: np.concatenate(values) for key, values in stored.items()}
        values = measure(result, data)
        if hasattr(method, "measure"):
            values.update(method.measure(result, data, state["config"]))
        values.update(parameters=sum(p.numel() for p in network.parameters()),
                      mean_latency_ms=1000 * float(np.mean(elapsed)),
                      p95_latency_ms=1000 * float(np.quantile(elapsed, .95)))
        np.savez_compressed(out / "predictions.npz", **result)
        dump(out / "metrics.json", values)
    elif args.command == "plot":
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        paths = list(Path(args.input).rglob("metrics.json"))
        if not paths:
            raise ValueError("No evaluated metrics")
        records = [json.loads(p.read_text()) for p in paths]
        for key in sorted(set().union(*(set(r) for r in records))):
            if not all(isinstance(r.get(key), (int, float)) for r in records):
                continue
            fig, ax = plt.subplots(figsize=(9, 4))
            ax.bar([str(p.parent.relative_to(args.input)) for p in paths], [r[key] for r in records], color="#4d8495")
            ax.set_ylabel(key)
            ax.tick_params(axis="x", labelrotation=30)
            fig.tight_layout()
            fig.savefig(out / (key + ".png"), dpi=180)
            plt.close(fig)


if __name__ == "__main__":
    main()


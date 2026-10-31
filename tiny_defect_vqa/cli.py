"""Train, calibrate and evaluate bounded mask-grounded VQA."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from scipy.optimize import minimize_scalar
from scipy.special import logsumexp, softmax
from sklearn.metrics import f1_score
from torch.nn import functional as F
from .dataset import Questions, prepare
from .method import Model
from .metrics import measure


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


def batch_device(batch, device):
    return {key: value.to(device) for key, value in batch.items()}


@torch.no_grad()
def collect(model, dataset, batch_size, device, grounding=False):
    model.eval()
    stored = {key: [] for key in ["logits", "y", "intent", "index"]}
    ious, seen = [], set()
    for batch in torch.utils.data.DataLoader(dataset, batch_size=batch_size):
        prediction = model(batch_device(batch, device))
        for key in stored:
            value = prediction[key] if key == "logits" else batch[key]
            stored[key].append(value.detach().cpu().numpy())
        if grounding:
            mask = F.interpolate(prediction["mask"], size=batch["mask"].shape[-2:], mode="bilinear", align_corners=False).cpu().numpy() >= .5
            target = batch["mask"].numpy() > 0
            for j, index in enumerate(batch["index"]):
                row = dataset.records[int(index)]
                if row["source_id"] not in seen and row["anomalous"]:
                    union = np.sum(mask[j] | target[j])
                    ious.append(float(np.sum(mask[j] & target[j])/max(union, 1)))
                    seen.add(row["source_id"])
    return {key: np.concatenate(values) for key, values in stored.items()}, ious


def load(path, device):
    state = torch.load(path, map_location=device, weights_only=True)
    model = Model(state["config"]).to(device)
    model.load_state_dict(state["state"])
    model.eval()
    return model, state


def scores(probability, gate):
    return np.sum(probability*np.log(np.maximum(probability, 1e-12)), axis=1) if gate == "entropy" else probability.max(1)


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare")
    prep.add_argument("--manifest", required=True)
    prep.add_argument("--seed", type=int, default=11)
    prep.add_argument("--output", required=True)
    training = sub.add_parser("train")
    training.add_argument("--config", default="config.json")
    training.add_argument("--variant", choices=["full", "no_mask", "no_gate", "no_localization", "mobilenet_gru"], default="full")
    training.add_argument("--epochs", type=int, default=60)
    training.add_argument("--patience", type=int, default=8)
    training.add_argument("--seed", type=int, default=11)
    training.add_argument("--lr", type=float, default=3e-4)
    training.add_argument("--pretrained", action="store_true")
    for command in [training, sub.add_parser("calibrate"), sub.add_parser("evaluate")]:
        command.add_argument("--input", required=True, help="Prepared records.json")
        command.add_argument("--output", required=True)
        command.add_argument("--batch-size", type=int, default=64)
        command.add_argument("--device", default="cpu")
        command.add_argument("--size", type=int, default=256)
        if command is not training:
            command.add_argument("--model", required=True)
        if command.prog.endswith("calibrate"):
            command.add_argument("--gate", choices=["intent", "global", "entropy"], default="intent")
    plotting = sub.add_parser("plot")
    plotting.add_argument("--input", required=True)
    plotting.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args.manifest, args.output, args.seed)
        return
    if args.command == "plot":
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        report = json.loads(Path(args.input).read_text(encoding="utf-8"))
        fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))
        values = report["per_intent_accuracy"]
        axes[0].bar(list(values), list(values.values()), color="#497d9c")
        axes[0].set(xlabel="Registered intent", ylabel="Answer accuracy", ylim=(0, 1))
        axes[1].bar(["coverage", "risk", "error capture"], [report["coverage"], report["selective_risk"] or 0, report["error_capture_rate"]], color="#53988e")
        axes[1].set(ylim=(0, 1))
        fig.tight_layout()
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.output, dpi=160)
        plt.close(fig)
        return
    if args.command == "train":
        torch.manual_seed(args.seed)
        config = json.loads(Path(args.config).read_text(encoding="utf-8"))
        config.update(variant=args.variant, pretrained=args.pretrained, size=args.size)
        dataset = Questions(args.input, "train", args.size, args.pretrained, augment=True)
        validation = Questions(args.input, "selection", args.size, args.pretrained)
        model = Model(config, initialize_pretrained=True).to(args.device)
        counts = np.bincount([row["y"] for row in dataset.records], minlength=32)
        weight = 1/np.sqrt(counts+1)
        model.answer_weight.copy_(torch.tensor(weight/weight.mean(), dtype=torch.float32, device=args.device))
        strata = [(row["dataset"], row["intent"]) for row in dataset.records]
        sizes = {key: strata.count(key) for key in set(strata)}
        sampler = torch.utils.data.WeightedRandomSampler([1/sizes[key] for key in strata], len(strata), replacement=True)
        loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, sampler=sampler)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
        best, waiting, history = -1., 0, []
        output = Path(args.output)
        output.mkdir(parents=True, exist_ok=True)
        tokenizer = (Path(args.input).parent/"tokenizer.json").read_text(encoding="utf-8")
        for epoch in range(args.epochs):
            model.train()
            losses = []
            for batch in loader:
                batch = batch_device(batch, args.device)
                optimizer.zero_grad()
                prediction = model(batch)
                loss = model.objective(batch, prediction)
                if not torch.isfinite(loss):
                    raise FloatingPointError("Nonfinite VQA objective")
                loss.backward()
                optimizer.step()
                losses.append(float(loss.detach()))
            scheduler.step()
            values, _ = collect(model, validation, args.batch_size, args.device)
            score = f1_score(values["y"], values["logits"].argmax(1), labels=np.arange(32), average="macro", zero_division=0)
            history.append({"epoch": epoch+1, "loss": float(np.mean(losses)), "selection_macro_f1": float(score)})
            if score > best:
                best, waiting = score, 0
                torch.save({"config": config, "state": model.state_dict(), "seed": args.seed, "tokenizer_json": tokenizer,
                            "training_sources": sorted({row["source_id"] for row in dataset.records}),
                            "selection_sources": sorted({row["source_id"] for row in validation.records})}, output/"model.pt")
            else:
                waiting += 1
                if waiting >= args.patience:
                    break
        dump(output/"history.json", history)
    else:
        model, state = load(args.model, args.device)
        split = "calibration" if args.command == "calibrate" else "test"
        if args.size != state["config"]["size"]:
            raise ValueError("Evaluation image size must equal the trained preprocessing size")
        dataset = Questions(args.input, split, args.size, state["config"]["pretrained"])
        excluded = set(state["training_sources"]+state["selection_sources"])
        if split == "test":
            excluded.update(state.get("calibration_sources", []))
        if excluded & {row["source_id"] for row in dataset.records}:
            raise ValueError("Calibration/test source IDs overlap fitting or checkpoint selection")
        values, ious = collect(model, dataset, args.batch_size, args.device, grounding=args.command == "evaluate")
        if args.command == "calibrate":
            result = minimize_scalar(lambda log_t: float(np.mean(logsumexp(values["logits"]/np.exp(log_t), axis=1)-values["logits"][np.arange(len(values["y"])), values["y"]]/np.exp(log_t))), bounds=(-3, 3), method="bounded")
            temperature = float(np.exp(result.x))
            probability = softmax(values["logits"]/temperature, axis=1)
            score = scores(probability, args.gate)
            if args.gate == "intent":
                thresholds = {str(i): float(np.quantile(score[values["intent"] == i], .2)) for i in range(5)}
            else:
                thresholds = {str(i): float(np.quantile(score, .2)) for i in range(5)}
            state.update(temperature=temperature, thresholds=thresholds, gate=args.gate,
                         calibration_sources=sorted({row["source_id"] for row in dataset.records}))
            Path(args.output).parent.mkdir(parents=True, exist_ok=True)
            torch.save(state, args.output)
        else:
            if "temperature" not in state:
                raise ValueError("Fit temperature and gate on the independent calibration split first")
            probability = softmax(values["logits"]/state["temperature"], axis=1)
            score = scores(probability, state["gate"])
            accepted = score >= np.array([state["thresholds"][str(i)] for i in values["intent"]])
            report = measure(probability, values["y"], values["intent"], accepted)
            report.update(mask_iou_anomalous_images=float(np.mean(ious)) if ious else None,
                          anomalous_images=len(ious), trainable_parameters=sum(value.numel() for value in model.parameters()),
                          serialized_model_bytes=Path(args.model).stat().st_size)
            out = Path(args.output)
            out.mkdir(parents=True, exist_ok=True)
            dump(out/"metrics.json", report)
            np.savez_compressed(out/"predictions.npz", probability=probability, truth=values["y"], intent=values["intent"], accepted=accepted)


if __name__ == "__main__":
    main()

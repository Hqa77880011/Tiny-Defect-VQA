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
from .method import build_model
from .metrics import measure


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


def batch_device(batch, device):
    return {key: value.to(device) if isinstance(value, torch.Tensor) else value for key, value in batch.items()}


@torch.no_grad()
def collect(model, dataset, batch_size, device, grounding=False):
    model.eval()
    stored = {key: [] for key in ["logits", "y", "intent", "index", "intrinsic_review"]}
    ious, seen = [], set()
    for batch in torch.utils.data.DataLoader(dataset, batch_size=batch_size):
        prediction = model(batch_device(batch, device))
        for key in stored:
            value = prediction[key] if key == "logits" else prediction.get(key, torch.zeros_like(batch["intent"], dtype=torch.bool)) if key == "intrinsic_review" else batch[key]
            stored[key].append(value.detach().cpu().numpy())
        if grounding and model.config.get("grounding_supported", True):
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
    model = build_model(state["config"]).to(device)
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
    training.add_argument("--variant", choices=["full", "no_mask", "no_gate", "no_localization", "mobilenet_gru", "mobivqa", "vqa_lite", "efficientad", "blip2", "anomalygpt"], default="full")
    training.add_argument("--epochs", type=int, default=60)
    training.add_argument("--patience", type=int, default=8)
    training.add_argument("--seed", type=int, default=11)
    training.add_argument("--lr", type=float, default=3e-4)
    training.add_argument("--pretrained", action="store_true")
    training.add_argument("--foundation-path", help="Local Hugging Face BLIP-2 model/tokenizer directory")
    training.add_argument("--init", help="Locally supplied compatible baseline checkpoint")
    training.add_argument("--feature-checkpoint", help="EfficientAD PDN teacher state dictionary")
    training.add_argument("--penalty-images", help="Independent general-image folder for EfficientAD penalty")
    training.add_argument("--teacher", help="Independent frozen typed-answer teacher for VQA-lite distillation")
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
        if args.variant == "efficientad" and args.feature_checkpoint:
            args.pretrained = True
        if args.foundation_path:
            if args.variant != "blip2":
                raise ValueError("This pretrained foundation backend is BLIP-2")
            config.update(foundation_backend="huggingface", foundation_path=args.foundation_path, normalization="clip")
        config.update(variant=args.variant, pretrained=args.pretrained, size=args.size)
        dataset = Questions(args.input, "train", args.size, args.pretrained, augment=True, normalization=config.get("normalization"))
        validation = Questions(args.input, "selection", args.size, args.pretrained, normalization=config.get("normalization"))
        model = build_model(config, initialize_pretrained=True).to(args.device)
        config = model.config
        if args.variant == "efficientad":
            dataset.records = [row for row in dataset.records if not row["anomalous"]]
            if not dataset.records:
                raise ValueError("EfficientAD trains only on normal images")
        if args.init:
            initial = torch.load(args.init, map_location=args.device, weights_only=True)
            if "config" in initial:
                for key in ["variant", "size", "pretrained"]:
                    if initial["config"].get(key) != config.get(key):
                        raise ValueError("Initialization checkpoint preprocessing/architecture mismatch: "+key)
            model.load_state_dict(initial.get("state", initial))
        if args.feature_checkpoint:
            if args.variant != "efficientad":
                raise ValueError("PDN feature checkpoints apply to EfficientAD")
            teacher_state = torch.load(args.feature_checkpoint, map_location=args.device, weights_only=True)
            model.teacher.load_state_dict(teacher_state.get("teacher", teacher_state.get("state_dict", teacher_state)))
        if hasattr(model, "fit_reference"):
            reference = Questions(args.input, "train", args.size, args.pretrained, augment=False, normalization=config.get("normalization"))
            model.fit_reference(reference, args.batch_size, args.device)
        teacher_model = None
        if args.teacher:
            teacher_model, teacher_state = load(args.teacher, args.device)
            if teacher_state["config"]["size"] != args.size or teacher_state["tokenizer_json"] != (Path(args.input).parent/"tokenizer.json").read_text(encoding="utf-8"):
                raise ValueError("Distillation teacher must share the frozen tokenizer and image size")
        penalty_paths = []
        if args.penalty_images:
            penalty_paths = sorted(path for path in Path(args.penalty_images).rglob("*") if path.suffix.lower() in {".jpg", ".jpeg", ".png"})
            if not penalty_paths:
                raise ValueError("No general penalty images found")
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
                if penalty_paths:
                    from PIL import Image
                    penalty = []
                    for choice in torch.randint(len(penalty_paths), (len(batch["x"]),)):
                        with Image.open(penalty_paths[int(choice)]) as image:
                            value = np.asarray(image.convert("RGB").resize((args.size, args.size)), dtype=np.float32).copy().transpose(2, 0, 1)/255
                        tensor = torch.tensor(value, device=args.device)
                        if args.pretrained:
                            tensor = (tensor-torch.tensor([.485, .456, .406], device=args.device)[:, None, None])/torch.tensor([.229, .224, .225], device=args.device)[:, None, None]
                        penalty.append(tensor)
                    batch["penalty_x"] = torch.stack(penalty)
                if teacher_model is not None:
                    with torch.no_grad():
                        batch["teacher_logits"] = teacher_model(batch)["logits"]
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
        dataset = Questions(args.input, split, args.size, state["config"]["pretrained"], normalization=state["config"].get("normalization"))
        excluded = set(state["training_sources"]+state["selection_sources"])
        if split == "test":
            excluded.update(state.get("calibration_sources", []))
        if excluded & {row["source_id"] for row in dataset.records}:
            raise ValueError("Calibration/test source IDs overlap fitting or checkpoint selection")
        if args.command == "calibrate" and hasattr(model, "calibrate_maps"):
            model.calibrate_maps(dataset, args.batch_size, args.device)
            state["state"] = model.state_dict()
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
            accepted = (score >= np.array([state["thresholds"][str(i)] for i in values["intent"]])) & ~values["intrinsic_review"]
            report = measure(probability, values["y"], values["intent"], accepted)
            report.update(intrinsic_review_rate=float(values["intrinsic_review"].mean()),
                          grounding_evaluated=model.config.get("grounding_supported", True),
                          mask_iou_anomalous_images=float(np.mean(ious)) if ious else None,
                          anomalous_images=len(ious), trainable_parameters=sum(value.numel() for value in model.parameters() if value.requires_grad),
                          total_parameters=sum(value.numel() for value in model.parameters()),
                          serialized_model_bytes=Path(args.model).stat().st_size)
            out = Path(args.output)
            out.mkdir(parents=True, exist_ok=True)
            dump(out/"metrics.json", report)
            np.savez_compressed(out/"predictions.npz", probability=probability, truth=values["y"], intent=values["intent"], accepted=accepted)


if __name__ == "__main__":
    main()

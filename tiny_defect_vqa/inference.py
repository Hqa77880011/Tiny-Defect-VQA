"""Single-image inference and uncached end-to-end profiling."""
import argparse
import time
from pathlib import Path
import numpy as np
import psutil
import torch
from PIL import Image
from tokenizers import Tokenizer
from .cli import load, scores, dump
from .dataset import INTENTS, MORPHOLOGIES

ANSWERS = (["presence:no", "presence:yes"] + ["morphology:no defect"]+["morphology:"+name for name in MORPHOLOGIES] +
           ["location:no defect"]+[f"location:r{row}c{col}" for row in range(3) for col in range(3)] +
           ["extent:"+name for name in ["none", "minor", "moderate", "major"]] +
           ["disposition:"+name for name in ["accept", "rework", "manual inspection"]])


@torch.no_grad()
def predict(model, state, image, question, intent, device, tokenizer=None):
    """Timeable RGB→resize→tokenize→model→mask resize→typed answer/gate pipeline."""
    if intent not in INTENTS:
        raise ValueError("Unregistered intent; manual review required")
    image = image.convert("RGB")
    original_size = image.size
    size = state["config"]["size"]
    array = np.asarray(image.resize((size, size), Image.Resampling.BILINEAR), dtype=np.float32).copy().transpose(2, 0, 1)/255
    x = torch.tensor(array, device=device)
    if state["config"].get("normalization") == "clip":
        x = (x-torch.tensor([.48145466, .4578275, .40821073], device=device)[:, None, None])/torch.tensor([.26862954, .26130258, .27577711], device=device)[:, None, None]
    elif state["config"]["pretrained"]:
        x = (x-torch.tensor([.485, .456, .406], device=device)[:, None, None])/torch.tensor([.229, .224, .225], device=device)[:, None, None]
    tokenizer = tokenizer if tokenizer is not None else Tokenizer.from_str(state["tokenizer_json"])
    tokens = tokenizer.encode(question).ids[:24]
    if not tokens:
        raise ValueError("Question is empty after tokenization")
    ids = torch.zeros((1, 24), dtype=torch.long, device=device)
    ids[0, :len(tokens)] = torch.tensor(tokens, device=device)
    index = INTENTS.index(intent)
    output = model({"x": x[None], "question": ids, "intent": torch.tensor([index], device=device), "raw_question": [question]})
    probability = (output["logits"]/state["temperature"]).softmax(-1).cpu().numpy()
    accepted = scores(probability, state["gate"])[0] >= state["thresholds"][str(index)]
    if "intrinsic_review" in output:
        accepted = bool(accepted) and not bool(output["intrinsic_review"][0])
    answer = int(probability.argmax(1)[0])
    mask = torch.nn.functional.interpolate(output["mask"], size=original_size[::-1], mode="bilinear", align_corners=False)[0, 0].cpu().numpy()
    return {"answer": ANSWERS[answer] if accepted else "manual review", "predicted_answer": ANSWERS[answer], "confidence": float(probability.max()), "accepted": bool(accepted)}, mask


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--question", required=True)
    parser.add_argument("--intent", choices=INTENTS, required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--repeats", type=int, default=1000)
    parser.add_argument("--power-records", help="Optional independently measured CSV with power_w column")
    args = parser.parse_args()
    model, state = load(args.model, args.device)
    if "temperature" not in state:
        raise ValueError("Use an independently calibrated checkpoint")
    tokenizer = Tokenizer.from_str(state["tokenizer_json"])
    with Image.open(args.image) as loaded:
        image = loaded.convert("RGB").copy()
    for _ in range(args.warmup):
        predict(model, state, image, args.question, args.intent, args.device, tokenizer)
    if args.device.startswith("cuda"):
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    elapsed = []
    macs = [0]
    handles = []
    def count(module, inputs, output):
        if isinstance(module, torch.nn.Conv2d):
            macs[0] += output.numel() * (module.in_channels//module.groups) * np.prod(module.kernel_size)
        elif isinstance(module, torch.nn.Linear):
            macs[0] += output.numel()*module.in_features
        elif isinstance(module, torch.nn.TransformerEncoderLayer):
            n, length, width = inputs[0].shape
            macs[0] += n*(4*length*width*width+2*length*length*width)
        elif isinstance(module, torch.nn.GRU):
            n, length, width = inputs[0].shape
            macs[0] += n*length*3*(width*module.hidden_size+module.hidden_size**2)
    for module in model.modules():
        if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear, torch.nn.TransformerEncoderLayer, torch.nn.GRU)):
            handles.append(module.register_forward_hook(count))
    result, mask = predict(model, state, image, args.question, args.intent, args.device, tokenizer)
    for handle in handles:
        handle.remove()
    for _ in range(args.repeats):
        if args.device.startswith("cuda"):
            torch.cuda.synchronize()
        start = time.perf_counter()
        result, mask = predict(model, state, image, args.question, args.intent, args.device, tokenizer)
        if args.device.startswith("cuda"):
            torch.cuda.synchronize()
        elapsed.append(time.perf_counter()-start)
    memory = psutil.Process().memory_info()
    power = None
    if args.power_records:
        import csv
        with open(args.power_records, encoding="utf-8") as handle:
            power = float(np.mean([float(row["power_w"]) for row in csv.DictReader(handle)]))
    result.update(median_latency_ms=float(np.median(elapsed)*1000), p95_latency_ms=float(np.quantile(elapsed, .95)*1000),
                  throughput_per_second=float(len(elapsed)/sum(elapsed)), trainable_parameters=sum(p.numel() for p in model.parameters()),
                  macs_convolution_linear_attention=int(macs[0]), serialized_model_bytes=Path(args.model).stat().st_size,
                  host_peak_or_observed_rss_mb=float(getattr(memory, "peak_wset", memory.rss)/1e6),
                  cuda_peak_allocated_mb=float(torch.cuda.max_memory_allocated()/1e6) if args.device.startswith("cuda") else None,
                  mean_device_power_w=power, warmup=args.warmup, requests=args.repeats, device=args.device)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    dump(out/"prediction.json", result)
    heatmap = np.asarray(image, dtype=np.float32)/255
    color = np.stack([mask, .3*mask, np.zeros_like(mask)], -1)
    Image.fromarray(np.uint8(np.clip(.65*heatmap+.35*color, 0, 1)*255)).save(out/"grounding.png")


if __name__ == "__main__":
    main()

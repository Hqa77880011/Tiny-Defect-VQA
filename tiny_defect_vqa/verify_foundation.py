"""Bounded, inference-only check of a supplied full local BLIP-2 checkpoint."""

import argparse
import json
import time
from pathlib import Path

import torch

from .dataset import Questions
from .method import build_model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--foundation-path", required=True)
    parser.add_argument("--source-url", required=True)
    parser.add_argument("--input", required=True, help="Prepared records.json")
    parser.add_argument("--split", default="test")
    parser.add_argument("--max-records", type=int, default=1)
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--dtype", choices=["float32", "float16"], default="float16")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", default="runs/foundation/validation.json")
    args = parser.parse_args()
    if args.max_records < 1:
        parser.error("--max-records must be positive")
    torch.set_num_threads(1)
    torch.manual_seed(17)
    model = (
        build_model(
            dict(
                variant="blip2",
                foundation_backend="huggingface",
                foundation_path=args.foundation_path,
                foundation_dtype=args.dtype,
                normalization="clip",
                size=args.size,
            )
        )
        .to(args.device)
        .eval()
    )
    dataset = Questions(args.input, args.split, args.size, normalization="clip")
    results = []
    with torch.inference_mode():
        for index in range(min(args.max_records, len(dataset))):
            item = dataset[index]
            batch = {
                key: (
                    value[None].to(args.device)
                    if isinstance(value, torch.Tensor)
                    else [value]
                )
                for key, value in item.items()
            }
            if batch["x"].is_cuda:
                torch.cuda.synchronize(batch["x"].device)
            start = time.perf_counter()
            prediction = model(batch)
            if batch["x"].is_cuda:
                torch.cuda.synchronize(batch["x"].device)
            elapsed = time.perf_counter() - start
            assert torch.isfinite(prediction["probability"]).all()
            assert prediction["probability"].shape == (1, 32)
            torch.testing.assert_close(
                prediction["probability"].sum(-1),
                torch.ones(1, device=args.device),
            )
            repeated = model(batch)
            torch.testing.assert_close(
                prediction["probability"], repeated["probability"], atol=1e-6, rtol=1e-6
            )
            results.append(
                dict(
                    record_id=dataset.records[index]["record_id"],
                    source_id=dataset.records[index]["source_id"],
                    raw_question=item["raw_question"],
                    intent=int(item["intent"]),
                    output_shape=list(prediction["probability"].shape),
                    finite=True,
                    repeated_output_matches=True,
                    seconds=elapsed,
                )
            )
    assert all(not p.requires_grad for p in model.model.vision_model.parameters())
    assert all(not p.requires_grad for p in model.model.language_model.parameters())
    assert model.model.query_tokens.dtype == torch.float32
    report = dict(
        supplied_source_url=args.source_url,
        local_checkpoint=str(Path(args.foundation_path).resolve()),
        parameters=sum(p.numel() for p in model.parameters()),
        vision_dtype=str(next(model.model.vision_model.parameters()).dtype),
        language_dtype=str(next(model.model.language_model.parameters()).dtype),
        bridge_dtype=str(model.model.query_tokens.dtype),
        original_tokenizer_loaded=model.tokenizer is not None,
        frozen_vision_and_language=True,
        device=args.device,
        size=args.size,
        split=args.split,
        training_updates=0,
        scope="full supplied pretrained foundation forward on real task inputs; task answer head is untrained, no task accuracy claimed",
        records=results,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

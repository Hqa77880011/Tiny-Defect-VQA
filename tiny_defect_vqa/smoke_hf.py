"""Optional actual Transformers BLIP-2 architecture/gradient/checkpoint check."""

from pathlib import Path
import argparse
import json
import torch
from .method import build_model
from .cli import load


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dtype", choices=["float32", "float16"], default="float32")
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.manual_seed(13)
    model = build_model(
        dict(
            variant="blip2",
            foundation_backend="huggingface",
            foundation_dtype=args.dtype,
            foundation_width=32,
            vision_layers=1,
            language_layers=1,
            size=32,
            vocabulary=128,
        )
    )
    batch = {
        "x": torch.rand(2, 3, 32, 32),
        "question": torch.tensor([[1, 4, 2, 0], [1, 5, 2, 0]]),
        "intent": torch.tensor([0, 1]),
        "y": torch.tensor([0, 2]),
    }
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=1e-4
    )
    before = model.model.query_tokens.detach().clone()
    frozen = model.model.vision_model.embeddings.patch_embedding.weight.detach().clone()
    for _ in range(2):
        optimizer.zero_grad()
        loss = model.objective(batch, model(batch))
        loss.backward()
        for value in model.parameters():
            if value.grad is not None:
                assert torch.isfinite(value.grad).all()
        assert model.model.query_tokens.grad.abs().sum() > 0
        assert model.model.language_projection.weight.grad.abs().sum() > 0
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
    assert not torch.equal(before, model.model.query_tokens)
    assert torch.equal(
        frozen, model.model.vision_model.embeddings.patch_embedding.weight
    )
    assert all(value.grad is None for value in model.model.language_model.parameters())
    root = Path("runs/smoke")
    root.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"config": model.config, "state": model.state_dict()}, root / "hf_blip2.pt"
    )
    restored, _ = load(root / "hf_blip2.pt", "cpu")
    assert not any(value.is_meta for value in restored.parameters())
    assert next(restored.model.vision_model.parameters()).dtype == getattr(
        torch, args.dtype
    )
    assert restored.model.query_tokens.dtype == torch.float32
    model.eval()
    restored.eval()
    with torch.no_grad():
        torch.testing.assert_close(
            model(batch)["probability"],
            restored(batch)["probability"],
            atol=1e-7,
            rtol=1e-6,
        )
    print(
        json.dumps(
            {
                "actual_transformers_blip2": True,
                "steps": 2,
                "frozen_dtype": args.dtype,
                "bridge_dtype": "float32",
                "meta_restore_verified": True,
                "loss": float(loss.detach()),
                "pretrained_weights_used": False,
            }
        )
    )


if __name__ == "__main__":
    main()

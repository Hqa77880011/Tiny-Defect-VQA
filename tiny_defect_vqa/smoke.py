"""Small five-intent end-to-end check; synthetic data are not paper evidence."""
import gc
import json
import sys
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from .dataset import prepare, Questions, answers
from .method import Model, ANSWER_OFFSETS
from .metrics import measure, paired_permutation
from . import cli, inference


def command(main, arguments):
    previous = sys.argv
    try:
        sys.argv = ["functional-check"]+arguments
        main()
    finally:
        sys.argv = previous


def main():
    torch.set_num_threads(1)
    torch.manual_seed(11)
    rng = np.random.default_rng(11)
    out = Path(__file__).resolve().parents[1]/"runs"/"smoke"
    out.mkdir(parents=True, exist_ok=True)
    sources = []
    for split in ["train", "selection", "calibration", "test"]:
        for anomalous in [False, True]:
            name = split+str(int(anomalous))
            image = rng.integers(10, 50, size=(64, 64, 3), dtype=np.uint8)
            mask = np.zeros((64, 64), dtype=np.uint8)
            if anomalous:
                mask[20:30, 30:40] = 255
                image[20:30, 30:40] = 220
            Image.fromarray(image).save(out/(name+".png"))
            Image.fromarray(mask).save(out/(name+"_mask.png"))
            sources.append(dict(id=name, dataset="functional", category="plate", image=name+".png", mask=name+"_mask.png", anomalous=anomalous, morphology="scratch", logical=False, split=split))
    manifest = out/"manifest.json"
    manifest.write_text(json.dumps(sources), encoding="utf-8")
    prepare(manifest, out/"qa", seed=11)
    records = out/"qa/records.json"
    train = Questions(records, "train", size=64)
    assert len(train) == 10
    all_rows = json.loads(records.read_text(encoding="utf-8"))
    assert len({row["record_id"] for row in all_rows}) == 40
    assert all(record["template"] == 2 for record in Questions(records, "test", size=64).records)
    batch = next(iter(torch.utils.data.DataLoader(train, batch_size=10)))
    config = {"width": 128, "rank": 128, "vocabulary": 8192, "variant": "full", "pretrained": False, "localization_weight": .5}
    model = Model(config)
    before = {name: value.detach().clone() for name, value in model.named_parameters()}
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    prediction = model(batch)
    assert prediction["logits"].shape == (10, 32) and prediction["mask"].shape == (10, 1, 8, 8)
    loss = model.objective(batch, prediction)
    assert torch.isfinite(loss)
    loss.backward()
    names = ["encoder.features.0.0.weight", "words.weight", "position.weight", "language.layers.0.self_attn.in_proj_weight", "language.layers.1.self_attn.in_proj_weight", "mask.0.weight", "alpha", "left.weight", "right.weight", "gate.weight", "answer.weight"]
    for name in names:
        grad = dict(model.named_parameters())[name].grad
        assert grad is not None and torch.isfinite(grad).all() and grad.abs().sum() > 0, name
    optimizer.step()
    for name in names:
        assert not torch.equal(before[name], dict(model.named_parameters())[name]), name
    model.eval()
    torch.save(model.state_dict(), out/"reload.pt")
    restored = Model(config)
    restored.load_state_dict(torch.load(out/"reload.pt", weights_only=True))
    restored.eval()
    with torch.no_grad():
        torch.testing.assert_close(model(batch)["probability"], restored(batch)["probability"], atol=1e-7, rtol=1e-6)
    for index, (lower, upper) in enumerate(zip(ANSWER_OFFSETS[:-1], ANSWER_OFFSETS[1:])):
        selected = batch["intent"] == index
        assert prediction["probability"][selected, :lower].sum() == 0
        assert prediction["probability"][selected, upper:].sum() == 0
    loss_value = float(loss.detach())
    del model, restored, optimizer, prediction, before, loss
    gc.collect()
    for variant in ["no_mask", "no_gate", "no_localization", "mobilenet_gru"]:
        model = Model(dict(config, width=32, rank=16, variant=variant))
        result = model(batch)
        objective = model.objective(batch, result)
        objective.backward()
        assert torch.isfinite(objective)
    del model, result, objective
    gc.collect()
    mask = np.zeros((100, 100), bool)
    mask.flat[:50] = True
    assert answers(mask, "scratch", True)[3] == 26
    mask.flat[50] = True
    assert answers(mask, "scratch", True)[3] == 27
    probabilities = np.zeros((4, 32))
    probabilities[:, :2] = [[.9, .1], [.2, .8], [.7, .3], [.4, .6]]
    report = measure(probabilities, [0, 1, 1, 1], [0]*4, [True, True, False, False])
    np.testing.assert_allclose([report["accuracy"], report["coverage"], report["selective_risk"], report["error_capture_rate"], report["ece15"]], [.75, .5, 0., 1., .35])
    assert paired_permutation([1]*5, [0]*5)["pvalue"] == 2/32
    command(cli.main, ["train", "--input", str(records), "--config", str(out.parents[1]/"config.json"), "--output", str(out/"training"), "--epochs", "1", "--batch-size", "5", "--size", "64", "--seed", "11"])
    for gate in ["intent", "global", "entropy"]:
        checkpoint = out/(gate+".pt")
        command(cli.main, ["calibrate", "--input", str(records), "--model", str(out/"training/model.pt"), "--output", str(checkpoint), "--batch-size", "5", "--size", "64", "--gate", gate])
        command(cli.main, ["evaluate", "--input", str(records), "--model", str(checkpoint), "--output", str(out/gate), "--batch-size", "5", "--size", "64"])
    command(cli.main, ["plot", "--input", str(out/"intent/metrics.json"), "--output", str(out/"metrics.png")])
    command(inference.main, ["--model", str(out/"intent.pt"), "--image", str(out/"test1.png"), "--question", "Can a defect be seen on the item?", "--intent", "presence", "--warmup", "1", "--repeats", "2", "--output", str(out/"inference")])
    print(json.dumps({"functional_check": "passed", "source_images": 8, "questions": 40, "image_size_functional_only": 64, "full_width": 128, "full_loss": loss_value, "calibration_gates": 3}))


if __name__ == "__main__":
    main()

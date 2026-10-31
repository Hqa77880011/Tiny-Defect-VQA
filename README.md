# Tiny Defect-VQA

Implementation of **Tiny Defect-VQA: Mask-Grounded Edge Visual Question Answering for Industrial Surface Inspection**. It implements the five registered intents, 32 typed answers, MobileNetV3-Small stride-8/16/32 features, a two-layer Transformer question encoder, predicted-mask attention, low-rank gated fusion, localization supervision and independently calibrated review gates. Historical manuscript result files are not results of this code.

## Install

Python 3.10+: `python -m pip install -e .`. Tested with Python 3.13.5, PyTorch 2.7.1+cu118, torchvision 0.22.1, NumPy 2.1.3, SciPy 1.15.3 and scikit-learn 1.6.1 on CPU with one torch thread. No shared environment installation is required. Default training initializes the visual encoder randomly; `--pretrained` explicitly requests torchvision ImageNet weights and uses ImageNet normalization. Keep that choice identical across compact comparisons. A weight download is required only for that option.

## Sources, ontology and image-disjoint preparation

Obtain the original MVTec AD, VisA and MVTec LOCO AD images/masks under their source terms. A JSON manifest contains one row per source image:

```json
[{"dataset":"MVTec AD","id":"bottle/example","category":"bottle","image":"images/example.png","mask":"masks/example.png","anomalous":true,"morphology":"scratch","logical":false,"split":"train"}]
```

Paths are relative to the manifest. Normal images may use `mask: null`; anomalous images need trusted nonempty pixel masks for grounded QA. `morphology` is one of the 12 groups in `ontology.json`; unknown/untrusted labels map to `unknown morphology`, which sends disposition to manual inspection. Logical anomalies always receive manual inspection. Do not infer physical failure causes from a dataset label.

Explicit splits are `train`, `selection`, `calibration`, `test`. Freeze source-image splits before question generation. If every row omits `split`, the generator stratifies by dataset/category/normality and assigns 70/7.5/7.5/15% with integer floor boundaries and seed 11; exact manuscript inventory counts require a supplied frozen manifest. Duplicate source IDs and cross-partition images are rejected. Each source generates exactly five records with a stable QA record ID, source ID, intent, template ID, typed answer, mask path, morphology and generator version. Training samples one of two wordings; selection/calibration/test use the third held-out wording. Tokenizer training sees training questions only. WordPiece has a maximum 8,192-token vocabulary and the embedding capacity is 8,192; the actual small template vocabulary size is recorded, rather than filled with invented tokens.

```bash
python -m tiny_defect_vqa.cli prepare --manifest data/sources.json --output data/qa --seed 11
```

Extent uses r=0 / (0,.005] / (.005,.02] / above .02; location uses the union-mask centroid in a 3×3 grid. Disposition is accept for normal, rework for trusted minor/moderate structural defects, otherwise manual inspection. The loader resizes RGB to 256×256 (bilinear), masks with nearest neighbor, applies synchronized training flip/±5° rotation and image-only brightness/contrast .9–1.1, then recomputes geometry-dependent answers. Eval has no augmentation. The supplied generator expects the declared shared morphology mapping in each manifest row; the original dataset-label mapping file is not available.

## Training and inference

```bash
python -m tiny_defect_vqa.cli train --input data/qa/records.json --config config.json --output runs/full --variant full --epochs 60 --patience 8 --batch-size 64 --size 256 --seed 11
python -m tiny_defect_vqa.cli calibrate --input data/qa/records.json --model runs/full/model.pt --output runs/full/calibrated.pt --gate intent --size 256
python -m tiny_defect_vqa.cli evaluate --input data/qa/records.json --model runs/full/calibrated.pt --output runs/full/evaluation --size 256
python -m tiny_defect_vqa.inference --model runs/full/calibrated.pt --image data/images/example.png --question "Can a defect be seen on the item?" --intent presence --output runs/prediction --warmup 100 --repeats 1000
```

Defaults: d=128, rank=128, two Transformer layers, four heads, feed-forward width 256, at most 24 tokens; AdamW lr3e-4/wd1e-4 with cosine decay, eight-epoch validation Macro-F1 patience. The rank and normalization choice (GroupNorm on projections) are explicit implementation settings where not fully specified. Mini-batches balance dataset×intent exposure, and answer loss uses inverse-square-root training frequencies. The localizer is depthwise-separable; BCE+per-image soft Dice has weight .5 and nearest-neighbor targets. The same predicted map biases attention through learned alpha. Registered intent masks all incompatible logits before softmax; unknown intents are rejected by single-image inference.

Checkpoint selection reads `selection` only. Scalar temperature minimizes calibration NLL, and each intent threshold is the empirical 20th confidence percentile with `>=` tie handling. `global` fits one confidence threshold; `entropy` fits one negative-entropy threshold. Both are frozen calibration controls. Source IDs and the training-only tokenizer are embedded in checkpoints; evaluation checks source disjointness. Calibration confidence ties can yield coverage above 80%; test coverage is measured.

## Evaluation, controls and outputs

```bash
python -m tiny_defect_vqa.cli train --input data/qa/records.json --config config.json --output runs/no_localization --variant no_localization --epochs 60 --size 256 --seed 11
python -m tiny_defect_vqa.cli calibrate --input data/qa/records.json --model runs/full/model.pt --output runs/full/global.pt --gate global --size 256
python -m tiny_defect_vqa.cli plot --input runs/full/evaluation/metrics.json --output runs/full/metrics.png
python -m tiny_defect_vqa.smoke
```

| Paper module/metric | Code entry | Command/setting |
|---|---|---|
| Multiscale MobileNet + Transformer | `layers.Encoder`, `method.Model` | full, width128, vocabulary8192 |
| Mask attention + learned alpha / gated fusion | `method.Model.forward` | full / `no_mask` / `no_gate` |
| Weighted answer + BCE/Dice localization | `method.Model.objective` | full / `no_localization` |
| Compact GRU/concatenation control | `method.Model` | `mobilenet_gru` (declared adaptation; no localization supervision) |
| Temperature / intent/global/entropy review | `cli.calibrate` | calibration-only gates above |
| Accuracy, fixed-32-token Macro-F1, per-intent accuracy, ECE15 | `metrics.measure` | evaluate `metrics.json` |
| Anomalous-image mask IoU | `cli.collect` | upsample map, threshold .5, one record per anomalous source, image-wise mean |
| Observed coverage, selective risk, error capture | `metrics.measure` | frozen calibrated gate; top-80%-test-ranking risk is a separate diagnostic |
| Exact paired permutation / mean effect | `metrics.paired_permutation` | actual paired seed arrays, 2**N sign assignments |
| Params/MACs/size/median/p95/throughput/memory/power | `inference` | uncached request preprocessing/tokenization/decoding, synchronized CUDA when used |

Training writes `model.pt,history.json`; calibration writes frozen temperature/thresholds; evaluation writes `predictions.npz,metrics.json`; inference writes `prediction.json,grounding.png`. Profiling starts from preloaded RGB pixels and runs resize, normalization, fresh tokenization, model, mask resize, answer decode and gate each request. MACs count conv/linear/attention products and exclude activation/normalization arithmetic. Host peak/observed RSS and CUDA allocated peak are separate fields; they do not measure unique shared Jetson memory. `--power-records measured.csv` accepts actual `power_w` readings; absent telemetry remains null. CPU measurements are functional measurements and do not establish the Jetson Nano 50-ms/512-MB envelope.

## Actual validation and remaining scope

`python -m tiny_defect_vqa.smoke` loads eight generated RGB/mask files, emits 40 image-disjoint five-intent questions, trains the train-only WordPiece, checks full width128 forward/loss/backward and key encoder/Transformer/localizer/alpha/fusion/answer parameter updates, typed-answer masking, checkpoint reload at atol1e-7/rtol1e-6, and each architectural ablation. It then runs one full-width epoch, all three independent calibration gates, locked test evaluation, metric plots and uncached single-image inference/profiling. Functional image size is reduced to 64, and ablation branch checks use width32/rank16; production defaults remain 256/128/128. Hand cases verify extent boundaries, accuracy/coverage/risk/error-capture/ECE and exact permutation p=.0625. These fixtures are code checks, not paper accuracy results.

The original image manifests/morphology mapping and pretrained experiment checkpoints, MobiVQA/VQA-lite/EfficientAD/BLIP-2/AnomalyGPT adapters, complete five-seed runs and TensorRT FP16 deployment are not supplied or executed. The original edge power/memory/latency claims are not asserted by this repository's CPU check. Preserve original dataset terms and use actual telemetry for device comparisons; no performance table is prefilled.

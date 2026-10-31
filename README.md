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

The original image manifests/morphology mapping, pretrained experiment checkpoints and complete five-seed performance runs remain unavailable. All contextual baselines below have runnable task adaptations. Functional fixtures do not establish the original edge power/memory/latency claims.

## Contextual baselines and matched runs

`train --variant` accepts the following distinct implementations in `baselines.py`; each uses the same frozen records/tokenizer, selection split, calibrated review gate and test denominator. The intended comparison identities are explicitly adaptations rather than claims of exact author checkpoints.

| Variant | Implemented mechanism | Explicit task convention |
|---|---|---|
| `mobivqa` | Grid features, question-guided token pruning, first-stage answerability classifier, per-layer answer/exit heads and actual sub-batch early exit | MobileNet/2-layer typed-answer task adaptation; answerability is any-layer-correct and exit target is earliest correct layer; thresholds and retention ratio are configuration parameters |
| `vqa_lite` | Soft-MoE dispatch/combine over 4 experts×4 slots, inter-slot attention, optional frozen teacher KL distillation | Common compact text/visual encoders replace original T5/SigLIP budgets; `--teacher` must share image size and the frozen tokenizer |
| `efficientad` | PDN teacher/student, normal-only teacher statistics, 99.9% hard loss, AE reconstruction/student consistency, general-image penalty and 90/99.5% normal-calibration map quantiles | `--feature-checkpoint` supplies actual PDN teacher weights; otherwise the random teacher is only a functional fixture. Deterministic map-to-answer decoder reports unknown morphology and conservative manual disposition; it receives no test masks |
| `blip2` | Frozen ViT/causal language encoder bridged by learned Q-Former queries and projection, typed answer head | Native small backend is a mechanism check; `--foundation-path` loads actual local Transformers BLIP-2 weights/tokenizer with CLIP normalization. Encoder-decoder and causal LM backends supported; tokenizer and HF architecture are serialized |
| `anomalygpt` | Frozen patch ViT/language blocks, normal-support patch retrieval, image decoder, anomaly-conditioned prompt learner and q/v LoRA, BCE/Dice localization | Compact native adaptation, not original ImageBind-Huge/Vicuna-7B. Frozen pretrained weights require a locally supplied compatible `--init`; random frozen weights are clearly functional only |

Only methods that produce a localization map enter mask-IoU comparisons. MobiVQA intrinsic review is combined with the independently calibrated confidence gate; its exit depth and processed-token counts are produced by the model. `total_parameters`, `trainable_parameters`, intrinsic review and grounding support are recorded for every evaluated checkpoint.

```bash
python -m tiny_defect_vqa.cli train --input data/qa/records.json --variant mobivqa --config config.json --epochs 1 --size 256 --output runs/mobivqa
python -m tiny_defect_vqa.cli train --input data/qa/records.json --variant blip2 --foundation-path /local/blip2 --epochs 1 --size 224 --output runs/blip2
python -m tiny_defect_vqa.benchmark --input data/qa/records.json --config config.json --methods full mobivqa vqa_lite efficientad blip2 anomalygpt --seeds 11 23 37 41 53 --epochs 1 --size 256 --output runs/paired
python -m tiny_defect_vqa.smoke_baselines
# Optional isolated environment: pip install -e '.[foundation,export]'
python -m tiny_defect_vqa.smoke_hf
```

`benchmark` runs training/selection, calibration and test sequentially for every explicitly requested epoch budget. It saves all checkpoints/predictions, paired exact permutation tests and Holm adjustment across methods, mean effects and plots. Freeze the prepared WordPiece once across methods/seeds. Rebuilding a WordPiece on tied frequencies can change IDs; a newly prepared tokenizer must never be mixed with an old checkpoint.

Mechanism sources: [MobiVQA author paper](https://netsys.cs.stonybrook.edu/sites/netsys.cs.stonybrook.edu/files/2024-06/mobivqa-compressed.pdf), [VQA-lite publisher](https://www.sciencedirect.com/science/article/pii/S0262885626000697), [EfficientAD](https://arxiv.org/abs/2303.14535), [BLIP-2](https://proceedings.mlr.press/v202/li23q.html), [AnomalyGPT author implementation](https://github.com/CASIA-LMC-Lab/AnomalyGPT). Native adapters were independently implemented; no third-party weights are bundled.

## Verified export and device entry

```bash
python -m tiny_defect_vqa.deploy export --model runs/full/calibrated.pt --output runs/edge/model.onnx
python -m tiny_defect_vqa.deploy build --onnx runs/edge/model.onnx --fp16 --workspace-mib 512 --output runs/edge/model.plan
python -m tiny_defect_vqa.deploy profile --engine runs/edge/model.plan --inputs runs/edge/model.inputs.npz --warmup 100 --iterations 1000 --output runs/edge/tensorrt.json
```

Export uses fixed image size/batch and 24 padded tokens, opset13 with decomposed attention/LayerNorm, and temporarily disables the unsupported fused PyTorch MHA fast path. It checks the ONNX graph and compares actual ONNX Runtime CPU logits/maps (atol/rtol2e-4). Sidecar JSON retains preprocessing and calibrated temperature/gates. Input-boundary checks remain the caller's responsibility in the fixed graph. Build/profile use actual TensorRT 8.2+ APIs, CUDA events, raw per-iteration times and numerical comparison with saved PyTorch outputs; the binding runner targets TensorRT8.x. This path measures preallocated device tensors only; use `inference` above for request-level timing/power. TensorRT8.2 parser support and FP16 latency/power must be checked on the actual Jetson, and were not executed here.

Additional validation: all five native baselines ran two optimization steps with mechanism-specific gradient/update/frozen-state/reload checks, then one epoch through the shared train→calibrate→evaluate→plot CLI. EfficientAD exercised real general-image penalty inputs, VQA-lite exercised teacher distillation, MobiVQA exercised true early exit and token reduction. Actual Transformers4.57.1 BLIP-2 ran two steps, Q-Former/query/projection updates, frozen vision/LM checks and serialized reconstruction. A two-seed full/MobiVQA one-epoch driver was run. ONNX1.19.1/ONNX Runtime1.22.1 matched the full 64px fixture logits/maps within 5e-7 on full-token and padded-question/intent4 inputs; no GPU deployment or paper performance is inferred.

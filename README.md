# Tiny Defect-VQA

Tiny Defect-VQA supports five inspection intents and 32 typed answers using multiscale MobileNetV3-Small features, a Transformer question encoder, predicted-mask attention, low-rank gated fusion and calibrated review gates.

## Install

Python 3.10+: `python -m pip install -e .`. Default training initializes the visual encoder randomly. `--pretrained` requests torchvision ImageNet weights and ImageNet normalization; use the same setting across compact comparisons.

## Sources, ontology and image-disjoint preparation

Obtain the original MVTec AD, VisA and MVTec LOCO AD images/masks under their source terms. A JSON manifest contains one row per source image:

```json
[{"dataset":"MVTec AD","id":"bottle/example","category":"bottle","image":"images/example.png","mask":"masks/example.png","anomalous":true,"morphology":"scratch","logical":false,"split":"train"}]
```

Paths are relative to the manifest. Normal images may use `mask: null`; anomalous images need trusted nonempty pixel masks for grounded QA. `morphology` is one of the 12 groups in `ontology.json`; unknown/untrusted labels map to `unknown morphology`, which sends disposition to manual inspection. Logical anomalies always receive manual inspection. Do not infer physical failure causes from a dataset label.

Explicit splits are `train`, `selection`, `calibration`, `test`. Freeze source-image splits before question generation. If every row omits `split`, the generator stratifies by dataset/category/normality and assigns 70/7.5/7.5/15% with integer floor boundaries and seed 11; supply explicit splits for a fixed inventory. Duplicate source IDs and cross-partition images are rejected. Each source generates exactly five records with a stable QA record ID, source ID, intent, template ID, typed answer, mask path, morphology and generator version. Training samples one of two wordings; selection/calibration/test use the third held-out wording. Tokenizer training sees training questions only. WordPiece has a maximum 8,192-token vocabulary and the embedding capacity is 8,192; the actual fitted vocabulary size is recorded.

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

| Module / metric | Code entry | Command/setting |
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

## Functional checks

Run `python -m tiny_defect_vqa.smoke` for the image/mask → QA → training → calibration → evaluation workflow. It uses generated functional inputs and reduced image sizes; production defaults remain 256px/width128/rank128.

## Contextual baselines and matched runs

`train --variant` selects the methods below. They share frozen records/tokenizer, selection split, calibration gate and test denominator.

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

Mechanism sources: [MobiVQA author paper](https://netsys.cs.stonybrook.edu/sites/netsys.cs.stonybrook.edu/files/2024-06/mobivqa-compressed.pdf), [VQA-lite publisher](https://www.sciencedirect.com/science/article/pii/S0262885626000697), [EfficientAD](https://arxiv.org/abs/2303.14535), [BLIP-2](https://proceedings.mlr.press/v202/li23q.html), [AnomalyGPT author implementation](https://github.com/CASIA-LMC-Lab/AnomalyGPT).

## Verified export and device entry

```bash
python -m tiny_defect_vqa.deploy export --model runs/full/calibrated.pt --output runs/edge/model.onnx
python -m tiny_defect_vqa.deploy build --onnx runs/edge/model.onnx --fp16 --workspace-mib 512 --output runs/edge/model.plan
python -m tiny_defect_vqa.deploy profile --engine runs/edge/model.plan --inputs runs/edge/model.inputs.npz --warmup 100 --iterations 1000 --output runs/edge/tensorrt.json
```

Export uses fixed image size/batch and 24 padded tokens, opset13 with decomposed attention/LayerNorm, and temporarily disables the unsupported fused PyTorch MHA fast path. It checks the ONNX graph and compares actual ONNX Runtime CPU logits/maps (atol/rtol2e-4). Sidecar JSON retains preprocessing and calibrated temperature/gates. Input-boundary checks remain the caller's responsibility in the fixed graph. Build/profile use actual TensorRT 8.2+ APIs, CUDA events, raw per-iteration times and numerical comparison with saved PyTorch outputs; the runner supports TensorRT8.x bindings and current named-tensor/execute_async_v3 APIs. This path measures preallocated device tensors only; use `inference` above for request-level timing/power. TensorRT8.2 parser support and original Jetson FP16 latency/power still require that target device; current desktop TensorRT is verified below.

## Real MVTec images, original masks and pretrained weights

```bash
python -m tiny_defect_vqa.public_mvtec --source data/mvtec_ad --source-url https://www.mvtec.com/research-teaching/datasets/mvtec-ad --output runs/real_mvtec --size 256
```

The source layout is `bottle/train/good`, `bottle/test/{broken_large,contamination}` and matching original masks. The bounded VQA example assigns one normal and one anomalous image to each train/selection/calibration/test role. Keep this supervised split when comparing methods. Images and masks can be obtained from [MVTec AD](https://www.mvtec.com/research-teaching/datasets/mvtec-ad) or the [foersben mirror](https://huggingface.co/datasets/foersben/mvtec-ad), under CC BY-NC-SA 4.0.

The full model supports ImageNet MobileNetV3 initialization. EfficientAD accepts a local PDN teacher through `--teacher`; the public example retrieves [teacher_small.pth](https://raw.githubusercontent.com/nelson1425/EfficientAD/fcab5146f84ae17597044ad5ddf1656ccf805401/models/teacher_small.pth) from the community EfficientAD implementation at revision `fcab5146f84ae17597044ad5ddf1656ccf805401` and freezes it. The example maps broken-large/contamination labels explicitly to the registered morphology ontology.

## Desktop TensorRT runtime

The deployment entry supports both legacy 8.x bindings and TensorRT10 named tensors. Profiling uses a dedicated CUDA stream, CUDA events, actual output comparisons, raw iteration times and separate IO/free-device-memory fields. Free-memory deltas include concurrent device activity and are not reported as a unique peak or shared Jetson memory.

The optional Windows runtime uses TensorRT 10.14.1.48/CUDA 13. `runtime_install` extracts GPU-specific DLLs from the NVIDIA distribution with wheel RECORD and Authenticode checks. Keep Python bindings and native SDK files in the configured local dependency directory.

```powershell
python -m pip install --no-deps --target .deps/tensorrt tensorrt-cu13==10.14.1.48 tensorrt-cu13-bindings==10.14.1.48
python -m tiny_defect_vqa.runtime_install --output .deps/tensorrt/native --sm 89
$env:PYTHONPATH = "$PWD/.deps/tensorrt;$PWD/.deps/baselines"
$env:PATH = "$PWD/.deps/tensorrt/native;C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.0/bin/x64;$env:PATH"
python -m tiny_defect_vqa.deploy export --model runs/real_mvtec/full/calibrated.pt --output runs/real_mvtec/full/model.onnx
python -m tiny_defect_vqa.deploy build --onnx runs/real_mvtec/full/model.onnx --fp16 --optimization-level 0 --output runs/real_mvtec/full/fp16.plan
python -m tiny_defect_vqa.deploy profile --engine runs/real_mvtec/full/fp16.plan --inputs runs/real_mvtec/full/model.inputs.npz --warmup 10 --iterations 30 --output runs/real_mvtec/full/gpu_profile.json
```

Builder optimization level defaults to 3; level 0 provides a bounded compile check. Set the actual GPU device and runtime for deployment measurements.

Large BLIP-2 checkpoints support `train --foundation-path PATH --foundation-dtype float16`. Frozen vision/language modules use FP16; Q-Former/projection/answer parameters use FP32 with explicit bridge casts. Restoration constructs the architecture on meta tensors before assigning saved weights. `python -m tiny_defect_vqa.smoke_hf --dtype float16` checks this path with a small offline configuration.

## Full original foundation checkpoints

```bash
python -m tiny_defect_vqa.verify_foundation --foundation-path /models/blip2-opt-2.7b --source-url https://huggingface.co/Salesforce/blip2-opt-2.7b/tree/59a1ef6c1e5117b3f65523d1c6066825bcf315e3 --input runs/real_mvtec/qa/records.json --max-records 1 --size 256 --dtype float16 --device cpu --output runs/foundation/official_blip2.json
```

`verify_foundation` performs frozen inference with supplied BLIP-2 vision/Q-Former/OPT/tokenizer weights and task records. It records source IDs, parameter counts and typed outputs. Its attached task-answer head is untrained. Download [Salesforce BLIP-2 OPT-2.7B](https://huggingface.co/Salesforce/blip2-opt-2.7b/tree/59a1ef6c1e5117b3f65523d1c6066825bcf315e3) at the pinned revision into the selected local model directory before running this command.

The original AnomalyGPT is exposed through a separate author runner. Use a checkout of [the author repository](https://github.com/CASIA-LMC-Lab/AnomalyGPT) at `f21c51b93de1094440172051e48f114fb630a389`, its own compatible dependency environment, and a legally restored full Vicuna-7B v0 directory. The public Vicuna v0 delta alone is insufficient. No base weights are downloaded or substituted by the runner.

```bash
python /path/to/this/repo/tiny_defect_vqa/author_anomalygpt.py --author-root /models/AnomalyGPT --vicuna-dir /models/vicuna-7b-v0 --imagebind /models/imagebind_huge.pth --pandagpt /models/pandagpt_7b_max_len_1024.pt --anomaly-checkpoint /models/anomalygpt_supervised.pt --image /data/bottle/test/contamination/000.png --normal-support /data/bottle/train/good/000.png --question "Is there a defect in the image?" --max-new-tokens 8 --output runs/author_anomalygpt/result.json --check-inputs
# In the author's separate environment, use the same arguments without --check-inputs to execute.
```

Run the standalone AnomalyGPT entry in the upstream dependency environment. It loads ImageBind-Huge/Vicuna, then PandaGPT and AnomalyGPT deltas, and saves original text plus raw anomaly-map NPY. `--check-inputs` validates paths and metadata without loading Torch; omit it for inference. The full FP16 model needs sufficient CUDA memory. The compact `anomalygpt` training variant uses the mechanism described in the comparison table.

## Download local frontend assets

Download the bottle image/mask directories under the dataset terms before preparing the public example:

```bash
python -m pip install huggingface_hub
python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='foersben/mvtec-ad', repo_type='dataset', allow_patterns=['bottle/**'], local_dir='data/mvtec_ad')"
python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='Salesforce/blip2-opt-2.7b', revision='59a1ef6c1e5117b3f65523d1c6066825bcf315e3', allow_patterns=['*.json', '*.txt', '*.model', '*.safetensors'], local_dir='models/blip2-opt-2.7b')"
```

Pass `models/blip2-opt-2.7b` to `--foundation-path`. `public_mvtec` downloads the configured MobileNet and PDN teacher on first use; `--teacher PATH` accepts an existing PDN checkpoint. AnomalyGPT's ImageBind/PandaGPT/delta links and upstream model setup are in the author repository linked above; supply a legally obtained full Vicuna v0 model in its separate environment.

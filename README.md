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

Export uses fixed image size/batch and 24 padded tokens, opset13 with decomposed attention/LayerNorm, and temporarily disables the unsupported fused PyTorch MHA fast path. It checks the ONNX graph and compares actual ONNX Runtime CPU logits/maps (atol/rtol2e-4). Sidecar JSON retains preprocessing and calibrated temperature/gates. Input-boundary checks remain the caller's responsibility in the fixed graph. Build/profile use actual TensorRT 8.2+ APIs, CUDA events, raw per-iteration times and numerical comparison with saved PyTorch outputs; the runner supports TensorRT8.x bindings and current named-tensor/execute_async_v3 APIs. This path measures preallocated device tensors only; use `inference` above for request-level timing/power. TensorRT8.2 parser support and original Jetson FP16 latency/power still require that target device; current desktop TensorRT is verified below.

Additional validation: all five native baselines ran two optimization steps with mechanism-specific gradient/update/frozen-state/reload checks, then one epoch through the shared train→calibrate→evaluate→plot CLI. EfficientAD exercised real general-image penalty inputs, VQA-lite exercised teacher distillation, MobiVQA exercised true early exit and token reduction. Actual Transformers4.57.1 BLIP-2 ran two steps, Q-Former/query/projection updates, frozen vision/LM checks and serialized reconstruction. A two-seed full/MobiVQA one-epoch driver was run. ONNX1.19.1/ONNX Runtime1.22.1 matched the full 64px fixture logits/maps within 5e-7 on full-token and padded-question/intent4 inputs; desktop GPU execution is verified below; original Jetson performance is not inferred.


## Real MVTec images, original masks and pretrained weights

```bash
python -m tiny_defect_vqa.public_mvtec --source /path/to/mvtec_ad --source-url https://www.mvtec.com/research-teaching/datasets/mvtec-ad --output runs/real_mvtec --size 256
```

The expected source layout contains `bottle/train/good`, `bottle/test/{broken_large,contamination}` and their original ground-truth masks. This bounded check uses four original normal images and four original anomalous images/masks. Each of train/selection/calibration/test receives one normal and one anomaly, source IDs remain disjoint, and the original pixel masks generate the five registered answers. This supervised VQA repartition is explicitly separate from the official unsupervised anomaly-detection split and the missing original manuscript manifest. The current check used original pixels/masks from the public foersben/mvtec-ad mirror, records that source, and retains the MVTec CC BY-NC-SA 4.0 terms. No source images or weights are committed.

The full width128/rank128 model actually ran one epoch at production 256px using the official ImageNet MobileNetV3 checkpoint (initial conv exact match, then actual training update), independent calibration, locked evaluation and saved-model inference. EfficientAD uses a real public distilled PDN teacher, loaded strictly and frozen throughout one normal-only training epoch, map calibration and test. The checkpoint is pinned to nelson1425/EfficientAD revision fcab5146f84ae17597044ad5ddf1656ccf805401; it is a community reproduction checkpoint, not an original-author release. `--teacher` can provide an existing copy. The two original source-mask classes map explicitly to broken part/contamination; no original hidden ontology mapping is claimed recovered. `validation.json` records metrics, sources, model sizes and teacher freeze checks. One epoch on eight images is an execution check, not a paper performance result.

## Actual desktop TensorRT verification

The deployment entry now supports both legacy 8.x bindings and TensorRT10 named tensors. Profiling uses a dedicated CUDA stream, CUDA events, actual output comparisons, raw iteration times and separate IO/free-device-memory fields. Free-memory deltas include concurrent device activity and are not reported as a unique peak or shared Jetson memory.

On this Windows/Python3.13 machine, the isolated runtime is TensorRT10.14.1.48/CUDA13 on RTX4060 Laptop 8GB. Standard pip runtime download was actually truncated on the network. For this environment, `runtime_install` extracts only official GPU-specific SDK DLLs with ZIP CRC and original wheel RECORD verification. NVIDIA signs these DLLs after writing RECORD: the verifier reconstructs the exact unsigned PE payload to match RECORD and also requires a valid NVIDIA Authenticode signature. A differing unsigned payload or invalid signature fails. Python bindings and the selected native SDK payload remain project-local; global packages and PATH are not changed.

```powershell
python -m pip install --no-deps --target .deps/tensorrt tensorrt-cu13==10.14.1.48 tensorrt-cu13-bindings==10.14.1.48
python -m tiny_defect_vqa.runtime_install --output .deps/tensorrt/native --sm 89
$env:PYTHONPATH = "$PWD/.deps/tensorrt;$PWD/.deps/baselines"
$env:PATH = "$PWD/.deps/tensorrt/native;C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.0/bin/x64;$env:PATH"
python -m tiny_defect_vqa.deploy export --model runs/real_mvtec/full/calibrated.pt --output runs/real_mvtec/full/model.onnx
python -m tiny_defect_vqa.deploy build --onnx runs/real_mvtec/full/model.onnx --fp16 --optimization-level 0 --output runs/real_mvtec/full/fp16.plan
python -m tiny_defect_vqa.deploy profile --engine runs/real_mvtec/full/fp16.plan --inputs runs/real_mvtec/full/model.inputs.npz --warmup 10 --iterations 30 --output runs/real_mvtec/full/gpu_profile.json
```

The builder optimization level is explicit (default3; level0 is a bounded compile/functional check). Current-desktop engine execution and numerical equivalence do not recover TensorRT8.2/Jetson power or its original latency/memory envelope. Original full-foundation execution and the separate AnomalyGPT author entry are described below; the missing manuscript-specific trained checkpoints remain separate from all public pretrained paths.


The production-size real-data checkpoint was exported and actually parsed/built/executed with TensorRT10.14.1.48 FP16, batch1, 256px, width128/rank128. Build level0 produced a 9,067,028-byte engine. After ten warmups, thirty measured CUDA-event iterations had median2.426ms/p95 3.149ms on this RTX4060 Laptop; preallocated inputs only, excluding request preprocessing/transfers/gates/power. The saved random-input numerical checks gave max absolute logit/map errors .002793/.007724 within the declared .02 tolerance. Float32 ONNX Runtime errors were below2.2e-6. Raw times and memory-scope fields remain in runs/real_mvtec/full/gpu_profile.json. These desktop figures do not establish the original Jetson envelope.

Two actual held-out normalized MVTec image/question records (morphology and disposition) were also executed through the engine, with saved PyTorch references: max absolute logits/map errors across these checks were .005216/.006696, inside .02. Original record IDs and per-case comparisons are retained in real_input_checks.json.

Large local BLIP-2 checkpoints also accept `train --foundation-path PATH --foundation-dtype float16`. Frozen vision/LM modules use FP16 while Q-Former/projection/answer parameters stay FP32; the two bridge boundaries cast explicitly. Checkpoint restoration constructs the HF architecture on meta tensors and assigns saved weights before moving to the selected device, avoiding a second initialized FP32 copy. `python -m tiny_defect_vqa.smoke_hf --dtype float16` verifies two actual updates, frozen modules, finite gradients and the actual CLI meta-checkpoint restoration. This small architecture check does not itself substitute for running the full original checkpoint.


## Full original foundation checkpoints

```bash
python -m tiny_defect_vqa.verify_foundation --foundation-path /models/blip2-opt-2.7b --source-url https://huggingface.co/Salesforce/blip2-opt-2.7b/tree/59a1ef6c1e5117b3f65523d1c6066825bcf315e3 --input runs/real_mvtec/qa/records.json --max-records 1 --size 256 --dtype float16 --device cpu --output runs/foundation/official_blip2.json
```

This bounded entry uses the actual supplied local pretrained BLIP-2 vision/Q-Former/OPT/tokenizer on real held-out task inputs. Frozen vision/LM can use FP16, while the bridge/head use FP32. It checks finite typed probabilities and repeated frozen inference and records actual parameter counts/source IDs; it performs zero training updates. The task-specific answer head is untrained in this check, so it produces no task-performance claim. The earlier small HF gradient/meta-restoration check remains separately labeled. The full original Salesforce/blip2-opt-2.7b pinned checkpoint was actually run through this entry at 256px on an image held out by the declared supervised MVTec repartition: 3,744,843,808 parameters including the attached head, actual original tokenizer, FP16 frozen vision/LM and FP32 bridge. The output was finite (1,32), repeated inference matched, and training_updates=0. Both original safetensors shards had already passed publisher LFS verification in the shared local model cache; no quantized/community replacement was used.

The original AnomalyGPT is exposed through a separate author runner. Use a checkout of [the author repository](https://github.com/CASIA-LMC-Lab/AnomalyGPT) at `f21c51b93de1094440172051e48f114fb630a389`, its own compatible dependency environment, and a legally restored full Vicuna-7B v0 directory. The public Vicuna v0 delta alone is insufficient. No base weights are downloaded or substituted by the runner.

```bash
python /path/to/this/repo/tiny_defect_vqa/author_anomalygpt.py --author-root /models/AnomalyGPT --vicuna-dir /models/vicuna-7b-v0 --imagebind /models/imagebind_huge.pth --pandagpt /models/pandagpt_7b_max_len_1024.pt --anomaly-checkpoint /models/anomalygpt_supervised.pt --image /data/bottle/test/contamination/000.png --normal-support /data/bottle/train/good/000.png --question "Is there a defect in the image?" --max-new-tokens 8 --output runs/author_anomalygpt/result.json --check-inputs
# In the author's separate environment, use the same arguments without --check-inputs to execute.
```

The standalone file avoids installing this project's Torch>=2.2 requirement into the author's older environment. It follows the original ImageBind-Huge/Vicuna construction, PandaGPT then AnomalyGPT delta loading, FP16 CUDA, first-turn question, web_demo=True and original sampling defaults. Outputs are unmodified text and raw anomaly-map NPY, with no invented five-intent scoring rule. Static validation checks the pinned real source/API, checkpoint roles, source images and full-model directory contract without importing Torch. Real missing-weight rejection and the declared metadata-only boundary fixtures passed; these are not original-model inference. Actual original AnomalyGPT execution still needs the full legally recovered base model and sufficient CUDA memory; the author's whole-model FP16 CUDA path exceeds this machine's 8GB even for the 7B language parameters alone. The native compact `anomalygpt` remains separately labeled as a mechanism adaptation.

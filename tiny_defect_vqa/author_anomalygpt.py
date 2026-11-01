"""Run the pinned original AnomalyGPT, separately from the compact adaptation.

Author source: https://github.com/CASIA-LMC-Lab/AnomalyGPT
Revision: f21c51b93de1094440172051e48f114fb630a389
The execution path follows code/web_demo.py and OpenLLAMAPEFTModel.generate:
ImageBind-Huge + legally restored Vicuna-7B v0, then PandaGPT and AnomalyGPT
delta state dictionaries, FP16 CUDA, and the author's first-turn prompt.
No foundation weights are downloaded or substituted by this module.

--check-inputs does not import Torch or author modules. It checks paths, pinned
source/API, image decoding and HF configuration, not weight tensor contents,
LLaMA authorization, or whether Vicuna deltas have really been merged.
Execution requires the author's separate compatible dependency environment.
"""

import argparse
import ast
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys


AUTHOR_URL = "https://github.com/CASIA-LMC-Lab/AnomalyGPT"
AUTHOR_REVISION = "f21c51b93de1094440172051e48f114fb630a389"
AUTHOR_FILES = (
    "code/header.py",
    "code/model/__init__.py",
    "code/model/openllama.py",
    "code/model/modeling_llama.py",
    "code/model/AnomalyGPT_models.py",
    "code/model/ImageBind/data.py",
    "code/model/ImageBind/models/imagebind_model.py",
    "code/model/ImageBind/bpe/bpe_simple_vocab_16e6.txt.gz",
    "code/web_demo.py",
    "requirements.txt",
)
VICUNA_7B_CONFIG = {
    "model_type": "llama",
    "hidden_size": 4096,
    "intermediate_size": 11008,
    "num_hidden_layers": 32,
    "num_attention_heads": 32,
    "vocab_size": 32001,
}


def _read_json(path):
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def _file_info(path, label, errors):
    if not path.is_file():
        errors.append(f"{label}: file missing: {path}")
        return {"path": str(path), "exists": False}
    size = path.stat().st_size
    if not size:
        errors.append(f"{label}: empty file: {path}")
    return {"path": str(path), "exists": True, "bytes": size}


def _check_source(root, errors):
    files = {name: _file_info(root / name, name, errors) for name in AUTHOR_FILES}
    revision = None
    try:
        result = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=15,
        )
        if result.returncode:
            errors.append(
                "author-root must be a Git checkout of the pinned author source"
            )
        else:
            revision = result.stdout.strip()
            if revision != AUTHOR_REVISION:
                errors.append(
                    f"author revision {revision} differs from {AUTHOR_REVISION}"
                )
            changes = subprocess.run(
                [
                    "git",
                    "-C",
                    str(root),
                    "diff",
                    "--name-only",
                    "HEAD",
                    "--",
                    "code",
                    "requirements.txt",
                ],
                capture_output=True,
                text=True,
                encoding="utf-8",
                timeout=15,
            )
            if changes.returncode:
                errors.append("cannot check author source working-tree changes")
            elif changes.stdout.strip():
                errors.append(
                    "author code differs from its pinned checkout: "
                    + changes.stdout.strip()
                )
    except (OSError, subprocess.TimeoutExpired) as exc:
        errors.append(f"cannot inspect author Git revision: {exc}")

    api = {}
    model_path = root / "code/model/openllama.py"
    demo_path = root / "code/web_demo.py"
    if model_path.is_file() and demo_path.is_file():
        try:
            tree = ast.parse(model_path.read_text(encoding="utf-8"))
            model = next(
                node
                for node in tree.body
                if isinstance(node, ast.ClassDef) and node.name == "OpenLLAMAPEFTModel"
            )
            methods = {
                node.name: node
                for node in model.body
                if isinstance(node, ast.FunctionDef)
            }
            generate = methods["generate"]
            api["generate_arguments"] = [arg.arg for arg in generate.args.args]
            if api["generate_arguments"] != ["self", "inputs", "web_demo"]:
                errors.append("unsupported author generate signature")
            for name in [
                "__init__",
                "prepare_generation_embedding",
                "extract_multimodal_feature",
            ]:
                if name not in methods:
                    errors.append(f"author model missing API: {name}")
            demo_tree = ast.parse(demo_path.read_text(encoding="utf-8"))
            demo_args = next(
                ast.literal_eval(node.value)
                for node in demo_tree.body
                if isinstance(node, ast.Assign)
                and any(
                    isinstance(t, ast.Name) and t.id == "args" for t in node.targets
                )
            )
            api["author_demo_config"] = demo_args
            expected = {"stage": 2, "lora_r": 32, "lora_alpha": 32, "lora_dropout": 0.1}
            if any(demo_args.get(key) != value for key, value in expected.items()):
                errors.append(
                    "author web_demo configuration differs from the supported pinned API"
                )
        except (SyntaxError, StopIteration, KeyError, ValueError) as exc:
            errors.append(f"cannot validate author source API: {exc}")
    requirements = root / "requirements.txt"
    return {
        "url": AUTHOR_URL,
        "required_revision": AUTHOR_REVISION,
        "observed_revision": revision,
        "files": files,
        "api": api,
        "requirements": (
            requirements.read_text(encoding="utf-8").splitlines()
            if requirements.is_file()
            else []
        ),
    }


def _check_vicuna(directory, errors):
    info = {
        "path": str(directory),
        "role": "merged_full_vicuna_7b_v0_not_delta",
        "tensor_contents_validated": False,
        "llama_authorization_verified": False,
        "merge_provenance_verified": False,
        "notice": "Supply legally restored Vicuna v0; static metadata cannot prove a delta was merged.",
    }
    for name in ["config.json", "tokenizer.model", "tokenizer_config.json"]:
        _file_info(directory / name, f"Vicuna {name}", errors)
    config_path = directory / "config.json"
    if config_path.is_file():
        try:
            config = _read_json(config_path)
            info["config"] = {key: config.get(key) for key in VICUNA_7B_CONFIG}
            if any(config.get(key) != value for key, value in VICUNA_7B_CONFIG.items()):
                errors.append(
                    "Vicuna config must match the original 7B-v0 architecture (including vocab_size=32001)"
                )
        except (json.JSONDecodeError, AttributeError) as exc:
            errors.append(f"invalid Vicuna config: {exc}")
    card = directory / "README.md"
    if (
        card.is_file()
        and 'This "delta model" cannot be used directly'
        in card.read_text(encoding="utf-8")
    ):
        errors.append("Vicuna directory identifies itself as an unmerged delta model")
    index = directory / "pytorch_model.bin.index.json"
    shards = []
    if index.is_file():
        try:
            weight_map = _read_json(index)["weight_map"]
            if not isinstance(weight_map, dict) or not weight_map:
                raise ValueError("empty or invalid weight_map")
            for name in sorted(set(weight_map.values())):
                shard = (directory / name).resolve()
                if not shard.is_relative_to(directory):
                    raise ValueError("weight shard escapes vicuna-dir")
                shards.append(_file_info(shard, "Vicuna weight shard", errors))
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            errors.append(f"invalid Vicuna shard index: {exc}")
    else:
        shards.append(
            _file_info(directory / "pytorch_model.bin", "Vicuna full weights", errors)
        )
    info["weight_files"] = shards
    return info


def check_inputs(args):
    """Return static input/source checks without importing Torch or author code."""
    errors = []
    paths = {
        name: Path(getattr(args, name)).expanduser().resolve()
        for name in [
            "author_root",
            "vicuna_dir",
            "imagebind",
            "pandagpt",
            "anomaly_checkpoint",
            "image",
            "output",
        ]
    }
    if args.normal_support:
        paths["normal_support"] = Path(args.normal_support).expanduser().resolve()
    report = {
        "mode": "check_inputs",
        "model_executed": False,
        "source": _check_source(paths["author_root"], errors),
        "vicuna": _check_vicuna(paths["vicuna_dir"], errors),
        "checkpoints": {},
        "images": {},
        "contract": {
            "batch_size": 1,
            "web_demo": True,
            "prompt": args.question,
            "history": [],
            "top_p": 0.01,
            "temperature": 1.0,
            "max_new_tokens": args.max_new_tokens,
            "precision": "float16",
            "device": "cuda (author single-device implementation)",
            "load_order": [
                "ImageBind-Huge full",
                "merged Vicuna-7B v0 full",
                "PandaGPT delta",
                "AnomalyGPT delta",
            ],
            "outputs": [
                "unmodified author text",
                "raw anomaly map NPY, normally [1,1,224,224]",
            ],
            "answer_mapping": None,
            "static_check_loads_weight_tensors": False,
        },
    }
    if not args.question.strip():
        errors.append("question must not be empty")
    if not 1 <= args.max_new_tokens <= 512:
        errors.append(
            "max-new-tokens must be in [1,512], matching the author demo range"
        )
    for name, role in [
        ("imagebind", "full ImageBind-Huge state_dict"),
        ("pandagpt", "PandaGPT 7b_max_len_1024 delta state_dict"),
        ("anomaly_checkpoint", "AnomalyGPT task delta state_dict"),
    ]:
        report["checkpoints"][name] = {
            **_file_info(paths[name], name, errors),
            "role": role,
            "tensor_contents_validated": False,
        }
    if paths["pandagpt"] == paths["anomaly_checkpoint"]:
        errors.append("PandaGPT and AnomalyGPT deltas must be separate files")
    from PIL import Image, UnidentifiedImageError

    for name in ["image", "normal_support"]:
        if name not in paths:
            continue
        image_info = _file_info(paths[name], name, errors)
        if image_info["exists"]:
            try:
                with Image.open(paths[name]) as image:
                    image_info.update(
                        {"size": list(image.size), "format": image.format}
                    )
                    image.verify()
            except (OSError, UnidentifiedImageError) as exc:
                errors.append(f"{name}: invalid image: {exc}")
        report["images"][name] = image_info
    if paths["output"] in {path for name, path in paths.items() if name != "output"}:
        errors.append("output must not overwrite an input")
    report["errors"] = errors
    report["static_checks_passed"] = not errors
    report["torch_imported"] = "torch" in sys.modules
    return report


def _load_delta(torch, model, path):
    state = torch.load(path, map_location=torch.device("cpu"))
    if not isinstance(state, dict) or not state:
        raise ValueError(f"expected a nonempty author delta state_dict: {path}")
    if not all(
        isinstance(key, str) and isinstance(value, torch.Tensor)
        for key, value in state.items()
    ):
        raise ValueError(
            f"expected plain tensor state_dict, not a training checkpoint wrapper: {path}"
        )
    matched = set(state).intersection(model.state_dict())
    if not matched:
        raise ValueError(
            f"delta has no keys matching the original author model: {path}"
        )
    incompatible = model.load_state_dict(state, strict=False)
    return {
        "path": str(path),
        "matched_keys": len(matched),
        "missing_key_count": len(incompatible.missing_keys),
        "unexpected_keys": incompatible.unexpected_keys,
        "strict": False,
    }


def run_author(args, report):
    """Run the original author model and save its raw outputs, without scoring."""
    import torch
    import numpy as np

    if not torch.cuda.is_available():
        raise RuntimeError(
            "The pinned original AnomalyGPT requires CUDA; no CPU/compact substitution is made."
        )
    author_code = Path(args.author_root).expanduser().resolve() / "code"
    previous_cwd = Path.cwd()
    sys.path.insert(0, str(author_code))
    try:
        # Relative BPE paths and the top-level header/model imports require this
        # exact author code working directory, also used by their web_demo.py.
        os.chdir(author_code)
        module = importlib.import_module("model.openllama")
        if Path(module.__file__).resolve() != author_code / "model/openllama.py":
            raise RuntimeError(
                "model.openllama resolved outside the requested author checkout"
            )
        model_args = dict(report["source"]["api"]["author_demo_config"])
        model_args.update(
            {
                "imagebind_ckpt_path": str(Path(args.imagebind).expanduser().resolve()),
                "vicuna_ckpt_path": str(Path(args.vicuna_dir).expanduser().resolve()),
            }
        )
        # CLI paths are resolved by main before the author working-directory change.
        model = module.OpenLLAMAPEFTModel(**model_args)
        loads = [
            _load_delta(torch, model, args.pandagpt),
            _load_delta(torch, model, args.anomaly_checkpoint),
        ]
        model = model.eval().half().cuda()
        inputs = {
            "prompt": args.question,
            "image_paths": [args.image],
            "normal_img_paths": [args.normal_support] if args.normal_support else [],
            "audio_paths": [],
            "video_paths": [],
            "thermal_paths": [],
            "top_p": 0.01,
            "temperature": 1.0,
            "max_tgt_len": args.max_new_tokens,
            "modality_embeds": [],
        }
        with torch.inference_mode():
            text, pixel_output = model.generate(inputs, web_demo=True)
        anomaly_map = pixel_output.detach().float().cpu().numpy()
    finally:
        os.chdir(previous_cwd)
        sys.path.pop(0)
    output = Path(args.output)
    map_path = output.with_name(output.stem + ".anomaly.npy")
    output.parent.mkdir(parents=True, exist_ok=True)
    np.save(map_path, anomaly_map, allow_pickle=False)
    report.update(
        {
            "mode": "author_inference",
            "model_executed": True,
            "torch_imported": True,
            "raw_text": text,
            "delta_loads": loads,
            "anomaly_map": {
                "path": str(map_path),
                "shape": list(anomaly_map.shape),
                "dtype": str(anomaly_map.dtype),
                "coordinate_frame": "author 224x224 preprocessing",
            },
            "postprocessing": "none: no threshold, smoothing, display normalization, or five-intent mapping",
        }
    )
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for option in [
        "author-root",
        "vicuna-dir",
        "imagebind",
        "pandagpt",
        "anomaly-checkpoint",
        "image",
    ]:
        parser.add_argument("--" + option, required=True)
    parser.add_argument(
        "--normal-support",
        help="Optional single normal image, as in the author web demo",
    )
    parser.add_argument("--question", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument(
        "--output",
        required=True,
        help="JSON result/check report; raw map uses a sibling .anomaly.npy",
    )
    parser.add_argument(
        "--check-inputs",
        action="store_true",
        help="Static checks only; no Torch or author/model imports",
    )
    args = parser.parse_args(argv)
    for name in [
        "author_root",
        "vicuna_dir",
        "imagebind",
        "pandagpt",
        "anomaly_checkpoint",
        "image",
        "normal_support",
        "output",
    ]:
        if getattr(args, name):
            setattr(args, name, str(Path(getattr(args, name)).expanduser().resolve()))
    report = check_inputs(args)
    if report["static_checks_passed"] and not args.check_inputs:
        report = run_author(args, report)
    output = Path(args.output)
    # A colliding output is an invalid input, so preserve that input file.
    if "output must not overwrite an input" not in report["errors"]:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
    print(json.dumps(report, indent=2, ensure_ascii=True))
    return 0 if report["static_checks_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

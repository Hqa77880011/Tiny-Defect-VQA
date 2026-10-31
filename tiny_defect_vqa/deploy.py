"""Export a fixed-shape edge model, verify ONNX, and build/profile TensorRT."""

import argparse
from collections import Counter
import json
from pathlib import Path
import numpy as np
import torch
from .cli import load


class EdgeWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, image, question, intent):
        output = self.model({"x": image, "question": question, "intent": intent})
        return output["logits"], output["mask"]


def export(model_path, output, batch=1):
    """Export TensorRT8.2-targeted opset13 primitives and verify CPU ORT.

    The 8.2-GA parser supports through opset13:
    https://github.com/onnx/onnx-tensorrt/blob/8.2-GA/docs/operators.md
    PyTorch2.7.1's public symbolic_opset14 SDPA function emits older primitive
    operators for this noncausal, dropout-free attention. Register it only for
    this export; opset13 also decomposes LayerNorm to reduction/arithmetic.
    This checks ONNX/CPU execution, not a TensorRT parser or GPU build.
    """
    import onnx
    import onnxruntime as ort
    from torch.onnx.symbolic_opset14 import scaled_dot_product_attention

    if batch < 1:
        raise ValueError("Export batch must be positive")
    model, state = load(model_path, "cpu")
    if state["config"]["variant"] not in {
        "full",
        "no_mask",
        "no_gate",
        "no_localization",
        "mobilenet_gru",
        "vqa_lite",
    }:
        raise ValueError(
            "Fixed edge export supports the native model and VQA-lite; dynamic early exits require their native runtime"
        )
    torch.manual_seed(13)
    size = state["config"]["size"]
    inputs = (
        torch.rand(batch, 3, size, size),
        torch.randint(
            1, min(128, state["config"].get("vocabulary", 8192)), (batch, 24)
        ),
        torch.arange(batch) % 5,
    )
    wrapped = EdgeWrapper(model).eval()
    target = Path(output)
    target.parent.mkdir(parents=True, exist_ok=True)
    fastpath = torch.backends.mha.get_fastpath_enabled()
    torch.onnx.register_custom_op_symbolic(
        "aten::scaled_dot_product_attention", scaled_dot_product_attention, 13
    )
    torch.backends.mha.set_fastpath_enabled(False)
    try:
        with torch.no_grad():
            expected = [value.numpy() for value in wrapped(*inputs)]
            padded_question = inputs[1].clone()
            padded_question[:, -6:] = 0
            padded_inputs = (inputs[0], padded_question, torch.full((batch,), 4))
            padded_expected = [value.numpy() for value in wrapped(*padded_inputs)]
            torch.onnx.export(
                wrapped,
                inputs,
                str(target),
                input_names=["image", "question", "intent"],
                output_names=["logits", "mask"],
                opset_version=13,
                dynamo=False,
                verbose=False,
            )
    finally:
        torch.onnx.unregister_custom_op_symbolic(
            "aten::scaled_dot_product_attention", 13
        )
        torch.backends.mha.set_fastpath_enabled(fastpath)
    graph = onnx.load(str(target))
    # Torch's MHA unflatten emits constant 2 % 3 for its shape axes. Mod is
    # unsupported by the 8.2 parser even at opset13, so evaluate only constant
    # Mod nodes exactly; a runtime-dependent Mod must not pass as compatible.
    constants = {
        value.name: onnx.numpy_helper.to_array(value)
        for value in graph.graph.initializer
    }
    nodes, folded_mod = [], 0
    for node in graph.graph.node:
        if node.op_type == "Constant":
            for attribute in node.attribute:
                if attribute.name == "value":
                    constants[node.output[0]] = onnx.numpy_helper.to_array(attribute.t)
        if node.op_type == "Mod":
            if not all(name in constants for name in node.input):
                raise ValueError("TensorRT8.2 cannot import a runtime-dependent Mod")
            fmod = any(
                attribute.name == "fmod" and attribute.i for attribute in node.attribute
            )
            operation = np.fmod if fmod else np.mod
            result = operation(*(constants[name] for name in node.input))
            constants[node.output[0]] = result
            node = onnx.helper.make_node(
                "Constant",
                [],
                list(node.output),
                name=node.name,
                value=onnx.numpy_helper.from_array(result),
            )
            folded_mod += 1
        nodes.append(node)
    del graph.graph.node[:]
    graph.graph.node.extend(nodes)
    onnx.checker.check_model(graph)
    onnx.save(graph, str(target))
    operators = Counter(node.op_type for node in graph.graph.node)
    if "LayerNormalization" in operators or "Trilu" in operators:
        raise ValueError(
            "TensorRT8.2 export must decompose LayerNorm and noncausal attention"
        )
    options = ort.SessionOptions()
    options.intra_op_num_threads = options.inter_op_num_threads = 1
    session = ort.InferenceSession(
        str(target), sess_options=options, providers=["CPUExecutionProvider"]
    )
    names = ["image", "question", "intent"]
    arrays = {name: value.numpy() for name, value in zip(names, inputs)}
    cases = []
    for name, values, reference in [
        ("all_tokens_first_intent", inputs, expected),
        ("padding_last_intent", padded_inputs, padded_expected),
    ]:
        observed = session.run(
            None, {key: value.numpy() for key, value in zip(names, values)}
        )
        case_errors = []
        for left, right in zip(reference, observed):
            np.testing.assert_allclose(left, right, atol=2e-4, rtol=2e-4)
            case_errors.append(float(np.max(np.abs(left - right))))
        cases.append({"name": name, "maximum_absolute_errors": case_errors})
    errors = [
        max(case["maximum_absolute_errors"][i] for case in cases) for i in range(2)
    ]
    np.savez_compressed(
        target.with_suffix(".inputs.npz"),
        **arrays,
        logits=expected[0],
        mask=expected[1],
    )
    report = {
        "onnx_verified": True,
        "maximum_absolute_errors": errors,
        "batch": batch,
        "image_size": size,
        "tokens": 24,
        "opset": 13,
        "target_parser": "TensorRT 8.2",
        "tensorrt_parser_verified": False,
        "validation_cases": cases,
        "operators": dict(sorted(operators.items())),
        "constant_mod_nodes_folded": folded_mod,
        "decomposition": {
            "scaled_dot_product_attention": "PyTorch symbolic_opset14 registered temporarily at opset13; noncausal eval attention -> Transpose/Mul/MatMul/Softmax (with mask Add)",
            "layer_norm": "opset13 reduction/arithmetic primitives; no LayerNormalization operator",
            "shape_mod": "exact NumPy evaluation of constant Mod shape nodes; runtime-dependent Mod rejected",
        },
        "parser_support_source": "https://github.com/onnx/onnx-tensorrt/blob/8.2-GA/docs/operators.md",
        "model": str(model_path),
        "onnx_bytes": target.stat().st_size,
        "preprocessing": state["config"].get(
            "normalization",
            "imagenet" if state["config"].get("pretrained") else "unit_rgb",
        ),
        "temperature": state.get("temperature"),
        "thresholds": state.get("thresholds"),
        "gate": state.get("gate"),
        "torch": torch.__version__,
        "onnxruntime": ort.__version__,
    }
    target.with_suffix(".json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    return report


def build(onnx_path, output, fp16=False, workspace_mib=512, optimization_level=3):
    """Build TensorRT 8.2+ from the verified fixed-shape ONNX model."""
    import tensorrt as trt

    logger = trt.Logger(trt.Logger.WARNING)
    with (
        trt.Builder(logger) as builder,
        builder.create_network(
            1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        ) as network,
        trt.OnnxParser(network, logger) as parser,
    ):
        if not parser.parse(Path(onnx_path).read_bytes()):
            raise RuntimeError(
                "\n".join(str(parser.get_error(i)) for i in range(parser.num_errors))
            )
        config = builder.create_builder_config()
        if not 0 <= optimization_level <= 5:
            raise ValueError("Builder optimization level must be 0..5")
        if hasattr(config, "builder_optimization_level"):
            config.builder_optimization_level = optimization_level
        elif optimization_level != 3:
            raise ValueError(
                "This TensorRT release does not expose builder optimization levels"
            )
        if hasattr(config, "set_memory_pool_limit"):
            config.set_memory_pool_limit(
                trt.MemoryPoolType.WORKSPACE, workspace_mib << 20
            )
        else:
            config.max_workspace_size = workspace_mib << 20
        if fp16:
            if not builder.platform_has_fast_fp16:
                raise RuntimeError("The selected GPU does not report fast FP16")
            config.set_flag(trt.BuilderFlag.FP16)
        plan = builder.build_serialized_network(network, config)
        if plan is None:
            raise RuntimeError("TensorRT engine build failed")
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        Path(output).write_bytes(bytes(plan))
    return {
        "tensorrt": trt.__version__,
        "fp16": fp16,
        "optimization_level": optimization_level,
        "engine_bytes": Path(output).stat().st_size,
    }


def profile(engine_path, inputs_path, warmup=50, iterations=200, tolerance=0.02):
    """Use CUDA events around synchronized TensorRT execution; retain raw times."""
    import tensorrt as trt

    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU and TensorRT runtime are required")
    arrays = np.load(inputs_path)
    logger = trt.Logger(trt.Logger.WARNING)
    with trt.Runtime(logger) as runtime:
        free_before, total_device_bytes = torch.cuda.mem_get_info()
        engine = runtime.deserialize_cuda_engine(Path(engine_path).read_bytes())
        if engine is None:
            raise RuntimeError("Cannot deserialize engine on this TensorRT/device")
        context = engine.create_execution_context()
        legacy = hasattr(engine, "num_bindings")
        tensors = {}
        count = engine.num_bindings if legacy else engine.num_io_tensors
        bindings = [0] * count
        for i in range(count):
            name = engine.get_binding_name(i) if legacy else engine.get_tensor_name(i)
            dtype = trt.nptype(
                engine.get_binding_dtype(i) if legacy else engine.get_tensor_dtype(name)
            )
            is_input = (
                engine.binding_is_input(i)
                if legacy
                else engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT
            )
            shape = tuple(
                context.get_binding_shape(i)
                if legacy
                else context.get_tensor_shape(name)
            )
            if any(dimension < 0 for dimension in shape):
                raise ValueError(
                    "This exported edge engine requires fixed tensor shapes"
                )
            if is_input:
                value = torch.as_tensor(
                    arrays[name].astype(dtype), device="cuda"
                ).contiguous()
                if tuple(value.shape) != shape:
                    raise ValueError(f"Input shape differs from the engine: {name}")
            else:
                value = torch.empty(
                    shape,
                    dtype=torch.from_numpy(np.empty((), dtype=dtype)).dtype,
                    device="cuda",
                )
            tensors[name], bindings[i] = value, value.data_ptr()
            if not legacy and not context.set_tensor_address(name, value.data_ptr()):
                raise RuntimeError(f"Cannot bind TensorRT tensor: {name}")
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())

        def execute():
            success = (
                context.execute_async_v2(
                    bindings=bindings, stream_handle=stream.cuda_stream
                )
                if legacy
                else context.execute_async_v3(stream_handle=stream.cuda_stream)
            )
            if not success:
                raise RuntimeError("TensorRT execution failed")

        for _ in range(warmup):
            execute()
        torch.cuda.synchronize()
        times = []
        for _ in range(iterations):
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(
                enable_timing=True
            )
            start.record(stream)
            execute()
            end.record(stream)
            end.synchronize()
            times.append(start.elapsed_time(end))
        for name in ["logits", "mask"]:
            np.testing.assert_allclose(
                tensors[name].cpu().numpy(),
                arrays[name],
                atol=tolerance,
                rtol=tolerance,
            )
        free_after, _ = torch.cuda.mem_get_info()
        return {
            "tensorrt": trt.__version__,
            "device": torch.cuda.get_device_name(),
            "warmup": warmup,
            "iterations": iterations,
            "latency_ms": times,
            "median_ms": float(np.median(times)),
            "p95_ms": float(np.quantile(times, 0.95)),
            "numerical_tolerance": tolerance,
            "maximum_absolute_errors": {
                name: float(np.max(np.abs(tensors[name].cpu().numpy() - arrays[name])))
                for name in ["logits", "mask"]
            },
            "io_api": (
                "bindings/execute_async_v2"
                if legacy
                else "named_tensors/execute_async_v3"
            ),
            "device_total_bytes": total_device_bytes,
            "observed_device_free_delta_bytes": free_before - free_after,
            "io_tensor_bytes": sum(
                value.numel() * value.element_size() for value in tensors.values()
            ),
            "memory_scope": "free-memory delta includes engine/context/IO and concurrent device activity; not a unique peak or shared Jetson memory measurement",
            "scope": "preallocated device tensors; preprocessing/transfers/QA gate excluded; no power measurement",
        }


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("export")
    p.add_argument("--model", required=True)
    p.add_argument("--batch", type=int, default=1)
    p = sub.add_parser("build")
    p.add_argument("--onnx", required=True)
    p.add_argument("--fp16", action="store_true")
    p.add_argument("--workspace-mib", type=int, default=512)
    p.add_argument("--optimization-level", type=int, default=3)
    p = sub.add_parser("profile")
    p.add_argument("--engine", required=True)
    p.add_argument("--inputs", required=True)
    p.add_argument("--warmup", type=int, default=50)
    p.add_argument("--iterations", type=int, default=200)
    p.add_argument("--tolerance", type=float, default=0.02)
    for p in sub.choices.values():
        p.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.command == "export":
        report = export(args.model, args.output, args.batch)
    elif args.command == "build":
        report = build(
            args.onnx,
            args.output,
            args.fp16,
            args.workspace_mib,
            args.optimization_level,
        )
    else:
        report = profile(
            args.engine, args.inputs, args.warmup, args.iterations, args.tolerance
        )
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report))


if __name__ == "__main__":
    main()

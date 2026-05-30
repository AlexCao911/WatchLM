#!/usr/bin/env python3
from pathlib import Path
import os
import shutil
import subprocess

import coremltools as ct
import numpy as np
import torch
from coremltools.models import datatypes
from coremltools.models.neural_network import NeuralNetworkBuilder


ROOT = Path(__file__).resolve().parents[2]
RESOURCES = ROOT / "Tests" / "WatchLMCoreTests" / "Resources"
SOURCE_MODEL = RESOURCES / "SmokeIdentity.mlmodel"
SOURCE_PREFILL_MODEL = RESOURCES / "SmokePrefill.mlmodel"
SOURCE_DECODE_MODEL = RESOURCES / "SmokeDecode.mlmodel"
SOURCE_LAYERED_PREFILL_MODEL = RESOURCES / "SmokeLayeredPrefill.mlpackage"
SOURCE_LAYERED_DECODE_MODEL = RESOURCES / "SmokeLayeredDecode.mlpackage"
SOURCE_STATEFUL_KV_MODEL = RESOURCES / "SmokeStatefulKV.mlpackage"


def main() -> None:
    RESOURCES.mkdir(parents=True, exist_ok=True)
    write_identity_model()
    write_prefill_model()
    write_decode_model()
    write_layered_prefill_model()
    write_layered_decode_model()
    write_stateful_kv_model()
    compile_variant(SOURCE_MODEL, "macOS", "13.0", "SmokeIdentity_macOS.mlmodelc")
    compile_variant(SOURCE_MODEL, "watchOS", "10.0", "SmokeIdentity_watchOS.mlmodelc")
    compile_variant(SOURCE_PREFILL_MODEL, "macOS", "13.0", "SmokePrefill_macOS.mlmodelc")
    compile_variant(SOURCE_PREFILL_MODEL, "watchOS", "10.0", "SmokePrefill_watchOS.mlmodelc")
    compile_variant(SOURCE_DECODE_MODEL, "macOS", "13.0", "SmokeDecode_macOS.mlmodelc")
    compile_variant(SOURCE_DECODE_MODEL, "watchOS", "10.0", "SmokeDecode_watchOS.mlmodelc")
    compile_variant(SOURCE_LAYERED_PREFILL_MODEL, "macOS", "13.0", "SmokeLayeredPrefill_macOS.mlmodelc")
    compile_variant(SOURCE_LAYERED_PREFILL_MODEL, "watchOS", "10.0", "SmokeLayeredPrefill_watchOS.mlmodelc")
    compile_variant(SOURCE_LAYERED_DECODE_MODEL, "macOS", "13.0", "SmokeLayeredDecode_macOS.mlmodelc")
    compile_variant(SOURCE_LAYERED_DECODE_MODEL, "watchOS", "10.0", "SmokeLayeredDecode_watchOS.mlmodelc")
    compile_variant(SOURCE_STATEFUL_KV_MODEL, "macOS", "15.0", "SmokeStatefulKV_macOS.mlmodelc")
    compile_variant(SOURCE_STATEFUL_KV_MODEL, "watchOS", "11.0", "SmokeStatefulKV_watchOS.mlmodelc")


def write_identity_model() -> None:
    builder = NeuralNetworkBuilder(
        input_features=[("token", datatypes.Array(1))],
        output_features=[("logits", datatypes.Array(1))],
    )
    builder.add_activation(
        name="identity",
        non_linearity="LINEAR",
        input_name="token",
        output_name="logits",
        params=[1.0, 0.0],
    )
    spec = builder.spec
    use_exact_array_mapping(spec)
    spec.description.metadata.shortDescription = "WatchLM identity smoke model"
    spec.description.metadata.author = "WatchLM"
    ct.models.MLModel(spec).save(SOURCE_MODEL)


def write_prefill_model() -> None:
    builder = NeuralNetworkBuilder(
        input_features=[("input_ids", datatypes.Array(4))],
        output_features=[
            ("next_token", datatypes.Array(1)),
            ("kv_cache", datatypes.Array(1)),
        ],
    )
    builder.add_reduce_sum(
        name="sum_prompt",
        input_name="input_ids",
        output_name="prompt_sum",
        axes=[0],
        keepdims=True,
    )
    builder.add_copy(name="copy_next_token", input_name="prompt_sum", output_name="next_token")
    builder.add_copy(name="copy_kv_cache", input_name="prompt_sum", output_name="kv_cache")
    spec = builder.spec
    use_exact_array_mapping(spec)
    spec.description.metadata.shortDescription = "WatchLM split prefill smoke model"
    spec.description.metadata.author = "WatchLM"
    ct.models.MLModel(spec).save(SOURCE_PREFILL_MODEL)


def write_decode_model() -> None:
    builder = NeuralNetworkBuilder(
        input_features=[
            ("token", datatypes.Array(1)),
            ("kv_cache", datatypes.Array(1)),
        ],
        output_features=[
            ("next_token", datatypes.Array(1)),
            ("updated_kv_cache", datatypes.Array(1)),
        ],
    )
    builder.add_load_constant_nd(
        name="one",
        output_name="one",
        constant_value=np.array([1.0], dtype=np.float32),
        shape=[1],
    )
    builder.add_elementwise(
        name="increment_token",
        input_names=["token", "one"],
        output_name="next_token",
        mode="ADD",
    )
    builder.add_elementwise(
        name="append_to_cache",
        input_names=["kv_cache", "token"],
        output_name="updated_kv_cache",
        mode="ADD",
    )
    spec = builder.spec
    use_exact_array_mapping(spec)
    spec.description.metadata.shortDescription = "WatchLM split decode smoke model"
    spec.description.metadata.author = "WatchLM"
    ct.models.MLModel(spec).save(SOURCE_DECODE_MODEL)


class LayeredPrefillSmokeModel(torch.nn.Module):
    def forward(self, input_ids, position_ids, causal_mask):
        del input_ids, position_ids, causal_mask
        logits = torch.full((1, 8), -1000.0, dtype=torch.float32)
        logits[:, 5] = 1000.0
        present_key = torch.zeros((1, 1, 4, 1), dtype=torch.float32)
        present_value = torch.zeros((1, 1, 4, 1), dtype=torch.float32)
        return logits, present_key, present_value


class LayeredDecodeSmokeModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("vocab", torch.arange(8, dtype=torch.float32).view(1, 8))

    def forward(self, token_id, position_id, causal_mask, past_key_0, past_value_0):
        del position_id, causal_mask, past_key_0, past_value_0
        target = token_id.to(torch.float32).view(1, 1) + 1.0
        logits = -torch.abs(self.vocab - target) * 100.0
        new_key = torch.ones((1, 1, 1, 1), dtype=torch.float32) * target.view(1, 1, 1, 1)
        new_value = new_key + 10.0
        return logits, new_key, new_value


class StatefulKVSmokeModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("last_token", torch.zeros((1,), dtype=torch.float16))
        self.register_buffer("vocab", torch.arange(8, dtype=torch.float32).view(1, 8))

    def forward(self, input_ids, position_ids, causal_mask):
        input_zero = input_ids.to(torch.float32).sum() * 0.0
        position_zero = position_ids.to(torch.float32).sum() * 0.0
        mask_zero = causal_mask.to(torch.float32).sum() * 0.0
        target = torch.maximum(
            self.last_token.to(torch.float32) + 1.0 + input_zero + position_zero + mask_zero,
            torch.tensor([5.0], dtype=torch.float32),
        )
        self.last_token += (target.to(torch.float16) - self.last_token)
        return -torch.abs(self.vocab - target.view(1, 1)) * 100.0


def write_layered_prefill_model() -> None:
    model = LayeredPrefillSmokeModel().eval()
    traced = torch.jit.trace(
        model,
        (
            torch.ones((1, 4), dtype=torch.int32),
            torch.arange(4, dtype=torch.int32).view(1, 4),
            torch.zeros((1, 1, 4, 4), dtype=torch.float32),
        ),
    )
    mlmodel = ct.convert(
        traced,
        convert_to="mlprogram",
        minimum_deployment_target=ct.target.watchOS10,
        inputs=[
            ct.TensorType(name="input_ids", shape=(1, 4), dtype=int),
            ct.TensorType(name="position_ids", shape=(1, 4), dtype=int),
            ct.TensorType(name="causal_mask", shape=(1, 1, 4, 4)),
        ],
        outputs=[
            ct.TensorType(name="logits"),
            ct.TensorType(name="present_key_0"),
            ct.TensorType(name="present_value_0"),
        ],
    )
    mlmodel.short_description = "WatchLM layered prefill smoke ML Program"
    mlmodel.author = "WatchLM"
    shutil.rmtree(SOURCE_LAYERED_PREFILL_MODEL, ignore_errors=True)
    mlmodel.save(SOURCE_LAYERED_PREFILL_MODEL)


def write_layered_decode_model() -> None:
    model = LayeredDecodeSmokeModel().eval()
    traced = torch.jit.trace(
        model,
        (
            torch.tensor([[5]], dtype=torch.int32),
            torch.tensor([[2]], dtype=torch.int32),
            torch.zeros((1, 1, 1, 5), dtype=torch.float32),
            torch.zeros((1, 1, 4, 1), dtype=torch.float32),
            torch.zeros((1, 1, 4, 1), dtype=torch.float32),
        ),
    )
    mlmodel = ct.convert(
        traced,
        convert_to="mlprogram",
        minimum_deployment_target=ct.target.watchOS10,
        inputs=[
            ct.TensorType(name="token_id", shape=(1, 1), dtype=int),
            ct.TensorType(name="position_id", shape=(1, 1), dtype=int),
            ct.TensorType(name="causal_mask", shape=(1, 1, 1, 5)),
            ct.TensorType(name="past_key_0", shape=(1, 1, 4, 1)),
            ct.TensorType(name="past_value_0", shape=(1, 1, 4, 1)),
        ],
        outputs=[
            ct.TensorType(name="logits"),
            ct.TensorType(name="new_key_0"),
            ct.TensorType(name="new_value_0"),
        ],
    )
    mlmodel.short_description = "WatchLM layered decode smoke ML Program"
    mlmodel.author = "WatchLM"
    shutil.rmtree(SOURCE_LAYERED_DECODE_MODEL, ignore_errors=True)
    mlmodel.save(SOURCE_LAYERED_DECODE_MODEL)


def write_stateful_kv_model() -> None:
    model = StatefulKVSmokeModel().eval()
    traced = torch.jit.trace(
        model,
        (
            torch.ones((1, 1), dtype=torch.int32),
            torch.ones((1, 1), dtype=torch.int32),
            torch.zeros((1, 1, 1, 2), dtype=torch.float32),
        ),
        check_trace=False,
    )
    query_length = ct.RangeDim(lower_bound=1, upper_bound=4, default=1)
    key_length = ct.RangeDim(lower_bound=1, upper_bound=5, default=2)
    mlmodel = ct.convert(
        traced,
        convert_to="mlprogram",
        minimum_deployment_target=ct.target.iOS18,
        inputs=[
            ct.TensorType(name="input_ids", shape=(1, query_length), dtype=int),
            ct.TensorType(name="position_ids", shape=(1, query_length), dtype=int),
            ct.TensorType(name="causal_mask", shape=(1, 1, query_length, key_length), dtype=np.float16),
        ],
        outputs=[
            ct.TensorType(name="logits"),
        ],
        states=[
            ct.StateType(
                wrapped_type=ct.TensorType(shape=(1,), dtype=np.float16),
                name="last_token",
            )
        ],
    )
    mlmodel.short_description = "WatchLM stateful KV smoke ML Program"
    mlmodel.author = "WatchLM"
    shutil.rmtree(SOURCE_STATEFUL_KV_MODEL, ignore_errors=True)
    mlmodel.save(SOURCE_STATEFUL_KV_MODEL)


def use_exact_array_mapping(spec) -> None:
    if spec.HasField("neuralNetwork"):
        spec.neuralNetwork.arrayInputShapeMapping = 1


def compile_variant(source_model: Path, platform: str, deployment_target: str, output_name: str) -> None:
    destination = RESOURCES / output_name
    temporary = RESOURCES / f".compile-{platform}"
    shutil.rmtree(destination, ignore_errors=True)
    shutil.rmtree(temporary, ignore_errors=True)
    temporary.mkdir(parents=True)

    environment = os.environ.copy()
    environment.setdefault("DEVELOPER_DIR", "/Applications/Xcode.app/Contents/Developer")
    subprocess.run(
        [
            "xcrun",
            "coremlc",
            "compile",
            str(source_model),
            str(temporary),
            "--platform",
            platform,
            "--deployment-target",
            deployment_target,
        ],
        check=True,
        env=environment,
    )

    compiled = temporary / f"{source_model.stem}.mlmodelc"
    shutil.move(str(compiled), str(destination))
    shutil.rmtree(temporary)


if __name__ == "__main__":
    main()

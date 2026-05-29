#!/usr/bin/env python3
from pathlib import Path
import os
import shutil
import subprocess

import coremltools as ct
import numpy as np
from coremltools.models import datatypes
from coremltools.models.neural_network import NeuralNetworkBuilder


ROOT = Path(__file__).resolve().parents[1]
RESOURCES = ROOT / "Tests" / "WatchLMCoreTests" / "Resources"
SOURCE_MODEL = RESOURCES / "SmokeIdentity.mlmodel"
SOURCE_PREFILL_MODEL = RESOURCES / "SmokePrefill.mlmodel"
SOURCE_DECODE_MODEL = RESOURCES / "SmokeDecode.mlmodel"


def main() -> None:
    RESOURCES.mkdir(parents=True, exist_ok=True)
    write_identity_model()
    write_prefill_model()
    write_decode_model()
    compile_variant(SOURCE_MODEL, "macOS", "13.0", "SmokeIdentity_macOS.mlmodelc")
    compile_variant(SOURCE_MODEL, "watchOS", "10.0", "SmokeIdentity_watchOS.mlmodelc")
    compile_variant(SOURCE_PREFILL_MODEL, "macOS", "13.0", "SmokePrefill_macOS.mlmodelc")
    compile_variant(SOURCE_PREFILL_MODEL, "watchOS", "10.0", "SmokePrefill_watchOS.mlmodelc")
    compile_variant(SOURCE_DECODE_MODEL, "macOS", "13.0", "SmokeDecode_macOS.mlmodelc")
    compile_variant(SOURCE_DECODE_MODEL, "watchOS", "10.0", "SmokeDecode_watchOS.mlmodelc")


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

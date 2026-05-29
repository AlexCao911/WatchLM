#!/usr/bin/env python3
from pathlib import Path
import os
import shutil
import subprocess

import coremltools as ct
from coremltools.models import datatypes
from coremltools.models.neural_network import NeuralNetworkBuilder


ROOT = Path(__file__).resolve().parents[1]
RESOURCES = ROOT / "Tests" / "WatchLMCoreTests" / "Resources"
SOURCE_MODEL = RESOURCES / "SmokeIdentity.mlmodel"


def main() -> None:
    RESOURCES.mkdir(parents=True, exist_ok=True)
    write_source_model()
    compile_variant("macOS", "13.0", "SmokeIdentity_macOS.mlmodelc")
    compile_variant("watchOS", "10.0", "SmokeIdentity_watchOS.mlmodelc")


def write_source_model() -> None:
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
    spec.description.metadata.shortDescription = "WatchLM identity smoke model"
    spec.description.metadata.author = "WatchLM"
    ct.models.MLModel(spec).save(SOURCE_MODEL)


def compile_variant(platform: str, deployment_target: str, output_name: str) -> None:
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
            str(SOURCE_MODEL),
            str(temporary),
            "--platform",
            platform,
            "--deployment-target",
            deployment_target,
        ],
        check=True,
        env=environment,
    )

    compiled = temporary / "SmokeIdentity.mlmodelc"
    shutil.move(str(compiled), str(destination))
    shutil.rmtree(temporary)


if __name__ == "__main__":
    main()

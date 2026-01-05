# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
from pathlib import Path

from setuptools import find_packages, setup

version_folder = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(version_folder, "verl/version/version"), encoding="utf-8") as f:
    __version__ = f.read().strip()

install_requires = [
    "accelerate",
    "codetiming",
    "datasets",
    "dill",
    "hydra-core",
    "numpy",
    "pandas",
    "peft",
    "pyarrow>=19.0.0",
    "pybind11",
    "pylatexenc",
    "ray[default]>=2.41.0",
    "torchdata",
    "tensordict<=0.6.2",
    "transformers",
    "wandb",
    "packaging>=20.0",
]

extras_require = {
    "test": ["pytest", "pre-commit", "py-spy"],
    "prime": ["pyext"],
    "geo": ["mathruler", "torchvision", "qwen_vl_utils"],
    "gpu": ["liger-kernel", "flash-attn"],
    "math": ["math-verify"],
    "vllm": ["tensordict<=0.6.2", "vllm<=0.8.5"],
    "sglang": [
        "tensordict<=0.6.2",
        "sglang[srt,openai]==0.4.6.post5",
        "torch-memory-saver>=0.0.5",
        "torch==2.6.0",
    ],
    "trl": ["trl<=0.9.6"],
}

this_directory = Path(__file__).parent
readme_path = this_directory / "README.md"
long_description = readme_path.read_text(encoding="utf-8") if readme_path.exists() else ""

setup(
    name="verl",
    version=__version__,
    package_dir={"": "."},
    packages=find_packages(where="."),
    license="Apache-2.0",
    description="Research implementation for RL training and evaluation of LLMs.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    install_requires=install_requires,
    extras_require=extras_require,
    package_data={
        "": ["version/*"],
        "verl": ["trainer/config/*.yaml"],
    },
    include_package_data=True,
)

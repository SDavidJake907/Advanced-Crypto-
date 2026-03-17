#!/usr/bin/env bash
set -euo pipefail

mkdir -p models
python -m pip install -U huggingface_hub
huggingface-cli download Qwen/Qwen3-4B-Instruct --local-dir models/Qwen3-4B-Instruct

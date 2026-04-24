#!/usr/bin/env bash
# Set up a project-local virtualenv with uv.
set -euo pipefail

cd "$(dirname "$0")/.."

if ! command -v uv >/dev/null; then
  echo "uv not found. Install: curl -LsSf https://astral.sh/uv/install.sh | sh"
  exit 1
fi

uv venv --python 3.12 .venv
source .venv/bin/activate
uv pip install -e '.[dev]'
# Torch is picked up via the wheel index; if you need a specific CUDA wheel:
uv pip install "torch>=2.3" --index-url https://download.pytorch.org/whl/cu128 || true

python -c "import torch; print('torch', torch.__version__, 'cuda?', torch.cuda.is_available())"
echo "done. activate with: source .venv/bin/activate"

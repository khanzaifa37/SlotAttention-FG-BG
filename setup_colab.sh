#!/usr/bin/env bash

set -euo pipefail

python -m pip install --upgrade pip

if ! python -c "import torch" >/dev/null 2>&1; then
  python -m pip install torch torchvision
fi

python -m pip install -r requirements-colab.txt

echo "Colab setup complete."
echo "If you want a different PyTorch build, install it before running this script."

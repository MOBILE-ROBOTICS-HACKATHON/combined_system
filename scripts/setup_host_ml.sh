#!/usr/bin/env bash
# Host-side Python env for ML / imitation learning (GPU training runs on the host, not in Docker).
# Uses uv to get Python 3.12 (system Python is 3.14, too new for torch/lerobot wheels).
set -e
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export UV_HTTP_TIMEOUT=600     # big CUDA wheels; slow/shared networks time out at the default
[ -d .venv-ml ] || uv venv --python 3.12 .venv-ml
source .venv-ml/bin/activate
# CUDA 12.x wheels work with the RTX 3050 driver already installed on the host
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
uv pip install numpy matplotlib tqdm gymnasium ultralytics opencv-python jupyterlab
uv pip install lerobot          # Hugging Face LeRobot (ACT / Diffusion / VQ-BeT policies)
# Optional sim for LeRobot demos:  uv pip install "lerobot[pusht]"
python - <<'EOF'
import torch; print("torch", torch.__version__, "| cuda available:", torch.cuda.is_available())
EOF
echo "Activate with:  source .venv-ml/bin/activate"

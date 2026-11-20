#!/usr/bin/env bash
set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"
export HF_HOME="${HF_HOME:-$HOME/hf}"
export HF_HUB_ENABLE_HF_TRANSFER=1

echo "[setup] started $(date -Is)"
uv venv "$HOME/venv" --python 3.12
uv pip install --python "$HOME/venv/bin/python" \
  vllm httpx pyyaml matplotlib pandas numpy scipy "huggingface_hub[hf_transfer]"
echo "[setup] python packages installed"
"$HOME/venv/bin/python" - <<'PY'
import vllm, torch, transformers, httpx
print("vllm", vllm.__version__)
print("torch", torch.__version__, "cuda", torch.version.cuda)
print("transformers", transformers.__version__)
print("httpx", httpx.__version__)
print("cuda_available", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu", torch.cuda.get_device_name(0))
PY
echo "[setup] downloading model weights"
HF_HUB_ENABLE_HF_TRANSFER=1 "$HOME/venv/bin/hf" download Qwen/Qwen2.5-14B-Instruct
echo "[setup] model downloaded"
du -sh "$HOME/hf" || true
echo "[setup] finished $(date -Is)"

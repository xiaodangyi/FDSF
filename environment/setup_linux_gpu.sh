#!/usr/bin/env bash
# 独立 Linux GPU 环境；保留系统 Python，复用已有模型。
set -euo pipefail
project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
tool_dir="$HOME/.local/share/deskvision-tools"
env_dir="$HOME/.venvs/deskvision-s1"
mkdir -p "$tool_dir"
uv_path="$tool_dir/uv-x86_64-unknown-linux-gnu/uv"
if [ ! -x "$uv_path" ]; then
  uv_release='https://github.com/astral-sh/uv/releases/download/0.9.5'
  archive='uv-x86_64-unknown-linux-gnu.tar.gz'
  curl -fL --retry 2 --max-time 180 "$uv_release/$archive" -o "$tool_dir/$archive"
  curl -fL --retry 2 --max-time 60 "$uv_release/$archive.sha256" -o "$tool_dir/$archive.sha256"
  (cd "$tool_dir" && sha256sum -c "$archive.sha256")
  tar -xzf "$tool_dir/$archive" -C "$tool_dir"
fi
"$uv_path" --version
"$uv_path" python install 3.12.12
if [ ! -x "$env_dir/bin/python" ]; then
  "$uv_path" venv --python 3.12.12 "$env_dir"
fi
# 只在 CPU 构建存在时卸载它，并清理 uv 中这两个包的下载缓存。
if "$env_dir/bin/python" -c 'from importlib.metadata import version; raise SystemExit(0 if version("torch").endswith("+cpu") else 1)' 2>/dev/null; then
  "$uv_path" pip uninstall --python "$env_dir/bin/python" torch torchvision
  "$uv_path" cache clean torch torchvision
fi
"$uv_path" pip install --python "$env_dir/bin/python" \
  'torch==2.7.1+cu128' 'torchvision==0.22.1+cu128' \
  --index-url https://download.pytorch.org/whl/cu128
# 使用相对依赖路径，避免旧工具解析包含空格的 Windows 挂载路径。
cd "$project_dir"
"$uv_path" pip install --python "$env_dir/bin/python" \
  -r environment/requirements.txt -c environment/constraints.txt \
  'setuptools==75.8.0' wheel pip 'pandas==2.3.3' 'matplotlib==3.9.2'
SAM2_BUILD_CUDA=0 "$uv_path" pip install --python "$env_dir/bin/python" \
  --no-build-isolation --no-deps \
  'git+https://github.com/facebookresearch/sam2.git@2b90b9f5ceec907a1c18123530e92e794ad901a4'
"$uv_path" pip check --python "$env_dir/bin/python"
"$env_dir/bin/python" --version

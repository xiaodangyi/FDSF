#!/usr/bin/env bash
# Install the NVIDIA container runtime only; do not install a Linux display driver in WSL.
set -euo pipefail
if [ "$(id -u)" -ne 0 ]; then
  echo 'Run this explicit installation script as root inside Ubuntu.' >&2
  exit 1
fi
keyring=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -fsSL --max-time 60 https://nvidia.github.io/libnvidia-container/gpgkey \
  | gpg --dearmor --yes -o "$keyring"
curl -fsSL --max-time 60 https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
  | sed "s#deb https://#deb [signed-by=$keyring] https://#g" \
  > /etc/apt/sources.list.d/nvidia-container-toolkit.list
apt-get update -qq
DEBIAN_FRONTEND=noninteractive apt-get install -y nvidia-container-toolkit
nvidia-ctk runtime configure --runtime=docker
systemctl restart docker
nvidia-ctk --version

#!/usr/bin/env bash
# One-time setup on a fresh Ubuntu 22.04/24.04 EC2 instance.
# Usage:  bash deployment/aws/setup.sh   (run as the "ubuntu" user, from the repo root)
set -euo pipefail

# pip unpacks big wheels (PyTorch) into TMPDIR; /tmp can be a small RAM disk, so use the real disk.
export TMPDIR="$HOME/pip-tmp"
mkdir -p "$TMPDIR"

APP_DIR=/opt/ai-cloud-resource-manager
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

# Small machines (1 GB) need swap space or the PyTorch install gets killed.
if [ "$(free -m | awk '/^Mem:/{print $2}')" -lt 1500 ] && [ ! -f /swapfile ]; then
    sudo fallocate -l 2G /swapfile
    sudo chmod 600 /swapfile
    sudo mkswap /swapfile
    sudo swapon /swapfile
    echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
fi

sudo apt-get update
sudo apt-get install -y python3-venv python3-pip nginx git rsync

if [ "$REPO_DIR" != "$APP_DIR" ]; then
    sudo mkdir -p "$APP_DIR"
    sudo chown ubuntu:ubuntu "$APP_DIR"
    rsync -a --exclude .git --exclude .venv "$REPO_DIR"/ "$APP_DIR"/
fi
cd "$APP_DIR"

python3 -m venv .venv
.venv/bin/pip install --no-cache-dir --upgrade pip
# CPU-only PyTorch keeps the install small; everything else comes from requirements.txt.
.venv/bin/pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
grep -v '^torch==' requirements.txt > /tmp/requirements-no-torch.txt
.venv/bin/pip install --no-cache-dir -r /tmp/requirements-no-torch.txt gunicorn

# Train the device anomaly detector (writes ml/device/models/anomaly_detector.pt).
.venv/bin/python -m scripts.train_device_anomaly_detector

sudo cp deployment/aws/ai-resource-manager.service /etc/systemd/system/
sudo cp deployment/aws/nginx-ai-resource-manager.conf /etc/nginx/sites-available/ai-resource-manager
sudo ln -sf /etc/nginx/sites-available/ai-resource-manager /etc/nginx/sites-enabled/ai-resource-manager
sudo rm -f /etc/nginx/sites-enabled/default

sudo systemctl daemon-reload
sudo systemctl enable --now ai-resource-manager
sudo nginx -t
sudo systemctl restart nginx

echo "Done. Open http://<instance-public-ip>/dashboard/"

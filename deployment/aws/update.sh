#!/usr/bin/env bash
# Deploy new code after a git pull. Run from the cloned repo as the "ubuntu" user:
#   git pull && bash deployment/aws/update.sh
set -euo pipefail

export TMPDIR="$HOME/pip-tmp"
mkdir -p "$TMPDIR"

APP_DIR=/opt/ai-cloud-resource-manager
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

rsync -a --exclude .git --exclude .venv --exclude ml/device/models "$REPO_DIR"/ "$APP_DIR"/
cd "$APP_DIR"
grep -v '^torch==' requirements.txt > /tmp/requirements-no-torch.txt
.venv/bin/pip install --no-cache-dir -r /tmp/requirements-no-torch.txt gunicorn
sudo systemctl restart ai-resource-manager
echo "Updated. Step history and device actions were reset by the restart."

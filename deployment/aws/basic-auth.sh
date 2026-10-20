#!/usr/bin/env bash
# Turn nginx Basic Auth (one shared username and password) on or off.
# Usage:  bash deployment/aws/basic-auth.sh on     (asks for a username and password)
#         bash deployment/aws/basic-auth.sh off
# The password is stored only on the server (hashed), never in git.
set -euo pipefail

SITE=/etc/nginx/sites-available/ai-resource-manager
HTPASSWD=/etc/nginx/.htpasswd-ai-rm

case "${1:-}" in
    on)
        read -r -p "Username: " user
        read -r -s -p "Password: " pw; echo
        read -r -s -p "Repeat password: " pw2; echo
        if [ -z "$user" ] || [ -z "$pw" ] || [ "$pw" != "$pw2" ]; then
            echo "Username missing or passwords do not match. Nothing changed."; exit 1
        fi
        printf '%s:%s\n' "$user" "$(openssl passwd -apr1 "$pw")" | sudo tee "$HTPASSWD" > /dev/null
        sudo chmod 640 "$HTPASSWD"
        sudo chgrp www-data "$HTPASSWD"
        if ! grep -q auth_basic "$SITE"; then
            sudo sed -i '/location \/ {/a\        auth_basic "AI Resource Manager";\n        auth_basic_user_file '"$HTPASSWD"';' "$SITE"
        fi
        ;;
    off)
        sudo sed -i '/auth_basic/d' "$SITE"
        ;;
    *)
        echo "Usage: bash deployment/aws/basic-auth.sh on|off"; exit 1
        ;;
esac

sudo nginx -t
sudo systemctl reload nginx
echo "Basic Auth is now: $1"

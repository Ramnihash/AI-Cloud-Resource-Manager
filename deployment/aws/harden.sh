#!/usr/bin/env bash
# Extra protection for a site that is open to the internet. Safe to run twice.
# Run on the EC2 instance:  bash deployment/aws/harden.sh
#  - nginx: per-IP rate limit, hides the nginx version, security headers
#  - fail2ban: bans an IP for 1 hour after 5 wrong passwords
set -euo pipefail

SITE=/etc/nginx/sites-available/ai-resource-manager
LIMITS=/etc/nginx/conf.d/ai-rm-limits.conf

sudo tee "$LIMITS" >/dev/null <<'CONF'
server_tokens off;
limit_req_zone $binary_remote_addr zone=ai_rm_req:10m rate=20r/s;
limit_conn_zone $binary_remote_addr zone=ai_rm_conn:10m;
CONF

if ! grep -q ai_rm_req "$SITE"; then
  sudo sed -i '/location \/ {/a\        limit_req zone=ai_rm_req burst=60 nodelay;\n        limit_conn ai_rm_conn 20;\n        add_header X-Content-Type-Options "nosniff" always;\n        add_header X-Frame-Options "SAMEORIGIN" always;\n        add_header Referrer-Policy "same-origin" always;' "$SITE"
fi

sudo apt-get install -y fail2ban
sudo tee /etc/fail2ban/jail.d/ai-rm.local >/dev/null <<'CONF'
[sshd]
enabled = true
maxretry = 5
bantime = 1h

[nginx-http-auth]
enabled = true
maxretry = 5
bantime = 1h
CONF
sudo systemctl enable --now fail2ban
sudo systemctl restart fail2ban

sudo nginx -t
sudo systemctl reload nginx
echo "Hardening applied."
sudo fail2ban-client status

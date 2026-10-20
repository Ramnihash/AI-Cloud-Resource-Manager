#!/usr/bin/env bash
# Get a free Let's Encrypt certificate and turn on HTTPS.
# Run on the EC2 instance:  bash deployment/aws/enable-https.sh your.domain.name you@example.com
# Needs: the domain's A record points at this server, and ports 80 and 443 are open.
set -euo pipefail

DOMAIN="${1:-}"
EMAIL="${2:-}"
SITE=/etc/nginx/sites-available/ai-resource-manager

if [ -z "$DOMAIN" ] || [ -z "$EMAIL" ]; then
  echo "Usage: bash deployment/aws/enable-https.sh <domain> <email>"
  exit 1
fi

MYIP="$(curl -s https://checkip.amazonaws.com || true)"
DNSIP="$(getent hosts "$DOMAIN" | awk '{print $1}' | head -n1 || true)"
echo "This server: ${MYIP:-unknown}   $DOMAIN points to: ${DNSIP:-nothing yet}"
if [ -n "$MYIP" ] && [ "$MYIP" != "$DNSIP" ]; then
  echo "The domain does not point at this server yet. Fix the A record, wait a few minutes, and run again."
  exit 1
fi

sudo sed -i "s/^\(\s*server_name\s\).*;/\1$DOMAIN;/" "$SITE"
sudo apt-get install -y certbot python3-certbot-nginx
sudo nginx -t
sudo systemctl reload nginx
sudo certbot --nginx -d "$DOMAIN" -m "$EMAIL" --agree-tos --no-eff-email --redirect --non-interactive
sudo nginx -t
sudo systemctl reload nginx
echo "Done. Open https://$DOMAIN/dashboard/"

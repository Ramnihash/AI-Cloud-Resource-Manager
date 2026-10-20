# Deployment

The live instance runs on AWS EC2 (Ubuntu, ap-south-1) behind nginx with HTTPS and Basic Auth,
at https://cloudproject.cv/dashboard/. The full step-by-step guide is
[deployment/aws/DEPLOY-AWS.md](../deployment/aws/DEPLOY-AWS.md). This page is the short version.

## Layout on the server

| Piece | Where |
|-------|-------|
| Git checkout (what you `git pull`) | `~/AI-Cloud-Resource-Manager` |
| Running copy (what gunicorn serves) | `/opt/ai-cloud-resource-manager` |
| Service | `ai-resource-manager` (systemd), gunicorn with 1 worker on 127.0.0.1:8000 |
| Reverse proxy | nginx, ports 80 and 443, Let's Encrypt certificate, Basic Auth |
| Config | `/etc/ai-resource-manager.env` (for example `BACKUP_S3_BUCKET`) |

`git pull` alone does not change the running site, because gunicorn serves the copy in `/opt`.

## Update the live site

```
cd ~/AI-Cloud-Resource-Manager
git pull
bash deployment/aws/update.sh
```

`update.sh` copies the code to `/opt`, installs requirements and restarts the service.
Restarting clears the in-memory step history. Check afterwards:

```
sudo systemctl status ai-resource-manager --no-pager | head -15
curl -s -o /dev/null -w "%{http_code}\n" -u USER:PASS https://cloudproject.cv/dashboard/
```

## One-time scripts (in `deployment/aws/`)

| Script | Purpose |
|--------|---------|
| `setup.sh` | First install: venv, gunicorn, nginx, systemd service |
| `update.sh` | Deploy new code |
| `basic-auth.sh on/off` | Turn the login prompt on or off |
| `harden.sh` | nginx rate limits, security headers, fail2ban |
| `enable-https.sh DOMAIN EMAIL` | Let's Encrypt certificate and HTTP to HTTPS redirect |

## Operations notes

- Ports: SSH (22) only from the admin's IP, 80 and 443 open to everyone. Close 80 after demos if not needed.
- Backups: device backups go to the S3 bucket in `BACKUP_S3_BUCKET` through the instance IAM role. No keys in git.
- Single worker by design (state is in memory). Requests are served one at a time.
- Logs: `sudo journalctl -u ai-resource-manager -n 100 --no-pager` and `/var/log/nginx/error.log`.

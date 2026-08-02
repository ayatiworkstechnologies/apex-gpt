#!/usr/bin/env bash
#
# One-shot setup for the Apex Construction Estimator API on a fresh Debian/
# Ubuntu Linux server: installs system + Python dependencies, trains the ML
# model if missing, installs it as a systemd service (auto-starts on boot,
# auto-restarts on crash), and wires up nginx as a reverse proxy.
#
# Usage (run from inside the cloned repo, as root):
#   sudo bash deploy/setup.sh
#
# Safe to re-run: every step is idempotent.

set -euo pipefail

WITH_OLLAMA=0
for arg in "$@"; do
  case "$arg" in
    --with-ollama) WITH_OLLAMA=1 ;;
  esac
done

APP_USER="apexapi"
APP_DIR="/opt/apex-gpt"
ENV_FILE="/etc/apex-estimator.env"
SERVICE_NAME="apex-estimator"
NGINX_SITE="apex-estimator"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

log() { echo -e "\n\033[1;32m==> $*\033[0m"; }
warn() { echo -e "\033[1;33m!! $*\033[0m"; }

if [[ $EUID -ne 0 ]]; then
  echo "This script must be run as root (sudo bash deploy/setup.sh)." >&2
  exit 1
fi

# ── 1. System packages ─────────────────────────────────────────────────────
log "Installing system packages (python3, venv, pip, nginx)..."
apt-get update -qq
apt-get install -y -qq python3 python3-venv python3-pip nginx rsync >/dev/null

# ── 2. Service account ─────────────────────────────────────────────────────
if ! id -u "$APP_USER" >/dev/null 2>&1; then
  log "Creating service user '$APP_USER' (no login, no home)..."
  useradd --system --no-create-home --shell /usr/sbin/nologin "$APP_USER"
else
  log "Service user '$APP_USER' already exists, skipping."
fi

# ── 3. Deploy application code ─────────────────────────────────────────────
log "Syncing application code to $APP_DIR..."
mkdir -p "$APP_DIR"
rsync -a --delete \
  --exclude='.git' \
  --exclude='__pycache__' \
  --exclude='**/__pycache__' \
  --exclude='tests/.tmp' \
  --exclude='venv' \
  "$REPO_ROOT"/ "$APP_DIR"/

# ── 4. Python virtualenv + dependencies ────────────────────────────────────
log "Creating virtualenv and installing Python dependencies..."
if [[ ! -d "$APP_DIR/venv" ]]; then
  python3 -m venv "$APP_DIR/venv"
fi
"$APP_DIR/venv/bin/pip" install --quiet --upgrade pip
"$APP_DIR/venv/bin/pip" install --quiet -r "$APP_DIR/requirements.txt"

# ── 5. Bootstrap data + model if missing ───────────────────────────────────
if [[ ! -f "$APP_DIR/data/data.csv" ]]; then
  log "Generating training data (first-time setup, ~27,000 rows)..."
  (cd "$APP_DIR" && ./venv/bin/python data/generate_data.py)
else
  log "Training data already present, skipping generation."
fi

if [[ ! -f "$APP_DIR/model/estimator_model.pkl" ]]; then
  log "Training the model (first-time setup)..."
  (cd "$APP_DIR" && ./venv/bin/python model/train.py)
else
  log "Trained model already present, skipping training."
fi

# ── 6. Environment file (API key, config) ──────────────────────────────────
if [[ ! -f "$ENV_FILE" ]]; then
  log "Creating $ENV_FILE from template (EDIT THIS to set a real REFRESH_API_KEY)..."
  cp "$APP_DIR/deploy/apex-estimator.env.example" "$ENV_FILE"
  chmod 640 "$ENV_FILE"
  chown "root:$APP_USER" "$ENV_FILE"
  warn "REFRESH_API_KEY in $ENV_FILE is still the placeholder value — edit it before relying on /api/model/refresh-live."
else
  log "$ENV_FILE already exists, leaving it untouched."
fi

# ── 6b. Optional: local LLM prompt parser (Ollama) ─────────────────────────
if [[ "$WITH_OLLAMA" -eq 1 ]]; then
  if ! command -v ollama >/dev/null 2>&1; then
    log "Installing Ollama (--with-ollama was passed)..."
    curl -fsSL https://ollama.com/install.sh | sh
  else
    log "Ollama already installed, skipping."
  fi
  systemctl enable ollama >/dev/null 2>&1 || true
  systemctl restart ollama
  log "Pulling llama3.2 model (this can take a few minutes)..."
  ollama pull llama3.2
  if ! grep -q '^PROMPT_PARSER=' "$ENV_FILE"; then
    { echo "PROMPT_PARSER=llm"; echo "OLLAMA_MODEL=llama3.2"; echo "OLLAMA_URL=http://localhost:11434"; } >> "$ENV_FILE"
  fi
  warn "PROMPT_PARSER=llm is enabled. Every /api/estimate-from-prompt call now depends on Ollama being up — see README.md for the resource/latency tradeoffs."
fi

# ── 7. Ownership ────────────────────────────────────────────────────────────
chown -R "$APP_USER:$APP_USER" "$APP_DIR"

# ── 8. systemd service (auto-start on boot, auto-restart on crash) ────────
log "Installing systemd service..."
cp "$APP_DIR/deploy/apex-estimator.service" "/etc/systemd/system/${SERVICE_NAME}.service"
systemctl daemon-reload
systemctl enable "$SERVICE_NAME" >/dev/null
systemctl restart "$SERVICE_NAME"

# ── 9. nginx reverse proxy ─────────────────────────────────────────────────
log "Installing nginx site config..."
cp "$APP_DIR/deploy/nginx-apex-estimator.conf" "/etc/nginx/sites-available/${NGINX_SITE}"
ln -sf "/etc/nginx/sites-available/${NGINX_SITE}" "/etc/nginx/sites-enabled/${NGINX_SITE}"
# Remove the default site if it's still the stock one, so it doesn't clash on port 80.
if [[ -L /etc/nginx/sites-enabled/default ]]; then
  rm -f /etc/nginx/sites-enabled/default
fi
nginx -t
systemctl reload nginx

# ── 10. Status ──────────────────────────────────────────────────────────────
log "Done."
sleep 1
systemctl --no-pager --lines=5 status "$SERVICE_NAME" || true

echo
echo "Next steps:"
echo "  1. Edit $ENV_FILE and set a real REFRESH_API_KEY, then: sudo systemctl restart $SERVICE_NAME"
echo "  2. Edit /etc/nginx/sites-available/${NGINX_SITE} and set server_name to your real domain."
echo "  3. Once DNS points at this server: sudo certbot --nginx -d your-domain (adds HTTPS)."
echo "  4. Check logs any time: sudo journalctl -u $SERVICE_NAME -f"
echo "  5. To redeploy after code changes: git pull, then re-run: sudo bash deploy/setup.sh"

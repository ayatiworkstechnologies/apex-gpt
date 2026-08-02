# Deployment Guide (Linux + nginx)

One script sets up a fresh Debian/Ubuntu server end-to-end: system packages,
Python venv, the trained model (trains it if missing), a systemd service that
auto-starts on boot and auto-restarts on crash, and nginx as the reverse proxy.

## Requirements

- A Debian/Ubuntu server (tested target: Ubuntu 22.04/24.04) with `sudo`/root access
- This repo checked out somewhere on that server (e.g. `git clone ... && cd apex-gpt`)

## One-shot setup

```bash
sudo bash deploy/setup.sh
```

This is **safe to re-run** — every step is idempotent. Re-run it after `git pull`
to redeploy code changes.

What it does, in order:

1. Installs `python3`, `python3-venv`, `python3-pip`, `nginx`, `rsync`
2. Creates a dedicated, unprivileged `apexapi` system user to run the service
3. Syncs the repo into `/opt/apex-gpt`
4. Creates a Python venv at `/opt/apex-gpt/venv` and installs `requirements.txt`
5. Generates training data (`data/data.csv`) and trains the model
   (`model/estimator_model.pkl`) — **only if they don't already exist**
6. Creates `/etc/apex-estimator.env` from the template (you must edit this —
   see below) — **only if it doesn't already exist**, so re-running never
   clobbers a secret you've already set
7. Installs and enables the `apex-estimator` systemd service
   (auto-start on boot, auto-restart on crash)
8. Installs the nginx reverse-proxy config and reloads nginx

## After setup: things you must do manually

1. **Set a real API key.** Edit `/etc/apex-estimator.env`:
   ```bash
   sudo nano /etc/apex-estimator.env
   # set REFRESH_API_KEY=<something random>
   sudo systemctl restart apex-estimator
   ```
   Without this, `/api/model/refresh-live` stays locked (fails closed by design).

2. **Point nginx at your real domain.** Edit
   `/etc/nginx/sites-available/apex-estimator`, change `server_name _;` to your
   domain, then:
   ```bash
   sudo nginx -t && sudo systemctl reload nginx
   ```

3. **Add HTTPS** once DNS points at the server:
   ```bash
   sudo apt-get install -y certbot python3-certbot-nginx
   sudo certbot --nginx -d your-domain.com
   ```

## Optional: local LLM prompt parsing (Ollama)

By default the API uses the regex prompt parser — fast, deterministic, no
extra infrastructure. If you want `/api/estimate-from-prompt` to use a local
LLM instead (see `app/llm_parser.py`), pass `--with-ollama`:

```bash
sudo bash deploy/setup.sh --with-ollama
```

This installs Ollama, pulls `llama3.2`, and sets `PROMPT_PARSER=llm` in the
env file. **Know the tradeoff before enabling this in production**: every
prompt-parsing request now depends on Ollama being up, adds real CPU/RAM
usage (~2-4GB) and multi-second latency per request, versus near-zero cost
for the regex parser. It automatically falls back to regex if Ollama is
unreachable, but it's not free to run. See the "Local LLM Prompt Parsing"
section in [README.md](README.md) for details.

## Operating the service

```bash
sudo systemctl status apex-estimator     # check it's running
sudo systemctl restart apex-estimator    # after editing the env file
sudo journalctl -u apex-estimator -f     # tail logs
```

## Files in deploy/

| File | Purpose |
|------|---------|
| `setup.sh` | One-shot install/redeploy script (see above) |
| `apex-estimator.service` | systemd unit — runs `uvicorn` under the `apexapi` user, auto-restarts on failure |
| `nginx-apex-estimator.conf` | nginx reverse proxy config (port 80 → `127.0.0.1:8000`) |
| `apex-estimator.env.example` | Template for `/etc/apex-estimator.env` (API key, root path, optional LLM config) |

## Known limitations

- `setup.sh` targets **Debian/Ubuntu** (`apt-get`). RHEL/Fedora/Amazon Linux
  would need a `dnf`/`yum` variant — ask if you need that.
- The systemd unit runs 2 uvicorn workers by default (`--workers 2` in
  `apex-estimator.service`); adjust for your server's CPU count.
- No Docker/container image yet — this is a bare-metal/VM systemd deployment.

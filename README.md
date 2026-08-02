# 🏗️ Apex Construction Estimator API
### M/S. Apex Steel Industries Limited

City-aware, ML-powered FastAPI service that predicts construction material quantities
(cement, sand, bricks, aggregate, steel) from built-up area, building type, quality
grade and city, and returns a full ₹ cost breakdown for 62 Indian cities.

---

## 📁 Project Structure

```
apex-gpt/
├── app/
│   ├── __init__.py
│   ├── main.py            ← FastAPI app + routes + Swagger docs
│   ├── schemas.py          ← Pydantic request/response models
│   ├── predictor.py        ← Model loader + inference logic
│   ├── nlp_parser.py       ← Plain-English prompt parser (regex, with optional local LLM path)
│   ├── llm_parser.py       ← Local Ollama-backed prompt parser (opt-in, offline)
│   ├── city_rates.py       ← City/state cost database (backed by data/city_rates.csv)
│   ├── live_refresh.py     ← Orchestrates price refresh + auto-tune + reload
│   ├── quantity_model.py   ← Custom sklearn QuantityRatioRegressor
│   └── static/             ← Frontend SPA assets
│
├── model/
│   ├── train.py             ← Training script (run once to generate .pkl)
│   ├── auto_tune.py         ← Daily auto-tuning across model candidates
│   ├── estimator_model.pkl  ← Trained sklearn pipeline (auto-generated, gitignored)
│   └── model_meta.json      ← R² / MAE metrics (auto-generated)
│
├── data/
│   ├── generate_data.py  ← Synthetic dataset generator (tier-stratified, writes data.csv)
│   ├── update_prices.py  ← Weekly live price updater
│   ├── verify.py         ← Quick sanity check of data/data.csv
│   ├── city_rates.csv    ← 62-city rate database (source of truth at runtime)
│   └── data.csv           ← ~27,000-row training dataset across 62 cities (auto-generated)
│
├── tests/                 ← pytest suite (endpoint + unit tests)
├── scripts/                ← Windows PowerShell automation (daily tune, weekly price update)
├── requirements.txt
└── README.md
```

---

## 🤖 Model Architecture

```
Input Features (7)
│
├── area_sqft          (float)  — per-floor area
├── floors             (int)    — number of floors
├── building_type      (int)    — 0=Residential | 1=Commercial | 2=Industrial
├── quality            (int)    — 0=Economy | 1=Standard | 2=Premium
├── total_area_sqft    (float)  — area_sqft × floors (derived)
├── foundation_factor  (float)  — 1.0 + min((floors-1)×0.018, 0.20) (derived)
└── city               (string) — city name (categorical)
         │
         ▼
  ColumnTransformer
  ├── numeric: passthrough (6 features)
  └── city:    OneHotEncoder(handle_unknown="ignore")
         │
         ▼
  QuantityRatioRegressor
  │   divides targets by total_area_sqft to learn per-sqft ratios,
  │   trains on the ratio targets, predicts ratio × total_area_sqft
  │
  └── MultiOutputRegressor
      └── RandomForestRegressor(n_estimators=160, max_depth=9,
                                 min_samples_leaf=8, random_state=42)
           ├── cement_bags
           ├── sand_cft
           ├── bricks
           ├── aggregate_cft
           └── steel_kg
```

### Why this design?
- **QuantityRatioRegressor** learns per-sqft material ratios instead of raw quantities, so large
  projects don't dominate the training loss the way they would with a naive regression target.
- **MultiOutputRegressor** trains one RandomForest per target — each material gets its own ensemble.
- **RandomForest** handles non-linear interactions between area, floors, building type and city
  without manual feature engineering.
- **OneHotEncoder** turns the categorical `city` feature into per-city splits the forest can use.

Current metrics (see `model/model_meta.json` for the live numbers): avg R² ≈ 0.99, avg MAE ≈ 2,370
across the 5 material targets, trained on ~27,000 synthetic samples across 62 cities.

---

## ⚙️ Setup

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Generate training data (~27,000 rows across 62 tiered cities)
python data/generate_data.py

# 3. Train the model
python model/train.py

# 4. (Optional) Set an API key for the live-refresh endpoint
export REFRESH_API_KEY=change-me      # PowerShell: $env:REFRESH_API_KEY = "change-me"

# 5. Run the API
uvicorn app.main:app --reload
```

API is now live at **http://localhost:8000**
Swagger UI at **http://localhost:8000/docs**

---

## 🚢 Production Deployment (Linux + nginx)

For a live server: one script installs system deps, Python deps, trains the
model if missing, and sets up systemd (auto-start on boot, auto-restart on
crash) + nginx (reverse proxy) — see **[DEPLOY.md](DEPLOY.md)**.

```bash
sudo bash deploy/setup.sh
```

---

## 🧠 Local LLM Prompt Parsing (Optional)

`POST /api/estimate-from-prompt` uses a hand-written regex parser (`app/nlp_parser.py`) by
default — no network calls, always available. You can optionally route prompt parsing through
a **locally-running open-source LLM via [Ollama](https://ollama.com)** instead — no cloud API,
no external calls, model runs entirely on your machine.

```bash
# 1. Install Ollama, then pull a model
ollama pull llama3.2

# 2. Enable the LLM parser
export PROMPT_PARSER=llm              # PowerShell: $env:PROMPT_PARSER = "llm"
export OLLAMA_MODEL=llama3.2          # optional, this is the default
export OLLAMA_URL=http://localhost:11434  # optional, this is the default

# 3. Run the API as usual
uvicorn app.main:app --reload
```

If Ollama isn't running, isn't reachable, or returns unusable output, requests **automatically
fall back to the regex parser** — the endpoint never fails just because the local LLM is down.
Prompts that are genuinely invalid (e.g. missing area with no BHK to infer from) still return a
`422` either way. See `app/llm_parser.py` for the JSON contract sent to/from the model.

---

## 🚀 API Endpoints (v3)

| Method | Path                         | Auth        | Description                              |
|--------|------------------------------|-------------|-------------------------------------------|
| GET    | `/`                          | –           | Serve frontend SPA                        |
| GET    | `/api/health`                | –           | Service health + model info               |
| GET    | `/api/cities`                | –           | List all 62 cities with local rates       |
| POST   | `/api/estimate-from-prompt`  | –           | Plain-English NLP estimate (city auto-detected) |
| POST   | `/api/estimate`               | –           | Structured JSON estimate                  |
| GET    | `/api/model/info`             | –           | Training metrics (R², MAE per target)     |
| GET    | `/api/model/catalog`          | –           | Active AI model + tuned candidates        |
| GET    | `/api/model/refresh-status`   | –           | Last live refresh status                  |
| POST   | `/api/model/refresh-live`     | `X-API-Key` | Fetch latest rates + auto-tune + reload   |

`POST /api/model/refresh-live` triggers external network fetches and model retraining, so it
requires an `X-API-Key` header matching the `REFRESH_API_KEY` environment variable. If that
variable is unset, the endpoint rejects every request (fails closed).

---

## 📤 Request Examples

### AI Prompt (NLP)
```json
POST /api/estimate-from-prompt
{
  "prompt": "3 BHK house 1500 sqft in Chennai standard"
}
```

### Structured Request (Manual)
```json
POST /api/estimate
{
  "area": 1200,
  "unit": "sqft",
  "floors": 2,
  "building_type": 0,
  "quality": 1,
  "city": "Chennai"
}
```

### Model Catalog
```json
GET /api/model/catalog
```

### Live Refresh
```json
POST /api/model/refresh-live
Headers: { "X-API-Key": "<REFRESH_API_KEY>" }
{
  "only_verified": true,
  "dry_run": false
}
```

---

## 📥 Sample Response

```json
{
  "input_area_sqft": 1200.0,
  "total_area_sqft": 2400.0,
  "building_type": "Residential",
  "quality": "Standard",
  "materials": {
    "cement_bags": 1005,
    "sand_cft": 2499,
    "bricks": 19078,
    "aggregate_cft": 1251,
    "steel_kg": 10685
  },
  "cost": {
    "city": "Chennai",
    "state": "Tamil Nadu",
    "tier": 1,
    "total_cost_inr": 5231400,
    "cost_per_sqft": 2180
  },
  "model_r2_scores": {
    "cement_bags": 0.9801,
    "sand_cft": 0.9919,
    "bricks": 0.9939,
    "aggregate_cft": 0.9930,
    "steel_kg": 0.9947
  }
}
```

---

## 🔢 Field Reference

### Request fields

| Field           | Type   | Default   | Values                                    |
|-----------------|--------|-----------|--------------------------------------------|
| `area`          | float  | required  | > 0                                         |
| `unit`          | string | `"sqft"`  | `"sqft"` or `"sqm"`                         |
| `floors`        | int    | `1`       | 1 – 15                                      |
| `building_type` | int    | `0`       | 0=Residential, 1=Commercial, 2=Industrial   |
| `quality`       | int    | `1`       | 0=Economy, 1=Standard, 2=Premium            |
| `city`          | string | none      | Any of the 62 supported cities (optional)   |

### Material outputs

| Field           | Unit          | Description                  |
|-----------------|---------------|-------------------------------|
| `cement_bags`   | bags (50 kg)  | Ordinary Portland Cement       |
| `sand_cft`      | cubic feet    | Fine aggregate / river sand    |
| `bricks`        | nos           | Standard modular bricks        |
| `aggregate_cft` | cubic feet    | 20mm stone aggregate           |
| `steel_kg`      | kilograms     | Fe500 reinforcement steel      |

---

## 🔁 Retraining

To retrain on new project data:

1. Add rows to `data/data.csv`
2. Run: `python model/train.py`
3. Restart the server — new model loads automatically on startup.

### Daily Auto-Tuning

For day-by-day model improvement and version tracking:

```bash
python -m model.auto_tune
```

This tries multiple model settings, selects the best by validation R²/MAE, updates
`model/estimator_model.pkl`, writes `model/model_meta.json`, and stores a dated
copy under `model/versions/`.

Windows daily scheduler command:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\daily_model_tuning.ps1
```

### Weekly Price Updates

To refresh material rates from configured city source URLs:

```bash
python -m data.update_prices
```

Dry run without writing the CSV:

```bash
python -m data.update_prices --dry-run
```

Run weekly price update and then auto-tune/version the model:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\weekly_price_update.ps1 -AutoTune
```

The updater creates a timestamped backup like
`data/city_rates.csv.YYYYMMDD-HHMMSS.bak` before writing new rates.

---

## 🧪 Testing

```bash
pytest tests/ -q
```

All test files use real pytest assertions (no print-only scripts). Endpoint tests use
`fastapi.testclient.TestClient`; external dependencies (price fetches, auto-tuning) are
mocked via `monkeypatch`.

---

## 🔒 Security notes

- CORS is currently wide open (`allow_origins=["*"]`) — restrict this to known frontend
  origins before deploying publicly.
- `POST /api/model/refresh-live` requires `X-API-Key` (see above); no other endpoint
  requires authentication.
- There is no rate limiting — put this behind a reverse proxy / API gateway if exposed
  publicly.

---

## 📜 License
Proprietary — All Rights Reserved © Rubankumar, Ayatiworks. See [LICENSE](LICENSE).

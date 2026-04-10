# 🏗️ Apex Construction Estimator API
### M/S. Apex Steel Industries Limited

ML-powered FastAPI service that predicts construction material quantities from built-up area.

---

## 📁 Project Structure

```
construction_estimator/
├── app/
│   ├── __init__.py
│   ├── main.py          ← FastAPI app + routes + Swagger docs
│   ├── schemas.py       ← Pydantic request/response models
│   └── predictor.py     ← Model loader + inference logic
│
├── model/
│   ├── train.py         ← Training script (run once to generate .pkl)
│   ├── estimator_model.pkl   ← Trained sklearn pipeline (auto-generated)
│   └── model_meta.json       ← R² / MAE metrics (auto-generated)
│
├── data/
│   ├── generate_data.py ← Synthetic dataset generator
│   └── data.csv         ← 27,000-row dataset across 56 cities (auto-generated)
│
├── tests/
│   └── test_predict.py  ← Direct inference tests (no server needed)
│
├── requirements.txt
└── README.md
```

---

## 🤖 Model Architecture

```
Input Features (5)
│
├── area_sqft       (float)  — per-floor area
├── floors          (int)    — number of floors
├── building_type   (int)    — 0=Residential | 1=Commercial | 2=Industrial
├── quality         (int)    — 0=Economy | 1=Standard | 2=Premium
└── total_area_sqft (float)  — area_sqft × floors (derived)
         │
         ▼
  StandardScaler   ← normalise all features to zero mean, unit variance
         │
         ▼
  MultiOutputRegressor
  └── RandomForestRegressor (n_estimators=200, max_depth=12)
       ├── cement_bags    R²=0.9973
       ├── sand_cft       R²=0.9988
       ├── bricks         R²=0.9991
       ├── aggregate_cft  R²=0.9966
       └── steel_kg       R²=0.9963
```

### Why RandomForest + MultiOutput?
- **MultiOutputRegressor** trains one RF per target — each material has its own tree ensemble
- **RandomForest** handles non-linear interactions between area, floors and building type without feature engineering
- **StandardScaler** keeps feature magnitudes comparable so the scaler won't skew splits

---

## ⚙️ Setup

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Generate training data (27,000 rows across 56 tiered cities)
python data/run_generate.py

# 3. Train the model
python model/train.py

# 4. Run the API
uvicorn app.main:app --reload
```

API is now live at **http://localhost:8000**
Swagger UI at **http://localhost:8000/docs**

---

## 🚀 API Endpoints

| Method | Path          | Description                            |
|--------|---------------|----------------------------------------|
| GET    | `/`           | Health check                           |
| GET    | `/health`     | Service health + model info            |
| POST   | `/estimate`   | **Main endpoint** — predict materials  |
| GET    | `/model/info` | Training metrics (R², MAE per target)  |

---

## 📤 Prompting — Request Examples

### Minimal request (defaults apply)
```json
POST /estimate
{
  "area": 1000,
  "unit": "sqft"
}
```

### Full residential (2 floors, standard quality)
```json
POST /estimate
{
  "area": 1200,
  "unit": "sqft",
  "floors": 2,
  "building_type": 0,
  "quality": 1
}
```

### Commercial in metric (sqm → sqft conversion auto-applied)
```json
POST /estimate
{
  "area": 250,
  "unit": "sqm",
  "floors": 3,
  "building_type": 1,
  "quality": 2
}
```

### Economy ground floor
```json
POST /estimate
{
  "area": 600,
  "unit": "sqft",
  "floors": 1,
  "building_type": 0,
  "quality": 0
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
  "model_r2_scores": {
    "cement_bags": 0.9973,
    "sand_cft": 0.9988,
    "bricks": 0.9991,
    "aggregate_cft": 0.9966,
    "steel_kg": 0.9963
  }
}
```

---

## 🔢 Field Reference

### Request fields

| Field           | Type   | Default      | Values                                          |
|----------------|--------|-------------|------------------------------------------------|
| `area`          | float  | required    | > 0                                             |
| `unit`          | string | `"sqft"`    | `"sqft"` or `"sqm"`                            |
| `floors`        | int    | `1`         | 1 – 10                                          |
| `building_type` | int    | `0`         | 0=Residential, 1=Commercial, 2=Industrial       |
| `quality`       | int    | `1`         | 0=Economy, 1=Standard, 2=Premium                |

### Material outputs

| Field           | Unit          | Description                         |
|----------------|---------------|-------------------------------------|
| `cement_bags`   | bags (50 kg)  | Ordinary Portland Cement            |
| `sand_cft`      | cubic feet    | Fine aggregate / river sand         |
| `bricks`        | nos           | Standard modular bricks             |
| `aggregate_cft` | cubic feet    | 20mm stone aggregate                |
| `steel_kg`      | kilograms     | Fe500 reinforcement steel           |

---

## 🔁 Retraining

To retrain on new project data:

1. Add rows to `data/data.csv`
2. Run: `python model/train.py`

3. Restart the server — new model loads automatically on startup.

---

## 📜 License
MIT — M/S. Apex Steel Industries Limited

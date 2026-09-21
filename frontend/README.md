# Mars Rover ML — Mission Control Frontend & API System

Automated Classification of Martian Surface Images Captured by NASA's Curiosity Rover using Machine Learning & Deep Learning.

This frontend is designed directly following the **Stitch AeroScientific Precision** design system and connects directly to the real Python backend serving verified project data.

---

## Key Features & User Corrections

1. **Default Test Set Workflow (1,305 Samples)**:
   - The Classification Lab loads official test samples by default because verified test predictions are preserved for all 9 models.
   - Train and Validation samples may only be classified when a local model checkpoint (`.joblib` or `.pth`) is available on disk for live forward inference. If no checkpoint exists, the UI disables inference with a clear warning explaining the requirement.

2. **Inference Latency & Telemetry**:
   - **Live Model**: Displays actual measured execution time (in milliseconds) with green highlight.
   - **Stored Test Prediction**: Explicitly returns `inference_time_ms = null` and displays `N/A — STORED PREDICTION`.

3. **Approved Model Naming**:
   - EfficientNet-B3 is displayed strictly as **`EfficientNet-B3`** across all cards, panels, tables, and charts (without any `Champion` tags).

4. **Canonical Benchmark Source**:
   - All benchmark metrics (Accuracy, Macro/Weighted F1, Precision, Recall) are dynamically loaded from `results/10_Overall_Comparison/final_test_comparison.json`. No benchmark metrics are hardcoded in the frontend.

5. **Lazy Loading of Deep-Learning Checkpoints**:
   - Checkpoints are not eagerly loaded at server startup. When a live inference request arrives, the required model is loaded on demand and cached in memory.

---

## Technology Stack

- **Frontend**: React 19 + TypeScript + Vite + React Router 7 + Lucide React
- **Design Tokens**: AeroScientific dark palette (`#080C14` space slate, `#F97316` Mars amber, `#06B6D4` NASA cyan, `#10B981` telemetry green) with HUD reticle crosshairs and scanning laser animations.
- **Backend**: FastAPI + Uvicorn + NumPy + PyTorch + Scikit-Learn + Joblib

---

## Project Structure

```
├── api/
│   ├── dataset_service.py     # Indexes 6,691 real images across test, val, train splits
│   ├── model_service.py       # Canonical benchmark loader and model metadata
│   ├── inference_service.py   # Stored predictions lookup + lazy-load checkpoint inference
│   ├── schemas.py             # Pydantic request/response schemas
│   └── server.py              # FastAPI server (port 8000)
├── frontend/
│   ├── src/
│   │   ├── components/
│   │   │   ├── Navbar.tsx           # Mission status and navigation
│   │   │   ├── ViewportReticle.tsx  # HUD reticle with laser scanner & telemetry
│   │   │   ├── ModelCard.tsx        # Model selector with checkpoint status badges
│   │   │   ├── ResultPanel.tsx      # Inference telemetry, probabilities & latency
│   │   │   ├── SampleDrawer.tsx     # Filterable sample browser (defaulting to Test)
│   │   │   └── BenchmarkTable.tsx   # Canonical test set comparison table
│   │   ├── pages/
│   │   │   ├── ClassificationLab.tsx # Main interactive classification workstation
│   │   │   ├── DatasetModelsPage.tsx # Dataset distributions & model architecture specs
│   │   │   └── DashboardPage.tsx     # Comparative accuracy rankings & benchmark tables
│   │   ├── services/api.ts          # Typed REST API client
│   │   ├── types/index.ts           # Schema interfaces
│   │   ├── index.css                # AeroScientific design system CSS
│   │   └── App.tsx                  # Main router setup
│   └── .env                         # VITE_API_BASE_URL=http://localhost:8000/api
└── tests/
    └── test_api.py            # Complete API test suite verifying all 5 corrections
```

---

## How to Run Locally

### 1. Start Backend API (FastAPI)
```bash
cd "Mars ML models"
./venv/bin/uvicorn api.server:app --host 127.0.0.1 --port 8000 --reload
```
API Documentation will be available at: `http://127.0.0.1:8000/api/docs`

### 2. Start Frontend (Vite)
```bash
cd "Mars ML models/frontend"
npm run dev
```
Open `http://localhost:5173` in your browser.

### 3. Run Backend API Tests
```bash
cd "Mars ML models"
./venv/bin/python -m unittest tests/test_api.py
```

# 🏗️ COMPLETE SYSTEM EXECUTION PLAN
### Pedestrian Distraction Detection System — Group 09 (M)
### Corrected & Final Version

---

## 📋 CONTINUATION PROMPT
> "Continue the execution plan for Pedestrian Distraction Detection System. We left off at [PHASE X / Step Y]. The key rule is: ML code lives in src/ and is never duplicated. backend/ is API layer only."

---

---

# 🔍 PHASE 1 — Project Analysis Summary

---

## 1.1 Frontend Structure (React)

The frontend is a React 19 SPA built with Vite 7, TailwindCSS v4, and React Router v7. It has 5 pages.

**Pages:**
- `/` — Dashboard: StatusCards, WebcamDetector, UsageMetrics, AlertsPanel
- `/detection` — Full WebcamDetector with detection controls
- `/devices` — Device list (currently static)
- `/reports` — Reports page (currently static)
- `/settings` — Settings page (currently static)

**Key Components:**
- `WebcamDetector` — Opens webcam, draws bounding boxes on canvas, simulates detection via setInterval
- `AlertsPanel` — Hardcoded static alerts
- `UsageMetrics` — Chart.js chart with hardcoded data
- `StatusCards` — 4 hardcoded stat cards
- `AppContext` — Global state via useReducer — defined but NEVER applied to app
- `useWebSocket` — Simulates WebSocket with setInterval, never connects to real backend

---

## 1.2 ML Model Structure (src/)

Already well-organized in MLOps style. Key components:

- `src/components/` — 16 core ML modules (pose estimation, feature engineering, classification, fusion, etc.)
- `src/pipeline/` — Orchestration pipelines for inference and training
- `src/serving/` — API predictor and Pydantic schemas
- `src/config/` — Config loading classes
- `src/entity/` — Pydantic config entities
- `src/utils/` — Helpers, logging, OpenCV utils
- `configs/` — All YAML configuration files
- `scripts/` — Training and inference entry points

**Inference Flow (already built in src/):**
1. YOLO detects pedestrian bounding boxes
2. RTMPose extracts 17 key-points per pedestrian
3. PoseFeatureGenerator builds 2-Channel 3D tensor
4. 3D-CNN classifies posture state
5. YOLOv11n detects phone in face/hand crop
6. DistractionFusion updates state machine
7. Face crop extracted on confirmed violation

---

## 1.3 Current Data Flow (Broken)

```
Camera → Frame → [src/ ML Pipeline] → Result
                                          ↓
                                ❌ No backend receives it
                                ❌ No database stores it
                                ❌ Frontend uses mock data only
```

---

## 1.4 All Issues Found

### Frontend Issues

| Issue | Detail |
|---|---|
| AppProvider never applied | AppContext defined but never wraps the app — global state is completely dead |
| useWebSocket is fake | Uses setInterval, never connects to any real WebSocket endpoint |
| All data is hardcoded | AlertsPanel, UsageMetrics, StatusCards all use static values |
| RealTimeMonitor commented out | Imported but commented out in Dashboard.jsx |
| Canvas size mismatch | Canvas fixed at 640×480, video is responsive — bounding boxes misalign at other sizes |
| Detection colors reversed | Green shown when phone IS detected — should be red for a safety system |
| No API service layer | No fetch or axios calls anywhere in the frontend codebase |
| node_modules committed | node_modules included in the RAR — should be in .gitignore |

### Model / Backend Issues

| Issue | Detail |
|---|---|
| app/ routes don't exist | app/api.py imports auth, detect, face, reports — none of these files exist |
| No WebSocket server | Frontend expects WebSocket stream, backend has none implemented |
| No database connection | No MongoDB connection anywhere in the codebase |
| Docker stubs are empty | docker/ folder exists but Dockerfile and docker-compose.yml are empty |
| MMPose dependencies commented out | mmpose and mmcv commented out in requirements — inference will fail |
| No .env or .env.example | Secrets and config paths completely undocumented |
| No inference endpoint | No /detect POST or /ws/stream WebSocket endpoint exists |
| Face saving incomplete | Face crops captured inside pipeline but never stored anywhere |
| No output schema enforced | Model output has no consistent Pydantic schema — random dict structure |
| No session state management | DistractionFusion state machine has no mechanism to persist state per camera session |
| Numpy types not serialized | Pipeline returns numpy float32 values — these crash JSON serialization |

### Integration Gaps

- No API contract defined between frontend and backend
- No WebSocket protocol defined
- No image/frame upload flow from frontend to model
- No real-time frame streaming implemented
- No alert creation logic connected to detections
- No training trigger from UI
- No progress feed from training to frontend
- No dashboard data flowing from MongoDB to frontend

---

## 1.5 What is Missing to Make the System Fully Functional

1. Complete FastAPI backend with all route files
2. MongoDB database with all collections, indexes, and seed data
3. WebSocket server for real-time frame streaming
4. Inference endpoint that wraps the existing src/ pipeline
5. Consistent DetectionResult Pydantic output schema
6. Session state management for DistractionFusion per camera session
7. Face embedding, deduplication, and MongoDB storage
8. Alert generation logic triggered by detections
9. Training API with background job and WebSocket log streaming
10. Dashboard APIs serving real aggregated data
11. Settings API with singleton MongoDB document
12. Frontend wired to all real APIs
13. .env configuration, Docker setup, CORS setup, README

---

---

# 🧠 PHASE 2 — System Architecture Plan

---

## 2.1 Golden Rule (MUST FOLLOW)

```
src/        ← ALL ML code lives here. NEVER touch or duplicate this.
backend/    ← API layer only. Imports from src/. No ML logic written here.
frontend/   ← React app. No logic changes, only wire to real APIs.
```

---

## 2.2 High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      FRONTEND (React + Vite)                     │
│  WebcamDetector ──WS──► Dashboard ──HTTP──► Alerts / Settings    │
└───────────────────────────┬─────────────────────────────────────┘
                            │ HTTP + WebSocket
┌───────────────────────────▼─────────────────────────────────────┐
│                     BACKEND (FastAPI)                            │
│  routes/ → services/ → ml/pipeline.py (bridge only)             │
│                              │                                   │
│                    imports from src/                             │
└──────────┬─────────────────────────────────┬────────────────────┘
           │                                 │
┌──────────▼──────────┐          ┌───────────▼──────────────────┐
│  src/ (ML Pipeline)  │          │       MongoDB                 │
│  Already built.      │          │  detections, alerts,          │
│  Just import it.     │          │  faces, settings,             │
└─────────────────────┘          │  devices, training_logs       │
                                  └──────────────────────────────┘
```

---

## 2.3 Correct Final Project Structure

```
PedestrianProject/                  ← Root (single repo)
│
├── src/                            ← ✅ YOUR EXISTING ML CODE — UNTOUCHED
│   ├── components/                 ← 16 ML modules — never modify
│   ├── pipeline/                   ← Inference + training pipelines
│   ├── serving/                    ← Predictor + schemas
│   ├── config/                     ← Config loaders
│   ├── entity/                     ← Pydantic entities
│   └── utils/                      ← ML utilities
│
├── configs/                        ← ✅ YOUR EXISTING YAML CONFIGS — UNTOUCHED
├── scripts/                        ← ✅ YOUR EXISTING SCRIPTS — UNTOUCHED
├── tests/                          ← ✅ YOUR EXISTING TESTS — UNTOUCHED
│
├── backend/                        ← 🆕 NEW — API layer only
│   ├── app/
│   │   ├── main.py                 ← FastAPI app, CORS, lifespan, router includes
│   │   ├── config.py               ← Reads .env via pydantic-settings
│   │   ├── database.py             ← Motor async MongoDB client + get_db()
│   │   │
│   │   ├── routes/                 ← HTTP + WebSocket handlers only, no logic
│   │   │   ├── detect.py           ← POST /api/detect, WS /ws/stream
│   │   │   ├── alerts.py           ← GET/POST/PATCH/DELETE alerts
│   │   │   ├── faces.py            ← GET/DELETE faces
│   │   │   ├── devices.py          ← GET devices
│   │   │   ├── dashboard.py        ← GET dashboard stats + timeline
│   │   │   ├── settings.py         ← GET/PUT settings
│   │   │   └── training.py         ← POST /api/train, WS /ws/train-logs
│   │   │
│   │   ├── services/               ← Business logic only, calls src/ and DB
│   │   │   ├── inference_service.py ← Decodes frame, calls pipeline, saves to DB
│   │   │   ├── alert_service.py    ← Creates alerts from DetectionResult
│   │   │   ├── face_service.py     ← Embedding, dedup, save/update face
│   │   │   ├── training_service.py ← Triggers src/ training, streams logs
│   │   │   └── dashboard_service.py ← MongoDB aggregation queries
│   │   │
│   │   ├── models/                 ← Pydantic API schemas only (request/response)
│   │   │   ├── detection.py        ← DetectionResult, PedestrianResult, BBox
│   │   │   ├── alert.py            ← AlertDocument, AlertResponse
│   │   │   ├── face.py             ← FaceDocument, FaceResponse
│   │   │   ├── device.py           ← DeviceDocument
│   │   │   └── settings.py         ← SettingsDocument
│   │   │
│   │   ├── ml/
│   │   │   └── pipeline.py         ← ⚠️ BRIDGE ONLY — imports from src/, zero ML logic
│   │   │
│   │   └── utils/
│   │       ├── image_utils.py      ← Base64 decode/encode, numpy convert, resize
│   │       ├── embedding_utils.py  ← Face embedding generation (cosine similarity)
│   │       └── response_utils.py   ← Standard API response builder
│   │
│   └── seeds/
│       └── seed_settings.py        ← Inserts default settings on first run
│
├── frontend/                       ← ✅ YOUR EXISTING FRONTEND — minimal changes
│
├── .env                            ← 🆕 All secrets and config paths
├── .env.example                    ← 🆕 Template for other developers
├── requirements.txt                ← Updated with backend + ML dependencies
├── docker-compose.yml              ← 🆕 FastAPI + MongoDB services
└── Dockerfile                      ← 🆕 Backend container
```

---

## 2.4 Module Responsibilities (Crystal Clear)

| Module | Responsibility | What it MUST NOT do |
|---|---|---|
| `routes/` | Receive request, validate input, call service, return response | Contain business logic or DB queries |
| `services/` | Business logic, DB operations, calls src/ pipeline | Contain FastAPI-specific code or ML code |
| `ml/pipeline.py` | Import and expose src/ pipeline to FastAPI | Contain any ML logic, rewrite anything from src/ |
| `models/` | Define Pydantic input/output schemas | Contain logic of any kind |
| `utils/` | Pure helper functions | Import from routes, services, or models |
| `src/` | All ML logic | Be modified by backend development |

---

## 2.5 Data Flow Diagram (Full System)

```
FRONTEND — WebcamDetector
  [1] Captures frame from webcam every 500ms
  [2] Converts frame to JPEG Base64 string
  [3] Sends JSON over WebSocket: { frame, timestamp, session_id }
        │
        ▼
BACKEND — /ws/stream (routes/detect.py)
  [4] Receives WebSocket message
  [5] Passes to inference_service.process_frame(frame_b64, session_id)
        │
        ▼
BACKEND — inference_service.py
  [6] Decodes Base64 → BGR numpy array
  [7] Calls ml/pipeline.py → which calls src/pipeline/InferencePipeline.run(frame)
        │
        ▼
src/ — InferencePipeline.run(frame)
  [8]  YOLO → pedestrian bounding boxes
  [9]  RTMPose → 17 key-points per pedestrian
  [10] PoseFeatureGenerator → 2-Ch 3D tensor
  [11] 3D-CNN → posture_state + posture_confidence
  [12] YOLOv11n → phone_detected + phone_confidence
  [13] DistractionFusion → fusion_state (per session)
  [14] Face crop extracted if fusion_state == USING
  [15] Returns raw result dict
        │
        ▼
BACKEND — inference_service.py (continues)
  [16] Converts raw dict → DetectionResult Pydantic model
  [17] Converts all numpy types → Python native types
  [18] Saves DetectionResult to MongoDB detections collection
  [19] If is_violation=True → alert_service.create_alert(result)
  [20] If is_violation=True → face_service.process_face(face_crop, result)
  [21] Serializes DetectionResult to JSON
  [22] Sends JSON back over WebSocket to frontend
        │
        ▼
FRONTEND — receives WebSocket message
  [23] Parses DetectionResult JSON
  [24] Draws bounding boxes on canvas using pedestrians[].bbox
  [25] Updates detection status text and color
  [26] Shows red alert overlay if is_violation=True
  [27] Dashboard polls /api/dashboard/stats every 5 seconds
  [28] AlertsPanel polls /api/alerts every 5 seconds
```

---

---

# ⚙️ PHASE 3 — Step-by-Step Implementation Plan

---

## Step 1 — Backend Project Setup

**What to do:**
- Create the `backend/` folder with the exact structure shown in Phase 2.3
- Create `backend/app/main.py` — initialize FastAPI app, add CORS middleware, include all routers, add lifespan event handler for model loading and DB connection
- Create `backend/app/config.py` — use pydantic-settings to read all values from `.env`
- Create `.env` at project root with these variables: `MONGODB_URI`, `DB_NAME`, `MODEL_WEIGHTS_DIR`, `POSTURE_MODEL_PATH`, `PHONE_MODEL_PATH`, `SECRET_KEY`, `CORS_ORIGINS`, `DEVICE` (cpu/cuda), `CONFIDENCE_THRESHOLD`, `LOG_LEVEL`
- Create `.env.example` with placeholder values for every variable in `.env`
- Update `requirements.txt` at project root — add: fastapi, uvicorn[standard], motor, pydantic-settings, python-multipart, python-dotenv, insightface (or facenet-pytorch), websockets
- Uncomment mmpose and mmcv in requirements.txt
- Create `Dockerfile` — Python 3.10 base, copy project root, install requirements, expose port 8000, start uvicorn
- Create `docker-compose.yml` — two services: `backend` (Dockerfile) and `mongodb` (mongo:6 image) with a volume for data persistence

**Why needed:**
Clean scaffold prevents circular imports, missing config errors, and CORS failures that will block all frontend calls from day one.

**Expected output:**
Running `uvicorn backend.app.main:app --reload` from project root starts server at `localhost:8000`. `/docs` shows empty Swagger UI. `/health` returns `{ "status": "ok" }`.

**Common mistakes to avoid:**
- Do NOT put `.env` inside the `backend/` folder — put it at project root so both `src/` and `backend/` can read it
- Do NOT forget to add `backend/` to Python path — either via `PYTHONPATH=.` or a `pyproject.toml`
- Do NOT hardcode any path, secret, or threshold — everything goes in `.env`
- Do NOT forget `allow_origins`, `allow_methods`, `allow_headers` in CORS — frontend will be blocked silently

---

## Step 2 — Database Setup (MongoDB)

**What to do:**
- Create `backend/app/database.py` — initialize Motor async client using `MONGODB_URI` from config, expose a `get_db()` async dependency function, open/close client inside FastAPI lifespan
- Create all 6 MongoDB collections: `detections`, `alerts`, `faces`, `settings`, `devices`, `training_logs`
- Create indexes (exact indexes defined in Phase 7)
- Create `backend/seeds/seed_settings.py` — checks if settings document exists, if not inserts the full default settings document
- Call seed script inside the FastAPI lifespan startup event after DB connection is confirmed

**Why needed:**
Motor is non-blocking async — critical for FastAPI performance. Indexes on timestamp and is_violation fields are essential for dashboard query speed. Without the settings seed, the settings API will return empty on first run.

**Expected output:**
After startup, MongoDB has 6 collections. `settings` collection has exactly 1 document with all default values. Indexes exist and can be verified via MongoDB Compass.

**Common mistakes to avoid:**
- Do NOT use `pymongo` (synchronous) — it blocks the event loop and kills performance under concurrent WebSocket connections
- Do NOT skip indexes — a full collection scan on 100,000 detections will take seconds per dashboard request
- Do NOT open a new MongoDB connection per request — open once in lifespan, reuse everywhere

---

## Step 3 — Model Integration (BRIDGE ONLY)

**What to do:**
- Create `backend/app/ml/pipeline.py` — this file does ONE thing only: imports the existing `InferencePipeline` from `src/pipeline/` and exposes a `load_pipeline()` function
- `load_pipeline()` reads model paths from `config.py`, instantiates `InferencePipeline` from `src/`, and returns the ready-to-use pipeline object
- In `backend/app/main.py` lifespan startup event: call `load_pipeline()` and store the result in `app.state.pipeline`
- Add a `GET /api/health` endpoint that returns model load status from `app.state`
- Verify that `src/pipeline/InferencePipeline` has a `run(frame: np.ndarray)` method that returns a result — if it returns a raw dict, note this for Step 4

**Why needed:**
Models must be loaded exactly once at startup. Loading per request adds 5–15 seconds per detection call. `app.state` is FastAPI's correct mechanism for storing app-wide shared objects.

**Expected output:**
`GET /api/health` returns `{ "status": "ok", "model_loaded": true, "device": "cpu" }`. Models load in under 15 seconds at startup.

**Common mistakes to avoid:**
- Do NOT write any ML logic inside `backend/app/ml/pipeline.py` — it is an import bridge only
- Do NOT load models in route handlers or service files
- Do NOT use global Python variables for model state — use `app.state` only
- Do NOT modify any file inside `src/` during this step

---

## Step 4 — Detection API Logic

**What to do:**

**Part A — Define DetectionResult schema** in `backend/app/models/detection.py`:
Define these Pydantic models: `BoundingBox` (x1, y1, x2, y2 absolute + x1_norm, y1_norm, x2_norm, y2_norm normalized), `FaceRegion` (x1, y1, x2, y2), `PedestrianResult` (pedestrian_id, bbox, posture_state, posture_confidence, phone_detected, phone_confidence, fusion_state, face_region, is_violation), `DetectionResult` (detection_id, timestamp, session_id, frame_id, is_violation, overall_confidence, processing_time_ms, pedestrians list, error_message)

**Part B — Create inference_service.py** in `backend/app/services/`:
This service receives a Base64 frame string and session_id. It: decodes Base64 to numpy BGR array, records frame dimensions, calls `app.state.pipeline.run(frame)`, receives raw output from src/, maps raw output into `DetectionResult` Pydantic model, converts ALL numpy types to Python native types (critical), computes normalized bbox coordinates, saves DetectionResult to MongoDB `detections` collection, calls `alert_service` if is_violation is True, calls `face_service` if is_violation is True, returns the DetectionResult object.

**Part C — Create routes/detect.py** with two endpoints:
- `POST /api/detect` — accepts `{ image: base64_string, session_id: string }`, calls inference_service, returns DetectionResult JSON. Used for single image testing.
- `WebSocket /ws/stream` — accepts continuous frames, loops receiving messages, calls inference_service per frame, sends DetectionResult JSON back, handles disconnect gracefully with try/except.

**Why needed:**
The POST endpoint allows testing detection without a camera. The WebSocket is what the live frontend camera feed uses. Both must exist and use the same inference_service.

**Expected input (WebSocket):**
```
{ "frame": "<base64_jpeg_string>", "timestamp": "2024-01-01T12:00:00Z", "session_id": "cam_01" }
```

**Expected output (DetectionResult JSON):**
```
{
  "detection_id": "uuid-string",
  "timestamp": "2024-01-01T12:00:00Z",
  "session_id": "cam_01",
  "is_violation": true,
  "overall_confidence": 0.92,
  "processing_time_ms": 85,
  "pedestrians": [
    {
      "pedestrian_id": "uuid-string",
      "bbox": { "x1": 100, "y1": 80, "x2": 280, "y2": 420,
                "x1_norm": 0.15, "y1_norm": 0.16, "x2_norm": 0.43, "y2_norm": 0.87 },
      "posture_state": "USING",
      "posture_confidence": 0.88,
      "phone_detected": true,
      "phone_confidence": 0.94,
      "fusion_state": "USING",
      "face_region": { "x1": 110, "y1": 80, "x2": 200, "y2": 170 },
      "is_violation": true
    }
  ]
}
```

**Common mistakes to avoid:**
- Do NOT return numpy float32, numpy int64, or numpy arrays in JSON — call `float()` and `int()` on every numeric value
- Do NOT block the WebSocket loop with synchronous DB writes — all DB operations must be async (await)
- Do NOT crash the entire WebSocket connection on one bad frame — wrap frame processing in try/except and send an error response instead
- Do NOT process frames faster than the model can handle — add a frame rate limiter or process every Nth frame

---

## Step 5 — Face Cropping + Duplicate Prevention

**What to do:**

**Part A — Face Cropping:**
- The face crop already happens inside `src/` pipeline when fusion_state == USING
- The `DetectionResult` will carry the `face_region` bbox
- In `inference_service.py`, after building DetectionResult: if is_violation is True, extract the face crop from the original numpy frame using `face_region` coordinates
- Pass face crop numpy array to `face_service.process_face(crop, detection_result)`

**Part B — Create face_service.py** in `backend/app/services/`:
This service: resizes face crop to 160×160, generates a 128-dimensional embedding using InsightFace or FaceNet (use `embedding_utils.py`), queries MongoDB `faces` collection for all existing embeddings, computes cosine similarity between new embedding and each stored one, applies this decision logic:
- If max similarity > 0.85 → same person → update existing document: increment detection_count, increment violation_count if is_violation, update last_seen, append detection_id to detection_ids array
- If max similarity ≤ 0.85 → new person → insert new face document with all fields

**Part C — Create routes/faces.py:**
- `GET /api/faces` — returns paginated list of all faces with metadata
- `DELETE /api/faces/{face_id}` — removes a face record

**Why needed:**
Without deduplication, the same pedestrian crossing the road 20 times creates 20 unrelated records. Embedding similarity ensures one person = one record with consolidated history.

**Expected face document saved to MongoDB:**
```
{
  "face_id": "uuid-string",
  "embedding": [0.12, -0.44, 0.33, ...],   // 128 floats
  "image_base64": "...",
  "first_seen": "ISO8601",
  "last_seen": "ISO8601",
  "detection_count": 4,
  "violation_count": 3,
  "detection_ids": ["uuid1", "uuid2", "uuid3", "uuid4"]
}
```

**Common mistakes to avoid:**
- Do NOT store raw binary image data directly in MongoDB — always use Base64 string or GridFS
- Do NOT run cosine similarity in a Python loop over thousands of faces — limit by querying only faces seen in the last 30 days, or use a vector search index
- Do NOT hardcode the similarity threshold 0.85 — read it from the settings document in MongoDB so operators can tune it
- Do NOT generate embedding on every frame — only when fusion_state == USING is confirmed

---

## Step 6 — Alert System

**What to do:**

**Create alert_service.py** in `backend/app/services/`:
This service receives a `DetectionResult` and applies this logic:
- If `is_violation=False` → do nothing, return
- Check if an unresolved alert already exists for this `face_id` within the last 60 seconds — if yes → skip (prevents duplicate alerts for same person same crossing)
- Determine severity:
  - `posture_confidence > 0.90` AND `phone_detected=True` AND `phone_confidence > 0.90` → HIGH
  - `posture_confidence > 0.75` AND `phone_detected=True` → MEDIUM
  - `fusion_state == SUSPICIOUS` → LOW
- Create alert document and insert into `alerts` collection

**Create routes/alerts.py:**
- `GET /api/alerts` — paginated, query params: `severity`, `resolved` (bool), `from_date`, `to_date`, `limit`, `page`
- `PATCH /api/alerts/{alert_id}/resolve` — sets resolved=True, resolved_at=now()
- `DELETE /api/alerts/{alert_id}` — removes alert

**Call alert_service** from `inference_service.py` after every detection where is_violation is True.

**Why needed:**
Alerts are the primary actionable output operators see. Without the dedup check, a 3-second crossing at 2 frames/second creates 6 identical alerts for the same person.

**Expected alert document:**
```
{
  "alert_id": "uuid-string",
  "detection_id": "uuid-string",
  "face_id": "uuid-string",
  "severity": "HIGH",
  "title": "Cell Phone Usage Detected",
  "description": "Pedestrian confirmed using cell phone while crossing road",
  "confidence": 0.92,
  "timestamp": "ISO8601",
  "resolved": false,
  "resolved_at": null,
  "snapshot_base64": "..."
}
```

**Common mistakes to avoid:**
- Do NOT create an alert for every single frame of the same violation — implement the 60-second dedup window
- Do NOT skip the confidence threshold — detections below threshold should never trigger alerts
- Do NOT block inference_service while saving the alert — use `asyncio.create_task()` to fire-and-forget the alert save

---

## Step 7 — Training Module (UI-Based)

**What to do:**

**Create training_service.py** in `backend/app/services/`:
This service: validates no training job is currently running (check `training_logs` collection for status=running), creates a new training job document in `training_logs` with status=running, launches training in a background thread using `asyncio.create_subprocess_exec` or `concurrent.futures.ThreadPoolExecutor`, calls the appropriate script from `scripts/` (already built in your repo), captures stdout/stderr line by line, writes each log line to the `training_logs` document in MongoDB, broadcasts each log line to all connected `/ws/train-logs` WebSocket clients, on completion: updates status=completed, saves final metrics, on error: updates status=error, saves error message.

**Create routes/training.py:**
- `POST /api/train` — body: `{ model_type: "posture_classifier"|"phone_detector", epochs: int, learning_rate: float, batch_size: int }` — starts training, returns `{ job_id, status: "started" }`
- `GET /api/train/status` — returns `{ status, current_epoch, total_epochs, job_id, started_at }`
- `GET /api/train/history` — returns list of past training jobs with metrics
- `DELETE /api/train/current` — cancels running training job
- `WebSocket /ws/train-logs` — streams log lines as they are generated

**Training log message format (WebSocket):**
```
{ "type": "progress", "epoch": 5, "total_epochs": 50, "loss": 0.312, "accuracy": 0.78 }
{ "type": "log", "message": "Epoch 5/50 complete" }
{ "type": "complete", "metrics": { "final_accuracy": 0.9178, "final_loss": 0.21 } }
{ "type": "error", "message": "CUDA out of memory" }
```

**Why needed:**
Training from UI allows operators to retrain models with new data without touching code. This is a core MLOps requirement and a strong FYP feature.

**Common mistakes to avoid:**
- Do NOT run training in the main FastAPI event loop — it will freeze all API responses for hours
- Do NOT lose logs if the WebSocket disconnects — always persist every log line to MongoDB so frontend can reload history
- Do NOT allow two training jobs simultaneously — the GPU cannot handle two training runs and results will be corrupted
- Do NOT forget to update model path in settings after training completes so inference_service uses the new model

---

## Step 8 — Dashboard APIs

**What to do:**

**Create dashboard_service.py** in `backend/app/services/`:
This service runs these MongoDB aggregation queries:
- Total detections today: count `detections` where timestamp >= today midnight
- Total violations today: count `detections` where is_violation=True and timestamp >= today
- Unresolved alert count: count `alerts` where resolved=False
- Unique faces detected today: count distinct face_ids in today's violations
- Hourly detection counts: group `detections` by hour for last 24 hours (for chart)
- Hourly violation counts: same but filtered by is_violation=True (second chart line)
- Recent alerts: last 5 unresolved alerts sorted by timestamp desc

**Create routes/dashboard.py:**
- `GET /api/dashboard/stats` — returns summary stats object
- `GET /api/dashboard/timeline` — returns hourly arrays for last 24 hours

**Expected response for /api/dashboard/stats:**
```
{
  "active_devices": 1,
  "detections_today": 47,
  "violations_today": 8,
  "unresolved_alerts": 3,
  "unique_violators_today": 6,
  "system_status": "online",
  "model_status": "loaded",
  "last_updated": "ISO8601"
}
```

**Expected response for /api/dashboard/timeline:**
```
{
  "labels": ["00:00", "01:00", ..., "23:00"],
  "detections": [0, 0, 2, 5, 12, ...],
  "violations": [0, 0, 0, 1, 3, ...]
}
```

**Why needed:**
The frontend Dashboard StatusCards and UsageMetrics chart are entirely hardcoded. These APIs replace static values with real aggregated data from the database.

**Common mistakes to avoid:**
- Do NOT run a full collection scan per stat — use aggregation pipelines with date range filters and covered indexes
- Do NOT forget to cache results for 5 seconds — dashboard polls every 5 seconds, no need to re-aggregate on every call
- Do NOT return null for any field — always return 0 for numeric fields if no data exists yet

---

## Step 9 — Settings APIs

**What to do:**

**Create routes/settings.py:**
- `GET /api/settings` — returns the single settings document
- `PUT /api/settings` — validates and updates the settings document (upsert pattern — always update the same single document)

**Settings document fields (all configurable at runtime):**
```
detection_confidence_threshold: 0.75
alert_confidence_threshold: 0.80
face_similarity_threshold: 0.85
alert_dedup_window_seconds: 60
auto_refresh_interval_ms: 5000
notifications_enabled: true
usage_threshold_percent: 80
active_model: "posture_classifier_v1"
camera_resolution: "640x480"
max_concurrent_cameras: 4
frame_sample_rate: 2
updated_at: ISO8601
```

**Why needed:**
All thresholds read by alert_service, face_service, and inference_service must come from this settings document — not from .env or hardcoded values. This allows operators to tune sensitivity without redeploying. The frontend Settings page saves and loads from here.

**Common mistakes to avoid:**
- Do NOT read thresholds from `.env` in services — `.env` is for infrastructure config (DB URI, ports). Runtime thresholds belong in MongoDB settings
- Do NOT allow the settings collection to grow beyond 1 document — always upsert the same document
- Do NOT forget to update `updated_at` on every PUT request

---

## Step 10 — Frontend Integration

**What to do (minimal changes — no design modifications):**

**Change 1 — Apply AppProvider:**
In `frontend/src/main.jsx`, wrap `<App />` with `<AppProvider>`. This activates the global state that is already defined but currently dead.

**Change 2 — Fix useWebSocket hook:**
Replace the setInterval simulation with a real WebSocket connection to `ws://localhost:8000/ws/stream`. The hook should: open connection when camera starts, send Base64 frames on a configurable interval, receive DetectionResult JSON, parse it, and dispatch to AppContext via `ADD_DETECTION` action.

**Change 3 — Create src/services/api.js:**
Create a single API service file that exports all fetch functions. Every API call in the app goes through this file. Functions to include: `getDashboardStats()`, `getDashboardTimeline()`, `getAlerts(params)`, `resolveAlert(id)`, `getSettings()`, `updateSettings(data)`, `getDevices()`, `getFaces()`, `deleteFace(id)`, `startTraining(config)`, `getTrainingStatus()`.

**Change 4 — Wire Dashboard:**
In Dashboard.jsx: fetch `/api/dashboard/stats` on mount and every 5 seconds, update StatusCards with real values, fetch `/api/dashboard/timeline` for chart data, uncomment `<RealTimeMonitor />` and `<ActivityFeed />`.

**Change 5 — Wire AlertsPanel:**
Replace hardcoded alerts array with a `GET /api/alerts` fetch on mount and every 5 seconds. Map API response to existing AlertItem component props (no design change).

**Change 6 — Fix WebcamDetector canvas:**
Replace fixed `width={640} height={480}` with `videoRef.current.videoWidth` and `videoRef.current.videoHeight` after video loads. Draw bounding boxes using `pedestrians[].bbox` from DetectionResult — use normalized coordinates scaled to canvas dimensions.

**Change 7 — Fix detection colors:**
Change the detection status indicator logic: `phone_detected=true` OR `is_violation=true` → red/danger color. `is_violation=false` → green/safe color.

**Change 8 — Wire Settings page:**
On mount, fetch `GET /api/settings`. On save button click, call `PUT /api/settings` with updated values.

**Change 9 — Wire Devices page:**
Fetch `GET /api/devices` on mount. Map device objects to existing DeviceList component.

**Change 10 — Add Training UI to Settings:**
Add a training section inside the existing Settings page (no new page). Include: model selector dropdown, epochs/learning_rate inputs, Start Training button (calls `POST /api/train`), progress bar and log output area (connects to `WS /ws/train-logs`).

**Frontend → Backend API mapping:**

| Frontend Action | API Endpoint |
|---|---|
| Dashboard loads | GET /api/dashboard/stats |
| Chart data loads | GET /api/dashboard/timeline |
| AlertsPanel loads | GET /api/alerts |
| Resolve alert button | PATCH /api/alerts/{id}/resolve |
| Camera starts | WS /ws/stream (connect) |
| Frame captured | WS /ws/stream (send frame) |
| Detection result received | WS /ws/stream (receive DetectionResult) |
| Settings page loads | GET /api/settings |
| Settings save button | PUT /api/settings |
| Devices page loads | GET /api/devices |
| Faces page loads | GET /api/faces |
| Delete face button | DELETE /api/faces/{id} |
| Start Training button | POST /api/train |
| Training logs view | WS /ws/train-logs |
| Training status check | GET /api/train/status |

**Common mistakes to avoid:**
- Do NOT change any component design, colors, or layout — only wire data
- Do NOT fetch data inside individual components — use the api.js service file exclusively
- Do NOT use raw WebSocket in components — only use the `useWebSocket` hook
- Do NOT forget to handle loading and error states for every API call

---
---

# 🎨 FRONTEND STYLING & COLOR PROMPT
### Paste this into your `requirements_rules.md` at the end of Phase 8 (Frontend Integration Plan)

---

---

## PHASE 12 — Frontend Visual Design System (MANDATORY)

> ⚠️ This phase is NOT optional. Every color, font, and style decision is defined here. The coding agent MUST follow this exactly. Do NOT use default Tailwind colors (blue-600, gray-200, etc.) for primary UI elements. Use CSS variables defined below.

---

### 12.1 — Design Personality

This dashboard is a **real-time AI safety system** used by operators to monitor pedestrian violations. The visual language must feel:

- **Authoritative** — this is serious surveillance software, not a toy app
- **High-tech** — dark theme with glowing cyan accents, like a command center
- **Alive** — pulsing indicators, animated status dots, smooth transitions
- **Readable** — every number, label, and status must be instantly legible at a glance
- **Wow factor** — when someone opens this dashboard for the first time, they say "wow"

The closest reference: **Tesla Autopilot dashboard + Vercel Analytics + Grafana dark theme**. Clean, dark, electric, professional.

---

### 12.2 — Global CSS Variables

Add ALL of these to `frontend/src/index.css` inside `:root {}`. Replace the current minimal CSS completely.

```css
@import "tailwindcss";

:root {
  /* ── PAGE BACKGROUNDS ── */
  --bg-page:        #0B1120;   /* darkest — main page background */
  --bg-primary:     #0F1724;   /* sidebar, header */
  --bg-secondary:   #1A2535;   /* cards, panels */
  --bg-surface:     #243044;   /* inputs, hover states, table rows */
  --bg-elevated:    #2D3B52;   /* dropdowns, tooltips, modals */

  /* ── ACCENT COLORS ── */
  --accent-cyan:    #00D4FF;   /* PRIMARY — active nav, primary buttons, chart lines */
  --accent-cyan-dim:#00D4FF22; /* cyan with low opacity — borders, backgrounds */
  --accent-teal:    #00B4A0;   /* SECONDARY — success actions, badges */
  --accent-blue:    #3B82F6;   /* INFO — info states, links */

  /* ── STATUS / SEMANTIC COLORS ── */
  --color-danger:   #FF3B3B;   /* violation detected, high alerts, stop button */
  --color-danger-dim:#FF3B3B22;/* danger with low opacity */
  --color-warning:  #F59E0B;   /* suspicious state, medium alerts, pause button */
  --color-warning-dim:#F59E0B22;
  --color-success:  #10B981;   /* safe/clear state, low alerts, system online */
  --color-success-dim:#10B98122;
  --color-neutral:  #6B7A99;   /* disabled, inactive, hints */

  /* ── TEXT COLORS ── */
  --text-primary:   #F1F5F9;   /* headings, important values, active labels */
  --text-secondary: #94A3B8;   /* descriptions, subtitles, labels */
  --text-muted:     #475569;   /* timestamps, hints, disabled text */

  /* ── BORDERS ── */
  --border-default: #1E2D42;   /* all card/panel borders */
  --border-hover:   #2D4060;   /* hover state borders */
  --border-accent:  #00D4FF33; /* focused inputs, highlighted cards */
  --border-danger:  #FF3B3B44; /* alert cards, violation panels */

  /* ── TYPOGRAPHY ── */
  --font-display: 'DM Sans', sans-serif;
  --font-mono:    'JetBrains Mono', monospace;

  /* ── ELEVATION / SHADOWS ── */
  --shadow-card:  0 4px 24px rgba(0, 0, 0, 0.4);
  --shadow-glow:  0 0 24px rgba(0, 212, 255, 0.10);
  --shadow-danger:0 0 24px rgba(255, 59, 59, 0.15);

  /* ── SPACING / RADII ── */
  --radius-sm:  6px;
  --radius-md:  10px;
  --radius-lg:  14px;
  --radius-xl:  20px;
}

/* ── BASE STYLES ── */
* { box-sizing: border-box; }

body {
  background-color: var(--bg-page);
  color: var(--text-primary);
  font-family: var(--font-display);
  font-size: 14px;
  line-height: 1.6;
  -webkit-font-smoothing: antialiased;
}

/* ── SCROLLBAR ── */
::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: var(--bg-primary); }
::-webkit-scrollbar-thumb { background: var(--bg-elevated); border-radius: 10px; }
::-webkit-scrollbar-thumb:hover { background: var(--accent-cyan); }

/* ── SELECTION ── */
::selection { background: var(--accent-cyan); color: var(--bg-primary); }
```

---

### 12.3 — Fonts (Add to `index.html` `<head>`)

```html
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=DM+Sans:ital,opsz,wght@0,9..40,300;0,9..40,400;0,9..40,500;0,9..40,600;0,9..40,700&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
```

---

### 12.4 — Component Color Rules

The coding agent MUST follow these rules for every component:

**Cards & Panels:**
```
background:    var(--bg-secondary)
border:        1px solid var(--border-default)
border-radius: var(--radius-lg)
box-shadow:    var(--shadow-card)
hover border:  var(--border-hover)
```

**Sidebar:**
```
background:         var(--bg-primary)
active item bg:     var(--bg-surface)
active item border: 3px solid var(--accent-cyan)  ← left side only
active item text:   var(--accent-cyan)
inactive text:      var(--text-secondary)
hover text:         var(--text-primary)
hover bg:           var(--bg-surface)
```

**Header:**
```
background: var(--bg-primary)
border-bottom: 1px solid var(--border-default)
text: var(--text-primary)
```

**Buttons:**

| Type | Background | Text | Border | Hover |
|---|---|---|---|---|
| Primary | `var(--accent-cyan)` | `var(--bg-primary)` | none | brightness 110% |
| Secondary | `var(--bg-surface)` | `var(--text-primary)` | `var(--border-default)` | `var(--border-hover)` |
| Danger | `var(--color-danger)` | white | none | brightness 110% |
| Warning | `var(--color-warning)` | white | none | brightness 110% |
| Success | `var(--color-success)` | white | none | brightness 110% |
| Ghost | transparent | `var(--text-secondary)` | `var(--border-default)` | text becomes `var(--accent-cyan)` |

**Status Indicators (pulsing dot):**
```
Online  → color: var(--color-success)  + CSS pulse animation
Offline → color: var(--color-danger)   + no pulse
Warning → color: var(--color-warning)  + slow pulse
```

**Pulsing animation (add to index.css):**
```css
@keyframes pulse-dot {
  0%, 100% { opacity: 1; transform: scale(1); }
  50%       { opacity: 0.5; transform: scale(1.4); }
}
.pulse { animation: pulse-dot 1.8s ease-in-out infinite; }
```

**Detection Status Colors in WebcamDetector:**
```
is_violation = true  → bg: var(--color-danger-dim),  border: var(--color-danger),  text: var(--color-danger)
fusion = SUSPICIOUS  → bg: var(--color-warning-dim), border: var(--color-warning), text: var(--color-warning)
safe / no detection  → bg: var(--color-success-dim), border: var(--color-success), text: var(--color-success)
camera off           → bg: var(--bg-surface),         border: var(--border-default), text: var(--text-muted)
```

**Bounding Box Colors on Canvas:**
```
USING / Violation  → stroke: #FF3B3B,  label: "VIOLATION"
SUSPICIOUS         → stroke: #F59E0B,  label: "SUSPICIOUS"
NOT_USING / Safe   → stroke: #10B981,  label: "SAFE"
Face region        → stroke: #F1F5F9,  label: "FACE"  (dashed line)
```

**Alert Severity Colors:**
```
HIGH   → left-border: var(--color-danger),  bg: var(--color-danger-dim)
MEDIUM → left-border: var(--color-warning), bg: var(--color-warning-dim)
LOW    → left-border: var(--accent-blue),   bg: rgba(59,130,246,0.08)
```

**Chart (UsageMetrics):**
```
Line 1 (Detections): color: var(--accent-cyan)
Line 2 (Violations): color: var(--color-danger)
Grid lines:          color: var(--border-default)
Labels:              color: var(--text-muted)
Background:          var(--bg-secondary)
Tooltip bg:          var(--bg-elevated)
```

**Status Cards (top 4 on Dashboard):**
```
background:   var(--bg-secondary)
border:       1px solid var(--border-default)
top-accent:   2px solid [status color]  ← colored top border per card type
value color:  var(--text-primary)
label color:  var(--text-secondary)
icon bg:      [status color at 10% opacity]
icon color:   [status color]

Card status color mapping:
  "active"  → var(--color-success)
  "warning" → var(--color-danger)
  "info"    → var(--accent-cyan)
  "neutral" → var(--accent-blue)
```

**Inputs & Form Fields:**
```
background:    var(--bg-surface)
border:        1px solid var(--border-default)
border-radius: var(--radius-sm)
color:         var(--text-primary)
placeholder:   var(--text-muted)
focus-border:  var(--accent-cyan)
focus-shadow:  0 0 0 3px var(--accent-cyan-dim)
height:        40px
padding:       0 12px
```

**Toggle Switches:**
```
active background:   var(--accent-cyan)
inactive background: var(--bg-elevated)
thumb color:         white
```

**Table Rows:**
```
header bg:       var(--bg-surface)
header text:     var(--text-secondary)  font-size: 11px  letter-spacing: 0.05em  uppercase
row bg:          var(--bg-secondary)
row hover:       var(--bg-surface)
row border:      1px solid var(--border-default)
row text:        var(--text-primary)
```

---

### 12.5 — Typography Rules

```
Page Title (h1):    font-size: 24px,  font-weight: 600, color: var(--text-primary)
Section Title (h2): font-size: 16px,  font-weight: 600, color: var(--text-primary)
Card Title (h3):    font-size: 14px,  font-weight: 500, color: var(--text-primary)
Body text:          font-size: 14px,  font-weight: 400, color: var(--text-secondary)
Small label:        font-size: 12px,  font-weight: 400, color: var(--text-muted)
Stat number:        font-size: 28px,  font-weight: 600, color: var(--text-primary)
Monospace data:     font-family: var(--font-mono), font-size: 12px
Badge/chip:         font-size: 11px,  font-weight: 500, letter-spacing: 0.04em
```

---

### 12.6 — Micro-Interactions (Add to index.css)

```css
/* smooth all transitions */
*, *::before, *::after {
  transition:
    background-color 0.15s ease,
    border-color     0.15s ease,
    color            0.15s ease,
    box-shadow       0.15s ease,
    transform        0.1s ease;
}

/* card hover lift */
.card-hover:hover {
  transform: translateY(-2px);
  box-shadow: var(--shadow-card), var(--shadow-glow);
  border-color: var(--border-hover);
}

/* button active press */
button:active { transform: scale(0.97); }

/* cyan glow on focused inputs */
input:focus, select:focus, textarea:focus {
  outline: none;
  border-color: var(--accent-cyan);
  box-shadow: 0 0 0 3px var(--accent-cyan-dim);
}
```

---

### 12.7 — "Wow Factor" Details (Do NOT Skip These)

These small details are what make the UI memorable. Agent MUST implement all of them:

**1. Live pulsing dot on System Online status in Header**
Small circle, `var(--color-success)`, with the `pulse` CSS animation. Tells operator the system is alive at a glance.

**2. Gradient top border on active sidebar item**
Use `border-left: 3px solid var(--accent-cyan)` on the active nav item. Combined with the dark background it creates a "selected beam" effect.

**3. Glowing camera feed border when violation detected**
When `is_violation = true` is received over WebSocket, add `box-shadow: 0 0 0 3px var(--color-danger), var(--shadow-danger)` to the camera container. Remove it when clear. Makes the violation visually "alarm" the screen.

**4. Colored top accent line on Status Cards**
Each StatusCard has a `2px` colored top border (`border-top`) using the card's status color. Gives each card a unique identity without being loud.

**5. Monospace font for all numeric data**
FPS, confidence scores, timestamps, coordinates — all use `var(--font-mono)`. Makes data feel precise and technical.

**6. Sidebar brand area**
At the very top of the sidebar, above nav items:
```
[🔵 glow icon]  PDS
                AI Monitor
```
"PDS" in `var(--accent-cyan)` at 18px bold. "AI Monitor" in `var(--text-muted)` at 11px. Creates instant brand identity.

**7. Chart gradient fill**
In UsageMetrics chart, fill below the detection line with a gradient from `rgba(0,212,255,0.15)` at the line down to `rgba(0,212,255,0)` at the bottom. Makes the chart feel rich and full.

**8. Training log terminal look**
The training log textarea uses `var(--bg-primary)` (darkest background) + `var(--font-mono)` + `12px` text. Feels like a real terminal. Add a blinking `▌` cursor CSS animation at the end of the last log line.

---

### 12.8 — Strict Rules for the Agent

```
❌ NEVER use raw hex colors in component JSX — always use CSS variables
❌ NEVER use Tailwind's default color scale (blue-600, gray-200, red-500, etc.)
    for primary UI — only for Tailwind utilities like padding, margin, flex
❌ NEVER use white (#fff) as a background color — use var(--bg-secondary) or above
❌ NEVER use black (#000) as text — use var(--text-primary) minimum
❌ NEVER use font-weight 300 for anything interactive — minimum 400
✅ ALWAYS use var(--font-mono) for: FPS, confidence scores, bbox coords, timestamps
✅ ALWAYS use var(--color-danger) for phone detected — never green
✅ ALWAYS add 1px solid var(--border-default) to every card and panel
✅ ALWAYS test that text is readable on dark background before committing
```

---


---

# 🤖 PHASE 4 — ML + API Integration Plan

---

## 4.1 How Model Output Connects to Backend

The connection is clean and one-directional:

```
backend/app/ml/pipeline.py
  ↓ imports
src/pipeline/InferencePipeline
  ↓ exposes
.run(frame: np.ndarray) → raw result dict
  ↓ received by
backend/app/services/inference_service.py
  ↓ converts to
DetectionResult (Pydantic model)
  ↓ consumed by
routes, alert_service, face_service, MongoDB
```

`src/` code is never modified. Never called directly from routes. Always goes through `inference_service.py`.

## 4.2 Session State for DistractionFusion

The DistractionFusion state machine tracks state per pedestrian across frames. The backend must maintain this across WebSocket frames.

**Solution:** In `inference_service.py`, maintain a dict keyed by `session_id`. Each entry stores the current DistractionFusion state per pedestrian track_id. Pass the correct state into `pipeline.run()` per frame. Clean up session state on WebSocket disconnect.

## 4.3 Real-Time Detection Frame Rate

- Frontend captures one frame every 500ms (configurable via settings `frame_sample_rate`)
- Backend processes one frame at a time per WebSocket connection
- If model inference takes 85ms, the system can handle ~11 frames/second — sufficient for 2 frames/second input
- For multiple concurrent cameras: each camera opens its own WebSocket connection with a unique `session_id`

---

---

# 🧬 PHASE 5 — Face Handling Logic

*(Fully detailed in Step 5 above — summary here for reference)*

**Flow:**
```
DetectionResult (is_violation=True)
  → face_region bbox extracted from full frame
  → crop resized to 160×160
  → 128-dim embedding generated
  → cosine similarity computed against all stored embeddings
  → if similarity > threshold → update existing face document
  → if similarity ≤ threshold → insert new face document
```

**Threshold:** Read from MongoDB settings document. Default 0.85. Tunable by operator.

**Embedding model:** InsightFace (recommended — fast, accurate, no GPU required for inference).

---

---

# 🧪 PHASE 6 — Training Module Plan

*(Fully detailed in Step 7 above — summary here for reference)*

**UI Flow:**
```
Settings Page → Training Section
  → Select model (posture_classifier or phone_detector)
  → Set epochs, learning_rate, batch_size
  → Click "Start Training"
  → POST /api/train
  → Connect to WS /ws/train-logs
  → Progress bar fills, log textarea scrolls in real time
  → "Training Complete" shown with final metrics
  → New model automatically active for inference
```

---

---

# 🍃 PHASE 7 — Database Design

---

## Collection: detections

| Field | Type | Description |
|---|---|---|
| `_id` | ObjectId | MongoDB auto |
| `detection_id` | string (UUID) | Unique identifier |
| `timestamp` | DateTime | When detection occurred |
| `session_id` | string | Camera session ID |
| `frame_id` | string | Frame within session |
| `is_violation` | bool | Any pedestrian violating |
| `overall_confidence` | float | Highest violation confidence |
| `processing_time_ms` | int | Inference duration |
| `pedestrians` | array | List of PedestrianResult objects |

**Indexes:** `timestamp` (desc), `is_violation`, `session_id`, compound `(timestamp, is_violation)`

---

## Collection: alerts

| Field | Type | Description |
|---|---|---|
| `_id` | ObjectId | MongoDB auto |
| `alert_id` | string (UUID) | Unique identifier |
| `detection_id` | string (UUID) | Links to detections |
| `face_id` | string (UUID) | Links to faces (nullable) |
| `severity` | string | HIGH / MEDIUM / LOW |
| `title` | string | Short description |
| `description` | string | Full description |
| `confidence` | float | Detection confidence |
| `timestamp` | DateTime | Creation time |
| `resolved` | bool | Operator resolved it |
| `resolved_at` | DateTime | When resolved (null if not) |
| `snapshot_base64` | string | Face crop at violation |

**Indexes:** `timestamp` (desc), `resolved`, `severity`, `face_id`, compound `(face_id, timestamp)` for dedup check

---

## Collection: faces

| Field | Type | Description |
|---|---|---|
| `_id` | ObjectId | MongoDB auto |
| `face_id` | string (UUID) | Unique identifier |
| `embedding` | float array | 128-dimensional vector |
| `image_base64` | string | Best quality face crop |
| `first_seen` | DateTime | First detection |
| `last_seen` | DateTime | Most recent detection |
| `detection_count` | int | Total detections |
| `violation_count` | int | Total confirmed violations |
| `detection_ids` | string array | Linked detection UUIDs |

**Indexes:** `face_id`, `last_seen` (desc), `violation_count` (desc)

---

## Collection: settings

**Singleton — always exactly 1 document. Use upsert.**

| Field | Type | Default |
|---|---|---|
| `detection_confidence_threshold` | float | 0.75 |
| `alert_confidence_threshold` | float | 0.80 |
| `face_similarity_threshold` | float | 0.85 |
| `alert_dedup_window_seconds` | int | 60 |
| `auto_refresh_interval_ms` | int | 5000 |
| `notifications_enabled` | bool | true |
| `usage_threshold_percent` | int | 80 |
| `active_model` | string | "posture_classifier_v1" |
| `camera_resolution` | string | "640x480" |
| `max_concurrent_cameras` | int | 4 |
| `frame_sample_rate` | int | 2 |
| `updated_at` | DateTime | auto-set |

**Index:** None needed (single document)

---

## Collection: devices

| Field | Type | Description |
|---|---|---|
| `device_id` | string | Camera/device ID |
| `name` | string | Display name |
| `location` | string | Physical location |
| `status` | string | online / offline / error |
| `last_seen` | DateTime | Last active timestamp |
| `total_detections` | int | Lifetime detection count |
| `session_id` | string | Current active session ID |

**Index:** `status`, `last_seen`

---

## Collection: training_logs

| Field | Type | Description |
|---|---|---|
| `job_id` | string (UUID) | Training job ID |
| `model_type` | string | posture_classifier or phone_detector |
| `status` | string | running / completed / error / cancelled |
| `started_at` | DateTime | Start time |
| `completed_at` | DateTime | End time (null if running) |
| `config` | object | epochs, lr, batch_size |
| `logs` | string array | All log lines in order |
| `final_metrics` | object | accuracy, loss, val_accuracy |
| `error_message` | string | Error details if failed |

**Index:** `status`, `started_at` (desc)

---

---

# 🧠 PHASE X — Model Logic Validation & Output Definition

---

## X.1 Model Logic Issues (All Issues)

### Issue 1 — No Enforced Output Schema
**Problem:** src/ pipeline returns a raw Python dict with no guaranteed structure. Different code paths return different keys.
**Why wrong:** Backend cannot reliably read keys — any key change in src/ will silently crash inference_service.
**Fix:** After `pipeline.run()` returns, map its output to a strict `DetectionResult` Pydantic model in inference_service. Any missing or unexpected field is caught immediately as a validation error.

### Issue 2 — Numpy Types Break JSON Serialization
**Problem:** Pipeline returns numpy float32, numpy int64, and numpy arrays. These types are not JSON-serializable.
**Why wrong:** FastAPI will crash with `TypeError: Object of type float32 is not JSON serializable` when trying to send WebSocket response.
**Fix:** In inference_service, convert every numeric value using `float()` and `int()` before building DetectionResult. Convert numpy arrays to Python lists using `.tolist()`.

### Issue 3 — DistractionFusion Has No Session Continuity
**Problem:** The state machine tracks posture state transitions across frames but there is no mechanism to pass the previous state when processing the next WebSocket frame.
**Why wrong:** Every frame is treated as the first frame — the state machine never advances past the initial state. SUSPICIOUS → USING transition never triggers.
**Fix:** In inference_service, maintain a `session_states` dict: `{ session_id: { pedestrian_id: fusion_state } }`. Pass the stored state into each pipeline call. Update stored state after each result.

### Issue 4 — Canvas Bounding Box Misalignment
**Problem:** Frontend canvas is hardcoded at 640×480. The actual video element is CSS-responsive and may render at different dimensions.
**Why wrong:** Boxes drawn at pixel coordinates (100, 80) on a 640px canvas appear at wrong positions on a 400px wide rendered video.
**Fix:** Backend includes both absolute pixel coordinates AND normalized 0.0–1.0 coordinates in DetectionResult. Frontend always uses normalized coordinates multiplied by actual canvas render dimensions.

### Issue 5 — Phone Detection Runs on Every Frame Regardless of Posture
**Problem:** In src/ pipeline, YOLOv11n phone detection runs on every pedestrian crop unconditionally.
**Why wrong:** Phone detection is expensive. Running it on pedestrians classified as NOT_USING wastes ~30ms per pedestrian per frame.
**Fix:** In the pipeline logic, only run YOLOv11n phone detection when posture_state is SUSPICIOUS or USING. Skip for NOT_USING and BACKSIDE states.

### Issue 6 — MMPose Dependencies Commented Out
**Problem:** `mmpose==1.3.2` and `mmcv==2.1.0` are commented out in requirements.txt.
**Why wrong:** Running inference will immediately raise `ModuleNotFoundError: No module named 'mmpose'`.
**Fix:** Uncomment these dependencies in requirements.txt. Add installation instructions to README since mmcv requires a specific torch-compatible version.

### Issue 7 — app/api.py Imports Non-Existent Routes
**Problem:** The existing `app/api.py` imports `from app.routes import auth, detect, face, reports` — none of these files exist.
**Why wrong:** Starting the app raises `ImportError` immediately.
**Fix:** Delete `app/` folder entirely and replace with the new `backend/` structure. Do not attempt to fix `app/api.py`.

### Issue 8 — No Face Save Logic Connected to Pipeline
**Problem:** The pipeline extracts face crops internally but has no mechanism to pass them out for storage.
**Why wrong:** Detected violator faces are never saved anywhere — the entire face tracking feature is non-functional.
**Fix:** Ensure `DetectionResult` includes `face_region` bbox. In inference_service, use the original frame + `face_region` to crop the face and pass to face_service. The crop does not need to come from src/ directly — it can be done in the backend using the coordinates.

---

## X.2 Expected Model Output (Exact Definition)

Every inference call MUST return exactly this structure — no exceptions:

```
DetectionResult
├── detection_id: string (UUID, generated by inference_service)
├── timestamp: string (ISO 8601, set by inference_service)
├── session_id: string (passed in from WebSocket)
├── frame_id: string (session_id + frame counter)
├── is_violation: bool (True if ANY pedestrian has fusion_state == USING)
├── overall_confidence: float (max posture_confidence among violating pedestrians)
├── processing_time_ms: int (measured by inference_service around pipeline call)
├── error_message: string | null (null on success, error description on failure)
└── pedestrians: List[PedestrianResult]
    └── PedestrianResult
        ├── pedestrian_id: string (track ID from YOLO, stable within session)
        ├── bbox
        │   ├── x1, y1, x2, y2: int (absolute pixel coords)
        │   └── x1_norm, y1_norm, x2_norm, y2_norm: float (0.0–1.0)
        ├── posture_state: string (NOT_USING | SUSPICIOUS | USING | BACKSIDE | OUT_OF_FRAME)
        ├── posture_confidence: float (0.0–1.0)
        ├── phone_detected: bool
        ├── phone_confidence: float (0.0 if phone not detected)
        ├── fusion_state: string (current DistractionFusion state)
        ├── is_violation: bool (True only if fusion_state == USING)
        └── face_region: FaceRegion | null
            └── x1, y1, x2, y2: int
```

---

## X.3 Complete Inference Pipeline (End-to-End)

```
INPUT
  Raw video frame as numpy array (BGR, any resolution)
  session_id (string, identifies which camera this frame is from)

STEP 1 — Validate Input
  Check frame is not None
  Check frame shape has 3 channels
  Record frame width and height for normalization
  If invalid → return empty DetectionResult with error_message

STEP 2 — Call src/ Pipeline
  Pass frame to InferencePipeline.run(frame, session_state)
  Receive raw result dict
  If exception → catch, return DetectionResult with error_message

STEP 3 — Convert Types
  Convert ALL numpy numeric types to Python float/int
  Convert ALL numpy arrays to Python lists
  This step is mandatory — never skip it

STEP 4 — Build PedestrianResult for each detected pedestrian
  Map raw bbox coords to BoundingBox (absolute + normalized)
  Map posture class to posture_state string
  Map phone detection output to phone_detected and phone_confidence
  Map fusion output to fusion_state
  Set is_violation = (fusion_state == "USING")
  Set face_region if is_violation is True, else null

STEP 5 — Build DetectionResult
  Set is_violation = any(p.is_violation for p in pedestrians)
  Set overall_confidence = max violation confidence, or 0.0 if no violations
  Set processing_time_ms from time.monotonic() measurement
  Set all metadata fields

STEP 6 — Save to MongoDB
  Insert full DetectionResult dict to detections collection
  This is async — await the insert

STEP 7 — Trigger Side Effects (async, non-blocking)
  If is_violation: asyncio.create_task(alert_service.create_alert(result))
  If is_violation: asyncio.create_task(face_service.process_face(frame, result))

STEP 8 — Return DetectionResult
  Return the Pydantic model object
  Route handler serializes to JSON and sends over WebSocket

OUTPUT
  DetectionResult (complete, consistent, all fields present, all types JSON-safe)
```

---

## X.4 Output Consistency Rules

**Rule 1:** ALWAYS return a DetectionResult object. Never return None, never raise unhandled exceptions to the route handler.

**Rule 2:** If no pedestrians detected → return DetectionResult with `pedestrians=[]`, `is_violation=False`, `overall_confidence=0.0`.

**Rule 3:** All float fields must be Python float (not numpy float32). All int fields must be Python int (not numpy int64). Enforce this by passing values through `float()` and `int()` explicitly.

**Rule 4 — Edge Cases:**

| Scenario | Behavior |
|---|---|
| Empty / black frame | Return DetectionResult, pedestrians=[], error_message=null |
| No pedestrians in frame | Return DetectionResult, pedestrians=[], is_violation=False |
| Multiple pedestrians | All included in pedestrians list, is_violation=True if any one violates |
| Confidence below threshold | phone_detected=False, is_violation=False, no alert created |
| Model inference exception | Return DetectionResult with error_message set, pedestrians=[] |
| Face region out of frame bounds | Clamp face_region coordinates to (0, 0, frame_width, frame_height) |
| WebSocket disconnect mid-frame | Catch exception, clean up session_state for this session_id |

---

## X.5 Integration Expectation

```
DetectionResult
  ↓ consumed by inference_service
  → .model_dump() converts to dict for MongoDB insert
  → Sent as JSON string over WebSocket to frontend

  ↓ consumed by alert_service
  → Checks is_violation, confidence
  → Creates alert document if conditions met

  ↓ consumed by face_service
  → Extracts face crop from frame using face_region
  → Generates embedding, deduplicates, saves to faces collection

  ↓ consumed by frontend (WebSocket message)
  → pedestrians[].bbox → canvas bounding box drawing
  → posture_state → detection status text
  → is_violation → overlay color (red/green)
  → Triggers dashboard refresh
```

---

---



# 🧱 PHASE 9 — Clean Code + Structure Plan

---

## 9.1 Dependency Direction (Never Break This)

```
routes/ → services/ → ml/pipeline.py → src/
routes/ → services/ → database.py
models/ ← used by routes/, services/, ml/
utils/  ← used by services/, ml/
```

A file deeper in the chain must never import from a file higher in the chain. `services/` never imports from `routes/`. `ml/pipeline.py` never imports from `services/`.

## 9.2 One Responsibility Per File

| File | Contains | Does NOT contain |
|---|---|---|
| routes/detect.py | WebSocket handler, POST handler | Business logic, DB queries |
| services/inference_service.py | Frame decode, pipeline call, DB save, trigger alerts | FastAPI Request/Response objects |
| ml/pipeline.py | Import from src/, expose run() | Any ML code |
| models/detection.py | Pydantic field definitions | Any logic or functions |
| utils/image_utils.py | Base64 decode, resize functions | DB calls, business logic |

## 9.3 Standard Response Format

All HTTP API responses must use the same envelope:

**Success:**
```
{ "data": { ... }, "meta": { "timestamp": "ISO8601", "request_id": "uuid" } }
```

**Error:**
```
{ "error": { "message": "Human readable error", "code": "ERROR_CODE" }, "meta": { ... } }
```

**Never** return raw Pydantic models directly from routes — always wrap in this envelope using `response_utils.py`.

## 9.4 Configuration Rule

- Infrastructure config (DB URI, ports, secrets) → `.env` → `config.py`
- Runtime operational config (thresholds, intervals, active model) → MongoDB `settings` collection
- ML model hyperparameters → `configs/` YAML files (already in src/)

Never mix these three layers.

---

---

# 🚀 PHASE 10 — Final Execution Strategy

---

## Exact Build Order (3 Weeks)

```
WEEK 1 — Foundation & Core Pipeline

Day 1:
  - Create backend/ folder structure (Phase 2.3)
  - Create .env and .env.example
  - Create main.py with CORS, lifespan skeleton, health endpoint
  - Verify: uvicorn starts, /health returns 200

Day 2:
  - Create database.py with Motor client
  - Create all 6 MongoDB collections and indexes
  - Create seed_settings.py and call from lifespan
  - Verify: MongoDB connected, settings document seeded

Day 3:
  - Create backend/app/ml/pipeline.py (bridge only, import from src/)
  - Load pipeline in lifespan startup event, store in app.state
  - Update /health to show model_loaded: true
  - Verify: models load without error, /health shows model status

Day 4:
  - Define all Pydantic models in backend/app/models/
  - Create inference_service.py (frame decode, pipeline call, type conversion, schema mapping)
  - Verify: can call inference_service directly in a test, receives valid DetectionResult

Day 5:
  - Create routes/detect.py POST /api/detect endpoint
  - Test with real image via curl or Postman
  - Verify: returns valid DetectionResult JSON with correct field types

WEEK 2 — Features

Day 6:
  - Create routes/detect.py WebSocket /ws/stream endpoint
  - Test with wscat sending base64 frame
  - Verify: receives DetectionResult back over WebSocket

Day 7:
  - Create face_service.py with embedding + deduplication logic
  - Create routes/faces.py
  - Verify: same face sent twice creates only 1 document in faces collection

Day 8:
  - Create alert_service.py with severity logic and dedup window
  - Create routes/alerts.py
  - Verify: violation detection creates alert, same violation within 60s does not duplicate

Day 9:
  - Create dashboard_service.py with aggregation queries
  - Create routes/dashboard.py
  - Verify: /api/dashboard/stats returns real counts matching MongoDB data

Day 10:
  - Create routes/settings.py with GET/PUT
  - Verify: GET returns seeded defaults, PUT updates document, GET again shows new values

WEEK 3 — Integration & Polish

Day 11:
  - Apply AppProvider in frontend main.jsx
  - Create frontend/src/services/api.js
  - Wire Dashboard StatusCards to GET /api/dashboard/stats
  - Verify: Dashboard shows real database numbers

Day 12:
  - Fix useWebSocket hook with real WebSocket connection
  - Fix canvas sizing (normalized coordinates)
  - Fix detection status colors
  - Verify: Live camera feed shows real bounding boxes from model

Day 13:
  - Create training_service.py and routes/training.py
  - Add training UI section to Settings page
  - Verify: Start Training button triggers real training, logs stream to UI

Day 14:
  - Wire all remaining frontend pages (Alerts, Devices, Settings, Faces)
  - End-to-end integration test: camera → detection → alert → dashboard update
  - Verify full flow works

Day 15:
  - Complete Dockerfile and docker-compose.yml
  - Write README with setup instructions
  - Fix CRLF line endings in frontend (.editorconfig or git config)
  - Add node_modules to .gitignore
  - Final smoke test of entire system via Docker
```

Now I have everything I need. Here is the coding agent prompt first, then the improvements.

---

# PHASE 11 — Source Code Issues & Required Improvements

*These are real issues found in the existing codebase that must be fixed during backend implementation. These improvements do NOT change any ML logic — they only fix integration gaps, missing return values, and incorrect patterns.*

---

### Issue 1 — `run_on_frame()` Does NOT Return Bounding Boxes

**Location:** `src/pipeline/inference_pipeline.py` → `run_on_frame()`

**Problem:** The method returns `{ frame, person_results, num_persons }`. Each `person_result` dict contains `posture`, `phone`, `state`, `display_text`, `score_text` — but **no `bbox` (xyxy coordinates)**. The bounding box `xyxy` is only used internally inside the loop and never passed out.

**Why it matters:** The backend cannot build `DetectionResult.pedestrians[].bbox` without these coordinates. The frontend cannot draw bounding boxes without them.

**Fix (in `src/pipeline/inference_pipeline.py`):** In the `run_on_frame()` loop, pass `xyxy` into `process_one_person()` and ensure `process_one_person()` returns it back inside the result dict as `"xyxy": xyxy.tolist()`. This is a one-line addition to the returned dict — it does not change any ML logic.

---

### Issue 2 — `person_result` Does NOT Return Face Region Coordinates

**Location:** `src/components/runtime_detector.py` → `process_one_person()`

**Problem:** When `fusion_result["final_label"] == "distracted"`, the code crops the face (`face_frame`, `face_xyxy`) and assigns `announced_face_frame`. However, `face_xyxy` is never included in the returned result dict. Only the cropped frame itself is returned as `announced_face_frame`.

**Why it matters:** The backend's `face_service.py` needs `face_xyxy` to know where the face is. Without coordinates, the backend cannot crop the face from the original frame independently or store the region.

**Fix (in `src/components/runtime_detector.py`):** Add `"face_xyxy": face_xyxy.tolist() if face_xyxy is not None else None` to the returned result dict. No ML logic changes.

---

### Issue 3 — `detect_frame()` Writes Temp File Instead of Direct numpy Inference

**Location:** `app/services/detection_service.py` → `detect_frame()`

**Problem:** When processing a Base64 frame, the code decodes it to numpy array, then **writes it to a temp `.jpg` file**, then calls `predict_image(temp_path)` which reads the file back again. This adds disk I/O on every single frame — at 2 frames/second this is wasteful and slow.

**Why it matters:** In the new backend, real-time WebSocket frames must be processed in under 100ms. Disk writes per frame will degrade performance significantly.

**Fix:** `InferencePipeline.run_on_frame()` already accepts a numpy array directly. In `inference_service.py`, decode Base64 to numpy array and call `app.state.pipeline.run_on_frame(frame)` directly. No temp file needed. The old `app/services/detection_service.py` file is replaced entirely by the new `backend/app/services/inference_service.py`.

---

### Issue 4 — `DistractionFusion` State is Stateless Per Frame

**Location:** `src/components/runtime_detector.py` → `process_one_person()` and `src/components/distraction_fusion.py`

**Problem:** `DistractionFusion.fuse()` is a pure stateless function — it takes `base_state` and returns a result. It does not remember the previous state between calls. The `RuntimeDetector` creates a new `DistractionFusion` instance once but `fuse()` itself has no memory. This means every frame is evaluated independently — the transition from SUSPICIOUS → USING requires phone detection in the same frame, but there is no concept of "this pedestrian was suspicious last frame, now confirm with phone."

**Why it matters:** The state machine in `constants.py` defines transitions but the current implementation does not persist state per pedestrian across frames. A pedestrian who is SUSPICIOUS in frame 1 and has phone confirmed in frame 3 will not correctly transition to USING.

**Fix (backend-side, no src/ changes):** In `backend/app/services/inference_service.py`, maintain a `session_states` dict: `{ session_id: { pedestrian_track_id: last_fusion_state } }`. On each frame, pass the stored previous state for that pedestrian to the fusion logic. This is backend responsibility — it does not require changes to `src/`.

---

### Issue 5 — Existing `src/serving/schemas.py` Is Incomplete

**Location:** `src/serving/schemas.py`

**Problem:** `PersonPrediction` schema has `posture`, `phone`, `state`, `display_text`, `score_text` — but is missing:
- `bbox` (absolute pixel coordinates)
- `posture_confidence` (numeric, not just text)
- `face_region` coordinates
- Normalized bbox coordinates

**Why it matters:** The `InferenceResponse` from `src/` cannot be used directly as the API response. It must be mapped to the fuller `DetectionResult` schema defined in the backend.

**Fix:** Do NOT modify `src/serving/schemas.py`. Instead, define the full `DetectionResult` and `PedestrianResult` Pydantic models in `backend/app/models/detection.py` as specified in Phase 3 Step 4. The backend `inference_service.py` maps the limited `PersonPrediction` output to the full `PedestrianResult` by combining it with the `xyxy` and `face_xyxy` data from Issues 1 and 2.

---

### Issue 6 — `app/services/face_service.py` Uses Wrong Stack

**Location:** `app/services/face_service.py`

**Problem:** The existing face service uses:
- `face_recognition` library (dlib-based, heavy, no async support)
- `SQLAlchemy` ORM and a relational DB
- `pickle` files on disk for embeddings
- `RegisteredFace` SQLAlchemy model

**Why it matters:** The new backend uses MongoDB + Motor async. The existing face service is completely incompatible with the new stack and cannot be reused.

**Fix:** Completely discard `app/services/face_service.py`. Build `backend/app/services/face_service.py` from scratch using `insightface` or `facenet-pytorch` for embeddings and Motor async for all MongoDB operations, as specified in Phase 3 Step 5.

---

### Issue 7 — `app/services/detection_service.py` Uses SQLAlchemy

**Location:** `app/services/detection_service.py`

**Problem:** Uses SQLAlchemy Session, `Report` ORM model, and PostgreSQL/SQLite. Also saves rendered output images to disk. Also uses `generate_unique_filename` from a non-existent `app.core.security`.

**Fix:** Completely discard this file. Build `backend/app/services/inference_service.py` using Motor async and MongoDB as specified in Phase 3 Step 4.

---

### Issue 8 — `runtime_parameters` Resets on Every Frame

**Location:** `src/pipeline/inference_pipeline.py` → `run_on_frame()`

**Problem:**
```python
runtime_parameters = {
    "time_last_record_framerate": 0.0,
    "time_last_announce_face": 0.0,
    "path_runtime_handframes": None,
}
```
This dict is created fresh on every `run_on_frame()` call. `time_last_announce_face` is always 0.0, so the face announce interval check in `process_one_person()` always passes — meaning face crops are announced on every frame the person is classified as distracted.

**Why it matters:** The face service will be called on every single frame of a violation instead of respecting the `face_announce_interval_seconds` config. This creates thousands of unnecessary face documents.

**Fix (backend-side):** In `inference_service.py`, maintain a `session_runtime_params` dict per `session_id`. Pass the stored params into `run_on_frame()` instead of letting the pipeline create fresh ones. Update stored params after each call using the timing values from the result. This does not require `src/` changes — it requires `run_on_frame()` to accept `runtime_parameters` as an argument (it already does in `process_one_person()`, just not at the `run_on_frame()` level).

---

### Summary Table

| Issue | Location | Src Change Needed | Severity |
|---|---|---|---|
| bbox not returned | `inference_pipeline.py` | ✅ One-line addition | 🔴 Critical |
| face_xyxy not returned | `runtime_detector.py` | ✅ One-line addition | 🔴 Critical |
| Temp file per frame | `detection_service.py` | ❌ Replace service | 🟠 High |
| Fusion has no state memory | `inference_service.py` (backend) | ❌ Backend-only fix | 🟠 High |
| schemas.py incomplete | `schemas.py` | ❌ Add backend models | 🟡 Medium |
| Face service wrong stack | `face_service.py` | ❌ Replace service | 🔴 Critical |
| Detection service wrong stack | `detection_service.py` | ❌ Replace service | 🔴 Critical |
| runtime_parameters resets | `inference_service.py` (backend) | ❌ Backend-only fix | 🟡 Medium |

---



## What to Test at Each Key Stage

| Stage | Test Command / Action | Expected Result |
|---|---|---|
| After Day 1 | `curl localhost:8000/health` | `{ "status": "ok" }` |
| After Day 2 | MongoDB Compass: list collections | 6 collections, settings has 1 doc |
| After Day 3 | `curl localhost:8000/health` | `{ "model_loaded": true }` |
| After Day 5 | POST /api/detect with real image | Valid DetectionResult JSON, no numpy type errors |
| After Day 6 | wscat connect, send frame | DetectionResult received back over WS |
| After Day 7 | Send same face twice | Only 1 face in MongoDB faces collection |
| After Day 8 | Send violation frame | Alert appears in GET /api/alerts |
| After Day 9 | GET /api/dashboard/stats | Numbers match MongoDB collection counts |
| After Day 11 | Open browser dashboard | StatusCards show real numbers, not hardcoded |
| After Day 12 | Start camera in browser | Real bounding boxes drawn at correct positions |
| After Day 14 | Full end-to-end in browser | Camera → boxes → alert panel updates → dashboard updates |
| After Day 15 | `docker-compose up` | Entire system starts, all features work in containers |

---

## Final Note for Coding Agent

**The three rules that must never be broken:**

1. `src/` is read-only. Import from it. Never modify it. Never duplicate it.
2. Every inference result must be converted to `DetectionResult` Pydantic model before leaving `inference_service.py`. No raw dicts ever reach routes or the database.
3. Build in order. Do not start Step 6 (alerts) before Step 4 (detection API) is tested and confirmed working. Each step depends on the one before it.

---

*End of Plan — Pedestrian Distraction Detection System — Group 09 (M)*
*Version: Corrected Final — src/ untouched, backend/ is API layer only*
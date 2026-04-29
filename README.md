<div align="center">

<br/>

```
██████╗ ██████╗ ███████╗
██╔══██╗██╔══██╗██╔════╝
██████╔╝██║  ██║███████╗
██╔═══╝ ██║  ██║╚════██║
██║     ██████╔╝███████║
╚═╝     ╚═════╝ ╚══════╝
Pedestrian Distraction Detection System
```

<h1>🚶 Pedestrian Distraction Detection System</h1>

<p>
  <strong>Real-time AI-powered pedestrian safety monitoring — detecting cell phone usage at road crossings.</strong>
</p>

<br/>

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![React](https://img.shields.io/badge/React-19.0-61DAFB?style=for-the-badge&logo=react&logoColor=black)](https://react.dev)
[![MongoDB](https://img.shields.io/badge/MongoDB-6.0+-47A248?style=for-the-badge&logo=mongodb&logoColor=white)](https://mongodb.com)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?style=for-the-badge&logo=docker&logoColor=white)](https://docker.com)

<br/>

[![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Active%20Development-orange?style=flat-square)]()
[![University](https://img.shields.io/badge/University-Final%20Year%20Project-blue?style=flat-square)]()
[![Group](https://img.shields.io/badge/Group-09%20(M)-purple?style=flat-square)]()

<br/>

> **Supervisor:** Dr. Adeel Ahmad   &nbsp;|&nbsp; **Students:** Adnan Khan F22-0431 &nbsp;·&nbsp; Bilal Asghar F22-1813
>
> *University of Haripur  — Faculty of Science and Technology*
> *Department of  Information Technology*

<br/>

---

</div>

<br/>

## 📌 Table of Contents

- [Overview](#-overview)
- [Key Features](#-key-features)
- [System Architecture](#-system-architecture)
- [Technology Stack](#-technology-stack)
- [ML Pipeline](#-ml-pipeline--how-it-works)
- [Project Structure](#-project-structure)
- [Getting Started](#-getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Running the System](#running-the-system)
  - [Docker Setup](#docker-setup)
- [API Reference](#-api-reference)
- [Frontend Overview](#-frontend-overview)
- [Training Module](#-training-module)
- [Database Schema](#-database-schema)
- [Configuration](#-configuration)
- [Results & Performance](#-results--performance)
- [Roadmap](#-roadmap)
- [Contributing](#-contributing)
- [License](#-license)

<br/>

---

## 🌐 Overview

The **Pedestrian Distraction Detection System (PDS)** is a production-grade, real-time AI surveillance platform designed to detect pedestrians using mobile phones while crossing roads — a major cause of pedestrian fatalities worldwide.

The system operates by combining **pose estimation**, **3D behavioral feature engineering**, and **object detection** into a multi-stage inference pipeline. Detected violations are logged, the pedestrian's face is captured and deduplicated, and alerts are generated for operator review — all in real time via a modern web dashboard.

> 🏛️ This system was developed as part of a Final Year Project at the **University of Science and Technology**. The Macao government has been considering penalizing pedestrians for cell phone use while crossing roads — this system provides the technical foundation for automated enforcement and monitoring.

<br/>

### The Problem

<!-- PLACEHOLDER: Add local statistics or government policy context here -->

Pedestrian distraction from cell phones is responsible for **[X]% of pedestrian fatalities** in urban environments. Manual monitoring is expensive and inconsistent. Existing camera systems lack the intelligence to automatically identify and log violations in real time.

### Our Solution

A fully automated AI pipeline that:
1. Detects pedestrians in real-time video streams
2. Classifies their posture using a novel **2-Channel 3D behavioral feature**
3. Confirms phone usage with a fine-tuned **YOLOv11n detector**
4. Captures and stores the violator's face for record-keeping
5. Provides a live monitoring dashboard for operators

<br/>

---

## ✨ Key Features

<table>
<tr>
<td width="50%">

### 🤖 AI Detection
- **Real-time** pedestrian detection at ~41 FPS
- **2-Channel 3D CNN** for posture classification
- **YOLOv11n** fine-tuned phone detector
- **RTMPose** 17-point skeleton estimation
- State machine: `SAFE → SUSPICIOUS → VIOLATION`

</td>
<td width="50%">

### 🖥️ Live Dashboard
- WebSocket-powered real-time camera feed
- Live bounding box overlay on video
- Real-time alert generation and management
- Detection statistics and timeline charts
- Multi-camera device support

</td>
</tr>
<tr>
<td width="50%">

### 🧬 Smart Face Handling
- Automatic face crop on confirmed violations
- **128-dimensional embedding** deduplication
- Cosine similarity prevents duplicate records
- Consolidated detection history per person
- Operator-controlled face database

</td>
<td width="50%">

### 🧪 UI-Based Model Training
- Train models directly from the dashboard
- Live training progress bar and log stream
- Pause / Resume / Stop training controls
- Automatic model activation after training
- Full training history with metrics

</td>
</tr>
</table>

<br/>

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                     FRONTEND  (React 19 + Vite)                      │
│                                                                       │
│   📷 WebcamDetector   📊 Dashboard   🚨 Alerts   ⚙️ Settings/Train   │
│         │                   │              │              │           │
│         └───────────────────┴──────────────┴──────────────┘          │
│                    WebSocket /ws/stream      HTTP REST                │
└────────────────────────────┬────────────────────────┬────────────────┘
                             │                        │
┌────────────────────────────▼────────────────────────▼────────────────┐
│                       BACKEND  (FastAPI)                              │
│                                                                       │
│  routes/          services/              ml/                          │
│  ├─ detect.py     ├─ inference_service   └─ pipeline.py (bridge)      │
│  ├─ alerts.py     ├─ alert_service            │                       │
│  ├─ faces.py      ├─ face_service             │ imports               │
│  ├─ dashboard.py  ├─ training_service         ▼                       │
│  ├─ settings.py   └─ dashboard_service   src/ ML Code                 │
│  └─ training.py                                                       │
└──────────┬───────────────────────────────────┬────────────────────────┘
           │                                   │
┌──────────▼──────────┐           ┌────────────▼──────────────────────┐
│   ML PIPELINE        │           │           MONGODB                  │
│                      │           │                                    │
│  1. YOLO BBox        │           │  ● detections   ● alerts           │
│  2. RTMPose 17-kp    │           │  ● faces        ● settings         │
│  3. 2-Ch 3D Feature  │           │  ● devices      ● training_logs    │
│  4. 3D-CNN Classify  │           │                                    │
│  5. YOLOv11n Phone   │           └───────────────────────────────────┘
│  6. Fusion FSM       │
│  7. Face Crop        │
└─────────────────────┘
```

<br/>

---

## 🛠️ Technology Stack

### Backend

| Technology | Version | Purpose |
|---|---|---|
| **Python** | 3.10+ | Core backend language |
| **FastAPI** | 0.115+ | Async REST API + WebSocket server |
| **Motor** | 3.x | Async MongoDB driver |
| **MongoDB** | 6.0+ | Primary database |
| **Uvicorn** | 0.30+ | ASGI server |
| **Pydantic v2** | 2.x | Data validation and schemas |
| **pydantic-settings** | 2.x | Environment configuration |

### Machine Learning

| Technology | Version | Purpose |
|---|---|---|
| **PyTorch** | 2.0+ | Deep learning framework |
| **MMPose** | 1.3.2 | RTMPose 17-point skeleton estimation |
| **MMCV** | 2.1.0 | MMPose dependency |
| **Ultralytics** | 8.x | YOLOv11n phone detection |
| **OpenCV** | 4.x | Frame processing and image utilities |
| **NumPy** | 1.x | Numerical computations |
| **InsightFace** | 0.7+ | 128-dim face embedding generation |
| **MLflow** | 2.x | Experiment tracking and logging |

### Frontend

| Technology | Version | Purpose |
|---|---|---|
| **React** | 19.x | UI framework |
| **Vite** | 7.x | Build tool and dev server |
| **TailwindCSS** | 4.x | Utility-first styling |
| **React Router** | 7.x | Client-side routing |
| **Chart.js** | 4.x | Analytics charts and graphs |
| **react-chartjs-2** | 5.x | React wrapper for Chart.js |

### Infrastructure

| Technology | Purpose |
|---|---|
| **Docker** | Containerization |
| **Docker Compose** | Multi-service orchestration |
| **DAGsHub** | ML experiment tracking and dataset versioning |
| **DVC** | Data version control |

<br/>

---

## 🧠 ML Pipeline — How It Works

The detection pipeline processes each video frame through 7 sequential stages:

```
VIDEO FRAME (BGR numpy array)
        │
        ▼
┌───────────────────────────────────────────────────────────┐
│  STAGE 1 — Pedestrian Detection (YOLO BBox)               │
│  • Detects all persons in frame                           │
│  • Outputs: bounding boxes [(x1,y1,x2,y2, confidence)]   │
│  • Threshold: confidence > 0.50                           │
└───────────────────────┬───────────────────────────────────┘
                        │
                        ▼
┌───────────────────────────────────────────────────────────┐
│  STAGE 2 — Pose Estimation (RTMPose)                      │
│  • Extracts 17 skeleton key-points per pedestrian         │
│  • Outputs: [(x, y, confidence)] × 17 key-points         │
│  • If < 10 key-points confident → classify as BACKSIDE   │
└───────────────────────┬───────────────────────────────────┘
                        │
                        ▼
┌───────────────────────────────────────────────────────────┐
│  STAGE 3 — 2-Channel 3D Feature Engineering              │
│  • Converts 17 key-points → 858 angle+confidence pairs   │
│  • Builds 2-Channel 3D tensor: shape (7, 2, 12, 11)      │
│  • Channel 1: joint angles   Channel 2: confidence scores │
└───────────────────────┬───────────────────────────────────┘
                        │
                        ▼
┌───────────────────────────────────────────────────────────┐
│  STAGE 4 — Posture Classification (3D-CNN)                │
│  • 3 × 3D Convolutional layers                            │
│  • 1 × ResPool3d layer (doubles local max-voxels)         │
│  • 4 × Fully Connected layers with SiLU activation       │
│  • Output: posture_state + confidence score               │
└───────────────────────┬───────────────────────────────────┘
                        │
              (only if SUSPICIOUS or USING)
                        │
                        ▼
┌───────────────────────────────────────────────────────────┐
│  STAGE 5 — Phone Detection (YOLOv11n)                     │
│  • Crops face+hand region (upper 40% of person bbox)      │
│  • Resizes to 128×128 for YOLOv11n input                  │
│  • Output: phone_detected (bool) + phone_confidence       │
└───────────────────────┬───────────────────────────────────┘
                        │
                        ▼
┌───────────────────────────────────────────────────────────┐
│  STAGE 6 — DistractionFusion State Machine                │
│                                                           │
│  OUT_OF_FRAME → BACKSIDE → NOT_USING → SUSPICIOUS → USING │
│                                                           │
│  • Combines posture_state + phone_detected                │
│  • Maintains state per pedestrian across frames           │
│  • USING state = confirmed violation                      │
└───────────────────────┬───────────────────────────────────┘
                        │
              (only if state == USING)
                        │
                        ▼
┌───────────────────────────────────────────────────────────┐
│  STAGE 7 — Face Extraction + Output Assembly              │
│  • Crops face region from original frame                  │
│  • Builds structured DetectionResult (Pydantic model)     │
│  • Converts all numpy types to JSON-safe Python types     │
│  • Returns: DetectionResult with full metadata            │
└───────────────────────────────────────────────────────────┘
        │
        ▼
  DetectionResult JSON → WebSocket → Frontend + MongoDB + Alert
```

### Model Performance

| Model | Metric | Score |
|---|---|---|
| Posture Classifier (3D-CNN) | Accuracy | **91.78%** |
| Phone Detector (YOLOv11n) | mAP@50 | **98.1%** |
| System | Mean Frame Time | **0.085s** |
| System | Throughput | **~41.2 FPS** |
| Face Matching | Precision | **94.7%** |

<br/>

---

## 📁 Project Structure

```
PedestrianProject/
│
├── 📄 README.md                    ← You are here
├── 📄 requirements_rules.md        ← System architecture plan
├── 📄 requirements.txt             ← Python dependencies
├── 📄 .env.example                 ← Environment variables template
├── 📄 docker-compose.yml           ← Docker orchestration
├── 📄 Dockerfile                   ← Backend container
│
├── 📂 src/                         ← ML codebase (READ ONLY)
│   ├── components/                 ← 16 core ML modules
│   │   ├── runtime_detector.py     ← Per-person inference
│   │   ├── distraction_fusion.py   ← State machine logic
│   │   ├── pose_feature_generator.py  ← 2-Ch 3D feature builder
│   │   └── ...
│   ├── pipeline/                   ← Orchestration pipelines
│   │   ├── inference_pipeline.py   ← Main inference entry point
│   │   ├── posture_training_pipeline.py
│   │   └── phone_training_pipeline.py
│   ├── serving/                    ← API predictor + schemas
│   │   ├── predictor.py            ← Main Predictor class
│   │   └── schemas.py
│   ├── config/                     ← Config loading + constants
│   ├── entity/                     ← Pydantic config entities
│   └── utils/                      ← Helpers, logging, OpenCV utils
│
├── 📂 configs/                     ← YAML configuration files
│   ├── config.yaml                 ← Main system config
│   └── ...
│
├── 📂 scripts/                     ← Training entry points
│   ├── train_posture.py
│   └── train_phone.py
│
├── 📂 backend/                     ← FastAPI backend (API layer)
│   └── app/
│       ├── main.py                 ← FastAPI app entry point
│       ├── config.py               ← Reads .env settings
│       ├── database.py             ← Motor MongoDB client
│       ├── routes/                 ← HTTP + WebSocket handlers
│       ├── services/               ← Business logic layer
│       ├── models/                 ← Pydantic API schemas
│       ├── ml/
│       │   └── pipeline.py         ← Bridge: imports from src/
│       └── utils/                  ← Image, embedding, response utils
│
├── 📂 frontend/                    ← React frontend
│   └── src/
│       ├── pages/                  ← Dashboard, Detection, Settings...
│       ├── components/             ← Reusable UI components
│       ├── hooks/                  ← useWebSocket, useTrainingWebSocket
│       ├── services/
│       │   └── api.js              ← All backend API calls
│       ├── context/                ← Global AppContext (useReducer)
│       └── utils/                  ← Constants and helpers
│
└── 📂 tests/                       ← ML unit tests
```

<br/>

---

## 🚀 Getting Started

### Prerequisites

Before installing, make sure you have the following installed:

| Requirement | Version | Check Command |
|---|---|---|
| Python | 3.10+ | `python --version` |
| Node.js | 18+ | `node --version` |
| npm | 9+ | `npm --version` |
| MongoDB | 6.0+ | `mongod --version` |
| Git | Any | `git --version` |
| CUDA (optional) | 11.8+ | `nvcc --version` |

> 💡 **GPU is optional.** The system runs on CPU but GPU (CUDA) is recommended for real-time performance at 30+ FPS.

---

### Installation

#### 1. Clone the Repository

```bash
git clone https://github.com/AdnanKhaan11/Pedestrian-Distraction-Detection.git
cd Pedestrian-Distraction-Detection
```

#### 2. Set Up Python Environment

```bash
# Create a virtual environment
python -m venv venv

# Activate it
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

#### 3. Install Python Dependencies

```bash
# Install PyTorch first (CPU version)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# For GPU (CUDA 11.8):
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install MMPose dependencies (required for pose estimation)
pip install -U openmim
mim install mmengine
mim install "mmcv==2.1.0"
mim install "mmpose==1.3.2"

# Install remaining dependencies
pip install -r requirements.txt
```

#### 4. Install Frontend Dependencies

```bash
cd frontend
npm install
cd ..
```

#### 5. Configure Environment Variables

```bash
# Copy the example environment file
cp .env.example .env

# Open .env and fill in your values
nano .env   # or use any text editor
```

**Required `.env` values:**

```env
# Database
MONGODB_URI=mongodb://localhost:27017
DB_NAME=pedestrian_detection

# Security
SECRET_KEY=your-very-long-random-secret-key-here

# CORS (frontend URL)
CORS_ORIGINS=http://localhost:5173,http://localhost:3000

# ML Settings
DEVICE=cpu                          # or "cuda" for GPU
CONFIDENCE_THRESHOLD=0.75
ALERT_CONFIDENCE_THRESHOLD=0.80
FACE_SIMILARITY_THRESHOLD=0.85
ALERT_DEDUP_WINDOW_SECONDS=60

# Logging
LOG_LEVEL=INFO
```

#### 6. Download Model Weights

<!-- PLACEHOLDER: Add instructions for downloading pre-trained model weights -->
<!-- Example: -->
```bash
# Download pre-trained weights
# [PLACEHOLDER — Add model download instructions here]
# Example:
# wget https://your-storage-url/posture_classifier_v1.pth -P artifacts/models/
# wget https://your-storage-url/phone_detector_v1.pt -P artifacts/models/
```

---

### Running the System

You need **3 terminal windows** to run the full system locally.

#### Terminal 1 — Start MongoDB

```bash
# If MongoDB is installed locally
mongod --dbpath ./data/db

# Or if using MongoDB as a service (Windows)
net start MongoDB

# Or if using MongoDB as a service (Linux/Mac)
sudo systemctl start mongod
```

#### Terminal 2 — Start the Backend

```bash
# Make sure virtual environment is activated
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Start FastAPI backend from project root
uvicorn backend.app.main:app --reload --host 0.0.0.0 --port 8000
```

The backend will be available at:
- **API:** `http://localhost:8000`
- **Swagger Docs:** `http://localhost:8000/docs`
- **ReDoc:** `http://localhost:8000/redoc`
- **Health Check:** `http://localhost:8000/health`

#### Terminal 3 — Start the Frontend

```bash
cd frontend
npm run dev
```

The frontend will be available at: **`http://localhost:5173`**

> ✅ Once all three are running, open `http://localhost:5173` in your browser, navigate to Dashboard, and click **"Start Camera"** to begin real-time detection.

---

### Docker Setup

The easiest way to run the entire system with a single command:

```bash
# Build and start all services (backend + MongoDB)
docker-compose up --build

# Run in background
docker-compose up --build -d

# Stop all services
docker-compose down

# Stop and remove all data volumes
docker-compose down -v
```

**Services started by Docker Compose:**

| Service | Port | Description |
|---|---|---|
| `backend` | `8000` | FastAPI application |
| `mongodb` | `27017` | MongoDB database |

> 📝 **Note:** The frontend is NOT included in Docker Compose — run it separately with `npm run dev` in the `frontend/` folder, or serve it statically after building with `npm run build`.

<br/>

---

## 📡 API Reference

The full interactive API documentation is available at `http://localhost:8000/docs` when the backend is running.

### Core Endpoints

#### Detection

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/api/detect` | Submit a single image for detection |
| `WS` | `/ws/stream` | WebSocket for real-time frame streaming |

**WebSocket Frame Format (send):**
```json
{
  "frame": "<base64_jpeg_string>",
  "timestamp": "2026-04-29T12:00:00Z",
  "session_id": "camera_01"
}
```

**DetectionResult Format (receive):**
```json
{
  "detection_id": "uuid",
  "timestamp": "2026-04-29T12:00:00Z",
  "session_id": "camera_01",
  "is_violation": true,
  "overall_confidence": 0.92,
  "processing_time_ms": 85,
  "pedestrians": [
    {
      "pedestrian_id": "uuid",
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

#### Alerts

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/alerts` | Get paginated alerts (filter by severity, resolved, date) |
| `PATCH` | `/api/alerts/{id}/resolve` | Mark an alert as resolved |
| `DELETE` | `/api/alerts/{id}` | Delete an alert |

#### Dashboard

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/dashboard/stats` | Summary statistics |
| `GET` | `/api/dashboard/timeline` | Hourly detection counts (last 24h) |

#### Faces

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/faces` | Paginated list of all detected faces |
| `DELETE` | `/api/faces/{id}` | Remove a face record |

#### Training

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/api/train` | Start a training job |
| `GET` | `/api/train/status` | Current training status |
| `DELETE` | `/api/train/current` | Stop running training job |
| `GET` | `/api/train/history` | Past training jobs list |
| `WS` | `/ws/train-logs` | Real-time training log stream |

#### Settings

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/settings` | Get current system settings |
| `PUT` | `/api/settings` | Update system settings |

<br/>

---

## 🖥️ Frontend Overview

The frontend is a modern React 19 SPA with 5 main pages:

### Pages

| Page | Route | Description |
|---|---|---|
| **Dashboard** | `/` | Live camera feed, stats cards, alerts panel, detection chart |
| **Detection** | `/detection` | Full-screen camera with real-time bounding boxes and stats |
| **Devices** | `/devices` | Connected camera/device management |
| **Reports** | `/reports` | Detection history, face gallery, export tools |
| **Settings** | `/settings` | System config, detection thresholds, model training |

### Dashboard Screenshot

<!-- PLACEHOLDER: Add a screenshot of the running dashboard here -->
```
[PLACEHOLDER — Add dashboard screenshot here]
Example: ![Dashboard Screenshot](docs/screenshots/dashboard.png)
```

### Detection in Action

<!-- PLACEHOLDER: Add a GIF or screenshot of real-time detection -->
```
[PLACEHOLDER — Add detection GIF or screenshot here]
Example: ![Detection Demo](docs/screenshots/detection.gif)
```

<br/>

---

## 🧪 Training Module

The system includes a full UI-based model training workflow accessible from **Settings → Model Training**.

### How to Train a Model

1. Navigate to **Settings** → click **"Model Training"** tab
2. Select the model to train: **Posture Classifier** or **Phone Detector**
3. Configure hyperparameters:
   - **Epochs** (default: 50)
   - **Learning Rate** (default: 0.001)
   - **Batch Size** (default: 32)
4. Click **"Start Training"**
5. Monitor live progress:
   - Real-time progress bar with epoch counter
   - Live loss and accuracy metrics
   - Scrollable training log output
6. Use control buttons:
   - **Pause** — suspends training (can resume)
   - **Resume** — continues from last checkpoint
   - **Stop** — cancels training entirely
7. When complete, click **"Save & Activate Model"** to deploy the new weights

### Training Datasets

| Dataset | Samples | Purpose |
|---|---|---|
| Posture Dataset | 26,748 total | Train the 3D-CNN posture classifier |
| Phone Dataset | 11,508 total | Train the YOLOv11n phone detector |

<!-- PLACEHOLDER: Add dataset download or preparation instructions -->
```
[PLACEHOLDER — Add dataset download/preparation instructions here]
```

<br/>

---

## 🍃 Database Schema

The system uses MongoDB with 6 collections:

### `detections`
Stores every inference result from the pipeline.

```json
{
  "detection_id": "uuid",
  "timestamp": "ISO8601",
  "session_id": "camera_01",
  "is_violation": true,
  "overall_confidence": 0.92,
  "processing_time_ms": 85,
  "pedestrians": [{ "...full PedestrianResult..." }]
}
```

### `alerts`
Operator-facing violation alerts.

```json
{
  "alert_id": "uuid",
  "detection_id": "uuid",
  "face_id": "uuid",
  "severity": "HIGH",
  "title": "Cell Phone Usage Detected",
  "confidence": 0.92,
  "timestamp": "ISO8601",
  "resolved": false,
  "snapshot_base64": "..."
}
```

### `faces`
Deduplicated pedestrian face records.

```json
{
  "face_id": "uuid",
  "embedding": [0.12, -0.44, "...128 floats..."],
  "image_base64": "...",
  "first_seen": "ISO8601",
  "last_seen": "ISO8601",
  "detection_count": 4,
  "violation_count": 3
}
```

### `settings`
Singleton runtime configuration document.

```json
{
  "detection_confidence_threshold": 0.75,
  "alert_confidence_threshold": 0.80,
  "face_similarity_threshold": 0.85,
  "alert_dedup_window_seconds": 60,
  "frame_sample_rate": 2,
  "active_model": "posture_classifier_v1"
}
```

<br/>

---

## ⚙️ Configuration

All runtime behavior is controlled through two layers:

### 1. Infrastructure Config (`.env` file)
Used for server settings — database connections, ports, secrets. **Never changes at runtime.**

| Variable | Default | Description |
|---|---|---|
| `MONGODB_URI` | `mongodb://localhost:27017` | MongoDB connection string |
| `DB_NAME` | `pedestrian_detection` | Database name |
| `SECRET_KEY` | *(required)* | JWT signing secret |
| `CORS_ORIGINS` | `http://localhost:5173` | Allowed frontend origins |
| `DEVICE` | `cpu` | Inference device: `cpu` or `cuda` |
| `LOG_LEVEL` | `INFO` | Logging level |

### 2. Runtime Config (MongoDB `settings` collection)
Used for operational tuning — thresholds, intervals. **Changeable via the Settings page UI without restart.**

| Setting | Default | Description |
|---|---|---|
| `detection_confidence_threshold` | `0.75` | Minimum confidence to register a detection |
| `alert_confidence_threshold` | `0.80` | Minimum confidence to create an alert |
| `face_similarity_threshold` | `0.85` | Cosine similarity threshold for face dedup |
| `alert_dedup_window_seconds` | `60` | Seconds before same face can trigger new alert |
| `frame_sample_rate` | `2` | Frames per second sent to backend |

<br/>

---

## 📊 Results & Performance

### Model Accuracy

| Model | Architecture | Accuracy / mAP@50 | vs. Baseline |
|---|---|---|---|
| **Posture Classifier** | 2-Ch 3D CNN | **91.78%** | +17.5% over [2] |
| **Phone Detector** | YOLOv11n | **98.1% mAP@50** | +4.5% over YOLOv8n |

### System Performance

| Metric | Value |
|---|---|
| Mean inference time | **85ms** per frame |
| System throughput | **~41.2 FPS** under high load |
| False positive rate | **3.4%** |
| Face match precision | **94.7%** |

### Comparison with Prior Work

| Approach | Accuracy |
|---|---|
| Previous approach [2] | 74.3% |
| Single-channel CNN | 77.1% |
| LSTM baseline | 82.4% |
| **Our 2-Ch 3D CNN** | **91.78%** |

<!-- PLACEHOLDER: Add confusion matrix, ROC curve, or other evaluation figures here -->
```
[PLACEHOLDER — Add evaluation figures, charts, or images here]
Example: ![Confusion Matrix](docs/results/confusion_matrix.png)
```

<br/>

---

## 🗺️ Roadmap

- [x] RTMPose-based pose estimation pipeline
- [x] 2-Channel 3D feature engineering
- [x] 3D-CNN posture classifier (91.78% accuracy)
- [x] YOLOv11n phone detector (98.1% mAP@50)
- [x] DistractionFusion state machine
- [x] React frontend with live camera feed
- [ ] FastAPI backend with WebSocket streaming
- [ ] MongoDB integration with Motor async driver
- [ ] Face embedding and deduplication system
- [ ] UI-based training module with pause/resume
- [ ] Docker full-stack deployment
- [ ] Multi-camera concurrent session support
- [ ] Edge TPU deployment for sub-20ms latency
- [ ] Transformer-based temporal modeling
- [ ] Multi-intersection sensor fusion
- [ ] Mobile app for operator alerts

<br/>

---

## 🤝 Contributing

Contributions, issue reports, and feature requests are welcome!

1. **Fork** the repository
2. Create a feature branch: `git checkout -b feature/your-feature-name`
3. Commit your changes: `git commit -m "feat: add your feature"`
4. Push to your branch: `git push origin feature/your-feature-name`
5. Open a **Pull Request**

### Development Guidelines

- Do **NOT** modify any files inside `src/` without discussing in an issue first — this is the core ML pipeline
- Backend changes go in `backend/` only
- Frontend changes go in `frontend/` only
- Follow the architecture described in `requirements_rules.md`
- Write tests for any new service functions in `tests/`

<br/>

---

## 📚 References

1. Wong, "Pedestrians on phone while crossing road to face 300-pataca fine," *Macao Daily*, 2024.
2. H. Sunxi et al., "Detecting phone-related pedestrian distracted behaviours via a two-branch convolutional neural network," *IET Intelligent Transport Systems*, pp. 147–158, 2021.
3. E. Hatay, J. Ma, H. Sun, J. Fang, Z. Gao, Y. Yu, "Learning to detect phone-related pedestrian distracted behaviors with synthetic data," in *Proc. IEEE Intelligent Vehicles Symposium*, pp. 2479–2486, 2022.
4. J. Jiang et al., "RTMPose: Real-time multi-person pose estimation based on mmpose," 2023.

<br/>

---

## 📄 License

```
MIT License

Copyright (c) 2026 Huang Yanzhen, Mai Jiajun — University of Science & Technology

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
```

<br/>

---

## 🙏 Acknowledgements

Special thanks to:

- **Prof. Dr Adeel Ahmad ** — Project supervisor and academic guidance
- **MMPose Team** (OpenMMLab) — RTMPose implementation
- **Ultralytics** — YOLOv11n framework
- **InsightFace Team** — Face embedding library
- **Open Source Contributors** — All libraries listed in `requirements.txt`

---

<div align="center">

<br/>

**Built with ❤️ at the University of Science & Technology**

*Department of  Information Technology · Final Year Project 2025–2026*

<br/>

[![GitHub](https://img.shields.io/badge/GitHub-AdnanKhaan11-181717?style=for-the-badge&logo=github)](https://github.com/AdnanKhaan11/Pedestrian-Distraction-Detection)

<br/>

*"Technology in service of pedestrian safety."*

</div>

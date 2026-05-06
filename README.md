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
[![University](https://img.shields.io/badge/University-University%20of%20Haripur-blue?style=flat-square)]()
[![Group](https://img.shields.io/badge/Group-09%20(M)-purple?style=flat-square)]()

<br/>

> **Supervisor:** Dr. Adeel Ahmad &nbsp;|&nbsp; **Students:** Adnan Khan *(F22-0431)* &nbsp;·&nbsp; Bilal Asghar *(F22-1813)*
>
> *University of Haripur — Faculty of Science and Technology*
> *Department of Information Technology*

<br/>

---

</div>

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
  - [GPU Setup](#gpu-setup)
  - [Docker Setup](#docker-setup)
- [API Reference](#-api-reference)
- [Frontend Overview](#-frontend-overview)
- [Training Module](#-training-module)
- [Configuration](#-configuration)
- [Results & Performance](#-results--performance)
- [Roadmap](#-roadmap)
- [Contributing](#-contributing)
- [License](#-license)

<br/>

---

## 🌐 Overview

The **Pedestrian Distraction Detection System (PDS)** is a production-grade, real-time AI surveillance platform designed to detect and monitor pedestrians who use mobile phones while crossing roads — one of the leading causes of pedestrian accidents and fatalities worldwide.

### The Problem

Pedestrian distraction caused by smartphones has become a critical public safety crisis in urban environments. Studies show that distracted pedestrians are **4 times more likely** to engage in dangerous behavior such as ignoring traffic signals, failing to look both ways, and crossing at unsafe locations. Each year, thousands of preventable accidents and deaths occur simply because a pedestrian was looking at their phone instead of the road.

Manual monitoring of pedestrian behavior at road crossings is expensive, inconsistent, and impossible to scale. Existing camera systems at intersections are passive — they record footage but cannot automatically identify, log, or alert authorities about distracted pedestrian violations in real time.

### Our Solution

A fully automated AI pipeline that addresses this problem at scale:

1. Detects pedestrians in real-time video streams using state-of-the-art object detection
2. Classifies pedestrian posture using a novel **2-Channel 3D Behavioral Feature** approach
3. Confirms phone usage with a fine-tuned **YOLOv11n detector**
4. Captures and stores the violator's face for record-keeping and deduplication
5. Provides a live monitoring dashboard for operators with real-time alerts and analytics

This system provides the technical foundation for automated pedestrian safety enforcement, enabling city authorities and traffic monitoring agencies to proactively identify and respond to dangerous distracted walking behavior before accidents occur.

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
- Violation analytics and reporting

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
│  3. 2-Ch 3D Feature  │           │  ● training_logs                   │
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
│   │   ├── pose_feature_generator.py
│   │   └── ...
│   ├── pipeline/                   ← Orchestration pipelines
│   │   ├── inference_pipeline.py   ← Main inference entry point
│   │   ├── posture_training_pipeline.py
│   │   └── phone_training_pipeline.py
│   ├── serving/                    ← API predictor + schemas
│   │   ├── predictor.py
│   │   └── schemas.py
│   ├── config/                     ← Config loading + constants
│   ├── entity/                     ← Pydantic config entities
│   └── utils/                      ← Helpers, logging, OpenCV utils
│
├── 📂 configs/                     ← YAML configuration files
│   └── config.yaml                 ← Main system config
│
├── 📂 scripts/                     ← Training entry points
│   ├── train_posture_model.py
│   └── train_phone_model.py
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

Before installing, make sure you have the following:

| Requirement | Version | Check Command |
|---|---|---|
| Python | 3.10+ | `python --version` |
| Node.js | 18+ | `node --version` |
| npm | 9+ | `npm --version` |
| MongoDB | 6.0+ | `mongod --version` |
| Git | Any | `git --version` |
| CUDA *(GPU users)* | 11.8 or 12.1 | `nvcc --version` |

> 💡 **GPU is optional but strongly recommended.** The system runs fully on CPU but achieving real-time 30+ FPS performance requires a CUDA-capable GPU.

---

### Installation

#### 1. Clone the Repository

```bash
git clone https://github.com/AdnanKhaan11/Pedestrian-Distraction-Detection.git
cd Pedestrian-Distraction-Detection
```

#### 2. Set Up Python Virtual Environment

```bash
# Create virtual environment
python -m venv youfocus

# Activate it
# On Windows:
youfocus\Scripts\activate

# On macOS/Linux:
source youfocus/bin/activate
```

#### 3. Install Python Dependencies

**CPU Users:**
```bash
# Install PyTorch (CPU version)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Install MMPose and its dependencies
pip install -U openmim
mim install mmengine==0.10.4
mim install "mmcv==2.1.0"   OR 
mim install "mmcv-lite==2.1.0"
mim install "mmpose==1.3.1"
mim install "mmdet==3.3.0"

# Install all remaining dependencies
pip install -r requirements.txt
```

**GPU Users — see [GPU Setup](#gpu-setup) section below before installing PyTorch.**

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

# Open and fill in your values
# Windows:
notepad .env
# macOS/Linux:
nano .env
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
DEVICE=cpu                          # Change to "cuda" for GPU
CONFIDENCE_THRESHOLD=0.75
ALERT_CONFIDENCE_THRESHOLD=0.80
FACE_SIMILARITY_THRESHOLD=0.85
ALERT_DEDUP_WINDOW_SECONDS=60

# Logging
LOG_LEVEL=INFO
```

---

### GPU Setup

> ⚡ **Skip this section if you are using CPU only.**

GPU acceleration dramatically improves inference speed. Follow these steps carefully to install the correct CUDA-compatible version of PyTorch.

#### Step 1 — Check Your CUDA Version

```bash
nvcc --version
# or
nvidia-smi
```

Look for the CUDA version in the output (e.g., `CUDA Version: 11.8` or `CUDA Version: 12.1`).

#### Step 2 — Install the Correct PyTorch Version

**For CUDA 11.8:**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

**For CUDA 12.1:**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

**For CUDA 12.4+:**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

> ⚠️ **Do not mix CUDA versions.** Install the PyTorch build that exactly matches your installed CUDA version.

#### Step 3 — Verify GPU Is Detected

```python
import torch
print(torch.cuda.is_available())       # Should print: True
print(torch.cuda.get_device_name(0))   # Should print your GPU name
```

#### Step 4 — Enable GPU in `.env`

```env
DEVICE=cuda
```

#### Step 5 — Install MMPose with GPU Support

```bash
pip install -U openmim
mim install "mmengine==0.10.4"
mim install "mmcv==2.1.0"
mim install "mmpose==1.3.2"
mim install "mmdet==3.2.0"
pip install -r requirements.txt
```

#### GPU Requirements Summary

| Component | Minimum | Recommended |
|---|---|---|
| GPU VRAM | 4 GB | 8 GB+ |
| CUDA Version | 11.8 | 12.1+ |
| cuDNN | 8.x | 9.x |
| Driver (Windows) | 452.39+ | Latest |
| Driver (Linux) | 450.80+ | Latest |

> 💡 **Tested GPUs:** NVIDIA RTX 3060, RTX 3080, RTX 4070, Tesla T4. AMD GPUs are not supported — CUDA is NVIDIA only.

---

### Running the System
 run the full system locally.
You need **2 terminal windows** to 
#### Terminal 1 — Start the Backend

```bash
# anaconda recommended
conda create --prefix <Path> venv

conda activate <path of venv>\venv
# Make sure virtual environment is activated first
youfocus\Scripts\activate        # Windows
source youfocus/bin/activate     # Linux/macOS

# Start FastAPI backend from project root
cd Pedestrian-Distraction-Detection
cd backend
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Available at:
- **API:** `http://127.0.0.1:8000`
- **Swagger Docs:** `http://127.0.0.1:8000/docs`
- **Health Check:** `http://127.0.0.1:8000/health`

#### Terminal 2 — Start the Frontend

```bash

step 1: go and download frontend of this project.
`https://github.com/AdnanKhaan11/Pedestrian-Distraction-Frontend.git` 

cd Pedestrian-Distraction-Frontend

#install dependencies
npm install

#run the frontend
npm run dev
```

Available at: **`http://localhost:5173`**

> ✅ Once all two are running, open `http://localhost:5173` in your browser and navigate to **Dashboard → Start Camera** to begin real-time detection.



---

## 📡 API Reference

Full interactive API documentation is available at `http://localhost:8000/docs` when the backend is running.

### Detection

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/api/detect` | Submit a single image for detection |
| `WS` | `/ws/stream` | WebSocket for real-time frame streaming |

### Alerts

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/alerts` | Get paginated alerts |
| `PATCH` | `/api/alerts/{id}/resolve` | Mark an alert as resolved |
| `DELETE` | `/api/alerts/{id}` | Delete an alert |

### Dashboard

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/dashboard/stats` | Summary statistics |
| `GET` | `/api/dashboard/timeline` | Hourly violation counts (last 24h) |

### Faces

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/faces` | List all detected faces |
| `DELETE` | `/api/faces/{id}` | Remove a face record |

### Training

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/api/train` | Start a training job |
| `GET` | `/api/train/status` | Current training status |
| `GET` | `/api/train/current` | Currently running training details |
| `DELETE` | `/api/train/current` | Stop running training job |
| `GET` | `/api/train/history?limit=20` | Past training jobs list |
| `WS` | `/ws/train-logs` | Real-time training log stream |

### Settings

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/settings` | Get current system settings |
| `PUT` | `/api/settings` | Update system settings |

<br/>

---

## 🖥️ Frontend Overview

The frontend is a modern React 19 SPA with the following main pages:

| Page | Route | Description |
|---|---|---|
| **Dashboard** | `/` | Live stats, active violators count, timeline chart, alerts panel |
| **Detection** | `/detection` | Full-screen camera with real-time bounding boxes |
| **Reports** | `/reports` | Detection history, face gallery, export tools |
| **Training** | `/training` | Model training with live progress, pause/resume/stop controls |
| **Settings** | `/settings` | System config, detection thresholds |

<br/>

---

## 🧪 Training Module

The system includes a full UI-based model training workflow.

### How to Train a Model

1. Navigate to the **Training** page from the sidebar
2. Select the model to train: **Posture Classifier** or **Phone Detector**
3. Configure hyperparameters (Epochs, Learning Rate, Batch Size)
4. Click **"Start Training"**
5. Monitor live progress:
   - Real-time progress bar with epoch counter
   - Live loss and accuracy metrics
   - Scrollable training log output
6. Use control buttons during training:
   - **Pause** — suspends training (resumable)
   - **Resume** — continues from last checkpoint
   - **Stop** — cancels training entirely
7. View completed runs in **Training History** and delete old records as needed

### Training Datasets

| Dataset | Samples | Purpose |
|---|---|---|
| Posture Dataset | 26,748 total | Train the 3D-CNN posture classifier |
| Phone Dataset | 11,508 total | Train the YOLOv11n phone detector |

<br/>

---

## ⚙️ Configuration

### Infrastructure Config (`.env` file)
Used for server-level settings. Never changes at runtime.

| Variable | Default | Description |
|---|---|---|
| `MONGODB_URI` | `your mongodb uri` | MongoDB connection string |
| `DB_NAME` | `pedestrian_detection` | Database name |
| `SECRET_KEY` | *(required) in production not in local* | Signing secret |
| `CORS_ORIGINS` | `http://localhost:5173` | Allowed frontend origins |
| `DEVICE` | `cpu` | Inference device: `cpu` or `cuda` |
| `LOG_LEVEL` | `INFO` | Logging level |

### Runtime Config (Settings Page)
Adjustable from the Settings UI without restarting the server.

| Setting | Default | Description |
|---|---|---|
| `detection_confidence_threshold` | `0.75` | Minimum confidence to register a detection |
| `alert_confidence_threshold` | `0.80` | Minimum confidence to create an alert |
| `face_similarity_threshold` | `0.85` | Cosine similarity threshold for face deduplication |
| `alert_dedup_window_seconds` | `60` | Seconds before same face can trigger a new alert |
| `frame_sample_rate` | `2` | Frames per second sent to backend |

<br/>

---

## 📊 Results & Performance

### Model Accuracy

| Model | Architecture | Accuracy / mAP@50 | vs. Baseline |
|---|---|---|---|
| **Posture Classifier** | 2-Ch 3D CNN | **91.78%** | +17.5% over prior work |
| **Phone Detector** | YOLOv11n | **98.1% mAP@50** | +4.5% over YOLOv8n |

### System Performance

| Metric | Value |
|---|---|
| Mean inference time | **85ms** per frame |
| System throughput | **~41.2 FPS** |
| False positive rate | **3.4%** |
| Face match precision | **94.7%** |

### Comparison with Prior Work

| Approach | Accuracy |
|---|---|
| Prior two-branch CNN approach | 74.3% |
| Single-channel CNN | 77.1% |
| LSTM baseline | 82.4% |
| **Our 2-Ch 3D CNN (this work)** | **91.78%** |

<br/>

---

## 🗺️ Roadmap

- [x] RTMPose-based pose estimation pipeline
- [x] 2-Channel 3D feature engineering
- [x] 3D-CNN posture classifier (91.78% accuracy)
- [x] YOLOv11n phone detector (98.1% mAP@50)
- [x] DistractionFusion state machine
- [x] React frontend with live camera feed
- [x] FastAPI backend with WebSocket streaming
- [x] MongoDB integration with Motor async driver
- [x] Face embedding and deduplication system
- [x] UI-based training module with pause/resume/stop
- [ ] Docker full-stack deployment
- [ ] Multi-camera concurrent session support
- [ ] Edge TPU deployment for sub-20ms latency
- [ ] Transformer-based temporal modeling
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

## 📄 License

```
MIT License

Copyright (c) 2026 Adnan Khan, Bilal Asghar — University of Haripur

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

- **Dr. Adeel Ahmad** — Project supervisor and academic guidance
- **MMPose Team** (OpenMMLab) — RTMPose implementation
- **Ultralytics** — YOLOv11n framework
- **facenet-pytorch Team** — Face embedding library
- **Open Source Contributors** — All libraries listed in `requirements.txt`

---

<div align="center">

<br/>

**Built with ❤️ at the University of Haripur**

*Department of Information Technology · Final Year Project 2025–2026*

<br/>

[![GitHub](https://img.shields.io/badge/GitHub-AdnanKhaan11-181717?style=for-the-badge&logo=github)](https://github.com/AdnanKhaan11/Pedestrian-Distraction-Detection)

<br/>

*"Technology in service of pedestrian safety."*

</div>

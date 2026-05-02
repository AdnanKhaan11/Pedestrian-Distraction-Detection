## PART 1 — FUTURE SESSION PROMPT

Paste this at the start of your next session:

---

```
I am working on a Pedestrian Distraction Detection System (Group 09).
Here is the current state of my project:

PROJECT ROOT: D:\pedestrian-phone-detection\

TECH STACK:
- ML: RTMDet + RTMPose + MLP3d + YOLO11 + DistractionFusion
- Backend: FastAPI + MongoDB Atlas + Motor
- Frontend: React + Vite + Chart.js + Tailwind
- MLOps: MLflow + DAGsHub

CURRENT STATUS:
Backend  → FULLY WORKING at http://localhost:8000
ML Pipeline → FULLY WORKING (all 4 models load and run)
Frontend → Dark theme done, API wired, needs E2E testing

BACKEND START COMMAND:
cd D:\pedestrian-phone-detection\backend
uvicorn app.main:app --reload --port 8000

FRONTEND START COMMAND:
cd D:\pedestrian-phone-detection\frontend
npm run dev

TEST INFERENCE COMMAND:
python scripts/run_inference.py --image "path/to/image.jpg"
python scripts/run_inference.py --video "path/to/video.mp4" --frame-step 10

VERIFIED WORKING PACKAGES:
Python 3.10.20
torch 2.2.2+cpu
mmcv 2.2.0 (torch2.2.0 wheel, mmdet version cap patched to 2.3.0)
mmdet 3.3.0
mmpose 1.3.1
setuptools 69.5.1

NEXT TASKS:
1. Upload frontend folder for final audit
2. Test camera → WebSocket → bounding boxes end-to-end
3. Verify dashboard shows real MongoDB data
4. Test training UI with WebSocket log streaming
5. Docker setup

IMPORTANT RULES:
- src/ folder is READ ONLY — never modify ML pipeline
- Always run backend from inside backend/ folder
- Frontend api.js sends "frame" not "image" key
- mmdet __init__.py has mmcv_maximum_version patched to 2.3.0
- .env has no comments and no "set" prefix
```

---

## PART 2 — ISSUES WE SOLVED (with steps)

---

**Issue 1 — MLflow not connecting to DAGsHub**
Cause: `.env` had `set` commands, `#` comments, and wrong variable names.
Fix:
1. Remove all `set ` prefixes from `.env`
2. Remove all `#` comment lines
3. Rename `DAGSHUB_TOKEN` → `MLFLOW_TRACKING_PASSWORD`
4. Add `from dotenv import load_dotenv` + `load_dotenv()` at top of every training script

---

**Issue 2 — YOLO training crashed with device=auto**
Cause: No GPU on machine, `device="auto"` tries CUDA and fails.
Fix:
1. Open `configs/config.yaml`
2. Change `device: "auto"` → `device: "cpu"`

---

**Issue 3 — YOLO data.yaml wrong paths**
Cause: Relative paths like `../data/processed/train` got doubled by YOLO.
Fix: Rewrite `data/processed/phone_dataset/data.yaml` as:
```yaml
path: D:/pedestrian-phone-detection/data/processed/phone_dataset
train: train/images
val: valid/images
test: test/images
nc: 1
names:
  0: phone
```

---

**Issue 4 — Posture training FileNotFoundError (.npy files)**
Cause: `.npy` files were in wrong folders (`data/1dnpy/`, `data/3dnpy/`). Also `1dnpy` files had wrong shape `(302, 1716)` — trainer needs 5D shape.
Fix:
1. Copy `3dnpy` folder into `data/processed/posture_features/`
2. Rename `1dnpy` folder to `1dnpy_backup` and move it out of `posture_features/`
3. Update config to point to `data/processed/posture_features/3dnpy/`

---

**Issue 5 — MMPose config .py files missing**
Cause: Only `.pth` checkpoint files existed. MMPose also needs `.py` config files.
Fix: Download using Python:
```python
import requests
# RTMDet config
url = 'https://raw.githubusercontent.com/open-mmlab/mmpose/main/projects/rtmpose/rtmdet/person/rtmdet_nano_320-8xb32_coco-person.py'
open('artifacts/mmpose/configs/rtmdet_nano_320-8xb32_coco-person.py', 'w').write(requests.get(url).text)

# RTMPose config
url = 'https://raw.githubusercontent.com/open-mmlab/mmpose/main/projects/rtmpose/rtmpose/body_2d_keypoint/rtmpose-t_8xb256-420e_coco-256x192.py'
open('artifacts/mmpose/configs/rtmpose-tiny_simcc-aic-coco_pt-aic-coco_420e-256x192.py', 'w').write(requests.get(url).text)
```
Then update both paths in `configs/mmpose_config.yaml` to point to `artifacts/mmpose/configs/`

---

**Issue 6 — mmcv._ext DLL load failed**
Cause: Wrong mmcv wheel installed (source `.tar.gz` instead of pre-built `.whl`).
Fix:
```python
import requests
url = 'https://download.openmmlab.com/mmcv/dist/cpu/torch2.2.0/mmcv-2.2.0-cp310-cp310-win_amd64.whl'
r = requests.get(url, stream=True, timeout=120)
with open('mmcv-2.2.0-cp310-cp310-win_amd64.whl', 'wb') as f:
    for chunk in r.iter_content(chunk_size=8192):
        f.write(chunk)
```
```bash
pip install mmcv-2.2.0-cp310-cp310-win_amd64.whl
```

---

**Issue 7 — mmdet rejects mmcv 2.2.0**
Cause: `mmdet 3.3.0` has hardcoded version cap `< 2.2.0`.
Fix:
1. Open in Notepad: `youfocus\lib\site-packages\mmdet\__init__.py`
2. Find: `mmcv_maximum_version = '2.2.0'`
3. Change to: `mmcv_maximum_version = '2.3.0'`
4. Save and close

---

**Issue 8 — pkg_resources missing**
Cause: `setuptools 82.x` removed `pkg_resources`.
Fix:
```bash
pip install --force-reinstall setuptools==69.5.1
```

---

**Issue 9 — Database not connected (RuntimeError)**
Cause: Backend was run with `python -m uvicorn backend.app.main:app` from project root. This made Python see `backend.app.database.Database` and `app.database.Database` as two different classes — one connected, one not.
Fix: Always run from inside `backend/` folder:
```bash
cd backend
uvicorn app.main:app --reload --port 8000
```

---

**Issue 10 — Training always returns "already running" after restart**
Cause: MongoDB kept old job with `status: "running"` from previous session. In-memory job ID resets to `None` on restart — they go out of sync.
Fix: Added to `main.py` lifespan after `SeedData.seed_all(db)`:
```python
await db.training_logs.update_many(
    {"status": "running"},
    {"$set": {"status": "error", "error_message": "Server restarted"}}
)
```

---

**Issue 11 — /api/devices returned 404**
Cause: `devices` route was never created or registered.
Fix:
1. Created `backend/app/routes/devices.py` with `GET /api/devices`
2. Added to `main.py`: `from app.routes import devices` and `app.include_router(devices.router)`

---

**Issue 12 — is_violation always False**
Cause: `inference_service.py` checked `fusion_state == "USING"` but pipeline returns integer state (`128=safe`, `32=distracted`).
Fix in `_build_pedestrian_result()`:
```python
posture_label = person_result.get("posture", "safe")
is_violation = posture_label in ("distracted", "using_or_suspicious")
```

---

**Issue 13 — Confidence values always hardcoded**
Cause: `posture_confidence` and `phone_confidence` not in pipeline output. Pipeline returns `score_text: "0.96"` instead.
Fix:
```python
score_text = person_result.get("score_text", "0.0")
posture_confidence = float(score_text) if score_text else 0.0
```

---

**Issue 14 — HTTP POST field name inconsistency**
Cause: `DetectRequest` used field `image` but WebSocket used key `frame`. Frontend was confused.
Fix in `detect.py`: Renamed `image: str` → `frame: str` in `DetectRequest`.

---

**Issue 15 — Frontend AppContext JSX error**
Cause: File named `AppContext.js` but contained JSX syntax. Vite requires `.jsx` extension.
Fix: Rename `AppContext.js` → `AppContext.jsx`

---

**Issue 16 — Camera sends frames but zero detections**
Cause: `useWebSocket.js` sent `{image: frameBase64}` but backend reads `message.get("frame")`.
Fix in `useWebSocket.js`:
```javascript
wsRef.current.send(JSON.stringify({
    frame: frameBase64,
    session_id: sessionId,
    timestamp: new Date().toISOString(),
}))
```

---

**Issue 17 — Bounding boxes flickering and never visible**
Cause: `captureAndSendFrame()` reset `canvas.width` every 100ms which clears the canvas entirely, erasing drawn boxes immediately.
Fix: Use a separate offscreen canvas for frame capture. Never touch display canvas inside `captureAndSendFrame()`.

---

**Issue 18 — Chart.js lines invisible**
Cause: Chart.js uses canvas API which cannot resolve CSS variables like `var(--accent-cyan)`.
Fix: Replace all CSS variables in Chart.js config with real hex values:
```
var(--accent-cyan)   → #00D4FF
var(--color-danger)  → #FF3B3B
var(--bg-elevated)   → #2D3B52
var(--text-primary)  → #F1F5F9
var(--border-default)→ #1E2D42
```

---

**Issue 19 — Training status never shows as running in UI**
Cause: Backend returns `{status: "running"}` but `Settings.jsx` reads `status.is_running` which is `undefined`.
Fix in `Settings.jsx`:
```javascript
setIsTraining(status.status === "running")
```

---

**Issue 20 — Reports and ActivityFeed show fake data**
Cause: Both components had hardcoded arrays and never called any API.
Fix: Add `useEffect` to both components calling `getDashboardTimeline()` and `getAlerts()` from `api.js`.
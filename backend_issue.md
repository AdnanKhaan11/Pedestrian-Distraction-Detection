Now I have the complete picture. Here's the full report:

---

## ✅ CHATGPT'S ISSUES — ALREADY FIXED IN YOUR CURRENT CODE

Before anything — ChatGPT's entire list is **outdated**. Every issue it raised is already fixed:

| ChatGPT Issue | Reality |
|---|---|
| job_id vs _id mismatch | Fixed — `job_data["_id"] = job.job_id` + `job.dict()` stores both |
| Face→Alert data flow | Fixed — `_enrich_violation_pedestrian` chains them with await |
| face_id not in pedestrians | Fixed — `pedestrian.face_id = face_id` before DB save |
| Memory leak (session TTL) | Fixed — TTL cleanup in `_get_session_runtime_params` |
| Training logs unbounded | Fixed — `logs = logs[-500:]` is present |
| Cancel not killing process | Fixed — `process.terminate()` via `terminate_current_process()` |

Do not waste time on any of ChatGPT's suggestions. They are already done.

---

## 🚨 REAL ISSUES FOUND (5 actual bugs)

---

**BUG 1 — CRITICAL: Training always says "already running" after server restart**

`is_training_running()` queries MongoDB for `{status: "running"}`. After a server restart, MongoDB still has a leftover document with `status: "running"` from the previous session, but the in-memory `_current_job_id` resets to `None`. So every `POST /api/train` returns `"Training job already running"` forever, and `GET /api/train/status` returns `"none"`. The two are permanently out of sync.

Fix — add this in `main.py` lifespan, after `SeedData.seed_all(db)`:
```python
await db.training_logs.update_many(
    {"status": "running"},
    {"$set": {"status": "error", "error_message": "Server restarted"}}
)
```

---

**BUG 2 — CRITICAL: `/api/devices` route does not exist**

`main.py` registers: `test, detect, faces, alerts, training, dashboard, settings`. No `devices` router. The frontend `Devices.jsx` calls `GET /api/devices` → gets 404 → devices page shows nothing. MongoDB even has seed data for devices but it's unreachable.

Fix — create `backend/app/routes/devices.py` with a `GET /api/devices` route querying `db.devices`, and add `app.include_router(devices.router)` to `main.py`.

---

**BUG 3 — CRITICAL: Camera canvas used for both frame capture AND bounding boxes — boxes get erased every 100ms**

`captureAndSendFrame()` sets `canvas.width = video.videoWidth` every `frameInterval` ms. Setting `.width` on a canvas **resets and clears it entirely**. So every 100ms the bounding boxes get wiped. The boxes are drawn by `drawDetectionBoxes()` but immediately erased by the next capture cycle. Users effectively never see boxes — they flash for <100ms.

Fix — use a separate **offscreen canvas** for frame capture, keep the visible canvas only for drawing boxes:
```javascript
// In captureAndSendFrame(), replace canvas usage with:
const offscreen = document.createElement('canvas');
offscreen.width = video.videoWidth || 640;
offscreen.height = video.videoHeight || 480;
const offCtx = offscreen.getContext('2d');
offCtx.drawImage(video, 0, 0, offscreen.width, offscreen.height);
const frameBase64 = offscreen.toDataURL('image/jpeg', 0.8).split(',')[1];
// Do NOT touch canvasRef at all here
```
And in `startCamera()`, set the display canvas size once:
```javascript
videoRef.current.onloadedmetadata = () => {
  canvasRef.current.width = videoRef.current.videoWidth;
  canvasRef.current.height = videoRef.current.videoHeight;
};
```

---

**BUG 4 — MEDIUM: Training status `is_running` field mismatch between backend and frontend**

Backend `GET /api/train/status` returns `{"status": "running", ...}`. Frontend `Settings.jsx` reads:
```javascript
setIsTraining(status.is_running);  // ← undefined always
```
`status.is_running` doesn't exist. The field is `status.status`. So `isTraining` is always `false` — the UI never shows training as active.

Fix — change one line in `Settings.jsx`:
```javascript
setIsTraining(status.status === "running");
```

---

**BUG 5 — MEDIUM: Reports page never fetches from API — always shows hardcoded dummy data**

`Reports.jsx` has hardcoded arrays `[65, 59, 80, 81, 56, 55, 40]` for detections and violations. It never calls `getDashboardTimeline()`. The real data is in MongoDB and served by `/api/dashboard/timeline` but Reports never asks for it.

Fix — add `useEffect` to `Reports.jsx` that calls `getDashboardTimeline()` and maps the hourly data to chart labels/datasets. Replace hardcoded arrays with the API response.

---

## 📋 FIX ORDER FOR YOUR AGENT

```
1. main.py        → add stuck training job reset in lifespan (4 lines)
2. routes/        → create devices.py + register in main.py
3. WebcamDetector → use offscreen canvas for capture, keep display canvas clean
4. Settings.jsx   → change status.is_running → status.status === "running"
5. Reports.jsx    → fetch from getDashboardTimeline() instead of hardcoded data
```


# agent prompt

```
You are a senior debugger, QA engineer, and software tester for a real-time AI pedestrian distraction detection system.

STEP 1 — READ ISSUE FILES FIRST
Read both files completely before touching any code:
- backend_issue.md  → all backend bugs to fix
- frontend_issue.md → all frontend bugs to fix

STEP 2 — FIX ALL ISSUES (no .md files, no documentation, code only)

Backend fixes (backend/app/):
- main.py lifespan: add stuck training job reset after SeedData.seed_all(db)
- routes/: create devices.py with GET /api/devices querying db.devices, register in main.py
- All other fixes listed in backend_issue.md

Frontend fixes (frontend/src/):
- hooks/useWebSocket.js: remove auto-connect useEffect
- components/detection/WebcamDetector.jsx: offscreen canvas for capture, keep display canvas only for boxes
- pages/Settings/Settings.jsx line 51: status.is_running → status.status === "running"
- pages/Detection/Detection.jsx: dark theme + real stats (no hardcoded values)
- components/dashboard/ActivityFeed.jsx: dark theme + fetch from getAlerts()
- components/dashboard/RealTimeMonitor.jsx: replace fake devices with getDevices() API call
- pages/Reports/Reports.jsx: add useEffect + getDashboardTimeline() real data
- All other fixes listed in frontend_issue.md

STEP 3 — MODEL LOADING AUDIT
After fixes, inspect and report:
- backend/app/ml/pipeline.py → what does load_pipeline() do exactly?
- How many models are loaded? List each one with its name and file path
- Which model files are used from artifacts/ folder:
    artifacts/mmpose/checkpoints/
    artifacts/posture_classifier/weights/best_posture_model.pth
    artifacts/phone_detector/weights/best.pt
- Confirm models load successfully on startup (check health endpoint: GET /health → model_loaded: true)
- If any model fails to load, report the exact error and file path

STEP 4 — INTEGRATION TESTING (test every module end-to-end)
Run backend: cd backend && uvicorn app.main:app --reload --port 8000
Run frontend: cd frontend && npm run dev

Test each endpoint directly:
1. GET  /health                    → database_connected: true, model_loaded: true
2. GET  /api/dashboard/stats       → real numbers, not 500 error
3. GET  /api/dashboard/timeline    → labels + detections + violations arrays
4. GET  /api/alerts                → list (even if empty)
5. GET  /api/devices               → list from MongoDB (not 404)
6. GET  /api/settings              → full settings object
7. PUT  /api/settings              → 200 success
8. GET  /api/train/status          → status field present (not is_running)
9. POST /api/train                 → starts OR gives clear error (not "already running" if nothing running)
10. WS  /ws/stream                 → connects, accepts frame, returns DetectionResult JSON

Frontend UI tests (open browser at localhost:5173):
1. Dashboard → StatusCards show real numbers from MongoDB
2. Dashboard → Chart shows real hourly data (not dummy)
3. Dashboard → RealTimeMonitor shows real devices from /api/devices
4. Dashboard → ActivityFeed shows real alerts (dark theme, not white)
5. Dashboard → Camera starts, sends frames, bounding boxes visible and stable (not flickering)
6. Detection page → fully dark themed, no white panels
7. Devices page → real device data, not empty
8. Reports page → real chart data from API
9. Settings page → loads values, Save button works
10. Settings page → Start Training works, progress bar appears, log streams

STEP 5 — REPORT RESULTS (on screen only, no files)
For each test: PASS ✅ or FAIL ❌ with exact error message
For model loading: list each model name, file path, load status
For any remaining failure: exact file + line number + what needs to change

RULES:
- No .md files. No documentation files. Zero.
- Fix code directly. Report results in chat only.
- Do not refactor working code. Only fix what is listed.
- If a fix breaks something else, revert and report it.
- src/ ML pipeline folder → READ ONLY, never modify.
```
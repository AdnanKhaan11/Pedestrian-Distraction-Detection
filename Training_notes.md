# training notes: (yolo models)
> i run this command to prove that training pipeline work or not so it work successfully. congratulation to me.

 `python scripts/train_phone_model.py`


---

## Which model is training?

It is training **`best.pt`** — your already trained model.

Look at the log:
```
Starting phone detector training from: artifacts\phone_detector\weights\best.pt
model=artifacts\phone_detector\weights\best.pt
```

---

## What this means:

| File | Role |
|---|---|
| `yolo11n.pt` | Original pretrained base model from Ultralytics (scratch weights) |
| `best.pt` | YOUR model — already fine-tuned on phone detection data |

The script loads `best.pt` as the starting point and **continues training on top of it**. This is called **fine-tuning** — not training from scratch.

So it is doing:
```
best.pt (your trained weights)
      ↓
Train more epochs on your phone dataset
      ↓
saves new improved best.pt → artifacts/phone_detector/weights/best.pt
```

---

## When would you use `yolo11n.pt`?

Only if you wanted to train from scratch — completely fresh with no prior phone knowledge. That would take much longer and need more data.

**Your current setup is correct.** Fine-tuning `best.pt` is the right approach — faster convergence, better accuracy, less data needed.

# posture model [Training]

`python scripts/train_posture_model.py`
  
this script run sucessfully and model training start 
so wverything fine.ok


# The Moste Headach Issue In My This Project Is This.

## Complete Solution Guide — MMPose Setup on Windows CPU

---

## The Core Problem

Your project uses MMPose (RTMDet + RTMPose) for person detection and pose estimation. The inference pipeline was failing because of a chain of 6 issues that needed to be solved in order.

---

## Issue 1 — MLflow not connecting to DAGsHub

**Cause:** `.env` file had `set` commands and `#` comments which `python-dotenv` cannot parse. Also used wrong variable names (`DAGSHUB_TOKEN` instead of `MLFLOW_TRACKING_PASSWORD`).

**Fix:** Clean `.env` file with no comments, no `set` prefix, correct variable names:
```env
MLFLOW_TRACKING_USERNAME=your_username
MLFLOW_TRACKING_PASSWORD=your_dagshub_token
MLFLOW_TRACKING_URI=https://dagshub.com/username/repo.mlflow
```
Add `load_dotenv()` at top of every training script.

---

## Issue 2 — Missing MMPose config `.py` files

**Cause:** Config files were not in the project. Only checkpoint `.pth` files existed. MMPose needs both the config file AND the checkpoint file to load.

**Fix:** Download correct config files from GitHub using Python requests:
```python
import requests

# RTMDet person detector config
url = 'https://raw.githubusercontent.com/open-mmlab/mmpose/main/projects/rtmpose/rtmdet/person/rtmdet_nano_320-8xb32_coco-person.py'
open('artifacts/mmpose/configs/rtmdet_nano_320-8xb32_coco-person.py', 'w').write(requests.get(url).text)

# RTMPose config
url = 'https://raw.githubusercontent.com/open-mmlab/mmpose/main/projects/rtmpose/rtmpose/body_2d_keypoint/rtmpose-t_8xb256-420e_coco-256x192.py'
open('artifacts/mmpose/configs/rtmpose-tiny_simcc-aic-coco_pt-aic-coco_420e-256x192.py', 'w').write(requests.get(url).text)
```

---

## Issue 3 — Wrong paths in `mmpose_config.yaml`

**Cause:** Config file paths pointed to `model_config/configs/` but files were downloaded to `artifacts/mmpose/configs/`.

**Fix:** Update `configs/mmpose_config.yaml`:
```yaml
detector:
  config_file: "artifacts/mmpose/configs/rtmdet_nano_320-8xb32_coco-person.py"
  checkpoint_file: "artifacts/mmpose/checkpoints/rtmdet_nano_8xb32-100e_coco-obj365-person-05d8511e.pth"

pose_estimator:
  config_file: "artifacts/mmpose/configs/rtmpose-tiny_simcc-aic-coco_pt-aic-coco_420e-256x192.py"
  checkpoint_file: "artifacts/mmpose/checkpoints/rtmpose-tiny_simcc-aic-coco_pt-aic-coco_420e-256x192-cfc8f33d_20230126.pth"
```

---

## Issue 4 — mmcv not installed correctly (`mmcv._ext` missing)

**Cause:** mmcv was installed from source (`.tar.gz`) instead of pre-built wheel. Source installation does not compile C++ extensions on Windows. Also `mmcv 2.1.0` wheel only exists for torch 2.1.0, not torch 2.2.0.

**Fix:** Download and install `mmcv 2.2.0` wheel built for torch 2.2.0:
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

## Issue 5 — mmdet rejects mmcv 2.2.0

**Cause:** `mmdet 3.3.0` has a hardcoded version cap `mmcv_maximum_version = '2.2.0'` which uses strict `<` comparison. So mmcv 2.2.0 is rejected even though it works fine.

**Fix:** Open this file in Notepad:
```
D:\pedestrian-phone-detection\youfocus\lib\site-packages\mmdet\__init__.py
```
Find line:
```python
mmcv_maximum_version = '2.2.0'
```
Change to:
```python
mmcv_maximum_version = '2.3.0'
```
Save and close.

---

## Issue 6 — `pkg_resources` missing

**Cause:** Newer `setuptools` (82.x) removed `pkg_resources` from default exports. mmengine internally uses it.

**Fix:**
```bash
pip install --force-reinstall setuptools==69.5.1
```

---

## Final Verified Working State

```
Python:   3.10.20
torch:    2.2.2+cpu
mmcv:     2.2.0      ← from torch2.2.0 wheel
mmdet:    3.3.0      ← with version cap patched to 2.3.0
mmpose:   1.3.1
setuptools: 69.5.1   ← pkg_resources accessible
```

---

## Quick Recovery Checklist (for future)

If MMPose breaks again, check in this order:

```
Step 1 → python -c "import mmcv._ext; print('OK')"
         FAIL → reinstall mmcv wheel from torch2.2.0

Step 2 → python -c "import mmdet; print(mmdet.__version__)"
         FAIL → patch mmcv_maximum_version in mmdet __init__.py

Step 3 → python -c "import pkg_resources; print('OK')"
         FAIL → pip install --force-reinstall setuptools==69.5.1

Step 4 → check configs/mmpose_config.yaml paths exist on disk
         FAIL → re-download config files from GitHub

Step 5 → python -c "from src.components.mmpose_loader import MMPoseLoader"
         Must load both checkpoints successfully
```

---

## Run Inference

```bash
python scripts/run_inference.py --image "path/to/any/image.jpg"
```


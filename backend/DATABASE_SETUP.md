# 📊 Database Setup — STEP 2 Completed

## Overview

MongoDB database has been fully configured with all necessary collections and indexes.

## Collections Created

### 1. **detections**
Stores all detection results (violations and non-violations).

**Fields:**
- `_id` — MongoDB ObjectId
- `device_id` — Device identifier
- `timestamp` — Detection timestamp
- `is_violation` — Boolean flag for violations
- `severity` — Alert severity level
- `frame_id` — Frame identifier
- `session_id` — Camera session ID
- `pedestrians` — Array of detected pedestrians
- `overall_confidence` — Combined confidence score
- `processing_time_ms` — Inference time

**Indexes:**
- `timestamp` (descending) — Time-range queries
- `is_violation` (ascending) — Filter violations
- `session_id` (ascending) — Group by session
- `timestamp + is_violation` (compound)

---

### 2. **alerts**
Stores alerts triggered by violations.

**Fields:**
- `_id` — MongoDB ObjectId
- `alert_id` — UUID
- `detection_id` — Linked detection
- `face_id` — Linked face (nullable)
- `severity` — HIGH / MEDIUM / LOW
- `title` — Short description
- `description` — Full description
- `confidence` — Detection confidence
- `timestamp` — Alert creation time
- `resolved` — Whether operator resolved it
- `snapshot_base64` — Face crop image

**Indexes:**
- `timestamp` (descending) — Recent alerts
- `resolved` (ascending) — Unresolved alerts
- `severity` (ascending) — Filter by severity
- `face_id` (ascending) — Alerts per person
- `timestamp + resolved` (compound)

---

### 3. **faces**
Stores unique face embeddings and crops.

**Fields:**
- `_id` — MongoDB ObjectId
- `face_id` — UUID
- `embedding` — 512-dim float array (InsightFace)
- `image_base64` — Face crop (160×160)
- `first_seen` — First detection timestamp
- `last_seen` — Most recent detection
- `detection_count` — Total detections
- `violation_count` — Violation count
- `detection_ids` — Array of linked detection UUIDs

**Indexes:**
- `face_id` (unique) — Unique face identifier
- `last_seen` (descending) — Recent faces
- `violation_count` (descending) — Most violating people

---

### 4. **settings**
System-wide settings (singleton pattern).

**Fields:**
- `_id` = "system_settings"
- `detection_confidence_threshold` — 0.75
- `alert_confidence_threshold` — 0.80
- `face_similarity_threshold` — 0.85
- `auto_refresh_interval_ms` — 5000
- `notifications_enabled` — true
- `usage_threshold_percent` — 80
- `active_model` — Model version
- `camera_resolution` — Default resolution
- `max_concurrent_cameras` — 4
- `frame_interval_ms` — 500
- Alert thresholds (HIGH, MEDIUM, LOW)
- Face embedding config
- System flags (enabled, debug_mode)
- `created_at` — Creation timestamp
- `updated_at` — Last update timestamp

**Indexes:**
- `_id` (unique) — Singleton pattern

---

### 5. **devices**
Registered camera/detection devices.

**Fields:**
- `_id` — MongoDB ObjectId
- `device_id` — Unique device ID
- `device_name` — Human-readable name
- `device_type` — webcam / IP camera / etc
- `location` — Physical location
- `is_active` — Active status
- `ip_address` — Device IP
- `port` — Device port
- `status` — online / offline
- `total_detections` — Counter
- `total_violations` — Counter
- `created_at` — Creation timestamp
- `updated_at` — Last update timestamp

**Indexes:**
- `device_id` (unique) — Unique identifier
- `is_active` (ascending) — Active devices

---

### 6. **training_logs**
Training job logs and history.

**Fields:**
- `_id` — MongoDB ObjectId
- `job_id` — UUID
- `model_type` — Model being trained
- `status` — running / completed / error / cancelled
- `started_at` — Job start time
- `completed_at` — Job end time (nullable)
- `config` — Hyperparameters
- `logs` — Array of log lines
- `final_metrics` — accuracy, loss, etc

**Indexes:**
- `job_id` (unique) — Unique job identifier
- `started_at` (descending) — Recent jobs
- `status` (ascending) — Filter by status

---

## Default Data Seeded

### **Default Settings**
- `_id`: "system_settings"
- All thresholds set to sensible defaults
- Face embedding: InsightFace 512-dim
- Auto-refresh: 5000ms (5 seconds)

### **Default Device**
- `device_id`: "default_camera_001"
- `device_name`: "Main Entrance Camera"
- `location`: "Main Entrance"
- `is_active`: true
- `status`: "online"

---

## API Endpoints for Database Verification

### **GET /test-db**
Test database by inserting and reading a document.

**Response (Success):**
```json
{
  "status": "success",
  "message": "Database verification successful",
  "inserted": true,
  "data": {
    "_id": "...",
    "device_id": "test_device",
    "timestamp": "2026-04-28T10:30:45.123456",
    "is_violation": false,
    "severity": "low",
    "test_run": true,
    "created_at": "2026-04-28T10:30:45.123456"
  },
  "database": "pedestrian_detection",
  "timestamp": "2026-04-28T10:30:45.123456"
}
```

### **GET /health**
Quick health check with database status.

**Response:**
```json
{
  "status": "healthy",
  "service": "pedestrian-distraction-detection-backend",
  "version": "1.0.0",
  "database_status": "connected",
  "database_name": "pedestrian_detection",
  "timestamp": "2026-04-28T10:30:45.123456"
}
```

### **GET /db-status**
Detailed database status with collection counts.

**Response:**
```json
{
  "status": "success",
  "database": "pedestrian_detection",
  "health": {
    "status": "healthy",
    "collections": {
      "detections": 5,
      "alerts": 2,
      "faces": 1,
      "settings": 1,
      "devices": 1,
      "training_logs": 0
    }
  },
  "timestamp": "2026-04-28T10:30:45.123456"
}
```

---

## How to Test

### 1. **Start Backend**
```powershell
cd backend
.\run.ps1
```

Expected output:
```
[2/5] Initializing database collections and indexes...
✅ Database initialization complete
[3/5] Seeding default settings...
✅ Seeding complete
```

### 2. **Verify in Swagger UI**
- Open: http://localhost:8000/docs
- Expand **testing** section
- Try endpoints:
  - `GET /db-status` → See all collections
  - `GET /test-db` → Test insert/read
  - `GET /health` → Quick status

### 3. **Check MongoDB Atlas**
- Log into MongoDB Atlas
- Select database: `pedestrian_detection`
- You should see 6 collections
- View documents in each collection

---

## Database Retention Policy

- **Detection records**: Keep for 30 days (configurable via DETECTION_RETENTION_DAYS)
- **Alert records**: Indefinite (can be manually archived)
- **Face records**: Indefinite (deduplication by embedding)
- **Settings**: Singleton, always kept
- **Devices**: Updated as needed
- **Training logs**: Kept for reference (archive old jobs)

### Cleanup Query (Optional - Run manually when needed)

```javascript
// Delete detections older than 30 days
db.detections.deleteMany({
  timestamp: {
    $lt: new Date(new Date().getTime() - 30*24*60*60*1000)
  }
})
```

---

## Files Created in Step 2

✅ `backend/app/utils/db_schema.py`
   - Collection schema definitions
   - Index specifications

✅ `backend/app/utils/db_init.py`
   - Database initialization functions
   - Index creation
   - Health check

✅ `backend/app/utils/seed_data.py`
   - Seed default settings
   - Seed default device
   - Seeding orchestration

✅ Updated `backend/app/main.py`
   - Added initialization call in lifespan startup
   - Added seeding call in lifespan startup

✅ Updated `backend/app/routes/test.py`
   - Added `/db-status` endpoint

---

## Next Step: STEP 3 — ML Model Integration

After confirming database setup works, proceed to:
- Load ML models from existing code
- Integrate pipeline into app.state
- Create inference wrapper
- Define DetectionResult Pydantic model

# 🚀 Pedestrian Distraction Detection — Backend

FastAPI backend for real-time pedestrian phone detection.

## Setup Instructions

### 1. Create Virtual Environment

```bash
python -m venv venv

# Activate
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure Environment

```bash
# Copy .env.example to .env
cp .env.example .env

# Edit .env with your actual values:
# - MONGODB_URI: Your MongoDB Atlas connection string
# - CORS_ORIGINS: Frontend URL(s)
# - Other settings as needed
```

### 4. Run Backend

```bash
# Development mode (with auto-reload)
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# Or use Python directly
python -m uvicorn app.main:app --reload
```

### 5. Verify Backend is Running

```bash
# Check health
curl http://localhost:8000/health

# View API documentation
# Open http://localhost:8000/docs in browser
```

## Project Structure

```
backend/
├── app/
│   ├── main.py              # FastAPI entry point, CORS, lifespan
│   ├── config.py            # Pydantic Settings from .env
│   ├── database.py          # MongoDB connection (Motor async)
│   │
│   ├── routes/              # HTTP & WebSocket endpoints
│   │   └── __init__.py
│   │
│   ├── services/            # Business logic
│   │   └── __init__.py
│   │
│   ├── models/              # Pydantic schemas (request/response)
│   │   └── __init__.py
│   │
│   ├── ml/                  # ML pipeline wrappers
│   │   └── __init__.py
│   │
│   └── utils/               # Utility functions
│       └── __init__.py
│
├── .env.example             # Environment template
├── requirements.txt         # Python dependencies
└── README.md               # This file
```

## Architecture

```
Frontend (React)
    ↓ HTTP/WS
FastAPI Backend
    ├── Routes (HTTP/WS handlers)
    ├── Services (business logic)
    ├── ML Pipeline (inference)
    └── Database (MongoDB via Motor)
```

## Configuration

All settings are read from `.env` file:

- **MONGODB_URI**: MongoDB Atlas connection string
- **DB_NAME**: Database name (default: pedestrian_detection)
- **API_HOST**: Server host (default: 0.0.0.0)
- **API_PORT**: Server port (default: 8000)
- **CORS_ORIGINS**: Comma-separated allowed origins
- **DEVICE**: ML device - auto, cpu, or cuda
- **LOG_LEVEL**: Logging level (INFO, DEBUG, etc)

## Development

### Format Code

```bash
black app/
```

### Run Linter

```bash
flake8 app/
```

### Run Tests

```bash
pytest
```

## Next Steps

1. Step 2: Database setup and collections creation
2. Step 3: ML model integration
3. Step 4: Detection API endpoints
4. Step 5+: Full feature implementation

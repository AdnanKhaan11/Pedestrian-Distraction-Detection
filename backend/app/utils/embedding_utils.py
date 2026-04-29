"""
Embedding and similarity utilities for face recognition.

Uses InsightFace for face embeddings (128-dimensional vectors).
"""

import numpy as np


def cosine_similarity(embedding1: list[float], embedding2: list[float]) -> float:
    """
    Compute cosine similarity between two 128-D embeddings.

    Args:
        embedding1: First embedding vector
        embedding2: Second embedding vector

    Returns:
        Similarity score (0.0 to 1.0, higher = more similar)
    """
    # Convert to numpy arrays
    e1 = np.array(embedding1, dtype=np.float32)
    e2 = np.array(embedding2, dtype=np.float32)

    # Normalize vectors
    e1_norm = e1 / (np.linalg.norm(e1) + 1e-8)
    e2_norm = e2 / (np.linalg.norm(e2) + 1e-8)

    # Compute cosine similarity
    similarity = float(np.dot(e1_norm, e2_norm))

    # Clamp to [0, 1] range
    return max(0.0, min(1.0, similarity))


def euclidean_distance(embedding1: list[float], embedding2: list[float]) -> float:
    """
    Compute Euclidean distance between two embeddings.

    Args:
        embedding1: First embedding vector
        embedding2: Second embedding vector

    Returns:
        Distance score (lower = more similar)
    """
    e1 = np.array(embedding1, dtype=np.float32)
    e2 = np.array(embedding2, dtype=np.float32)

    distance = float(np.linalg.norm(e1 - e2))
    return distance


def generate_face_embedding(face_crop: np.ndarray) -> list[float]:
    """
    Generate 128-D embedding from face crop using InsightFace.

    Args:
        face_crop: Face image as numpy array (BGR, 160x160 recommended)

    Returns:
        List of 128 floats representing the face embedding

    Raises:
        ImportError: If InsightFace not installed
        RuntimeError: If face not detected in crop
    """
    try:
        from insightface.app import FaceAnalysis
    except ImportError:
        raise ImportError(
            "InsightFace not installed. Install via: pip install insightface"
        )

    # Initialize InsightFace model (lazy load, cached on first call)
    if not hasattr(generate_face_embedding, "_face_analyzer"):
        # Use CPU-only context
        import os

        os.environ["ONNX_COMPAT_MODE"] = "1"
        app = FaceAnalysis(
            providers=["CPUExecutionProvider"],
            allowed_modules=["detection", "recognition"],
        )
        app.prepare(ctx_id=-1)  # -1 = CPU
        generate_face_embedding._face_analyzer = app

    analyzer = generate_face_embedding._face_analyzer

    # Ensure input is BGR
    if len(face_crop.shape) != 3 or face_crop.shape[2] != 3:
        raise ValueError("Face crop must be BGR image with shape (H, W, 3)")

    # Detect and extract embedding
    faces = analyzer.get(face_crop)
    if not faces:
        raise RuntimeError("No face detected in crop")

    # Return embedding as list of floats
    embedding = faces[0].embedding
    return embedding.tolist()

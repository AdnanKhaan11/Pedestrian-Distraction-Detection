"""
Embedding and similarity utilities for face recognition.

Uses OpenCV HSV color histograms for face embeddings (114-dimensional vectors).
No external deep learning libraries required — cv2 and numpy only.

Note: facenet-pytorch has been removed. Face cropping is handled by the ML model
      (runtime_detector.py). Embeddings here are used only for deduplication.
"""

import cv2
import numpy as np


def cosine_similarity(embedding1: list[float], embedding2: list[float]) -> float:
    """
    Compute cosine similarity between two embeddings.

    Args:
        embedding1: First embedding vector
        embedding2: Second embedding vector

    Returns:
        Similarity score (0.0 to 1.0, higher = more similar)
    """
    e1 = np.array(embedding1, dtype=np.float32)
    e2 = np.array(embedding2, dtype=np.float32)

    e1_norm = e1 / (np.linalg.norm(e1) + 1e-8)
    e2_norm = e2 / (np.linalg.norm(e2) + 1e-8)

    similarity = float(np.dot(e1_norm, e2_norm))

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

    return float(np.linalg.norm(e1 - e2))


def generate_face_embedding(face_crop: np.ndarray) -> list[float]:
    """
    Generate a 114-D embedding from a face crop using HSV color histograms.

    Uses OpenCV HSV histograms — no external deep learning libraries needed.
    Sufficient for face deduplication within a session.

    Embedding breakdown:
        - H channel: 50 bins
        - S channel: 32 bins
        - V channel: 32 bins
        - Total: 114 dimensions, L2-normalized

    Args:
        face_crop: Face image as numpy array (BGR, any size — resized internally)

    Returns:
        List of 114 floats representing the face embedding

    Raises:
        ValueError: If face_crop is None or invalid
    """
    if face_crop is None or face_crop.size == 0:
        raise ValueError("generate_face_embedding: face_crop is None or empty")

    # Resize to fixed size for consistent histograms
    face_resized = cv2.resize(face_crop, (64, 64), interpolation=cv2.INTER_LINEAR)

    # Convert BGR → HSV
    # HSV is more robust to lighting changes than raw RGB
    face_hsv = cv2.cvtColor(face_resized, cv2.COLOR_BGR2HSV)

    # Compute per-channel histograms
    # H: 0–179 in OpenCV,  S: 0–255,  V: 0–255
    h_hist = cv2.calcHist([face_hsv], [0], None, [50], [0, 180]).flatten()
    s_hist = cv2.calcHist([face_hsv], [1], None, [32], [0, 256]).flatten()
    v_hist = cv2.calcHist([face_hsv], [2], None, [32], [0, 256]).flatten()

    # Concatenate into one vector
    embedding = np.concatenate([h_hist, s_hist, v_hist]).astype(np.float32)

    # L2-normalize so cosine similarity works correctly
    norm = np.linalg.norm(embedding) + 1e-8
    embedding = embedding / norm

    return embedding.tolist()

"""
Embedding and similarity utilities for face recognition.

Uses facenet-pytorch for face embeddings (512-dimensional vectors).
"""

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
    Generate 512-D embedding from face crop using facenet-pytorch.

    Args:
        face_crop: Face image as numpy array (BGR, any size — resized internally)

    Returns:
        List of 512 floats representing the face embedding

    Raises:
        ImportError: If facenet-pytorch not installed
        RuntimeError: If embedding generation fails
    """
    try:
        import torch
        from facenet_pytorch import InceptionResnetV1, MTCNN
    except ImportError:
        raise ImportError(
            "facenet-pytorch not installed. Install via: pip install facenet-pytorch"
        )

    # ── Lazy load models — only initialized once on first call ──
    if not hasattr(generate_face_embedding, "_resnet"):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # MTCNN — face detector + aligner
        generate_face_embedding._mtcnn = MTCNN(
            image_size=160,
            margin=10,
            min_face_size=20,
            thresholds=[0.6, 0.7, 0.7],
            factor=0.709,
            post_process=True,
            device=device,
            keep_all=False,  # only keep the most confident face
        )

        # InceptionResnetV1 — embedding model pretrained on VGGFace2
        generate_face_embedding._resnet = (
            InceptionResnetV1(pretrained="vggface2").eval().to(device)
        )

        generate_face_embedding._device = device
        print("Info: facenet-pytorch models loaded successfully")

    mtcnn = generate_face_embedding._mtcnn
    resnet = generate_face_embedding._resnet
    device = generate_face_embedding._device

    # ── Convert BGR (OpenCV) → RGB (PyTorch expects RGB) ──
    import cv2

    face_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)

    # ── Detect and align face using MTCNN ──
    # Returns a (1, 160, 160) float tensor, or None if no face found
    face_tensor = mtcnn(face_rgb)

    if face_tensor is None:
        # MTCNN found no face — use the raw crop resized to 160x160
        # This is the fallback when the crop is already a tight face region
        from PIL import Image
        import torchvision.transforms as transforms

        face_resized = cv2.resize(face_rgb, (160, 160))
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ]
        )
        face_tensor = transform(Image.fromarray(face_resized))

    # ── Generate embedding ──
    import torch

    with torch.no_grad():
        # Add batch dimension → (1, 3, 160, 160)
        embedding_tensor = resnet(face_tensor.unsqueeze(0).to(device))

    # Convert to plain Python list and return
    embedding = embedding_tensor.squeeze().cpu().numpy()
    return embedding.tolist()

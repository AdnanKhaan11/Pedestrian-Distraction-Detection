import base64
import json
import time
from typing import Any, List, Tuple, Union

import cv2
import numpy as np

try:
    import websocket
except ImportError:
    websocket = None


COLOR_BGR = {
    "green": (0, 255, 0),
    "orange": (51, 140, 232),
    "red": (0, 0, 255),
    "pink": (93, 57, 240),
    "gray": (155, 155, 155),
    "white": (255, 255, 255),
}

UI_TEXT_STYLE = {
    "fontFace": cv2.FONT_HERSHEY_SIMPLEX,
    "fontScale": 0.5,
    "color": COLOR_BGR["white"],
    "thickness": 2,
}

DETECTION_RECT_STYLE = {
    "thickness": 2,
}


def render_detection_rectangle(
    frame: np.ndarray,
    text: str,
    xyxy: Union[List[float], np.ndarray],
    color: str,
) -> None:
    """
    Draw one detection rectangle and label text on frame.
    """
    if xyxy is None:
        return

    cv2.putText(
        frame,
        text,
        org=(int(xyxy[0]), int(xyxy[1]) - 5),
        fontFace=UI_TEXT_STYLE["fontFace"],
        fontScale=0.6,
        color=COLOR_BGR[color],
        thickness=UI_TEXT_STYLE["thickness"],
    )

    cv2.rectangle(
        frame,
        pt1=(int(xyxy[0]), int(xyxy[1])),
        pt2=(int(xyxy[2]), int(xyxy[3])),
        color=COLOR_BGR[color],
        thickness=DETECTION_RECT_STYLE["thickness"],
    )


def render_ui_text(
    frame: np.ndarray,
    text: str,
    frame_wh: Tuple[int, int],
    margin_wh: Tuple[int, int],
    align: str,
    order: int,
) -> None:
    """
    Draw UI text on frame.

    This is the cleaned version of your current UI renderer.
    """
    frame_w, _ = frame_wh
    margin_w, margin_h = margin_wh

    (text_width, text_height), _ = cv2.getTextSize(
        text=text,
        fontFace=UI_TEXT_STYLE["fontFace"],
        fontScale=UI_TEXT_STYLE["fontScale"],
        thickness=UI_TEXT_STYLE["thickness"],
    )

    if align == "left":
        origin = (margin_w, margin_h + order * (text_height + 5))
    elif align == "right":
        origin = (
            frame_w - margin_w - int((0.01 if frame_w <= 600 else 1.0) * text_width),
            margin_h + order * (text_height + 5),
        )
    else:
        raise ValueError("align must be either 'left' or 'right'")

    cv2.putText(
        frame,
        text,
        org=origin,
        fontFace=UI_TEXT_STYLE["fontFace"],
        fontScale=UI_TEXT_STYLE["fontScale"],
        color=UI_TEXT_STYLE["color"],
        thickness=UI_TEXT_STYLE["thickness"],
    )


def crop_frame(
    frame: np.ndarray,
    center_xy: np.ndarray,
    crop_hw: Tuple[int, int],
) -> tuple[np.ndarray | None, list[int] | None]:
    """
    Crop a sub-frame using center coordinate and crop size.

    Returns:
    - cropped frame
    - absolute xyxy box
    """
    frame_h, frame_w, _ = frame.shape
    x_center, y_center = center_xy
    crop_h, crop_w = crop_hw

    if not (0 <= x_center <= frame_w and 0 <= y_center <= frame_h):
        return None, None

    xs = np.array([x_center - (crop_w // 2), x_center + (crop_w // 2)], dtype=np.int32)
    ys = np.array([y_center - (crop_h // 2), y_center + (crop_h // 2)], dtype=np.int32)

    np.clip(xs, 0, frame_w, out=xs)
    np.clip(ys, 0, frame_h, out=ys)

    xyxy = [int(xs[0]), int(ys[0]), int(xs[1]), int(ys[1])]
    cropped = frame[ys[0] : ys[1], xs[0] : xs[1], :]

    return cropped, xyxy


def resize_frame_to_square(
    frame: np.ndarray,
    edge_length: int,
    ratio_threshold: float = 9 / 16,
) -> np.ndarray:
    """
    Resize a frame to square.

    If frame ratio is too wide/tall, center-crop first.
    Otherwise, direct resize.

    This follows the same idea used in your current project.
    """
    if edge_length <= 0:
        raise ValueError("edge_length must be greater than 0")

    if not (0 < ratio_threshold <= 1):
        raise ValueError("ratio_threshold must be in (0, 1]")

    height, width = frame.shape[:2]
    if height <= 0 or width <= 0:
        raise ValueError(f"Invalid frame shape: {frame.shape}")

    ratio = height / (width + np.finfo(np.float32).eps)

    if ratio_threshold < ratio < 1 / ratio_threshold:
        return cv2.resize(
            frame, (edge_length, edge_length), interpolation=cv2.INTER_AREA
        )

    if width > height:
        start_x = (width - height) // 2
        cropped = frame[:, start_x : start_x + height]
    else:
        start_y = (height - width) // 2
        cropped = frame[start_y : start_y + width, :]

    return cv2.resize(cropped, (edge_length, edge_length), interpolation=cv2.INTER_AREA)


def relative_to_absolute(
    from_mother_wh: Tuple[int, int],
    to_mother_wh: Tuple[int, int],
    from_child_xyxy: Union[List[float], np.ndarray],
    to_mother_xy: Tuple[int, int] = (0, 0),
) -> list[int]:
    """
    Convert relative box coordinates from resized sub-frame to absolute coordinates.

    Example:
    - detection happens on resized square hand frame
    - convert detected phone box back to original frame coordinates
    """
    from_mother_w, from_mother_h = from_mother_wh
    to_mother_w, to_mother_h = to_mother_wh
    offset_x, offset_y = to_mother_xy

    scale_x = to_mother_w / (from_mother_w + np.finfo(np.float32).eps)
    scale_y = to_mother_h / (from_mother_h + np.finfo(np.float32).eps)

    x1, y1, x2, y2 = from_child_xyxy[:4]

    abs_x1 = int(x1 * scale_x + offset_x)
    abs_y1 = int(y1 * scale_y + offset_y)
    abs_x2 = int(x2 * scale_x + offset_x)
    abs_y2 = int(y2 * scale_y + offset_y)

    return [abs_x1, abs_y1, abs_x2, abs_y2]


def init_websocket(server_url: str) -> Any | None:
    """
    Initialize websocket connection safely.
    """
    if websocket is None:
        return None

    try:
        ws = websocket.WebSocket()
        ws.connect(server_url)
        return ws
    except Exception:
        return None


def yield_video_feed(frame_to_yield: np.ndarray, title: str = "", ws=None) -> None:
    """
    Show local OpenCV window and optionally send frame through websocket.
    """
    if ws is not None:
        _, jpeg_encoded = cv2.imencode(".jpg", frame_to_yield)
        jpeg_bytes = jpeg_encoded.tobytes()
        jpeg_base64 = base64.b64encode(jpeg_bytes).decode("utf-8")

        ws.send(
            json.dumps(
                {
                    "frameBase64": jpeg_base64,
                    "timestamp": f"{float(time.time()):.3f}",
                }
            )
        )

    cv2.imshow(title, frame_to_yield)


def announce_face_frame(face_frames: list[np.ndarray], ws) -> None:
    """
    Send face crops to websocket client.
    """
    encoded_frames = []

    for frame in face_frames:
        try:
            _, jpeg_encoded = cv2.imencode(".jpg", frame)
            jpeg_bytes = jpeg_encoded.tobytes()
            jpeg_base64 = base64.b64encode(jpeg_bytes).decode("utf-8")
            encoded_frames.append(jpeg_base64)
        except Exception:
            continue

    if len(encoded_frames) <= 0:
        return

    ws.send(
        json.dumps(
            {
                "announced_face_frames": encoded_frames,
                "timestamp": f"{float(time.time()):.3f}",
            }
        )
    )

"""
This file combines posture output and phone detection output into one final decision.
It follows the same idea as your current project: posture first, then phone confirmation.
This makes the final decision logic explicit, reusable, and easy to debug later.
It is the clean replacement for decision logic previously mixed inside processing.py.
"""

from pathlib import Path

from src.config.constants import (
    STATE_BACKSIDE,
    STATE_NOT_USING,
    STATE_OUT_OF_FRAME,
    STATE_SUSPICIOUS,
    STATE_TO_BE_CLASSIFIED,
    STATE_USING,
    STATE_DISPLAY,
)
from src.utils.logger import get_logger


class DistractionFusion:
    """
    Fuse posture and phone signals into final runtime state.
    """

    def __init__(self, log_dir: Path | None = None, log_level: str = "INFO") -> None:
        self.logger = get_logger(
            self.__class__.__name__, log_dir=log_dir, level=log_level
        )

    def fuse(
        self,
        base_state: int,
        posture_score_text: str,
        phone_detected: bool,
    ) -> dict:
        """
        Combine posture state and phone result into final output.

        Rules:
        - if already out of frame or backside, keep that state
        - if posture says suspicious and phone is detected -> using
        - if posture says suspicious but no phone -> suspicious
        - if posture says not_using -> not_using
        """
        final_state = base_state

        if base_state in {STATE_OUT_OF_FRAME, STATE_BACKSIDE}:
            final_state = base_state
        elif base_state == STATE_SUSPICIOUS:
            final_state = STATE_USING if phone_detected else STATE_SUSPICIOUS
        elif base_state == STATE_NOT_USING:
            final_state = STATE_NOT_USING
        elif base_state == STATE_TO_BE_CLASSIFIED:
            final_state = STATE_TO_BE_CLASSIFIED

        display_meta = STATE_DISPLAY.get(final_state, {"color": "gray", "text": "-"})

        result = {
            "state": final_state,
            "display_color": display_meta["color"],
            "display_text": display_meta["text"],
            "score_text": posture_score_text,
            "phone_detected": phone_detected,
            "final_label": self._map_state_to_label(final_state),
        }

        self.logger.info("Fusion result: %s", result)
        return result

    @staticmethod
    def _map_state_to_label(state: int) -> str:
        """
        Map internal runtime state to final business label.
        """
        if state == STATE_USING:
            return "distracted"
        if state == STATE_SUSPICIOUS:
            return "suspicious"
        if state == STATE_NOT_USING:
            return "safe"
        if state == STATE_OUT_OF_FRAME:
            return "out_of_frame"
        if state == STATE_BACKSIDE:
            return "backside"
        return "unknown"

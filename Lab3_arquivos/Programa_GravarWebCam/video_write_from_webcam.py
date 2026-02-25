from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Tuple

import cv2


logger = logging.getLogger("video_write_wc")


@dataclass(frozen=True)
class CaptureConfig:
    camera_index: int = 0
    output_path: str = "output.mp4"
    fps: int = 20
    fourcc: str = "h264"          # equivalente a fourcc('h','2','6','4')
    window_name: str = "Frame"
    wait_ms: int = 20
    quit_key: str = "q"


def _setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def _open_capture(camera_index: int) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"Não foi possível abrir a webcam (index={camera_index}).")
    return cap


def _get_frame_size(cap: cv2.VideoCapture) -> Tuple[int, int]:
    # Em C++: get(3) e get(4)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if width <= 0 or height <= 0:
        # fallback razoável se o driver não expõe dimensões
        width, height = 640, 480
    return width, height


def _open_writer(path: str, fourcc: str, fps: int, size: Tuple[int, int]) -> cv2.VideoWriter:
    codec = cv2.VideoWriter_fourcc(*fourcc)
    writer = cv2.VideoWriter(path, codec, float(fps), size)
    if not writer.isOpened():
        raise RuntimeError(
            f"Não foi possível abrir o VideoWriter para '{path}'. "
            f"Verifique fourcc='{fourcc}', fps={fps}, size={size}. "
            f"(Em muitos ambientes, H.264 pode não estar disponível.)"
        )
    return writer


def main() -> int:
    _setup_logging()
    cfg = CaptureConfig()

    cap = None
    writer = None
    try:
        cap = _open_capture(cfg.camera_index)

        frame_width, frame_height = _get_frame_size(cap)
        frame_size = (frame_width, frame_height)

        writer = _open_writer(cfg.output_path, cfg.fourcc, cfg.fps, frame_size)

        while cap.isOpened():
            ok, frame = cap.read()

            if not ok:
                print("Web camera is disconnected")
                break

            # equivalente a output.write(frame) + imshow(...)
            writer.write(frame)
            cv2.imshow(cfg.window_name, frame)

            # equivalente ao waitKey(20) e checar 'q'
            key = cv2.waitKey(cfg.wait_ms) & 0xFF
            if key == ord(cfg.quit_key):
                print("Key q is pressed by the user. Stopping the video")
                break

        return 0

    finally:
        cv2.destroyAllWindows()
        if cap is not None:
            cap.release()
        if writer is not None:
            writer.release()


if __name__ == "__main__":
    raise SystemExit(main())

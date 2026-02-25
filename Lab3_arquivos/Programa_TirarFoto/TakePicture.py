from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional, Tuple

import cv2


logger = logging.getLogger("webcam_snapshot")


@dataclass(frozen=True)
class SnapshotConfig:
    camera_index: int = 0
    output_path: str = "output.png"
    fourcc: str = "MJPG"
    fps: int = 20
    window_name: str = "Frame"
    wait_ms: int = 20
    quit_key: str = "q"


def _setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def _open_camera(index: int) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        raise RuntimeError(f"Não foi possível abrir a webcam (index={index}).")
    return cap


def _get_frame_size(cap: cv2.VideoCapture) -> Tuple[int, int]:
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if width <= 0 or height <= 0:
        return (640, 480)
    return (width, height)


def main() -> int:
    _setup_logging()
    cfg = SnapshotConfig()

    cap: Optional[cv2.VideoCapture] = None
    writer: Optional[cv2.VideoWriter] = None

    try:
        cap = _open_camera(cfg.camera_index)
        frame_size = _get_frame_size(cap)

        # Inicializa VideoWriter (mesma lógica do C++)
        codec = cv2.VideoWriter_fourcc(*cfg.fourcc)
        writer = cv2.VideoWriter(cfg.output_path, codec, float(cfg.fps), frame_size)

        while cap.isOpened():
            ok, frame = cap.read()

            if not ok:
                print("Web camera is disconnected")
                break

            cv2.imshow(cfg.window_name, frame)

            key = cv2.waitKey(cfg.wait_ms) & 0xFF
            if key == ord(cfg.quit_key):
                # Escreve apenas o último frame quando 'q' é pressionado
                writer.write(frame)
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

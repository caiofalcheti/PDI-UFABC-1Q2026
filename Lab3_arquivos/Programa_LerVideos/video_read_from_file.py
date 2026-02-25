from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import cv2


logger = logging.getLogger("video_reader")


@dataclass(frozen=True)
class VideoConfig:
    video_path: str = "Resources/big_buck_bunny.mp4"
    window_name: str = "Frame"
    wait_ms: int = 20
    quit_key: str = "q"


def _setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def _open_video(path: str) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        print("Error opening video stream or file")
    return cap


def _print_video_info(cap: cv2.VideoCapture) -> None:
    if cap.isOpened():
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"Frames per second :{fps}  Frame count :{frame_count}")


def main() -> int:
    _setup_logging()
    cfg = VideoConfig()

    vid_capture: Optional[cv2.VideoCapture] = None
    try:
        vid_capture = _open_video(cfg.video_path)
        _print_video_info(vid_capture)

        while vid_capture.isOpened():
            ok, frame = vid_capture.read()

            if ok:
                cv2.imshow(cfg.window_name, frame)
            else:
                print("Video camera is disconnected")
                break

            key = cv2.waitKey(cfg.wait_ms) & 0xFF
            if key == ord(cfg.quit_key):
                print("q key is pressed by the user. Stopping the video")
                break

        return 0

    finally:
        if vid_capture is not None:
            vid_capture.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    raise SystemExit(main())

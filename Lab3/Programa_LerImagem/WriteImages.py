from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import cv2


logger = logging.getLogger("image_viewer")


@dataclass(frozen=True)
class ImageConfig:
    image_name: str = "messi5.jpg"
    window_title: str = "Display window"
    save_key: str = "s"
    output_name: str = "messi5.png"


def _setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def _find_sample_file(image_name: str) -> str:
    # Equivalente a samples::findFile("messi5.jpg")
    return cv2.samples.findFile(image_name)


def _load_image(path: str) -> Optional[cv2.Mat]:
    return cv2.imread(path, cv2.IMREAD_COLOR)


def main() -> int:
    _setup_logging()
    cfg = ImageConfig()

    image_path = _find_sample_file(cfg.image_name)
    img = _load_image(image_path)

    if img is None or img.size == 0:
        print(f"Could not read the image: {image_path}")
        return 1

    cv2.imshow(cfg.window_title, img)
    k = cv2.waitKey(0) & 0xFF  # Wait for a keystroke

    if k == ord(cfg.save_key):
        cv2.imwrite(cfg.output_name, img)

    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python

"""
Local preview viewer: shows latest frames saved by the dataset policy infer server.
"""

import time
from dataclasses import dataclass

import cv2  # type: ignore

from lerobot.configs import parser


@dataclass
class PreviewViewerConfig:
    preview_dir: str
    cams: str = "top,gripper"
    fps: int = 5


@parser.wrap()
def run(cfg: PreviewViewerConfig) -> None:
    cam_names = [c.strip() for c in cfg.cams.split(",") if c.strip()]
    delay_ms = int(1000 / max(1, cfg.fps))

    while True:
        for name in cam_names:
            path = f"{cfg.preview_dir}/latest_{name}.jpg"
            img = cv2.imread(path)
            if img is None:
                continue
            cv2.imshow(name, img)
        if cv2.waitKey(delay_ms) == 27:  # ESC to exit
            break
        time.sleep(0.001)

    cv2.destroyAllWindows()


if __name__ == "__main__":
    run()

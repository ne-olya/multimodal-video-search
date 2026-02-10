from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image


@dataclass
class SampledFrame:
    timestamp: float
    image: Image.Image


def uniform_positions(frame_count: int, count: int):
    if frame_count <= 0 or count <= 0:
        return np.array([], dtype=int)
    return np.unique(np.linspace(0, frame_count - 1, min(count, frame_count)).astype(int))


def sample_video(path: str | Path, count=12, strategy="uniform", scene_threshold=0.35):
    import cv2

    capture = cv2.VideoCapture(str(path))
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = capture.get(cv2.CAP_PROP_FPS) or 25.0
    positions = uniform_positions(frame_count, count * 4 if strategy == "scene" else count)
    result, last_hist = [], None
    for pos in positions:
        capture.set(cv2.CAP_PROP_POS_FRAMES, int(pos))
        ok, bgr = capture.read()
        if not ok:
            continue
        if strategy == "scene":
            hist = cv2.calcHist([bgr], [0, 1], None, [16, 16], [0, 256] * 2)
            cv2.normalize(hist, hist)
            distance = (
                1.0
                if last_hist is None
                else cv2.compareHist(last_hist, hist, cv2.HISTCMP_BHATTACHARYYA)
            )
            last_hist = hist
            if distance < scene_threshold:
                continue
        result.append(
            SampledFrame(float(pos / fps), Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)))
        )
        if len(result) == count:
            break
    capture.release()
    return result

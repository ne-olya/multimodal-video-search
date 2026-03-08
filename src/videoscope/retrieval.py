import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from .encoders import normalize


@dataclass
class SegmentRecord:
    segment_id: str
    video_id: str
    path: str
    start: float
    end: float
    keyframe: str
    transcript: str = ""
    ocr_text: str = ""


class MultimodalSegmentIndex:
    """Late-fusion index retaining per-modality evidence instead of hiding it in one vector."""

    modalities = ("visual", "speech", "ocr")

    def __init__(self, records, vectors, available=None):
        self.records = records
        self.vectors = {name: normalize(vectors[name]) for name in self.modalities}
        self.available = available or {
            name: np.ones(len(records), dtype=bool) for name in self.modalities
        }
        if any(len(self.vectors[name]) != len(records) for name in self.modalities):
            raise ValueError("Every modality matrix must align with segment records")

    def search_segments(self, query, k=50, weights=None):
        weights = weights or {"visual": 0.55, "speech": 0.30, "ocr": 0.15}
        query = normalize(np.asarray(query).reshape(1, -1))[0]
        numerator = np.zeros(len(self.records), dtype=np.float32)
        denominator = np.zeros(len(self.records), dtype=np.float32)
        contributions = {}
        for name in self.modalities:
            raw = self.vectors[name] @ query
            active = np.asarray(self.available[name], dtype=bool)
            contribution = np.where(active, raw * weights[name], 0)
            contributions[name] = contribution
            numerator += contribution
            denominator += active * weights[name]
        scores = numerator / np.maximum(denominator, 1e-12)
        order = np.argsort(-scores)[:k]
        return [
            {
                "record": self.records[i],
                "score": float(scores[i]),
                "contributions": {
                    name: float(contributions[name][i] / max(denominator[i], 1e-12))
                    for name in self.modalities
                    if self.available[name][i]
                },
            }
            for i in order
        ]

    def search_videos(self, query, k=10, weights=None, candidate_segments=100):
        segments = self.search_segments(query, candidate_segments, weights)
        best = {}
        for result in segments:
            video_id = result["record"].video_id
            if video_id not in best or result["score"] > best[video_id]["score"]:
                best[video_id] = result
        return sorted(best.values(), key=lambda row: row["score"], reverse=True)[:k]

    def save(self, directory):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        arrays = {f"vectors_{name}": self.vectors[name] for name in self.modalities}
        arrays.update(
            {
                f"available_{name}": np.asarray(self.available[name], dtype=bool)
                for name in self.modalities
            }
        )
        np.savez_compressed(directory / "segments.npz", **arrays)
        (directory / "segments.json").write_text(
            json.dumps([asdict(record) for record in self.records], ensure_ascii=False, indent=2)
        )

    @classmethod
    def load(cls, directory):
        directory = Path(directory)
        data = np.load(directory / "segments.npz")
        records = [
            SegmentRecord(**row) for row in json.loads((directory / "segments.json").read_text())
        ]
        return cls(
            records,
            {name: data[f"vectors_{name}"] for name in cls.modalities},
            {name: data[f"available_{name}"] for name in cls.modalities},
        )


def group_evidence(frames, asr_segments, ocr_rows, video_id, path, frame_files, segment_seconds):
    groups = []
    if not frames:
        return groups
    duration = max(frame.timestamp for frame in frames) + segment_seconds
    for start in np.arange(0, duration, segment_seconds):
        end = start + segment_seconds
        frame_indices = [i for i, frame in enumerate(frames) if start <= frame.timestamp < end]
        if not frame_indices:
            continue
        speech = " ".join(
            row["text"] for row in asr_segments if row["start"] < end and row["end"] > start
        )
        ocr = " ".join(row["text"] for row in ocr_rows if start <= row["timestamp"] < end)
        groups.append(
            {
                "record": SegmentRecord(
                    f"{video_id}:{start:.1f}",
                    video_id,
                    str(path),
                    float(start),
                    float(end),
                    frame_files[frame_indices[0]],
                    speech,
                    ocr,
                ),
                "frame_indices": frame_indices,
            }
        )
    return groups

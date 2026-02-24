import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from .encoders import normalize


@dataclass
class VideoRecord:
    video_id: str
    path: str
    duration: float
    keyframes: list[dict]
    transcript: str = ""
    ocr_text: str = ""


class VectorIndex:
    def __init__(self, vectors, records):
        self.vectors = normalize(vectors)
        self.records = records
        self.faiss_index = None
        try:
            import faiss

            self.faiss_index = faiss.IndexFlatIP(self.vectors.shape[1])
            self.faiss_index.add(self.vectors)
        except ImportError:
            pass

    def search(self, query, k=10):
        query = normalize(np.asarray(query).reshape(1, -1))
        if self.faiss_index is not None:
            scores, order = self.faiss_index.search(query, min(k, len(self.records)))
            return [
                (self.records[i], float(score)) for i, score in zip(order[0], scores[0]) if i >= 0
            ]
        scores = self.vectors @ query[0]
        order = np.argsort(-scores)[:k]
        return [(self.records[i], float(scores[i])) for i in order]

    def save(self, directory):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        np.save(directory / "vectors.npy", self.vectors)
        if self.faiss_index is not None:
            import faiss

            faiss.write_index(self.faiss_index, str(directory / "index.faiss"))
        (directory / "manifest.json").write_text(
            json.dumps([asdict(record) for record in self.records], ensure_ascii=False, indent=2)
        )

    @classmethod
    def load(cls, directory):
        directory = Path(directory)
        records = [
            VideoRecord(**row) for row in json.loads((directory / "manifest.json").read_text())
        ]
        return cls(np.load(directory / "vectors.npy"), records)


def fuse_embeddings(visual, speech=None, ocr=None, weights=None):
    weights = weights or {"visual": 0.55, "speech": 0.30, "ocr": 0.15}
    available = [(visual, weights["visual"])]
    if speech is not None:
        available.append((speech, weights["speech"]))
    if ocr is not None:
        available.append((ocr, weights["ocr"]))
    total = sum(weight for _, weight in available)
    return normalize(sum(vector * weight / total for vector, weight in available))

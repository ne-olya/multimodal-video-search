from dataclasses import dataclass


@dataclass
class SearchResult:
    record: object
    score: float
    explanation: str


def explain(record, mode):
    evidence = ["визуальные ключевые кадры"]
    if mode in ("video_speech", "all") and record.transcript:
        evidence.append("совпадение с распознанной речью")
    if mode == "all" and record.ocr_text:
        evidence.append("совпадение с текстом в кадре")
    return ", ".join(evidence)


def search(index, encoder, query, k=10, mode="all"):
    vector = encoder.encode_texts([query])[0]
    return [
        SearchResult(record, score, explain(record, mode))
        for record, score in index.search(vector, k)
    ]

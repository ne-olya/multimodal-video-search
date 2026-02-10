from pathlib import Path


def transcribe(path: str | Path, model_name="small", language=None):
    from faster_whisper import WhisperModel

    model = WhisperModel(model_name, device="auto", compute_type="int8")
    segments, info = model.transcribe(str(path), language=language)
    rows = [{"start": s.start, "end": s.end, "text": s.text.strip()} for s in segments]
    return " ".join(row["text"] for row in rows), rows, info.language


def recognize_frames(frames, languages=("ru", "en")):
    import easyocr
    import numpy as np

    reader = easyocr.Reader(list(languages))
    rows = []
    for frame in frames:
        parts = reader.readtext(np.asarray(frame.image), detail=0)
        if parts:
            rows.append({"timestamp": frame.timestamp, "text": " ".join(parts)})
    unique = list(dict.fromkeys(row["text"] for row in rows))
    return " ".join(unique), rows

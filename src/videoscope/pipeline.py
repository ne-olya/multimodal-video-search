import argparse
import json
from pathlib import Path

import numpy as np
import yaml

from .encoders import OpenClipEncoder
from .retrieval import MultimodalSegmentIndex, group_evidence
from .modalities import recognize_frames, transcribe
from .sampling import sample_video


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    clip = OpenClipEncoder(cfg["encoders"]["visual"], cfg["encoders"]["visual_weights"])
    records, vectors = [], {"visual": [], "speech": [], "ocr": []}
    available = {"visual": [], "speech": [], "ocr": []}
    for path in sorted(Path(cfg["video_dir"]).glob("*")):
        if path.suffix.lower() not in {".mp4", ".mov", ".avi", ".mkv", ".webm"}:
            continue
        frames = sample_video(
            path,
            cfg["sampling"]["frames"],
            cfg["sampling"]["strategy"],
            cfg["sampling"]["scene_threshold"],
        )
        if not frames:
            continue
        frame_vectors = clip.encode_images([frame.image for frame in frames])
        transcript, segments, _ = (
            transcribe(path, cfg["asr"]["model"], cfg["asr"]["language"])
            if cfg["asr"]["enabled"]
            else ("", [], None)
        )
        ocr_text, ocr_rows = (
            recognize_frames(frames, cfg["ocr"]["languages"]) if cfg["ocr"]["enabled"] else ("", [])
        )
        out = Path(cfg["output_dir"])
        (out / "frames").mkdir(parents=True, exist_ok=True)
        frame_files = [f"frames/{path.stem}_{i}.jpg" for i in range(len(frames))]
        for frame, filename in zip(frames, frame_files):
            frame.image.save(out / filename)
        groups = group_evidence(
            frames,
            segments,
            ocr_rows,
            path.stem,
            path,
            frame_files,
            cfg["sampling"]["segment_seconds"],
        )
        for group in groups:
            record = group["record"]
            records.append(record)
            visual = frame_vectors[group["frame_indices"]].mean(0)
            vectors["visual"].append(visual)
            available["visual"].append(True)
            for name, text in (("speech", record.transcript), ("ocr", record.ocr_text)):
                available[name].append(bool(text))
                vectors[name].append(
                    clip.encode_texts([text])[0] if text else np.zeros_like(visual)
                )
        (out / f"{path.stem}.details.json").write_text(
            json.dumps({"asr_segments": segments, "ocr": ocr_rows}, ensure_ascii=False, indent=2)
        )
    if not records:
        raise RuntimeError(f"No videos found in {cfg['video_dir']}")
    MultimodalSegmentIndex(
        records, {name: np.vstack(rows) for name, rows in vectors.items()}, available
    ).save(cfg["output_dir"])


if __name__ == "__main__":
    main()

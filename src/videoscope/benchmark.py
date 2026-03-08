import argparse
import csv
import json
import time

from .encoders import OpenClipEncoder
from .evaluate import aggregate, ranking_metrics
from .retrieval import MultimodalSegmentIndex


MODES = {
    "video": {"visual": 1, "speech": 0, "ocr": 0},
    "video_speech": {"visual": 0.65, "speech": 0.35, "ocr": 0},
    "all": {"visual": 0.55, "speech": 0.30, "ocr": 0.15},
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("annotations", help="JSONL with query and relevant_ids")
    parser.add_argument("--index", default="artifacts/default")
    parser.add_argument("--output", default="reports/results.csv")
    parser.add_argument("--k", type=int, default=10)
    args = parser.parse_args()
    annotations = [json.loads(line) for line in open(args.annotations)]
    index = MultimodalSegmentIndex.load(args.index)
    encoder = OpenClipEncoder()
    query_vectors = encoder.encode_texts([row["query"] for row in annotations])
    report = []
    for mode, weights in MODES.items():
        metrics, latencies = [], []
        for row, vector in zip(annotations, query_vectors):
            started = time.perf_counter()
            found = index.search_videos(vector, args.k, weights)
            latencies.append((time.perf_counter() - started) * 1000)
            metrics.append(
                ranking_metrics(
                    [item["record"].video_id for item in found], set(row["relevant_ids"]), args.k
                )
            )
        values = aggregate(metrics)
        ordered = sorted(latencies)
        report.append(
            {
                "experiment": mode,
                **values,
                "search_p50_ms": ordered[len(ordered) // 2],
                "search_p95_ms": ordered[int(0.95 * (len(ordered) - 1))],
            }
        )
    with open(args.output, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=report[0].keys())
        writer.writeheader()
        writer.writerows(report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

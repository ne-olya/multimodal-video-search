import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def column(data, *names):
    return next((name for name in names if name in data.columns), None)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="reports/results.csv")
    parser.add_argument("--output", default="assets/retrieval-results.png")
    args = parser.parse_args()
    data = pd.read_csv(args.input).dropna(how="all")
    recall, precision = (
        column(data, "recall@10", "recall_at_10"),
        column(data, "precision@10", "precision_at_10", "precision_at_5"),
    )
    if data.empty or not recall or "mrr" not in data:
        raise SystemExit("Сначала выполните videoscope-benchmark и получите Recall/MRR")
    label = column(data, "experiment", "modalities")
    figure, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
    metrics = [recall, "mrr"] + ([precision] if precision else [])
    data.set_index(label)[metrics].plot.bar(
        ax=axes[0], color=["#28666e", "#fedc97", "#b5b682"][: len(metrics)]
    )
    axes[0].set(title="Качество мультимодального поиска", ylabel="Значение", ylim=(0, 1))
    axes[0].tick_params(axis="x", rotation=0)
    latency = column(data, "search_p95_ms", "search_p50_ms")
    axes[1].bar(data[label], data[latency], color="#7c3f58")
    axes[1].set(title="Время поиска", ylabel="Latency, ms", xlabel="Режим")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180)


if __name__ == "__main__":
    main()

import argparse
import json


def ranking_metrics(results, relevant, k=10):
    top = results[:k]
    hits = [int(item in relevant) for item in top]
    recall = sum(hits) / max(len(relevant), 1)
    precision = sum(hits) / max(len(top), 1)
    reciprocal_rank = next((1 / (i + 1) for i, hit in enumerate(hits) if hit), 0.0)
    return {f"recall@{k}": recall, f"precision@{k}": precision, "mrr": reciprocal_rank}


def aggregate(rows):
    keys = rows[0].keys()
    return {key: sum(row[key] for row in rows) / len(rows) for key in keys}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("annotations", help="JSONL: query, relevant_ids, retrieved_ids")
    parser.add_argument("--k", type=int, default=10)
    args = parser.parse_args()
    rows = [json.loads(line) for line in open(args.annotations)]
    print(
        aggregate(
            [
                ranking_metrics(row["retrieved_ids"], set(row["relevant_ids"]), args.k)
                for row in rows
            ]
        )
    )


if __name__ == "__main__":
    main()

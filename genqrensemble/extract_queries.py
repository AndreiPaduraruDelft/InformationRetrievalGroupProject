"""
Extract queries from a dataset and save them as a JSON file.

Usage:
    python extract_queries.py --dataset msmarco-passage/trec-dl-2019/judged
    python extract_queries.py --dataset beir/dbpedia-entity/test --output my_queries.json
"""

import argparse
import json
import pyterrier as pt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, help="Dataset name (e.g. msmarco-passage/trec-dl-2019/judged)")
    parser.add_argument("--output",  default=None,  help="Output JSON file (default: {dataset_safe}.json)")
    args = parser.parse_args()

    if not pt.java.started():
        pt.java.init()

    dataset = pt.get_dataset(f"irds:{args.dataset}")
    topics  = dataset.get_topics()
    qrels   = dataset.get_qrels()

    # Keep only judged topics
    judged_qids = set(qrels["qid"].astype(str).unique())
    topics = topics[topics["qid"].astype(str).isin(judged_qids)].reset_index(drop=True)

    queries = [{"qid": str(row["qid"]), "query": row["query"]} for _, row in topics.iterrows()]

    output_path = args.output or f"{args.dataset.replace('/', '_')}_queries.json"
    with open(output_path, "w") as f:
        json.dump(queries, f, indent=2)

    print(f"Saved {len(queries)} queries to {output_path}")


if __name__ == "__main__":
    main()

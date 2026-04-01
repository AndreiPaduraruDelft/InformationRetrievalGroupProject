"""
Reformulate queries using HFReformulator and save results in cache format.

Usage:
    python reformulate.py --queries msmarco-passage_trec-dl-2019_judged_queries.json \
                          --model google/flan-t5-xxl \
                          --device cuda \
                          --output cache/google_flan-t5-xxl__msmarco-passage_trec-dl-2019_judged.json
"""

import argparse
import json
from tqdm import tqdm

from config import INSTRUCTIONS
from reformulator import HFReformulator
from genqr_methods import flanqr_reformulate, genqr_ensemble_reformulate
from cache import load_cache, save_cache, get_cache_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--queries",  required=True, help="JSON file with queries (from extract_queries.py)")
    parser.add_argument("--model",    required=True, help="HuggingFace model ID (e.g. google/flan-t5-xxl)")
    parser.add_argument("--device",   default="cpu", help="Device: cpu, cuda, mps")
    parser.add_argument("--output",   default=None,  help="Output cache JSON file")
    parser.add_argument("--dataset",  default=None,  help="Dataset name (used to auto-generate output path)")
    parser.add_argument("--cache_dir", default="cache", help="Cache directory (default: cache)")
    args = parser.parse_args()

    with open(args.queries) as f:
        queries = json.load(f)

    if args.output:
        output_path = args.output
    elif args.dataset:
        output_path = get_cache_path(args.cache_dir, args.model, args.dataset)
    else:
        safe_model = args.model.replace("/", "_")
        output_path = f"{args.cache_dir}/{safe_model}__reformulations.json"

    cache = load_cache(output_path)

    print(f"Loaded {len(queries)} queries from {args.queries}")
    print(f"Output cache: {output_path}")

    already_done = sum(
        1 for q in queries
        if f"flanqr__{q['qid']}" in cache and f"ensemble__{q['qid']}" in cache
    )
    if already_done == len(queries):
        print("All queries already reformulated in cache. Nothing to do.")
        return

    print(f"Loading model {args.model} on {args.device} ...")
    reformulator = HFReformulator(model_id=args.model, device=args.device)

    for q in tqdm(queries, desc="reformulating"):
        qid, query = q["qid"], q["query"]

        if f"flanqr__{qid}" not in cache:
            cache[f"flanqr__{qid}"] = flanqr_reformulate(query, reformulator, INSTRUCTIONS[0])
            save_cache(output_path, cache)

        if f"ensemble__{qid}" not in cache:
            cache[f"ensemble__{qid}"] = genqr_ensemble_reformulate(query, reformulator, INSTRUCTIONS)
            save_cache(output_path, cache)

    print(f"Done. {len(queries)} queries saved to {output_path}")


if __name__ == "__main__":
    main()

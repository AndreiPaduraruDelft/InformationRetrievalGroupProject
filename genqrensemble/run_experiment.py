import argparse, os
from datetime import datetime
import pyterrier as pt
from config import INSTRUCTIONS, DATASETS
from reformulator import HFReformulator
from cache import get_cache_path, build_all_reformulated_topics
from evaluate import run_experiment

# Index fields per dataset (title is absent in msmarco-passage)
_DATASET_FIELDS = {
    "msmarco-passage/trec-dl-2019/judged": ["text"],
    "beir/dbpedia-entity/test":            ["text", "title"],
}


def load_pt_dataset(dataset_name):
    """Load a dataset via PyTerrier's irds wrapper and return corpus iter fn, topics, qrels, and fields."""
    dataset = pt.get_dataset(f"irds:{dataset_name}")
    fields  = _DATASET_FIELDS.get(dataset_name, ["text"])
    topics  = dataset.get_topics()
    qrels   = dataset.get_qrels()
    return dataset.get_corpus_iter, topics, qrels, fields


def get_or_build_index(corpus_iter_fn, index_path, fields):
    index_path = os.path.abspath(index_path)
    props = os.path.join(index_path, "data.properties")
    if not os.path.exists(props):
        os.makedirs(index_path, exist_ok=True)
        indexer = pt.IterDictIndexer(index_path, overwrite=True, fields=fields)
        indexer.index(corpus_iter_fn())
    return pt.IndexFactory.of(index_path + "/data.properties")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",       required=True)
    parser.add_argument("--device",      default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--datasets",    nargs="+", default=DATASETS)
    parser.add_argument("--cache_dir",   default="cache")
    parser.add_argument("--output",      default="results")
    parser.add_argument("--num_samples", type=int, default=None,
                        help="Limit evaluation to the first k queries")
    parser.add_argument("--use_cache",          action="store_true",
                        help="Load cached reformulations and skip regeneration for cached queries")
    parser.add_argument("--log_reformulations", action="store_true",
                        help="Write per-query reformulation log to logs/<run_name>.log")
    args = parser.parse_args()

    if not pt.java.started():
        pt.java.init()

    reformulator = HFReformulator(model_id=args.model, device=args.device)
    os.makedirs(args.output, exist_ok=True)

    for dataset_name in args.datasets:
        print(f"\n=== {dataset_name} | {args.model} ===")

        corpus_iter_fn, topics, qrels, fields = load_pt_dataset(dataset_name)

        # Keep only topics that have relevance judgements
        judged_qids = set(qrels["qid"].astype(str).unique())
        topics = topics[topics["qid"].astype(str).isin(judged_qids)].reset_index(drop=True)

        index_path = os.path.join(args.cache_dir, "indices", dataset_name.replace("/", "_"))
        index      = get_or_build_index(corpus_iter_fn, index_path, fields)
        bm25       = pt.terrier.Retriever(index, wmodel="BM25", num_results=100)

        if args.num_samples is not None:
            topics = topics.head(args.num_samples).reset_index(drop=True)

        safe_model   = args.model.replace("/", "_").replace("google_", "")
        safe_dataset = dataset_name.replace("/", "_")
        k_tag        = f"k{len(topics)}"
        timestamp    = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        run_name     = f"{safe_model}__{safe_dataset}__{k_tag}__{timestamp}"

        cache_path = get_cache_path(args.cache_dir, args.model, dataset_name)

        log_file = None
        if args.log_reformulations:
            os.makedirs("logs", exist_ok=True)
            log_path = os.path.join("logs", f"{run_name}.log")
            log_file = open(log_path, "w")
            print(f"  logging reformulations → {log_path}")

        try:
            flanqr_topics, ensemble_topics = build_all_reformulated_topics(
                topics, reformulator, INSTRUCTIONS, cache_path,
                use_cache=args.use_cache, log_file=log_file,
            )
        finally:
            if log_file is not None:
                log_file.close()

        results_df = run_experiment(bm25, topics, qrels, flanqr_topics, ensemble_topics)
        results_df["num_samples"] = len(topics)
        print(results_df.to_string())

        out_path = os.path.join(args.output, f"{run_name}.csv")
        results_df.to_csv(out_path, index=False)
        print(f"\n  saved → {out_path}")
        print(results_df.to_csv(index=False))


if __name__ == "__main__":
    main()

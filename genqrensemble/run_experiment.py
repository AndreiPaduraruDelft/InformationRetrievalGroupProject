import argparse, os
from datetime import datetime
import pyterrier as pt
from config import INSTRUCTIONS, DATASETS, DATASET_CONFIG
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


def _concat_corpus_iter(corpus_iter_fn, concat_fields):
    """Wrap a corpus iterator to concatenate specified fields into 'text'."""
    for doc in corpus_iter_fn():
        doc["text"] = " ".join(doc.get(f, "") or "" for f in concat_fields)
        yield doc


def get_or_build_index(corpus_iter_fn, index_path, fields):
    index_path = os.path.abspath(index_path)
    props = os.path.join(index_path, "data.properties")
    if not os.path.exists(props):
        print(f"  building index at {index_path} ...")
        os.makedirs(index_path, exist_ok=True)
        indexer = pt.IterDictIndexer(index_path, overwrite=True, fields=fields,
                                     meta={'docno': 512})
        indexer.index(corpus_iter_fn())
        print("  index built.")
    else:
        print(f"  loading existing index from {index_path}")
    return pt.IndexFactory.of(index_path + "/data.properties")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",       required=True)
    parser.add_argument("--device",      default="cpu", choices=["cpu", "cuda", "mps"])
    parser.add_argument("--datasets",    nargs="+", default=DATASETS)
    parser.add_argument("--cache_dir",   default="cache")
    parser.add_argument("--output",      default="results")
    parser.add_argument("--num_samples", type=int, default=None,
                        help="Limit evaluation to the first k queries")
    parser.add_argument("--use_cache",          action="store_true",
                        help="Load cached reformulations and skip regeneration for cached queries")
    parser.add_argument("--log_reformulations", action="store_true",
                        help="Write per-query reformulation log to logs/<run_name>.log")
    parser.add_argument("--rerank", action="store_true",
                        help="Add MonoT5 reranking pipelines to the experiment")
    parser.add_argument("--rerank_depth", type=int, default=1000,
                        help="Number of BM25 results to rerank with MonoT5 (default: 1000)")
    args = parser.parse_args()

    if not pt.java.started():
        pt.java.init()

    reformulator = HFReformulator(model_id=args.model, device=args.device)
    os.makedirs(args.output, exist_ok=True)

    for dataset_name in args.datasets:
        print(f"\n=== {dataset_name} | {args.model} ===")

        print("  loading topics and qrels ...")
        corpus_iter_fn, topics, qrels, fields = load_pt_dataset(dataset_name)

        # Keep only topics that have relevance judgements
        judged_qids = set(qrels["qid"].astype(str).unique())
        topics = topics[topics["qid"].astype(str).isin(judged_qids)].reset_index(drop=True)
        print(f"  {len(topics)} judged topics, {len(qrels)} qrels")

        ds_cfg        = DATASET_CONFIG.get(dataset_name, {"rel_threshold": 2, "num_results": 1000,
                                                           "bm25_k1": 1.2, "bm25_b": 0.75})
        rel_threshold = ds_cfg["rel_threshold"]
        num_results   = ds_cfg["num_results"]
        bm25_k1       = ds_cfg["bm25_k1"]
        bm25_b        = ds_cfg["bm25_b"]
        concat_fields = ds_cfg.get("concat_fields")

        if concat_fields:
            effective_corpus = lambda cf=concat_fields: _concat_corpus_iter(corpus_iter_fn, cf)
            effective_fields = ["text"]
        else:
            effective_corpus = corpus_iter_fn
            effective_fields = fields

        index_path = os.path.join(args.cache_dir, "indices", dataset_name.replace("/", "_"))
        index      = get_or_build_index(effective_corpus, index_path, effective_fields)
        bm25       = pt.terrier.Retriever(index, wmodel="BM25", num_results=num_results,
                                          controls={"bm25.k_1": bm25_k1, "bm25.k_3": 8, "c": bm25_b})
        print(f"  BM25 retriever ready (num_results={num_results}, k1={bm25_k1}, b={bm25_b}, rel_threshold={rel_threshold})")

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
            print("  reformulating queries ...")
            flanqr_topics, ensemble_topics = build_all_reformulated_topics(
                topics, reformulator, INSTRUCTIONS, cache_path,
                use_cache=args.use_cache, log_file=log_file,
            )
            print("  reformulations done.")
        finally:
            if log_file is not None:
                log_file.close()

        results_df = run_experiment(bm25, dataset_name, topics, qrels, flanqr_topics, ensemble_topics,
                                    rerank=args.rerank, rerank_depth=args.rerank_depth,
                                    rel_threshold=rel_threshold)
        results_df["num_samples"] = len(topics)
        numeric_cols = results_df.select_dtypes(include="number").columns
        results_df[numeric_cols] = results_df[numeric_cols].apply(
            lambda col: col.map(lambda x: float(f"{x:.3g}") if x == x else x)
        )
        print(results_df.to_string())

        out_path = os.path.join(args.output, f"{run_name}.csv")
        results_df.to_csv(out_path, index=False)
        print(f"\n  saved → {out_path}")


if __name__ == "__main__":
    main()

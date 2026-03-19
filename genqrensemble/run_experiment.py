import argparse, os
import pyterrier as pt
from config import INSTRUCTIONS, DATASETS
from reformulator import HFReformulator
from cache import get_cache_path, build_reformulated_topics
from evaluate import run_experiment


def get_or_build_index(corpus_dataset, index_path):
    index_path = os.path.abspath(index_path)
    props = os.path.join(index_path, "data.properties")
    if not os.path.exists(props):
        os.makedirs(index_path, exist_ok=True)
        indexer = pt.IterDictIndexer(index_path, overwrite=True, fields=["text", "title"])
        indexer.index(corpus_dataset.get_corpus_iter())
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
    args = parser.parse_args()

    if not pt.java.started():
        pt.java.init()

    reformulator = HFReformulator(model_id=args.model, device=args.device)
    os.makedirs(args.output, exist_ok=True)

    for dataset_name in args.datasets:
        print(f"\n=== {dataset_name} | {args.model} ===")

        corpus_dataset = pt.get_dataset(f"irds:{dataset_name}")
        test_dataset   = pt.get_dataset(f"irds:{dataset_name}/test")

        index_path = os.path.join(args.cache_dir, "indices", dataset_name.replace("/", "_"))
        index      = get_or_build_index(corpus_dataset, index_path)
        bm25       = pt.terrier.Retriever(index, wmodel="BM25", num_results=100)

        topics = test_dataset.get_topics()
        qrels  = test_dataset.get_qrels()

        if args.num_samples is not None:
            topics = topics.head(args.num_samples).reset_index(drop=True)

        cache_path      = get_cache_path(args.cache_dir, args.model, dataset_name)
        flanqr_topics   = build_reformulated_topics(topics, reformulator, INSTRUCTIONS, cache_path, "flanqr")
        ensemble_topics = build_reformulated_topics(topics, reformulator, INSTRUCTIONS, cache_path, "ensemble")

        results_df = run_experiment(bm25, topics, qrels, flanqr_topics, ensemble_topics)
        print(results_df.to_string())

        safe_model   = args.model.replace("/", "_")
        safe_dataset = dataset_name.replace("/", "_")
        out_path = os.path.join(args.output, f"{safe_model}__{safe_dataset}.csv")
        results_df.to_csv(out_path, index=False)


if __name__ == "__main__":
    main()

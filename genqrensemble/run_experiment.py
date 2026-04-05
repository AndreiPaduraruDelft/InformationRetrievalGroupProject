import argparse, os, re
from datetime import datetime
import torchvision  # must be eagerly loaded before pyserini triggers lazy transformers import
import pandas as pd
import pyterrier as pt
from config import INSTRUCTIONS, DATASETS
from reformulator import HFReformulator
from cache import get_cache_path, build_all_reformulated_topics
from evaluate import run_experiment

# Index fields per dataset (title is absent in msmarco-passage)
_DATASET_FIELDS = {
    "msmarco-passage/trec-dl-2019/judged": ["text"],
    "beir/dbpedia-entity/test":            ["text", "title"],
    "trec-robust04":                       ["text"],
}

# Datasets whose corpus is not freely downloadable but whose topics/qrels
# are available via pt.get_dataset().  Value = PyTerrier dataset name.
_PT_DATASET_ALIAS = {
    "trec-robust04": "trec-robust-2004",
}

# Pyserini prebuilt index names for extracting the corpus when building
# a Terrier index from scratch (used only once, then cached).
_PYSERINI_CORPUS_SOURCE = {
    "trec-robust04": "robust04",
}

# Datasets that should use Pyserini's prebuilt Lucene index for BM25
# retrieval instead of a native Terrier index.
_PYSERINI_PREBUILT = {
}


class PyseriniRetriever(pt.Transformer):
    """PyTerrier-compatible BM25 retriever backed by a Pyserini prebuilt Lucene index."""

    def __init__(self, prebuilt_index_name: str, num_results: int = 1000,
                 k1: float = None, b: float = None):
        self._prebuilt = prebuilt_index_name
        self._num_results = num_results
        self._k1 = k1
        self._b = b
        self._searcher = None

    def _get_searcher(self):
        if self._searcher is None:
            from pyserini.search.lucene import LuceneSearcher
            print(f"  downloading/loading Pyserini prebuilt index '{self._prebuilt}' ...")
            self._searcher = LuceneSearcher.from_prebuilt_index(self._prebuilt)
            if self._k1 is not None and self._b is not None:
                self._searcher.set_bm25(self._k1, self._b)
                print(f"  BM25 params: k1={self._k1}, b={self._b}")
            print(f"  Pyserini index ready ({self._searcher.num_docs:,} docs)")
        return self._searcher

    @staticmethod
    def _row_to_query(row) -> str:
        qt = getattr(row, "query_toks", None)
        if isinstance(qt, dict) and qt:
            return " ".join(
                term for term, _ in sorted(qt.items(), key=lambda kv: -kv[1])
            )
        return str(row.query)

    def transform(self, inp: pd.DataFrame) -> pd.DataFrame:
        searcher = self._get_searcher()
        qids    = [str(r.qid) for r in inp.itertuples()]
        queries = [self._row_to_query(r) for r in inp.itertuples()]

        hits_map = searcher.batch_search(
            queries, qids, k=self._num_results, threads=4
        )

        rows = [
            {"qid": qid, "docno": hit.docid, "score": float(hit.score), "rank": rank}
            for qid, hits in hits_map.items()
            for rank, hit in enumerate(hits)
        ]
        return (
            pd.DataFrame(rows)
            if rows
            else pd.DataFrame(columns=["qid", "docno", "score", "rank"])
        )


def _strip_sgml(text: str) -> str:
    """Remove SGML/XML tags from raw TREC document text."""
    return re.sub(r"<[^>]+>", " ", text)


def load_pt_dataset(dataset_name):
    """Load a dataset via PyTerrier's irds wrapper and return corpus iter fn, topics, qrels, and fields."""
    if dataset_name in _PT_DATASET_ALIAS:
        # Robust04 etc.: corpus is not freely downloadable, but topics/qrels
        # are available via PyTerrier's built-in dataset registry.
        pt_name = _PT_DATASET_ALIAS[dataset_name]
        dataset = pt.get_dataset(pt_name)
        fields  = _DATASET_FIELDS.get(dataset_name, ["text"])
        topics  = dataset.get_topics()
        qrels   = dataset.get_qrels()
        return None, topics, qrels, fields

    dataset = pt.get_dataset(f"irds:{dataset_name}")
    fields  = _DATASET_FIELDS.get(dataset_name, ["text"])
    topics  = dataset.get_topics()
    qrels   = dataset.get_qrels()
    return dataset.get_corpus_iter, topics, qrels, fields


def _corpus_iter_from_pyserini(prebuilt_name):
    """Yield {docno, text} dicts by iterating over a Pyserini prebuilt Lucene index."""
    from pyserini.index.lucene._base import LuceneIndexReader
    reader = LuceneIndexReader.from_prebuilt_index(prebuilt_name)
    total = reader.stats()["documents"]
    from tqdm import tqdm
    for i in tqdm(range(total), desc="extracting docs from Lucene index"):
        docid = reader.convert_internal_docid_to_collection_docid(i)
        doc = reader.doc(docid)
        raw = doc.raw() if doc else ""
        text = _strip_sgml(raw).strip()
        yield {"docno": docid, "text": text}


def get_or_build_index(corpus_iter_fn, index_path, fields, dataset_name=None):
    index_path = os.path.abspath(index_path)
    props = os.path.join(index_path, "data.properties")
    if not os.path.exists(props):
        print(f"  building index at {index_path} ...")
        os.makedirs(index_path, exist_ok=True)
        # Store document text in meta so pt.text.get_text(index, "text") works for reranking
        meta = {'docno': 512, 'text': 4096}
        indexer = pt.IterDictIndexer(index_path, overwrite=True, fields=fields,
                                     meta=meta)
        if corpus_iter_fn is not None:
            indexer.index(corpus_iter_fn())
        elif dataset_name in _PYSERINI_CORPUS_SOURCE:
            print(f"  extracting corpus from Pyserini prebuilt index '{_PYSERINI_CORPUS_SOURCE[dataset_name]}' ...")
            indexer.index(_corpus_iter_from_pyserini(_PYSERINI_CORPUS_SOURCE[dataset_name]))
        else:
            raise RuntimeError(f"No corpus source available for {dataset_name}")
        print("  index built.")
    else:
        print(f"  loading existing index from {index_path}")
    return pt.IndexFactory.of(index_path + "/data.properties")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",       required=True)
    parser.add_argument("--device",      default="cpu",
                        help="Device for the LLM: cpu, cuda, mps, or dml (AMD/DirectML on Windows)")
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
    parser.add_argument("--bm25_only", action="store_true",
                        help="Skip LLM reformulation and run BM25 baseline only (fast validation)")
    parser.add_argument("--reformulations_file", default=None,
                        help="Path to a pre-built reformulations JSON (flanqr__<qid>/ensemble__<qid> keys). "
                             "Skips LLM reformulation entirely.")
    parser.add_argument("--bm25_k1", type=float, default=1.2,
                        help="BM25 k1 parameter (default: 1.2)")
    parser.add_argument("--bm25_b", type=float, default=0.75,
                        help="BM25 b parameter (default: 0.75)")
    args = parser.parse_args()

    if not pt.java.started():
        # If any dataset needs Pyserini for corpus extraction, register
        # Anserini's JARs on the JVM classpath BEFORE pt.java.init().
        if any(ds in _PYSERINI_CORPUS_SOURCE for ds in args.datasets):
            import glob as _glob
            import jnius_config as _jc
            _pyserini_root = os.path.join(os.path.dirname(__import__('pyserini').__file__), 'resources', 'jars')
            _anserini_jars = _glob.glob(os.path.join(_pyserini_root, 'anserini-*-fatjar.jar'))
            if _anserini_jars:
                _jc.add_classpath(max(_anserini_jars, key=os.path.getctime))
                _jc.add_options('--add-modules=jdk.incubator.vector')
        pt.java.init()

    reformulator = None if (args.bm25_only or args.reformulations_file) else HFReformulator(model_id=args.model, device=args.device)
    os.makedirs(args.output, exist_ok=True)

    for dataset_name in args.datasets:
        print(f"\n=== {dataset_name} | {args.model} ===")

        print("  loading topics and qrels ...")
        corpus_iter_fn, topics, qrels, fields = load_pt_dataset(dataset_name)

        # Keep only topics that have relevance judgements
        judged_qids = set(qrels["qid"].astype(str).unique())
        topics = topics[topics["qid"].astype(str).isin(judged_qids)].reset_index(drop=True)
        print(f"  {len(topics)} judged topics, {len(qrels)} qrels")

        if dataset_name in _PYSERINI_PREBUILT:
            bm25 = PyseriniRetriever(_PYSERINI_PREBUILT[dataset_name], num_results=1000,
                                     k1=args.bm25_k1, b=args.bm25_b)
            print(f"  Pyserini BM25 retriever ready (k1={args.bm25_k1}, b={args.bm25_b})")
            get_text_pipe = None  # will fall back to ir_datasets in evaluate.py
        else:
            index_path = os.path.join(args.cache_dir, "indices", dataset_name.replace("/", "_"))
            index      = get_or_build_index(corpus_iter_fn, index_path, fields, dataset_name=dataset_name)
            bm25       = pt.terrier.Retriever(
                index, wmodel="BM25", num_results=1000,
                controls={"bm25.k_1": str(args.bm25_k1), "bm25.b": str(args.bm25_b)},
            )
            print(f"  Terrier BM25 retriever ready (k1={args.bm25_k1}, b={args.bm25_b})")
            # Use the Terrier index metadata for doc text if available (needed for reranking)
            if args.rerank and "text" in index.getMetaIndex().getKeys():
                get_text_pipe = pt.text.get_text(index, "text")
            elif args.rerank and dataset_name not in _PT_DATASET_ALIAS:
                get_text_pipe = pt.text.get_text(pt.get_dataset(f"irds:{dataset_name}"), "text")
            else:
                get_text_pipe = None

        if args.num_samples is not None:
            topics = topics.head(args.num_samples).reset_index(drop=True)

        safe_model   = args.model.replace("/", "_").replace("google_", "")
        safe_dataset = dataset_name.replace("/", "_")
        k_tag        = f"k{len(topics)}"
        timestamp    = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        run_name     = f"{safe_model}__{safe_dataset}__{k_tag}__{timestamp}"

        cache_path = get_cache_path(args.cache_dir, args.model, dataset_name)

        log_file = None
        if args.bm25_only:
            print("  --bm25_only: skipping reformulation, using original queries for all pipelines")
            flanqr_topics   = topics.copy()
            ensemble_topics = topics.copy()
        elif args.reformulations_file:
            import json as _json
            print(f"  loading reformulations from {args.reformulations_file} ...")
            with open(args.reformulations_file) as _f:
                _ref = _json.load(_f)
            flanqr_rows   = []
            ensemble_rows = []
            for _, _row in topics.iterrows():
                _qid = str(_row["qid"])
                flanqr_rows.append({"qid": _qid, "query": _ref.get(f"flanqr__{_qid}", _row["query"])})
                ensemble_rows.append({"qid": _qid, "query": _ref.get(f"ensemble__{_qid}", _row["query"])})
            flanqr_topics   = pd.DataFrame(flanqr_rows)
            ensemble_topics = pd.DataFrame(ensemble_rows)
            print(f"  loaded {len(flanqr_rows)} reformulated queries.")
        else:
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
                                    get_text_pipe=get_text_pipe, device=args.device)
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

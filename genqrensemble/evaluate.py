import re

import ir_measures
import pyterrier as pt
from tqdm import tqdm


def _query_swapper(reformulated_topics):
    """Transformer that replaces queries with pre-reformulated ones, keyed by qid."""
    qid_to_query = dict(zip(
        reformulated_topics["qid"].astype(str),
        reformulated_topics["query"],
    ))

    def swap(df):
        df = df.copy()
        df["query"] = df["qid"].astype(str).map(qid_to_query).fillna(df["query"])
        return df

    return pt.apply.generic(swap)


def _weighted_ensemble_swapper(original_topics, ensemble_topics, beta=0.05):
    """Build structured query_toks from original + reformulated queries.

    Original query terms contribute +1.0 per occurrence.
    Reformulated/ensemble query terms contribute +beta per occurrence.

    This means duplicates from both original and reformulated queries are kept
    and accumulated.

    Example:
        original:  "black hole formation"
        ensemble:  "black hole collapse black dwarf stellar"
        beta=0.05

        result:
        {
            "black": 1.05,
            "hole": 1.05,
            "formation": 1.0,
            "collapse": 0.05,
            "dwarf": 0.05,
            "stellar": 0.05
        }
    """
    orig_map = dict(zip(original_topics["qid"].astype(str), original_topics["query"]))
    ens_map = dict(zip(ensemble_topics["qid"].astype(str), ensemble_topics["query"]))

    def tokenise(text):
        return re.sub(r"[^\w\s]", "", text).lower().split()

    def build_query_toks(qid):
        orig_q = orig_map.get(qid, "")
        ens_q = ens_map.get(qid, orig_q)

        query_toks = {}

        # Original terms: +1.0 per occurrence
        for token in tokenise(orig_q):
            query_toks[token] = query_toks.get(token, 0.0) + 1.0

        # Reformulated terms: +beta per occurrence
        for token in tokenise(ens_q):
            query_toks[token] = query_toks.get(token, 0.0) + beta

        return query_toks

    def swap(df):
        df = df.copy()
        df["query_toks"] = df["qid"].astype(str).map(build_query_toks)
        return df

    return pt.apply.generic(swap)


def run_experiment(bm25, index, topics, qrels, flanqr_topics, ensemble_topics):
    from pyterrier_t5 import MonoT5ReRanker
    mono_t5  = MonoT5ReRanker(model="castorini/monot5-base-msmarco", verbose=False)
    get_text = pt.text.get_text(index, "text")

    flanqr_pipe          = _query_swapper(flanqr_topics)                                  >> bm25
    ensemble_pipe        = _query_swapper(ensemble_topics)                                >> bm25
    flanqr_weighted_pipe = _weighted_ensemble_swapper(topics, flanqr_topics,   beta=0.05) >> bm25
    weighted_pipe        = _weighted_ensemble_swapper(topics, ensemble_topics, beta=0.05) >> bm25

    bm25_mono     = bm25        >> get_text >> mono_t5
    flanqr_mono   = flanqr_pipe >> get_text >> mono_t5
    ensemble_mono = ensemble_pipe >> get_text >> mono_t5

    return pt.Experiment(
        [
            bm25, flanqr_pipe, flanqr_weighted_pipe, ensemble_pipe, weighted_pipe,
            bm25_mono, flanqr_mono, ensemble_mono,
        ],
        topics,
        qrels,
        eval_metrics=[ir_measures.nDCG@10, ir_measures.RR(rel=2), ir_measures.AP(rel=2)],
        names=[
            "BM25", "FlanQR", "FlanQR_beta_0_05", "GenQREnsemble", "GenQREnsemble_beta_0_05",
            "BM25+MonoT5", "FlanQR+MonoT5", "GenQREnsemble+MonoT5",
        ],
        baseline=1,
        correction="holm",
    )

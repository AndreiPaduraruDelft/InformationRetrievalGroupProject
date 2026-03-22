import re

import pyterrier as pt


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
    """Builds structured query_toks combining original and ensemble reformulation.

    Original terms start with weight 1.0. Expansion terms contribute beta per
    occurrence across the ensemble output, accumulating if repeated. Terms
    present in both original and expansion have their weights summed.

    Uses query_toks (dict) instead of TerrierQL term^weight strings, bypassing
    the query parser entirely and giving Terrier direct access to term weights.

    Example:
        original:  "black hole formation"
        ensemble:  "black hole collapse black dwarf stellar"
        beta=0.05
        result:    {"black": 1.1, "hole": 1.05, "formation": 1.0,
                    "collapse": 0.05, "dwarf": 0.05, "stellar": 0.05}
    """
    orig_map = dict(zip(original_topics["qid"].astype(str), original_topics["query"]))
    ens_map  = dict(zip(ensemble_topics["qid"].astype(str), ensemble_topics["query"]))

    def tokenise(text):
        return re.sub(r"[^\w\s]", "", text).lower().split()

    def build_query_toks(qid):
        orig_q = orig_map.get(qid, "")
        ens_q  = ens_map.get(qid, orig_q)

        query_toks = {}

        # Original terms: base weight 1.0
        for token in tokenise(orig_q):
            query_toks[token] = 1.0

        # Expansion terms: accumulate beta per occurrence
        for token in tokenise(ens_q):
            query_toks[token] = query_toks.get(token, 0.0) + beta

        return query_toks

    def swap(df):
        df = df.copy()
        df["query_toks"] = df["qid"].astype(str).map(build_query_toks)
        return df

    return pt.apply.generic(swap)


def run_experiment(bm25, topics, qrels, flanqr_topics, ensemble_topics):
    flanqr_pipe   = _query_swapper(flanqr_topics)   >> bm25
    ensemble_pipe = _query_swapper(ensemble_topics) >> bm25
    weighted_pipe = _weighted_ensemble_swapper(topics, ensemble_topics, beta=0.05) >> bm25

    return pt.Experiment(
        [bm25, flanqr_pipe, ensemble_pipe, weighted_pipe],
        topics,
        qrels,
        eval_metrics=["ndcg_cut_10", "map", "recip_rank", "P_10"],
        names=["BM25", "FlanQR", "GenQREnsemble", "GenQREnsemble_beta_0_05"],
        baseline=0,
    )

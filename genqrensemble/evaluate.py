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
    """Builds a Terrier weighted query combining original and ensemble reformulation.

    Terrier supports per-term boosting via the `term^weight` query syntax.
    Original query terms carry implicit weight 1.0; expansion terms unique to
    the ensemble output are appended with ^beta (default 0.05), downweighting
    them relative to the original signal.

    Example:
        original:  "cancer treatment"
        ensemble:  "cancer treatment oncology chemotherapy radiation"
        result:    "cancer treatment oncology^0.05 chemotherapy^0.05 radiation^0.05"
    """
    orig_map = dict(zip(original_topics["qid"].astype(str), original_topics["query"]))
    ens_map  = dict(zip(ensemble_topics["qid"].astype(str), ensemble_topics["query"]))

    def build_weighted(qid):
        orig_q = orig_map.get(qid, "")
        ens_q  = ens_map.get(qid, orig_q)

        # Tokenise both queries (strip punctuation, lowercase) to find expansion terms
        orig_tokens = set(re.sub(r"[^\w\s]", "", orig_q).lower().split())
        ens_tokens  = re.sub(r"[^\w\s]", "", ens_q).lower().split()

        # Collect unique expansion terms not already in the original query
        seen = set(orig_tokens)
        extra = []
        for token in ens_tokens:
            if token not in seen:
                seen.add(token)
                extra.append(token)

        if not extra:
            return orig_q
        weighted_extras = " ".join(f"{t}^{beta}" for t in extra)
        return f"{orig_q} {weighted_extras}"

    def swap(df):
        df = df.copy()
        df["query"] = df["qid"].astype(str).map(build_weighted)
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

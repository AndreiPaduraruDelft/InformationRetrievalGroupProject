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


def run_experiment(bm25, topics, qrels, flanqr_topics, ensemble_topics):
    flanqr_pipe   = _query_swapper(flanqr_topics)   >> bm25
    ensemble_pipe = _query_swapper(ensemble_topics) >> bm25

    return pt.Experiment(
        [bm25, flanqr_pipe, ensemble_pipe],
        topics,
        qrels,
        eval_metrics=["ndcg_cut_10", "map", "recip_rank"],
        names=["BM25", "FlanQR", "GenQREnsemble"],
        baseline=0,
    )

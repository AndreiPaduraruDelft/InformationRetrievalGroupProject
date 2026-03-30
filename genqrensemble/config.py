INSTRUCTIONS = [
    "Improve the search effectiveness by suggesting expansion terms for the query",
    "Recommend expansion terms for the query to improve search results",
    "Improve the search effectiveness by suggesting useful expansion terms for the query",
    "Maximize search utility by suggesting relevant expansion phrases for the query",
    "Enhance search efficiency by proposing valuable terms to expand the query",
    "Elevate search performance by recommending relevant expansion phrases for the query",
    "Boost the search accuracy by providing helpful expansion terms to enrich the query",
    "Increase the search efficacy by offering beneficial expansion keywords for the query",
    "Optimize search results by suggesting meaningful expansion terms to enhance the query",
    "Enhance search outcomes by recommending beneficial expansion terms to supplement the query",
]

DATASETS = ["msmarco-passage/trec-dl-2019/judged", "beir/dbpedia-entity/test"]

# Per-dataset retrieval and evaluation settings.
# rel_threshold: relevance level used for RR and AP (binary cutoff)
# num_results:   number of BM25 candidates retrieved per query
# bm25_k1/b:     BM25 term-frequency saturation and length normalisation parameters
DATASET_CONFIG = {
    "msmarco-passage/trec-dl-2019/judged": {
        "rel_threshold": 2,     # TREC DL 2019: 0-3 graded; rel≥2 is standard
        "num_results":   1000,
        "bm25_k1":       1.2,   # Terrier default
        "bm25_b":        0.75,  # Terrier default
    },
    "beir/dbpedia-entity/test": {
        "rel_threshold": 2,            # DBpedia: 0-2 graded; matches paper (rel≥2 = relevant)
        "num_results":   1000,
        "bm25_k1":       0.9,
        "bm25_b":        0.65,
        "concat_fields": ["title", "text"],  # Concatenate title+text into a single index field
    },
}

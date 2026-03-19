import json, os


def get_cache_path(cache_dir: str, model_id: str, dataset_name: str) -> str:
    safe_model   = model_id.replace("/", "_")
    safe_dataset = dataset_name.replace("/", "_")
    os.makedirs(cache_dir, exist_ok=True)
    return os.path.join(cache_dir, f"{safe_model}__{safe_dataset}.json")


def load_cache(path: str) -> dict:
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}


def save_cache(path: str, cache: dict):
    with open(path, "w") as f:
        json.dump(cache, f)


def build_reformulated_topics(topics, reformulator, instructions, cache_path, mode,
                              use_cache: bool = False):
    from genqr_methods import flanqr_reformulate, genqr_ensemble_reformulate
    import pandas as pd

    cache = load_cache(cache_path) if use_cache else {}
    rows  = []

    for _, row in topics.iterrows():
        qid, query = str(row["qid"]), row["query"]
        key = f"{mode}__{qid}"
        if key not in cache:
            if mode == "flanqr":
                reformed = flanqr_reformulate(query, reformulator, instructions[0])
            else:
                reformed = genqr_ensemble_reformulate(query, reformulator, instructions)
            cache[key] = reformed
            if use_cache:
                save_cache(cache_path, cache)
        rows.append({"qid": qid, "query": cache[key]})

    return pd.DataFrame(rows)

import json, os
from tqdm import tqdm


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
            save_cache(cache_path, cache)
        rows.append({"qid": qid, "query": cache[key]})

    return pd.DataFrame(rows)


def build_all_reformulated_topics(topics, reformulator, instructions, cache_path,
                                  use_cache: bool = False, log_file=None):
    """
    Generates FlanQR and GenQREnsemble reformulations for each query in a single
    per-query loop. When log_file is provided, writes a log entry for each query
    immediately after both reformulations are generated.
    """
    from genqr_methods import flanqr_reformulate, genqr_ensemble_reformulate
    import pandas as pd

    cache        = load_cache(cache_path) if use_cache else {}
    flanqr_rows  = []
    ensemble_rows = []

    for _, row in tqdm(topics.iterrows(), total=len(topics), desc="reformulating queries"):
        qid, query = str(row["qid"]), row["query"]

        flanqr_key = f"flanqr__{qid}"
        if flanqr_key not in cache:
            cache[flanqr_key] = flanqr_reformulate(query, reformulator, instructions[0])
            save_cache(cache_path, cache)

        ensemble_key = f"ensemble__{qid}"
        if ensemble_key not in cache:
            cache[ensemble_key] = genqr_ensemble_reformulate(query, reformulator, instructions)
            save_cache(cache_path, cache)

        if log_file is not None:
            log_file.write(f"[qid: {qid}]\n")
            log_file.write(f"original:   {query}\n")
            log_file.write(f"flanqr:     {cache[flanqr_key]}\n")
            log_file.write(f"ensemble:   {cache[ensemble_key]}\n")
            log_file.write("---\n")
            log_file.flush()

        flanqr_rows.append({"qid": qid, "query": cache[flanqr_key]})
        ensemble_rows.append({"qid": qid, "query": cache[ensemble_key]})

    return pd.DataFrame(flanqr_rows), pd.DataFrame(ensemble_rows)

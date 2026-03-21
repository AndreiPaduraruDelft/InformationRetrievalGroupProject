import pandas as pd, glob, os

dfs = []
for path in glob.glob("results/*.csv"):
    df    = pd.read_csv(path)
    fname = os.path.basename(path).replace(".csv", "")
    model, dataset = fname.split("__", 1)
    df["model"]   = model.replace("google_", "")
    df["dataset"] = dataset
    dfs.append(df)

final = (pd.concat(dfs, ignore_index=True)
           .rename(columns={"name": "setting", "recip_rank": "mrr"})
           [["dataset", "model", "setting", "ndcg_cut_10", "map", "mrr", "P_10"]]
           .sort_values(["dataset", "model", "setting"]))

final.to_csv("results/results_final.csv", index=False)
final.to_latex("results/results_final.tex", index=False, float_format="%.3f")
print(final.to_string(index=False))

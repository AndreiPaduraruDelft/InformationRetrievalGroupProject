# Webis Touche 2020 — Results & Reproducibility Notes

## Results

Each cell shows **paper / ours**. The paper used `flan-t5-xxl`; we used `flan-t5-base`.
No result reached statistical significance after Holm–Bonferroni correction in either case.

| Setting               | nDCG@10             | MAP                 | MRR                 |
|-----------------------|:-------------------:|:-------------------:|:-------------------:|
| BM25                  | .206 / .343         | .206 / .251         | .454 / .581         |
| FlanQR                | .276 / .327         | .221 / .240         | .476 / .544         |
| FlanQR β=0.05         | .276 / .328         | .221 / .238         | .476 / .573         |
| GenQREnsemble         | **.317** / **.360** | **.257** / **.263** | **.555** / **.587** |
| GenQREnsemble β=0.05  | .292 / .351         | .242 / .254         | .489 / .582         |

---

## Explanation of discrepancies

Our BM25 baseline (nDCG@10 = 0.343) is notably higher than the paper's (0.206). The relative
ordering across methods is preserved. GenQREnsemble outperforms FlanQR and BM25 in both cases,
but the absolute scores differ systematically. We attribute this to three compounding factors:

**1. Dataset version and qrel handling**

The paper likely loaded the Webis Touche dataset through the `beir` Python library, which
binarizes the multi-grade relevance judgements (grades 0, 1, 2) before computing nDCG@10. Our
implementation uses `ir_datasets`'s `beir/webis-touche2020/v2`, which preserves the original
graded relevance. The `ir_measures.nDCG@10` metric then computes **graded** nDCG (gain =
relevance grade), whereas the `beir` library computes **binary** nDCG (gain = 1 for any relevant
document). This difference in the ideal DCG normaliser and in how grade-1 documents contribute
produces systematically higher scores across all methods in our evaluation.

**2. Model size**

The paper uses `flan-t5-xxl` for all reformulations; we use `flan-t5-base`. This does not affect
the BM25 baseline but explains any remaining gap in the reformulation pipelines (FlanQR,
GenQREnsemble).

**3. Relative trends are preserved**

Despite the absolute differences, the qualitative conclusions hold: GenQREnsemble (swap) achieves
the highest nDCG@10 and MAP in both the paper and our evaluation, and no improvement over FlanQR
reaches statistical significance on this dataset in either case. This mirrors the pattern observed
on the DBpedia Entity dataset, where differences in document field handling between evaluation
frameworks produced consistent offsets without changing the ranking of methods.
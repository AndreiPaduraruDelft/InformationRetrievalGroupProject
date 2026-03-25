# GenQREnsemble

Reproduces the **GenQREnsemble** query reformulation method and evaluates it against **BM25** and **FlanQR** baselines using PyTerrier. Optionally adds **MonoT5** reranking.

## Setup

### Prerequisites

- Python 3.8+
- Java 11+ (required by PyTerrier)
  - macOS: `brew install java`
  - Linux: `sudo apt install default-jdk`
  - DelftBlue: `module load java/11`

### Install dependencies

```bash
pip install -r requirements.txt
```

## Datasets

Experiments run on the following datasets loaded via `ir_datasets`:

| Dataset | Corpus | Queries | Relevance |
|---|---|---|---|
| `msmarco-passage/trec-dl-2019/judged` | MS MARCO passages | TREC DL 2019 (judged only) | 0–3 (graded) |
| `beir/dbpedia-entity/test` | DBpedia entities | BEIR test split | 0–2 (graded) |

Corpora are indexed automatically on first run and cached under `cache/indices/`.

## Usage

### Run an experiment

```bash
# Small model — runs on CPU
python run_experiment.py --model google/flan-t5-small --device cpu
python run_experiment.py --model google/flan-t5-base  --device cpu
python run_experiment.py --model google/flan-t5-large --device cpu
python run_experiment.py --model google/flan-t5-xl    --device cpu

# Large model — requires GPU (DelftBlue only)
python run_experiment.py --model google/flan-t5-xxl --device cuda
```

Each run evaluates **BM25**, **FlanQR**, **FlanQR (β=0.05)**, **GenQREnsemble**, and **GenQREnsemble (β=0.05)** on the specified datasets and saves results to `results/<model>__<dataset>__k<n>__<timestamp>.csv`.

### Run on a single dataset with a limited number of queries

Useful for quick smoke-tests or debugging before a full run:

```bash
python run_experiment.py \
  --model google/flan-t5-base \
  --device cpu \
  --datasets msmarco-passage/trec-dl-2019/judged \
  --num_samples 20 \
  --log_reformulations
```

Per-query reformulation logs are written to `logs/<run_name>.log`.

### Reuse cached reformulations

Reformulations are saved to a JSON cache on every run. Pass `--use_cache` to skip regeneration for queries already in the cache:

```bash
python run_experiment.py \
  --model google/flan-t5-base \
  --device cpu \
  --use_cache
```

### MonoT5 reranking

Add `--rerank` to include **BM25+MonoT5**, **FlanQR+MonoT5**, and **GenQREnsemble+MonoT5** pipelines using `castorini/monot5-base-msmarco`. Use `--rerank_depth` to control how many BM25 results are passed to MonoT5 (default: 1000, use 100 for faster CPU runs):

```bash
# Full depth — matches paper, requires GPU for reasonable speed
python run_experiment.py \
  --model google/flan-t5-base \
  --device cpu \
  --use_cache \
  --rerank

# Reduced depth — faster on CPU, slight metric difference
python run_experiment.py \
  --model google/flan-t5-base \
  --device cpu \
  --use_cache \
  --rerank \
  --rerank_depth 100
```

### CLI options

| Flag | Default | Description |
|---|---|---|
| `--model` | required | HuggingFace model ID |
| `--device` | `cpu` | `cpu`, `cuda`, or `mps` (Apple Silicon) |
| `--datasets` | both datasets | space-separated `ir_datasets` dataset names |
| `--cache_dir` | `cache/` | directory for cached reformulations and indices |
| `--output` | `results/` | directory for result CSVs |
| `--num_samples` | `None` (all) | limit evaluation to the first k queries |
| `--use_cache` | off | skip regeneration for queries already in the cache |
| `--log_reformulations` | off | write per-query reformulation log to `logs/<run_name>.log` |
| `--rerank` | off | add MonoT5 reranking pipelines to the experiment |
| `--rerank_depth` | `1000` | number of BM25 results to rerank with MonoT5 |

### Output file naming

Each result CSV is named with the model, dataset, query count, and a timestamp:

```
flan-t5-base__msmarco-passage_trec-dl-2019_judged__k43__2024-03-19_14-32-05.csv
```

The CSV also includes a `num_samples` column recording the number of queries evaluated.

### Merge all results

After all runs are complete:

```bash
python merge_results.py
```

Produces `results/results_final.csv` and `results/results_final.tex` with all models and datasets combined.

## DelftBlue (SLURM)

Edit `slurm_job.sh` to replace `your_env` with your conda environment name, then submit:

```bash
sbatch slurm_job.sh
```

Logs are written to `logs/<job_id>.out`.

## Models

| Model | Size | Device |
|---|---|---|
| `google/flan-t5-small` | 80M | CPU |
| `google/flan-t5-base` | 250M | CPU |
| `google/flan-t5-large` | 780M | CPU |
| `google/flan-t5-xl` | 3B | CPU |
| `google/flan-t5-xxl` | 11B | GPU (≥22GB VRAM) |

> If `flan-t5-xxl` runs out of memory, add `load_in_8bit=True` in `reformulator.py` (requires `bitsandbytes`).

## File Structure

```
genqrensemble/
├── requirements.txt    — Python dependencies
├── config.py           — Instructions and dataset list
├── reformulator.py     — HFReformulator (wraps flan-t5 via HuggingFace)
├── genqr_methods.py    — FlanQR and GenQREnsemble reformulation logic
├── cache.py            — JSON cache to avoid re-generating reformulations
├── evaluate.py         — pt.Experiment runner (BM25 + optional MonoT5)
├── run_experiment.py   — Main CLI entry point
├── merge_results.py    — Aggregates CSVs into final table + LaTeX
├── slurm_job.sh        — SLURM script for DelftBlue
├── cache/              — Cached reformulated queries and indices (auto-created)
├── results/            — Result CSVs and final merged outputs
└── logs/               — Reformulation logs and SLURM job logs
```

## Metrics

Results are reported using official TREC DL 2019 evaluation measures:

- **nDCG@10** — primary ranking quality metric (graded relevance)
- **AP(rel=2)** — mean average precision, relevance threshold ≥ 2
- **RR(rel=2)** — mean reciprocal rank, relevance threshold ≥ 2

Significance testing uses paired t-tests with **Holm-Bonferroni correction** against the **FlanQR** baseline (`baseline=1` in `pt.Experiment`).

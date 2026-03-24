# GenQREnsemble

Reproduces the **GenQREnsemble** query reformulation method and evaluates it against **BM25** and **FlanQR** baselines using PyTerrier.

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

Each run evaluates **BM25**, **FlanQR**, **GenQREnsemble**, and **GenQREnsemble (weighted β=0.05)** on both datasets and saves results to `results/<model>__<dataset>__k<n>__<timestamp>.csv`.

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

By default, reformulations are **always regenerated from scratch**. Pass `--use_cache` to skip regeneration for queries that already have a cached result:

```bash
python run_experiment.py \
  --model google/flan-t5-base \
  --device cpu \
  --use_cache
```

### CLI options

| Flag | Default | Description |
|---|---|---|
| `--model` | required | HuggingFace model ID |
| `--device` | `cpu` | `cpu` or `cuda` |
| `--datasets` | both datasets | space-separated `ir_datasets` dataset names |
| `--cache_dir` | `cache/` | directory for cached reformulations and indices |
| `--output` | `results/` | directory for result CSVs |
| `--num_samples` | `None` (all) | limit evaluation to the first k queries |
| `--use_cache` | off | reuse cached reformulations instead of regenerating |
| `--log_reformulations` | off | write per-query reformulation log to `logs/<run_name>.log` |

### Output file naming

Each result CSV is named with the model, dataset, query count, and a timestamp:

```
flan-t5-base__msmarco-passage_trec-dl-2019__k43__2024-03-19_14-32-05.csv
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
├── evaluate.py         — pt.Experiment runner
├── run_experiment.py   — Main CLI entry point
├── merge_results.py    — Aggregates CSVs into final table + LaTeX
├── slurm_job.sh        — SLURM script for DelftBlue
├── cache/              — Cached reformulated queries and indices (auto-created)
├── results/            — Result CSVs and final merged outputs
└── logs/               — Reformulation logs and SLURM job logs
```

## Metrics

Results are reported as:

- **nDCG@10** — primary ranking quality metric
- **MAP** — mean average precision
- **MRR** — mean reciprocal rank
- **P@10** — precision at 10

Significance testing uses paired t-tests with **Holm-Bonferroni correction** against the BM25 baseline (`baseline=0` in `pt.Experiment`).

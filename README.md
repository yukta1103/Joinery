# Joinery

Text-to-SQL with a small LLM (Qwen2.5-1.5B) trained by reinforcement learning (GRPO) on
**execution-based rewards**. The model writes a SQL query, the query runs against a real SQLite
database, and the reward depends on whether the returned rows match those of the gold query.
Everything runs on free Kaggle/Colab GPUs.

## Status

**Phase 1 (evaluation harness and baselines) is in progress.** The harness is built and tested.
No baseline has been run yet, so this README reports no results.

| Phase | Scope | Status |
|---|---|---|
| 1 | Evaluation harness, reward draft, baselines | Harness done, baselines not run |
| 2 | SFT | Not started |
| 3 | GRPO with execution rewards | Not started |
| 4 | 4-bit GGUF, Hugging Face Space demo, write-up | Not started |

## Results

None yet. This table gets numbers only from recorded runs (`results/<run>/metrics.json`).
Execution accuracy uses the default column-permutation-invariant metric. The strict column-order
accuracy appears beside it.

| Model | Exec. accuracy (dev) | Strict | Easy | Medium | Hard | Extra |
|---|---|---|---|---|---|---|
| Qwen2.5-1.5B-Instruct | not run | | | | | |
| Qwen2.5-Coder-1.5B-Instruct | not run | | | | | |
| Qwen2.5-Coder-7B-Instruct (4-bit) | not run | | | | | |

## Repository layout

```
src/joinery/
  config.py          typed YAML configs and --set overrides
  data.py            Spider loading, database checks, gold-query audit
  spider_hardness.py official Spider difficulty levels
  prompts.py         schema + question prompt with a fixed output format
  extraction.py      final-SQL extraction from model output
  sqltext.py         lexical helpers (quotes, comments, top-level ORDER BY)
  execution.py       read-only SQLite execution with timeout and row limit
  comparison.py      result-set comparison rules
  scoring.py         extract -> execute -> compare for one completion
  rewards.py         tiered execution reward (for GRPO in Phase 3)
  metrics.py         accuracy aggregation (overall, by difficulty, by error type)
  eval.py            resumable evaluation runner
  generation.py      generator interface used by the runner
  hf_generator.py    Hugging Face batched greedy generation
scripts/             download_spider.py, check_data.py, run_eval.py, score_predictions.py
configs/             one YAML per run
notebooks/           thin Kaggle wrappers
tests/               pytest suite
results/             run outputs (predictions.jsonl is gitignored)
```

## Setup

```bash
python -m venv .venv
.venv/Scripts/activate        # Windows; use .venv/bin/activate on Linux/macOS
pip install -e ".[dev]"       # harness and tests only, no GPU needed
pip install -e ".[model]"     # add torch/transformers to run models
python -m pytest
```

## Data setup

### Download (recommended)

```bash
pip install -e ".[dev]"            # includes gdown
python scripts/download_spider.py  # add --keep-zip to keep spider_data.zip
```

The script fetches the official `spider_data.zip` from Google Drive (file id
`1403EGqzIDoHMdQF4c9Bkyl7dZLZ5Wt6J`) and checks its SHA256 against
`00636695dabed6b5f4b8328a16b13e069a2f16591d5efcce57660669c85b121b`. It then extracts it, with
path checks, to:

```
data/spider_data/
  dev.json  train_spider.json  train_others.json  tables.json  ...
  database/<db_id>/<db_id>.sqlite
```

- **Rerunning is safe.** A verified extract is detected and skipped. A kept zip with the right hash
  is reused without downloading again.
- **On a hash mismatch nothing is deleted.** The script prints the actual hash and stops.
- **If Google Drive refuses** (quota exceeded or a confirmation page), the script stops with a
  pointer to the manual steps below. It never retries.

`data/` and `*.zip` are gitignored. The extracted dataset is about 1.8 GB and is never committed.

### Manual download

1. Open https://yale-lily.github.io/spider and download `spider_data.zip` from its Google Drive
   link.
2. Optionally check its SHA256 (the value above), for example with `sha256sum spider_data.zip` or
   `certutil -hashfile spider_data.zip SHA256` on Windows.
3. Extract it so you get the `data/spider_data/` layout shown above. You can also put the zip at
   `data/spider_data.zip` and run `python scripts/download_spider.py`, which verifies and extracts
   it without downloading.

If your copy sits elsewhere, set `data.spider_dir` (and `data.database_dir` if the databases live
somewhere else again) in the config or with `--set`.

### Check the data

```bash
python scripts/check_data.py --config configs/eval_qwen15b_instruct.yaml
```

This prints the number of examples per difficulty, any gold queries that fail, how many return no
rows or depend on row order, and the largest gold result (`max_gold_rows`). It saves the report to
`results/data_check_dev.json` and exits non-zero if any gold query exceeds `execution.max_rows`.

For the dev split this check passes with 1,034 examples over 20 databases: 248 easy, 446 medium,
174 hard, 166 extra. 0 gold queries fail, 49 return no rows, 231 have a top-level `ORDER BY`, and
`max_gold_rows` is 20,662. The largest is `dev_00455`/`dev_00456`, `ORDER BY birth_date` over all
`wta_1` players, where 15,834 of the 20,662 rows tie on `birth_date`; see "Ties under `ORDER BY`"
below.

**Kaggle:** upload `spider_data.zip` once as a private Kaggle dataset (Kaggle extracts it) and
attach it to the notebook. Its path looks like `/kaggle/input/<dataset-slug>/spider_data`.

## Running a baseline

Local (GPU) or any machine with the `model` extra:

```bash
# smoke test on 20 examples
python scripts/run_eval.py --config configs/eval_qwen15b_instruct.yaml \
    --set data.limit=20 --set run_name=smoke_qwen15b_instruct
# full dev set
python scripts/run_eval.py --config configs/eval_qwen15b_instruct.yaml
```

On Kaggle, open `notebooks/eval_baseline.ipynb`, set `CONFIG` and `SPIDER_DIR`, and run all cells.
The notebook clones this repo, installs it without touching Kaggle's torch, and calls
`scripts/run_eval.py`.

**Resuming.** Predictions are appended and flushed after every batch. Rerunning the same command
skips examples that are already done. `/kaggle/working` does not survive between sessions, so to
resume on Kaggle you save the notebook version, attach its output as an input to the next session
and set `RESUME_FROM`. A resumed run must use the same config, except for paths and W&B settings,
or the runner refuses to mix results.

**Re-scoring without a GPU.** After any change to extraction, execution or comparison rules:

```bash
python scripts/score_predictions.py --config results/<run>/config.yaml \
    --predictions results/<run>/predictions.jsonl --out results/<run>/rescored
```

Both `run_eval.py` and `score_predictions.py` finish with a table of the default and strict
execution accuracy, overall and per difficulty.

### Outputs (`results/<run_name>/`)

| File | Contents |
|---|---|
| `config.yaml` | Resolved config |
| `generation_settings.json` | Model, commit hash, model-default and effective decoding config, library versions, GPU |
| `predictions.jsonl` | One record per example (fields below) |
| `metrics.json` | Overall and per-difficulty accuracy (default and strict), error-type counts, `complete` flag |

Each record contains: `example_id, db_id, difficulty, question, gold_sql, prompt, prompt_tokens,
output, output_tokens, truncated, extracted_sql, extraction_status, parses, exec_error_type,
exec_error_message, pred_row_count, gold_row_count, gold_ordered, correct, correct_strict,
permutation_limit_hit, error_type, reward, excluded, exclusion_reason`.

`error_type` names the first stage that failed: an extraction status (`no_block`,
`empty_block`), an execution error (`syntax`, `schema`, `timeout`, `not_allowed`,
`multiple_statements`, `too_many_rows`, `other`), `permutation_limit` (the query ran but the
column-permutation search hit its budget, so the result is undecided rather than proven wrong),
or `wrong_result`. It is `null` when the prediction is correct.

## Evaluation design

### Prompt

The system prompt asks for brief reasoning, then exactly one query in a ```` ```sql ```` block.
The user message contains the database's own `CREATE TABLE` statements (from `sqlite_master`, in
creation order). Each table can be followed by its first `sample_rows` rows in a SQL comment
(default 3). Long values are truncated, and prompts longer than `warn_prompt_tokens` are counted
in the metrics, never silently cut.

### Extraction

1. A ```` ```sql ```` block that is still open when the output ends (usually hit the token limit)
   is used if it comes after every closed block.
2. Otherwise the **last** closed block tagged `sql`/`sqlite`, in any case.
3. Otherwise the last untagged closed block.
4. Otherwise no SQL (`no_block`). Bare SQL in the reasoning is not scraped, so the format is
   enforced.

Comments and trailing semicolons are stripped. Blocks with more than one statement are flagged
and fail execution.

### Execution safety

Every query gets a fresh connection opened `mode=ro&immutable=1` with `PRAGMA query_only`, plus an
authorizer that allows only reads, SELECT, functions and recursive CTEs. Writes, DDL, `ATTACH`,
`PRAGMA`, transactions and `VACUUM` are rejected before they run. A progress handler enforces a
wall-clock timeout (default 5 s), and results with more than `max_rows` rows (default 100,000) are
rejected. Such a result is discarded, never truncated, so it can never match gold by
coincidence. Invalid UTF-8 text in a database is decoded with replacement characters.

### Comparison rules

| Case | Rule |
|---|---|
| Row order | Multiset (order-insensitive) unless the gold query has a **top-level** `ORDER BY`, then an ordered list. `ORDER BY` inside subqueries, `OVER (...)` or aggregate arguments doesn't count. |
| Duplicate rows | Counted: `[(1,), (1,)]` does not match `[(1,)]`. This applies after any column permutation and with either row-order rule. |
| Column order (default metric) | Ignored: a prediction is correct if **some permutation of its columns** matches the gold result, as in test-suite-sql-eval. The row-order and duplicate rules apply after permuting. The permutation must be one mapping shared by every row: `[(1,'b'), (2,'a')]` does not match `[(1,'a'), (2,'b')]`, even though each column holds the right values. This is `correct` / `execution_accuracy`, and the reward uses it. |
| Column order (strict metric) | Must match the gold order exactly. Reported alongside the default as `correct_strict` / `execution_accuracy_strict` (and per difficulty) in records, `metrics.json` and script output. A strict match is always also a default match. |
| Column count | Must be equal under both metrics. An extra or missing column never matches, even when the shared columns agree. Column names are always ignored. |
| Duplicate column values | Columns holding identical values are interchangeable, so `SELECT a, a, b` matches gold `SELECT b, a, a` whenever the values line up. |
| Permutation search budget | The search prunes candidate columns by their values, tries interchangeable columns once, and (for unordered rows) checks partial rows column by column. With ordered rows it is O(width²) and needs no budget. With unordered rows there is no polynomial bound: in the worst case found so far, even- vs odd-parity bit tables (any cross join of boolean columns), nothing can be pruned before the last column. That took 13 s at 8 columns and 255 s at 9 before the cap. The search therefore spends a deterministic budget (`comparison.permutation_search_budget`, default 1,000,000 row operations, about 0.3 s worst case on a laptop). Running out gives `permutation_limit_hit: true` and `error_type: permutation_limit`. The example counts as not correct, earns at most `execute_reward`, and is counted in `n_permutation_limit`. The budget counts work rather than time, so the outcome is the same on every machine. |
| NULL | `NULL` equals `NULL`. It does not equal `0` or `''`. |
| Numbers | `1 == 1.0`. Floats are rounded to `float_digits` (default 6) decimals, so `0.1 + 0.2` matches `0.3`. Limitation: two values 1e-6 apart on either side of a rounding boundary (e.g. 2.5000006 vs 2.4999996) do not match. |
| Text | Exact and case-sensitive. The text `'1'` does not equal the integer `1`. |
| Empty gold result | Two empty results match whatever their columns, so any wrong query that returns nothing also "matches". Metrics report accuracy separately for empty and non-empty gold results. |
| Ties under `ORDER BY` | Compared as ordered lists, so a correct query that breaks ties differently is marked wrong. This is a known limitation, also present in test-suite-sql-eval. |
| Results over `execution.max_rows` | A prediction over the limit has its rows discarded, never truncated (a truncated result could match gold by accident). It is labelled `too_many_rows`, counts as not correct, earns at most `parse_reward`, and is counted in `n_pred_too_many_rows`. A result of exactly `max_rows` rows is scored normally. A gold query over the limit would be excluded, so `check_data.py` reports `n_gold_too_many_rows` and `max_gold_rows` and exits non-zero unless the count is 0. |
| Failing gold queries | Excluded from scoring and listed with their error. Metrics report accuracy over scored examples and also counting excluded examples as wrong. |

### Difficulty levels

Spider's easy/medium/hard/extra levels are computed from the parsed SQL stored in Spider's JSON.
The logic is copied unchanged from the official evaluation script (see attribution below).

### Reward (draft for Phase 3)

`joinery.rewards.execution_reward(completion, gold_sql, db_path, ...) -> float` is a pure function
whose tiers come from the config: `match_reward` (1.0) if the rows match under the default
(column-permutation-invariant) metric, `execute_reward` (0.3)
if the query runs but the rows differ, `parse_reward` (0.1) if SQL was extracted and parses
(sqlglot) but fails to run, and 0 otherwise. It calls the same scoring code as the evaluation, so
the top reward always means the evaluation counts the completion as correct.

### Reproducibility

Decoding is greedy with `repetition_penalty` set explicitly in every config (1.0, i.e. none). This
overrides the model's shipped generation config: the Qwen2.5 Instruct models default to sampling
with a penalty of 1.1. `generation_settings.json` records both the model's default config and the
effective decoding config, along with the model's commit hash. Pending examples are batched in a deterministic order (longest prompt first),
and a resumed run builds the same batches. Batched fp16 decoding can differ in the last bits from
unbatched decoding, so results are reproducible for a given batch size and GPU type.

## Attribution

- Spider dataset: Yu et al., "Spider: A Large-Scale Human-Labeled Dataset for Complex and
  Cross-Domain Semantic Parsing and Text-to-SQL Task", EMNLP 2018. https://yale-lily.github.io/spider
- `src/joinery/spider_hardness.py` is adapted from the official Spider evaluation script,
  https://github.com/taoyds/spider, Apache License 2.0.

# Joinery

Text-to-SQL with a small LLM trained by reinforcement learning (GRPO) on execution-based rewards.
The model writes a SQL query, the query runs against a real SQLite database, and the reward depends
on whether its result rows match those of the gold query.

## Goal and deliverables

- Base model: Qwen2.5-1.5B-Instruct or Qwen2.5-Coder-1.5B-Instruct.
- Pipeline: baseline eval -> SFT -> GRPO with execution rewards -> 4-bit quantization.
- Deliverables:
  - Results table: base vs SFT vs SFT+GRPO vs quantized vs a larger baseline model.
  - Analysis by Spider difficulty level (easy / medium / hard / extra).
  - A 4-bit GGUF model.
  - A Hugging Face Space demo.

## Phases

1. **Evaluation harness + baselines** (current phase). No SFT or GRPO work yet.
2. SFT.
3. GRPO with the execution reward drafted and tested in Phase 1.
4. Quantization (GGUF), demo, write-up.

Do not start a later phase until the user approves it.

## Hard constraints

- **$0 budget.** All training and evaluation run on free Kaggle notebooks (T4 16GB or P100) or free
  Colab. No paid APIs, no paid GPUs.
- **Neither T4 nor P100 supports bf16.** Use fp16 (or 4-bit) on these GPUs.
- **Kaggle sessions time out.** Every long-running job must checkpoint and be resumable.
- **Thin notebooks.** Code is developed locally in this repo and run on Kaggle. Notebooks only install
  the package, set paths and call entry points. All real logic lives in `src/joinery/`.
- **Stack:** Python, PyTorch, Hugging Face transformers/datasets, Unsloth + TRL (`GRPOTrainer`) for
  training, `sqlite3` for execution, Weights & Biases (free tier) for tracking.
- If an approach won't fit in T4 memory or Kaggle time limits, tell the user before building it.

## Repo structure

```
src/joinery/   package: data, prompts, extraction, execution, metrics, rewards, eval
scripts/       CLI entry points
notebooks/     thin Kaggle/Colab wrappers
configs/       YAML configs for runs
tests/         pytest suite
results/       JSONL predictions and metrics (large files gitignored)
README.md
CLAUDE.md
```

## Engineering standards

- Type hints everywhere, small focused modules, clear docstrings, no dead code.
- Config-driven runs: no hardcoded paths or hyperparameters in library logic.
- Fixed random seeds, and the full resolved config is saved next to every result so each run is
  reproducible.
- The evaluation harness underlies every later number. Changes to extraction, execution, comparison
  or rewards need tests, including tricky edge cases.
- **Never fabricate, estimate or placeholder any result or metric.** If something hasn't been run,
  say so. Results tables contain only numbers produced by a recorded run.

## Authorship and git rules

- Yukta is the sole author of this project.
- Do not list Claude, Claude Code or Anthropic as an author, co-author or contributor anywhere.
- No `Co-Authored-By` trailers in commit messages.
- No "Generated with Claude Code" or similar lines in commits, PR descriptions, README, docstrings,
  comments or any other file.
- Commit messages use plain conventional style describing the change (e.g. `feat: add SQL extraction`).

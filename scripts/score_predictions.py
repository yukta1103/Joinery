"""Re-score saved predictions with the current harness (no GPU needed).

Use this after changing extraction, execution or comparison rules, or to score Kaggle outputs
locally. The config must be the one the predictions were generated with (its config.yaml).

Example:
    python scripts/score_predictions.py --config results/eval_qwen15b_instruct/config.yaml \
        --predictions results/eval_qwen15b_instruct/predictions.jsonl \
        --out results/eval_qwen15b_instruct/rescored --set data.spider_dir=data/spider_data
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from joinery.config import load_eval_config
from joinery.eval import rescore
from joinery.metrics import format_headline


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True, help="config the predictions were generated with")
    parser.add_argument("--predictions", required=True, type=Path, help="predictions.jsonl to re-score")
    parser.add_argument("--out", required=True, type=Path, help="output directory for re-scored files")
    parser.add_argument("--set", dest="overrides", action="append", default=[], help="override, e.g. data.spider_dir=...")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    config = load_eval_config(args.config, args.overrides)
    metrics = rescore(args.predictions, config, args.out)
    print(json.dumps(metrics, indent=2))
    print(format_headline(metrics))


if __name__ == "__main__":
    main()

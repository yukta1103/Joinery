"""Run or resume a baseline evaluation (needs a GPU and the ``model`` extra).

Example:
    python scripts/run_eval.py --config configs/eval_qwen15b_instruct.yaml --set data.limit=20 \
        --set run_name=smoke_qwen15b_instruct
"""

from __future__ import annotations

import argparse
import json
import logging

from joinery.config import load_eval_config
from joinery.eval import run_eval
from joinery.hf_generator import HFGenerator, set_seed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True, help="eval YAML config")
    parser.add_argument("--set", dest="overrides", action="append", default=[], help="override, e.g. data.limit=20")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    config = load_eval_config(args.config, args.overrides)
    set_seed(config.seed)
    metrics = run_eval(config, lambda: HFGenerator(config.model, config.generation))
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()

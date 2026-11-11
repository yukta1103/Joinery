"""Check the Spider setup and audit every gold query of a split (no GPU needed).

Example:
    python scripts/check_data.py --config configs/eval_qwen15b_instruct.yaml
    python scripts/check_data.py --config configs/eval_qwen15b_instruct.yaml --set data.split=train
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from joinery.config import load_eval_config
from joinery.data import audit_gold, load_examples, require_databases


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True, help="eval YAML config (only data/execution are used)")
    parser.add_argument("--set", dest="overrides", action="append", default=[], help="override, e.g. data.split=train")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    config = load_eval_config(args.config, args.overrides)
    examples = load_examples(config.data)
    database_dir = config.data.resolved_database_dir()
    require_databases(examples, database_dir)
    report = audit_gold(examples, database_dir, config.execution)

    out_path = Path(config.output_dir) / f"data_check_{config.data.split}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    summary = {k: v for k, v in report.items() if k != "gold_failures"}
    print(json.dumps(summary, indent=2))
    for failure in report["gold_failures"]:
        print(f"gold failure {failure['example_id']} ({failure['db_id']}): {failure['error_message']}")
    print(f"Full report written to {out_path}")


if __name__ == "__main__":
    main()

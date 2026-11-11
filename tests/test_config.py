from __future__ import annotations

from pathlib import Path

import pytest

from joinery.config import EvalConfig, apply_overrides, build_dataclass, load_eval_config, save_config

CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"


@pytest.mark.parametrize("path", sorted(CONFIG_DIR.glob("*.yaml")), ids=lambda p: p.name)
def test_shipped_configs_load(path: Path) -> None:
    cfg = load_eval_config(path)
    assert cfg.run_name == path.stem
    assert cfg.model.dtype == "float16", "T4/P100 have no bf16"


def test_overrides_are_typed(tmp_path: Path) -> None:
    path = sorted(CONFIG_DIR.glob("*.yaml"))[0]
    cfg = load_eval_config(path, ["data.limit=20", "data.database_dir=null", "wandb.mode=offline"])
    assert cfg.data.limit == 20 and cfg.data.database_dir is None and cfg.wandb.mode == "offline"


def test_round_trip(tmp_path: Path) -> None:
    cfg = load_eval_config(sorted(CONFIG_DIR.glob("*.yaml"))[0])
    save_config(cfg, tmp_path / "c.yaml")
    assert load_eval_config(tmp_path / "c.yaml") == cfg


def test_unknown_key_rejected() -> None:
    path = sorted(CONFIG_DIR.glob("*.yaml"))[0]
    with pytest.raises(ValueError, match="unknown config keys under 'generation.'"):
        load_eval_config(path, ["generation.temperature=0.7"])


def test_missing_required_key_rejected() -> None:
    with pytest.raises(ValueError, match="invalid config"):
        build_dataclass(EvalConfig, {"run_name": "x"})


def test_apply_overrides_does_not_mutate() -> None:
    data = {"a": {"b": 1}}
    assert apply_overrides(data, ["a.b=2", "c.d=[1, 2]"]) == {"a": {"b": 2}, "c": {"d": [1, 2]}}
    assert data == {"a": {"b": 1}}


@pytest.mark.parametrize("bad", ["novalue", "=3", "a.b.c=1"])
def test_bad_overrides(bad: str) -> None:
    with pytest.raises(ValueError):
        apply_overrides({"a": {"b": 1}}, [bad])

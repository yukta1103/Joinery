"""Typed run configuration loaded from YAML, with dotted command-line overrides.

Every tunable value used by the library lives in one of these dataclasses. Run configs are YAML
files under ``configs/``; the fully resolved config is saved next to each run's outputs.
"""

from __future__ import annotations

import copy
import dataclasses
import typing
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TypeVar

import yaml

T = TypeVar("T")


@dataclass(frozen=True)
class DataConfig:
    """Location of the Spider release and which split to use.

    ``spider_dir`` is the extracted Spider folder containing ``dev.json``, ``train_spider.json``
    and ``train_others.json``. ``database_dir`` defaults to ``<spider_dir>/database``.
    """

    spider_dir: str
    database_dir: str | None = None
    split: str = "dev"
    limit: int | None = None

    def __post_init__(self) -> None:
        if self.split not in ("dev", "train"):
            raise ValueError(f"data.split must be 'dev' or 'train', got {self.split!r}")
        if self.limit is not None and self.limit <= 0:
            raise ValueError(f"data.limit must be positive, got {self.limit}")

    def resolved_database_dir(self) -> Path:
        """Return the directory holding ``<db_id>/<db_id>.sqlite`` files."""
        if self.database_dir is not None:
            return Path(self.database_dir)
        return Path(self.spider_dir) / "database"


@dataclass(frozen=True)
class ModelConfig:
    """Which model to load and how."""

    name_or_path: str
    revision: str | None = None
    dtype: str = "float16"
    load_in_4bit: bool = False

    def __post_init__(self) -> None:
        if self.dtype not in ("float16", "float32", "bfloat16"):
            raise ValueError(f"model.dtype must be float16, float32 or bfloat16, got {self.dtype!r}")


@dataclass(frozen=True)
class GenerationConfig:
    """Batched greedy decoding settings for evaluation."""

    batch_size: int
    max_new_tokens: int

    def __post_init__(self) -> None:
        if self.batch_size <= 0 or self.max_new_tokens <= 0:
            raise ValueError("generation.batch_size and generation.max_new_tokens must be positive")


@dataclass(frozen=True)
class PromptConfig:
    """Prompt construction settings.

    ``sample_rows`` example rows are shown per table (0 disables them). Sample values longer than
    ``max_value_chars`` are truncated. Prompts longer than ``warn_prompt_tokens`` are counted and
    reported, never silently truncated.
    """

    sample_rows: int
    max_value_chars: int
    warn_prompt_tokens: int

    def __post_init__(self) -> None:
        if self.sample_rows < 0 or self.max_value_chars <= 0 or self.warn_prompt_tokens <= 0:
            raise ValueError("invalid prompt config values")


@dataclass(frozen=True)
class ExecutionConfig:
    """Limits applied to every SQL execution."""

    timeout_s: float = 5.0
    max_rows: int = 100_000

    def __post_init__(self) -> None:
        if self.timeout_s <= 0 or self.max_rows <= 0:
            raise ValueError("execution.timeout_s and execution.max_rows must be positive")


@dataclass(frozen=True)
class ComparisonConfig:
    """Result comparison settings. Floats are rounded to ``float_digits`` decimal places."""

    float_digits: int = 6

    def __post_init__(self) -> None:
        if self.float_digits < 0:
            raise ValueError("comparison.float_digits must be non-negative")


@dataclass(frozen=True)
class RewardConfig:
    """Tiered execution reward values. A completion with no extractable SQL scores 0."""

    parse_reward: float
    execute_reward: float
    match_reward: float

    def __post_init__(self) -> None:
        if not 0.0 <= self.parse_reward <= self.execute_reward <= self.match_reward:
            raise ValueError("rewards must satisfy 0 <= parse_reward <= execute_reward <= match_reward")


@dataclass(frozen=True)
class WandbConfig:
    """Weights & Biases logging. ``mode`` is online, offline or disabled."""

    mode: str = "disabled"
    project: str = "joinery"
    entity: str | None = None

    def __post_init__(self) -> None:
        if self.mode not in ("online", "offline", "disabled"):
            raise ValueError(f"wandb.mode must be online, offline or disabled, got {self.mode!r}")


@dataclass(frozen=True)
class EvalConfig:
    """Complete configuration of one evaluation run."""

    run_name: str
    seed: int
    output_dir: str
    data: DataConfig
    model: ModelConfig
    generation: GenerationConfig
    prompt: PromptConfig
    execution: ExecutionConfig
    comparison: ComparisonConfig
    reward: RewardConfig
    wandb: WandbConfig = field(default_factory=WandbConfig)

    def run_dir(self) -> Path:
        """Directory holding this run's predictions, metrics and resolved config."""
        return Path(self.output_dir) / self.run_name


def build_dataclass(cls: type[T], data: Mapping[str, Any], path: str = "") -> T:
    """Construct ``cls`` from a nested mapping, rejecting unknown keys.

    Args:
        cls: A dataclass type whose fields may themselves be dataclasses.
        data: Mapping of field names to values.
        path: Dotted prefix used in error messages.

    Raises:
        ValueError: On unknown keys, missing required keys or a non-mapping nested section.
    """
    hints = typing.get_type_hints(cls)
    fields = {f.name: f for f in dataclasses.fields(cls)}  # type: ignore[arg-type]
    unknown = sorted(set(data) - set(fields))
    if unknown:
        raise ValueError(f"unknown config keys under '{path or '<root>'}': {unknown}")
    kwargs: dict[str, Any] = {}
    for name, value in data.items():
        hint = hints[name]
        if dataclasses.is_dataclass(hint):
            if not isinstance(value, Mapping):
                raise ValueError(f"config section '{path}{name}' must be a mapping")
            value = build_dataclass(hint, value, f"{path}{name}.")  # type: ignore[arg-type]
        kwargs[name] = value
    try:
        return cls(**kwargs)
    except TypeError as exc:
        raise ValueError(f"invalid config under '{path or '<root>'}': {exc}") from exc


def apply_overrides(data: Mapping[str, Any], overrides: Sequence[str]) -> dict[str, Any]:
    """Return a copy of ``data`` with ``dotted.key=value`` overrides applied.

    Values are parsed as YAML, so ``limit=20`` gives an int and ``database_dir=null`` gives None.
    """
    result = copy.deepcopy(dict(data))
    for item in overrides:
        key, sep, raw = item.partition("=")
        if not sep or not key:
            raise ValueError(f"override must look like 'section.key=value', got {item!r}")
        node = result
        parts = key.split(".")
        for part in parts[:-1]:
            child = node.setdefault(part, {})
            if not isinstance(child, dict):
                raise ValueError(f"override {item!r} descends into non-mapping key {part!r}")
            node = child
        node[parts[-1]] = yaml.safe_load(raw)
    return result


def load_eval_config(path: str | Path, overrides: Sequence[str] = ()) -> EvalConfig:
    """Load an :class:`EvalConfig` from YAML and apply command-line overrides."""
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return build_dataclass(EvalConfig, apply_overrides(data, overrides))


def config_to_dict(config: Any) -> dict[str, Any]:
    """Convert a config dataclass to plain nested dicts."""
    return dataclasses.asdict(config)


def save_config(config: Any, path: str | Path) -> None:
    """Write a config dataclass to YAML."""
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(config_to_dict(config), f, sort_keys=False)

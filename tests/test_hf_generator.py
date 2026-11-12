"""Decoding-config overrides. Skipped when transformers/torch are not installed."""

from __future__ import annotations

import pytest

pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from joinery.config import GenerationConfig  # noqa: E402
from joinery.hf_generator import decoding_config  # noqa: E402


@pytest.fixture
def model_default() -> "transformers.GenerationConfig":
    # Shaped like a chat model's shipped config: sampling on, a repetition penalty above 1.
    return transformers.GenerationConfig(
        do_sample=True,
        temperature=0.7,
        top_p=0.8,
        top_k=20,
        repetition_penalty=1.1,
        bos_token_id=1,
        eos_token_id=[2, 3],
        pad_token_id=3,
    )


def test_overrides_repetition_penalty_and_sampling(model_default) -> None:
    cfg = GenerationConfig(batch_size=4, max_new_tokens=128, repetition_penalty=1.0)
    effective = decoding_config(model_default, cfg, pad_token_id=3)
    assert effective.repetition_penalty == 1.0
    assert effective.do_sample is False
    assert effective.num_beams == 1
    assert effective.max_new_tokens == 128
    assert effective.eos_token_id == [2, 3]


def test_model_default_is_not_mutated(model_default) -> None:
    cfg = GenerationConfig(batch_size=4, max_new_tokens=128, repetition_penalty=1.0)
    decoding_config(model_default, cfg, pad_token_id=3)
    assert model_default.repetition_penalty == 1.1 and model_default.do_sample is True


def test_effective_config_is_recordable(model_default) -> None:
    cfg = GenerationConfig(batch_size=4, max_new_tokens=128, repetition_penalty=1.0)
    recorded = decoding_config(model_default, cfg, pad_token_id=3).to_dict()
    assert recorded["repetition_penalty"] == 1.0 and recorded["do_sample"] is False

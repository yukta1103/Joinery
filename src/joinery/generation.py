"""Model-agnostic generation interface used by the evaluation runner.

The runner depends only on this protocol, so it can be tested without a GPU or torch. The Hugging
Face implementation lives in :mod:`joinery.hf_generator`.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from joinery.prompts import Message


@dataclass(frozen=True)
class Generation:
    """One decoded completion.

    ``truncated`` is True when decoding stopped at the token limit instead of an end-of-sequence
    token.
    """

    text: str
    num_tokens: int
    truncated: bool


class Generator(Protocol):
    """What the evaluation runner needs from a model."""

    def render_prompt(self, messages: list[Message]) -> str:
        """Apply the chat template and return the text the model continues from."""
        ...

    def count_tokens(self, text: str) -> int:
        """Number of tokens in ``text``."""
        ...

    def generate(self, prompts: Sequence[str]) -> list[Generation]:
        """Greedily decode one completion per prompt, in order."""
        ...

    def describe(self) -> dict[str, Any]:
        """JSON-serializable description of the model and decoding settings, for the run record."""
        ...

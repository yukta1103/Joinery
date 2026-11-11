from __future__ import annotations

from pathlib import Path

from joinery.config import PromptConfig
from joinery.extraction import extract_sql
from joinery.prompts import SYSTEM_PROMPT, build_messages, format_value, quote_identifier, schema_text


def test_schema_contains_create_statements_in_order(db_path: Path) -> None:
    text = schema_text(str(db_path), 0, 50)
    assert text.index("CREATE TABLE singer") < text.index("CREATE TABLE concert")
    assert "Sample rows" not in text
    assert text.rstrip().endswith(";")


def test_schema_sample_rows(db_path: Path) -> None:
    text = schema_text(str(db_path), 2, 50)
    assert "/* Sample rows from singer:\nsinger_id | name | country | age | net_worth\n" in text
    assert "1 | Joe | USA | 30 | 1.5\n2 | Ann | UK | 25 | NULL\n*/" in text
    assert "3 | Bo" not in text


def test_format_value() -> None:
    assert format_value(None, 10) == "NULL"
    assert format_value(b"\x00\x01", 10) == "<blob 2 bytes>"
    assert format_value("a\nb   c", 10) == "a b c"
    assert format_value("abcdefghijkl", 8) == "abcde..."
    assert format_value(1.5, 10) == "1.5"


def test_quote_identifier() -> None:
    assert quote_identifier('we"ird') == '"we""ird"'


def test_build_messages(db_path: Path) -> None:
    cfg = PromptConfig(sample_rows=1, max_value_chars=20, warn_prompt_tokens=4096)
    messages = build_messages(db_path, "  How many singers?  ", cfg)
    assert messages[0] == {"role": "system", "content": SYSTEM_PROMPT}
    assert messages[1]["role"] == "user"
    assert messages[1]["content"].startswith("Database schema:\nCREATE TABLE singer")
    assert messages[1]["content"].endswith("\n\nQuestion: How many singers?")


def test_prompt_is_deterministic(db_path: Path) -> None:
    cfg = PromptConfig(sample_rows=3, max_value_chars=20, warn_prompt_tokens=4096)
    assert build_messages(db_path, "q", cfg) == build_messages(db_path, "q", cfg)


def test_system_prompt_format_is_extractable() -> None:
    # The format the system prompt asks for must round-trip through extraction.
    example = "Reasoning: count rows in singer.\n```sql\nSELECT count(*) FROM singer\n```"
    assert "```sql" in SYSTEM_PROMPT
    assert extract_sql(example).sql == "SELECT count(*) FROM singer"

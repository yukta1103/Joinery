from __future__ import annotations

import pytest

from joinery.extraction import ExtractionStatus, clean_sql, extract_sql


def test_standard_format() -> None:
    out = "We need to count singers.\n```sql\nSELECT count(*) FROM singer;\n```"
    result = extract_sql(out)
    assert result.sql == "SELECT count(*) FROM singer"
    assert result.status is ExtractionStatus.OK


def test_last_sql_block_wins() -> None:
    out = (
        "First try:\n```sql\nSELECT name FROM singer\n```\n"
        "Actually we need the age too.\n```sql\nSELECT name, age FROM singer\n```\nDone."
    )
    assert extract_sql(out).sql == "SELECT name, age FROM singer"


@pytest.mark.parametrize("tag", ["SQL", "Sql", "sqlite", " sql ", "SQLite"])
def test_tag_variants(tag: str) -> None:
    result = extract_sql(f"```{tag}\nSELECT 1\n```")
    assert result.sql == "SELECT 1"
    assert result.status is ExtractionStatus.OK


def test_untagged_fence_fallback() -> None:
    result = extract_sql("Reasoning.\n```\nSELECT 1\n```")
    assert result.sql == "SELECT 1"
    assert result.status is ExtractionStatus.UNTAGGED_FENCE


def test_tagged_block_preferred_over_later_untagged() -> None:
    out = "```sql\nSELECT 1\n```\nOutput:\n```\n1\n```"
    result = extract_sql(out)
    assert result.sql == "SELECT 1"
    assert result.status is ExtractionStatus.OK


def test_other_language_blocks_ignored() -> None:
    out = "```python\nprint('SELECT 1')\n```"
    assert extract_sql(out).status is ExtractionStatus.NO_BLOCK


def test_unclosed_fence_from_truncation() -> None:
    out = "Reasoning.\n```sql\nSELECT name FROM singer WHERE age >"
    result = extract_sql(out)
    assert result.sql == "SELECT name FROM singer WHERE age >"
    assert result.status is ExtractionStatus.UNCLOSED_FENCE


def test_unclosed_fence_after_closed_block_wins() -> None:
    out = "```sql\nSELECT 1\n```\nBetter:\n```sql\nSELECT 2"
    result = extract_sql(out)
    assert result.sql == "SELECT 2"
    assert result.status is ExtractionStatus.UNCLOSED_FENCE


@pytest.mark.parametrize(
    "out",
    [
        "",
        "I don't know.",
        "SELECT count(*) FROM singer",
        "The answer is `SELECT count(*) FROM singer`.",
    ],
)
def test_no_block(out: str) -> None:
    result = extract_sql(out)
    assert result.sql is None
    assert result.status is ExtractionStatus.NO_BLOCK


@pytest.mark.parametrize("body", ["", "   \n", ";", "-- nothing here", "/* empty */ ;"])
def test_empty_block(body: str) -> None:
    result = extract_sql(f"```sql\n{body}\n```")
    assert result.sql is None
    assert result.status is ExtractionStatus.EMPTY_BLOCK


def test_crlf_line_endings() -> None:
    result = extract_sql("Reasoning.\r\n```sql\r\nSELECT 1\r\nFROM t;\r\n```\r\n")
    assert result.sql == "SELECT 1\nFROM t"
    assert result.status is ExtractionStatus.OK


def test_text_after_block_ignored() -> None:
    out = "```sql\nSELECT 1\n```\nThis query returns one."
    assert extract_sql(out).sql == "SELECT 1"


def test_multiple_statements_flagged() -> None:
    result = extract_sql("```sql\nSELECT 1;\nSELECT 2;\n```")
    assert result.sql == "SELECT 1;\nSELECT 2"
    assert result.status is ExtractionStatus.MULTIPLE_STATEMENTS


def test_semicolon_inside_string_is_one_statement() -> None:
    result = extract_sql("```sql\nSELECT name FROM singer WHERE name = 'a;b';\n```")
    assert result.sql == "SELECT name FROM singer WHERE name = 'a;b'"
    assert result.status is ExtractionStatus.OK


def test_comments_removed_but_string_dashes_kept() -> None:
    out = "```sql\n-- count them\nSELECT count(*) FROM t WHERE x = '--'; -- end\n```"
    assert extract_sql(out).sql == "SELECT count(*) FROM t WHERE x = '--'"


def test_indented_fence() -> None:
    out = "Answer:\n    ```sql\n    SELECT 1\n    ```"
    assert extract_sql(out).sql == "SELECT 1"


def test_multiline_query_preserved() -> None:
    body = "SELECT name\nFROM singer\nWHERE age > 20"
    assert extract_sql(f"```sql\n{body}\n```").sql == body


def test_clean_sql_strips_multiple_trailing_semicolons() -> None:
    assert clean_sql("SELECT 1 ; ;\n") == "SELECT 1"

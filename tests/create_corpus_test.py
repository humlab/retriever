from pathlib import Path
from typing import Any, LiteralString

import pandas as pd
import pytest
from pandas import Timestamp

import retriever.create_corpus


@pytest.fixture(name="input_file")
def fixture_input_file(tmp_path: Path) -> Path:
    file = tmp_path / "input.txt"
    content = """Some text
Datum 2024-01-30

Innehållsförteckning:

> Title One, Source One, 2020-12-04 05:26

> Title Two, Source Two, 2023-12-04 04:32


Title One
Source One, 2020-12-04 05:26
Publicerat på webb.

Text one.
==============================================================================

Title Two
Source Two, 2020-12-04 04:32
Publicerat i print.

Text two.

"""

    file.write_text(content)
    return file


@pytest.fixture(name="expected_corpus")
def fixture_corpus() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "title": {0: "Title One", 1: "Title Two"},
            "source": {0: "Source One", 1: "Source Two"},
            "date": {0: "2020-12-04 05:26", 1: "2023-12-04 04:32"},
            "toc_line_number": {0: 6, 1: 8},
            "full_text": {
                0: "Title One\nSource One, 2020-12-04 05:26\nPublicerat på webb.\n\nText one.",
                1: "Title Two\nSource Two, 2020-12-04 04:32\nPublicerat i print.\n\nText two.",
            },
            "header_lenght": {0: 3, 1: 3},
            "header": {
                0: "Title One\nSource One, 2020-12-04 05:26\nPublicerat på webb.",
                1: "Title Two\nSource Two, 2020-12-04 04:32\nPublicerat i print.",
            },
            "media": {0: "webb", 1: "print"},
            "pages": {0: None, 1: None},
            "url": {0: None, 1: None},
            "article_text": {0: "Text one.", 1: "Text two."},
            "date_time": {0: Timestamp("2020-12-04 05:26:00"), 1: Timestamp("2023-12-04 04:32:00")},
        }
    )


def test_get_toc(input_file: Any) -> None:
    toc, offset = retriever.create_corpus.get_toc(input_file, "Innehållsförteckning:", 2)

    expected_toc: list[list[str | int]] = [
        ["Title One", "Source One", "2020-12-04 05:26", 6],
        ["Title Two", "Source Two", "2023-12-04 04:32", 8],
    ]
    expected_offset = 11

    assert toc == expected_toc
    assert offset == expected_offset


def test_get_articles(input_file: Any) -> None:
    _, offset = retriever.create_corpus.get_toc(input_file, "Innehållsförteckning:", 2)
    articles = retriever.create_corpus.get_articles(input_file, offset)

    expected_articles = [
        "Title One\nSource One, 2020-12-04 05:26\nPublicerat på webb.\n\nText one.",
        "Title Two\nSource Two, 2020-12-04 04:32\nPublicerat i print.\n\nText two.",
    ]

    assert articles == expected_articles


def test_create_corpus(input_file: Any, expected_corpus: pd.DataFrame) -> None:
    toc, offset = retriever.create_corpus.get_toc(input_file, "Innehållsförteckning:", 2)
    articles = retriever.create_corpus.get_articles(input_file, offset)
    corpus = retriever.create_corpus.create_corpus(toc, articles)

    assert corpus.equals(expected_corpus)


def test_create_corpus_without_body_has_empty_article_text() -> None:
    toc: list[list[Any]] = [["Title", "Source", "2020-12-04 05:26", 6]]
    article = "Title\nSource, 2020-12-04 05:26\nPublicerat på webb."

    corpus = retriever.create_corpus.create_corpus(toc, [article])

    assert corpus["article_text"].tolist() == [""]


def test_create_corpus_raises_when_toc_and_articles_differ() -> None:
    toc: list[list[Any]] = [["Title", "Source", "2020-12-04 05:26", 6]]

    with pytest.raises(ValueError, match="1 table of contents entries but 2 articles"):
        retriever.create_corpus.create_corpus(toc, ["article one", "article two"])


def test_create_corpus_keeps_row_with_unparsable_date() -> None:
    toc: list[list[Any]] = [["Title", "Source", "okänt datum", 6]]
    article = "Title\nSource, okänt datum\nPublicerat på webb.\n\nBody."

    corpus = retriever.create_corpus.create_corpus(toc, [article])

    assert corpus["date_time"].isna().all()


def test_get_toc_warns_about_unparsed_line(tmp_path: Path, caplog: Any) -> None:
    file = tmp_path / "input.txt"
    file.write_text("Innehållsförteckning:\n\n> A Title, A Source, 2020-12-04 05:26\n> B Title, B Source\n\nend\n")

    toc, _ = retriever.create_corpus.get_toc(str(file), "Innehållsförteckning:", 2)

    assert [entry[0] for entry in toc] == ["A Title"]
    assert "WARNING" in caplog.text
    assert "Unparsed toc line 4: '> B Title, B Source'" in caplog.text


@pytest.fixture(name="duplicates")
def fixture_duplicates() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "title": {0: "Title", 1: "Title"},
            "source": {0: "Source", 1: "Source"},
            "date": {0: "2020-12-04", 1: "2020-12-04"},
            "toc_line_number": {0: 6, 1: 8},
            "full_text": {
                0: "Title\nSource, 2020-12-04 05:26\nPublicerat på webb.\n\nText.",
                1: "Title\nSource, 2020-12-04 05:26\nPublicerat på webb.\n\nDifferent text.",
            },
            "header_lenght": {0: 3, 1: 3},
            "header": {
                0: "Title\nSource, 2020-12-04 05:26\nPublicerat på webb.",
                1: "Title\nSource, 2020-12-04 05:26\nPublicerat på webb.",
            },
            "media": {0: "webb", 1: "webb"},
            "pages": {0: None, 1: None},
            "url": {0: None, 1: None},
            "article_text": {0: "Text.", 1: "nDifferent text."},
            "date_time": {0: Timestamp("2020-12-04 05:26:00"), 1: Timestamp("2020-12-04 05:26:00")},
            "document_name": {0: "Document_20201204_webb", 1: "Document_20201204_webb"},
            "filename": {0: "Document_20201204_webb.txt", 1: "Document_20201204_webb.txt"},
            "year": {0: "2020", 1: "2020"},
            "input_file": {0: "input.txt", 1: "input.txt"},
            "id": {0: "001001", 1: "001002"},
            "document_id": {0: "001", 1: "002"},
        }
    )


def test_log_diffs(caplog: Any, duplicates: pd.DataFrame, tmp_path: Path) -> None:
    retriever.create_corpus.log_diffs(duplicates, output_folder=str(tmp_path), save_diffs=False)

    assert caplog.text

    expected_output = "Differences for ('Document_20201204_webb', 'Source', '2020-12-04', 'webb'):\n- Text.\n+ nDifferent text."

    assert "INFO" in caplog.text
    assert caplog.text.count("INFO") == 1
    assert expected_output in caplog.text


def test_log_diffs_save_diffs(duplicates: pd.DataFrame, tmp_path: Path) -> None:
    retriever.create_corpus.log_diffs(duplicates, output_folder=str(tmp_path), save_diffs=True)

    expected_output = {
        "filename": "Document_20201204_webb_Source_2020-12-04_webb_0.diff",
        "content": "- Text.\n+ nDifferent text.",
    }

    with open(tmp_path / "diff" / expected_output["filename"], "r", encoding="utf-8") as file:
        assert file.read() == expected_output["content"]


@pytest.fixture(name="article_text")
def fixture_article_text() -> LiteralString:
    text = """Title

Source, 2020-12-04 05:26
Publicerat på webb.

Text."""
    return text


def test_fix_header_for_article_with_empty_line_after_title(article_text: LiteralString) -> None:
    expected_output = """Title
Source, 2020-12-04 05:26
Publicerat på webb.

Text."""

    assert retriever.create_corpus.fix_header(article_text)[0] == expected_output


@pytest.fixture(name="article_text_multiline")
def fixture_article_text_multiline() -> LiteralString:
    text = """Title line one
Title line two
Title line three

Source, 2020-12-04 05:26
Sida 23
Publicerat på webb.

Paragraph.

Another paragraph."""
    return text


def test_fix_header_for_article_with_multiline_title(article_text_multiline: LiteralString) -> None:
    expected_output = """Title line one
Title line two
Title line three
Source, 2020-12-04 05:26
Sida 23
Publicerat på webb.

Paragraph.

Another paragraph."""

    assert retriever.create_corpus.fix_header(article_text_multiline)[0] == expected_output


def test_extract_pages_without_a_number() -> None:
    toc: list[list[Any]] = [["Title", "Source", "2020-12-04 05:26", 6]]

    retriever.create_corpus.extract_pages(toc, 0, ["Title", "Sida", "Publicerat på webb."])
    retriever.create_corpus.extract_pages(toc, 0, ["Title", "Sida 23", "Publicerat på webb."])

    assert toc[0][4:6] == [None, "23"]


def test_check_title_warns_on_mismatch(caplog: Any) -> None:
    toc: list[list[Any]] = [["Article title", "Source", "2020-12-04 05:26", 6]]

    retriever.create_corpus.check_title(toc, 0, "Different title\nBody.")

    assert "WARNING" in caplog.text
    assert "Title mismatch in article 0: toc 'Article title' but article starts with 'Different title'" in caplog.text


def test_check_title_stays_quiet_on_match(caplog: Any) -> None:
    toc: list[list[Any]] = [["Article title", "Source", "2020-12-04 05:26", 6]]

    retriever.create_corpus.check_title(toc, 0, "Article title\nBody.")

    assert caplog.text == ""

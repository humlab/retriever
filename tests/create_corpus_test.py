import os
import shutil
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


def test_main_warns_when_different_articles_share_a_filename(tmp_path: Path, caplog: Any) -> None:
    (tmp_path / "export.txt").write_text(
        "Innehållsförteckning:\n"
        "\n"
        "> Blixten: Här är läget, Source One, 2020-12-04 05:26\n"
        "\n"
        "> Blixten - Här är läget, Source One, 2020-12-04 05:26\n"
        "\n"
        "\n"
        "Blixten: Här är läget\nSource One, 2020-12-04 05:26\nPublicerat på webb.\n"
        "\nText one.\n"
        "==============================================================================\n"
        "\nBlixten - Här är läget\nSource One, 2020-12-04 05:26\nPublicerat på webb.\n"
        "\nText two.\n",
        encoding="utf-8",
    )

    retriever.create_corpus.main(str(tmp_path))

    assert "WARNING" in caplog.text
    assert "Different articles share the file" in caplog.text
    saved = [f for f in (tmp_path / "output").iterdir() if f.suffix == ".txt"]
    assert len(saved) == 1
    assert "Text two." in saved[0].read_text(encoding="utf-8")


class _ReversedListdir:
    """Stands in for the os module and hands back files in reverse order."""

    def __getattr__(self, name: str) -> Any:
        return getattr(os, name)

    def listdir(self, path: str) -> list[str]:
        return list(reversed(os.listdir(path)))


@pytest.fixture(name="export_dir")
def fixture_export_dir(tmp_path: Path) -> Path:
    """Two exports with three articles, so file order decides the ids."""
    toc = "Innehållsförteckning:\n\n{entries}\n\n\n"
    entry = "> Article {n}, Source One, 2020-12-0{day} 0{hour}:00\n"
    article = "Article {n}\nSource One, 2020-12-0{day} 0{hour}:00\nPublicerat på webb.\n\nText {n}.\n"
    two = (
        toc.format(entries=entry.format(n="One", day=4, hour=8) + entry.format(n="Two", day=4, hour=9))
        + article.format(n="One", day=4, hour=8)
        + "=" * 78
        + "\n\n"
        + article.format(n="Two", day=4, hour=9)
    )
    one = toc.format(entries=entry.format(n="Two", day=5, hour=9)) + article.format(n="Two", day=5, hour=9)
    (tmp_path / "a_export.txt").write_text(two, encoding="utf-8")
    (tmp_path / "b_export.txt").write_text(one, encoding="utf-8")
    return tmp_path


def test_main_writes_articles_and_index(export_dir: Path) -> None:
    retriever.create_corpus.main(str(export_dir))

    output = export_dir / "output"
    saved = sorted(f.name for f in output.iterdir() if f.suffix == ".txt")
    assert saved == [
        "source_one_article_one_202012040800_webb.txt",
        "source_one_article_two_202012040900_webb.txt",
        "source_one_article_two_202012050900_webb.txt",
    ]
    assert (output / saved[0]).read_text(encoding="utf-8") == "Article One\n\nText One."

    index = pd.read_csv(output / "document_index.csv", sep=";", dtype={"year": str})  # year is a label in the file
    assert list(index.columns) == [
        "title",
        "source",
        "date",
        "media",
        "pages",
        "url",
        "date_time",
        "document_name",
        "filename",
        "year",
        "input_file",
        "document_id",
    ]
    assert list(index.document_id) == [0, 1, 2]
    assert list(index.input_file) == ["a_export.txt", "a_export.txt", "b_export.txt"]
    assert list(index.year.unique()) == ["2020"]
    assert not index.date_time.isna().any()
    assert (output / "duplicates.csv").read_text(encoding="utf-8-sig").strip() == "document_name;source;date;media;urls;count"


def test_main_document_id_does_not_follow_filesystem_order(export_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    retriever.create_corpus.main(str(export_dir))
    before = pd.read_csv(export_dir / "output/document_index.csv", sep=";")

    shutil.rmtree(export_dir / "output")
    monkeypatch.setattr(retriever.create_corpus, "os", _ReversedListdir())
    retriever.create_corpus.main(str(export_dir))
    after = pd.read_csv(export_dir / "output/document_index.csv", sep=";")

    assert dict(zip(after.document_name, after.document_id)) == dict(zip(before.document_name, before.document_id))

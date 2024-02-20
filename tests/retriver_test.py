import pandas as pd
import pytest
from pandas import Timestamp
from retriever.retriever import create_corpus, fix_header, get_articles, get_toc, log_diffs


@pytest.fixture(name='input_file')
def fixture_input_file(tmp_path):
    file = tmp_path / 'input.txt'
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


@pytest.fixture(name='expected_corpus')
def fixture_corpus():
    return pd.DataFrame(
        {
            'title': {0: 'Title One', 1: 'Title Two'},
            'source': {0: 'Source One', 1: 'Source Two'},
            'date': {0: '2020-12-04 05:26', 1: '2023-12-04 04:32'},
            'toc_line_number': {0: 6, 1: 8},
            'full_text': {
                0: 'Title One\nSource One, 2020-12-04 05:26\nPublicerat på webb.\n\nText one.',
                1: 'Title Two\nSource Two, 2020-12-04 04:32\nPublicerat i print.\n\nText two.',
            },
            'header_lenght': {0: 3, 1: 3},
            'header': {
                0: 'Title One\nSource One, 2020-12-04 05:26\nPublicerat på webb.',
                1: 'Title Two\nSource Two, 2020-12-04 04:32\nPublicerat i print.',
            },
            'media': {0: 'webb', 1: 'print'},
            'pages': {0: None, 1: None},
            'url': {0: None, 1: None},
            'article_text': {0: 'Text one.', 1: 'Text two.'},
            'date_time': {0: Timestamp('2020-12-04 05:26:00'), 1: Timestamp('2023-12-04 04:32:00')},
        }
    )


def test_get_toc(input_file):
    toc, offset = get_toc(input_file, 'Innehållsförteckning:', 2)

    expected_toc = [
        ['Title One', 'Source One', '2020-12-04 05:26', 6],
        ['Title Two', 'Source Two', '2023-12-04 04:32', 8],
    ]
    expected_offset = 11

    assert toc == expected_toc
    assert offset == expected_offset


def test_get_articles(input_file):
    _, offset = get_toc(input_file, 'Innehållsförteckning:', 2)
    articles = get_articles(input_file, offset)

    expected_articles = [
        'Title One\nSource One, 2020-12-04 05:26\nPublicerat på webb.\n\nText one.',
        'Title Two\nSource Two, 2020-12-04 04:32\nPublicerat i print.\n\nText two.',
    ]

    assert articles == expected_articles


def test_create_corpus(input_file, expected_corpus):
    toc, offset = get_toc(input_file, 'Innehållsförteckning:', 2)
    articles = get_articles(input_file, offset)
    corpus = create_corpus(toc, articles)

    assert corpus.equals(expected_corpus)


@pytest.fixture(name='duplicates')
def fixture_duplicates():
    return pd.DataFrame(
        {
            'title': {0: 'Title', 1: 'Title'},
            'source': {0: 'Source', 1: 'Source'},
            'date': {0: '2020-12-04', 1: '2020-12-04'},
            'toc_line_number': {0: 6, 1: 8},
            'full_text': {
                0: 'Title\nSource, 2020-12-04 05:26\nPublicerat på webb.\n\nText.',
                1: 'Title\nSource, 2020-12-04 05:26\nPublicerat på webb.\n\nDifferent text.',
            },
            'header_lenght': {0: 3, 1: 3},
            'header': {
                0: 'Title\nSource, 2020-12-04 05:26\nPublicerat på webb.',
                1: 'Title\nSource, 2020-12-04 05:26\nPublicerat på webb.',
            },
            'media': {0: 'webb', 1: 'webb'},
            'pages': {0: None, 1: None},
            'url': {0: None, 1: None},
            'article_text': {0: 'Text.', 1: 'nDifferent text.'},
            'date_time': {0: Timestamp('2020-12-04 05:26:00'), 1: Timestamp('2020-12-04 05:26:00')},
            'document_name': {0: 'Document_20201204_webb', 1: 'Document_20201204_webb'},
            'filename': {0: 'Document_20201204_webb.txt', 1: 'Document_20201204_webb.txt'},
            'year': {0: '2020', 1: '2020'},
            'input_file': {0: 'input.txt', 1: 'input.txt'},
            'id': {0: '001001', 1: '001002'},
            'document_id': {0: '001', 1: '002'},
        }
    )


def test_log_diffs(caplog, duplicates, tmp_path):
    log_diffs(duplicates, output_folder=tmp_path, save_diffs=False)

    assert caplog.text

    expected_output = (
        "Differences for ('Document_20201204_webb', 'Source', '2020-12-04', 'webb'):\n- Text.\n+ nDifferent text."
    )

    assert 'INFO' in caplog.text
    assert caplog.text.count('INFO') == 1
    assert expected_output in caplog.text


def test_log_diffs_save_diffs(duplicates, tmp_path):
    log_diffs(duplicates, output_folder=tmp_path, save_diffs=True)

    expected_output = {
        'filename': 'Document_20201204_webb_Source_2020-12-04_webb_0.diff',
        'content': '- Text.\n+ nDifferent text.',
    }

    with open(tmp_path / 'diff' / expected_output['filename'], 'r', encoding='utf-8') as file:
        assert file.read() == expected_output['content']


@pytest.fixture(name='article_text')
def fixture_article_text():
    text = """Title

Source, 2020-12-04 05:26
Publicerat på webb.

Text."""
    return text


def test_fix_header_for_article_with_empty_line_after_title(article_text):
    expected_output = """Title
Source, 2020-12-04 05:26
Publicerat på webb.

Text."""

    assert fix_header(article_text)[0] == expected_output


@pytest.fixture(name='article_text_multiline')
def fixture_article_text_multiline():
    text = """Title line one
Title line two
Title line three

Source, 2020-12-04 05:26
Sida 23
Publicerat på webb.

Paragraph.

Another paragraph."""
    return text


def test_fix_header_for_article_with_multiline_title(article_text_multiline):
    expected_output = """Title line one
Title line two
Title line three
Source, 2020-12-04 05:26
Sida 23
Publicerat på webb.

Paragraph.

Another paragraph."""

    assert fix_header(article_text_multiline)[0] == expected_output

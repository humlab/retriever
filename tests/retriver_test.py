import pandas as pd
import pytest
from pandas import Timestamp
from retriever.retriever import create_corpus, get_articles, get_toc


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

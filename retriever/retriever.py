# pylint: disable=redefined-outer-name
import difflib
import os
import re
import sys
from typing import Any

import pandas as pd
import roman
from loguru import logger


def get_toc(filename: str, toc_name: str, max_not_matched_lines: int) -> tuple[list[list[int | str | Any]], int]:
    """Extracts the table of contents from a Retriever export file.

    Args:
        filename (str): Input file.
        toc_name (str): Name of the table of contents.
        max_not_matched_lines (int): Number of lines not starting with '>' to read before breaking.

    Returns:
        tuple[list[list[int | str | Any]], int]: Table of contents and offset.
    """
    data = []
    with open(filename, "r", encoding="utf-8") as file:
        lines = file.readlines()

    # Set a flag to know when we've reached the relevant part of the file
    flag = False

    # Counter for lines not starting with '>'
    not_matched_lines = 0

    toc_line_number = 0
    for toc_line_number, line in enumerate(lines, start=1):
        # If the line contains `toc_name`, set the flag to True
        if toc_name in line:
            flag = True

        if not flag:
            continue

        if line.startswith(">"):
            match = re.match(r">\s*(.*),\s*(.*),\s*(.*)", line)
            if match:
                title, source, date_str = match.groups()
                date = date_str
                data.append([title, source, date, toc_line_number])
                # Reset not matched lines counter
                not_matched_lines = 0
        else:
            not_matched_lines += 1
        # Break if more than max_not_matched_lines does not start with >
        if not_matched_lines > max_not_matched_lines:
            break

    logger.debug(f"Found {len(data)} articles in {filename}")

    return data, toc_line_number  # "title", "source", "date", "toc_line_number"


def get_articles(filename: str, offset: int) -> list[str]:
    """Extracts the articles from a Retriever export file.

    Args:
        filename (str): Input file.
        offset (int): The line number to start reading from.

    Returns:
        list[str]: List of articles.
    """
    with open(filename, "r", encoding="utf-8") as file:
        data = "".join(file.readlines()[offset - 1 :])
    articles = data.split("==============================================================================")
    articles = [article.strip() for article in articles]

    return articles


def create_corpus(
    toc: list[list[int | str | Any]],
    articles: list[str],
    stop_words: str | None = None,
    remove_captions: bool = True,
    remove_copyright: bool = True,
) -> pd.DataFrame:
    """Create a corpus from the table of contents and articles.

    Args:
        toc (list[list[int  |  str  |  Any]]): Table of contents.
        articles (list[str]): List of articles.
        stop_words (str, optional): Stop words. A string with stop words separated by '|'. Defaults to None.
        remove_captions (bool, optional): Remove captions from article. Defaults to True.
        remove_copyright (bool, optional): Remove copyright string from article. Defaults to True.

    Returns:
        pd.DataFrame: Corpus.
    """
    for i, article in enumerate(articles):
        logger.debug(f"Processing article {i}: '{toc[i][0]}', {toc[i][1]}, {toc[i][2]}")

        toc[i].append(article)  # append full text to toc entry

        article, header_lenght = fix_header(article)

        toc[i].append(header_lenght)

        headers = extract_headers(toc, i, article)

        check_title(toc, i, article)

        extract_media(toc, i, headers)

        extract_pages(toc, i, headers)

        extract_url(toc, i, article)

        # Remove url from article
        article = re.sub("(?:Läs hela artikeln på|Se webartikeln på) .*", "", article).strip()

        article = remove_stop_words_from_article(article, stop_words) if stop_words else article

        article = remove_captions_from_article(article) if remove_captions else article

        article = remove_copyright_string(article) if remove_copyright else article

        assert "Retriever" not in article

        logger.debug(f"Removing header from article {i}")
        article = article[article.find("\n\n") :].strip()

        toc[i].append(article)

    # articles to dataframe
    columns: list[str] = [
        "title",
        "source",
        "date",
        "toc_line_number",
        "full_text",
        "header_lenght",
        "header",
        "media",
        "pages",
        "url",
        "article_text",
    ]
    corpus = pd.DataFrame(toc, columns=columns)
    corpus["date_time"] = pd.to_datetime(corpus["date"], format="mixed")

    # Check for missing values
    if len(empty := corpus[corpus.drop(columns=["pages", "media"]).isnull().any(axis=1)]):
        logger.info(f"Missing values in df:\n{empty}")
    return corpus


def remove_copyright_string(article: str) -> str:
    """Remove copyright string from article. The copyright string is assumed to be in the last line. If not, a warning is logged.

    Args:
        article (str): Article text.

    Returns:
        str: The article text without the copyright string.
    """
    logger.debug("Removing copyright string")

    if "©" not in article.split("\n")[-1:][0]:
        logger.warning("Copyright string not in last line")

    article = re.sub(r"©.*$", "", article).strip()

    return article


def remove_captions_from_article(article: str) -> str:
    """Remove captions from article.

    Args:
        article (str): Article text.

    Returns:
        str: Article text without captions.
    """
    logger.debug("Removing captions like 'Bild: [Name [Name]]'")
    article = re.sub(r"Bild: ([A-Z][a-z]+(?: [A-Z][a-z]+)*)(/TT)?\s?", "", article).strip()
    return article


def remove_stop_words_from_article(article: str, stop_words: str | None) -> str:
    """Remove stop words from article.

    Args:
        article (str): Article text.
        stop_words (str): Stop words. A string with stop words separated by '|'.

    Returns:
        str: Article text without stop words.
    """
    if stop_words:
        logger.debug(f"Removing stop words: '{', '.join(stop_words.split('|'))}'")
        article = re.sub(rf"({stop_words}):\s?", "", article).strip()
    return article


def check_title(toc: list[list[int | str | Any]], i: int, article: str) -> None:
    """Test that the title in the toc matches the title in the article.

    Args:
        toc (list[list[int  |  str  |  Any]]): Table of contents.
        i (int): Article index.
        article (str): Article text.
    """
    article_title = re.sub(r"\W+", " ", str(article.split("\n", maxsplit=1)[0].strip())).lower().strip()
    toc_title = re.sub(r"\W+", " ", str(toc[i][0])).lower().strip()
    assert toc_title.startswith(article_title[:25])


def extract_headers(toc: list[list[int | str | Any]], i: int, article: str) -> list[str]:
    """Extract headers from article.

    Args:
        toc (list[list[int  |  str  |  Any]]): Table of contents.
        i (int): Article index.
        article (str): Article text.

    Returns:
        list[str]: Headers.
    """
    headers = str(article).split("\n\n", maxsplit=1)[0].strip().split("\n")
    toc[i].append("\n".join(headers))  # append header to toc entry

    # Log ERROR if headers length is less than 3
    if len(headers) < 3:
        logger.error(f"Headers length is less than 3 in article {i}: '{toc[i][0]}', {toc[i][1]}, {toc[i][0]}")
    return headers


def extract_url(toc: list[list[int | str | Any]], i: int, article: str) -> None:
    """Extract url from article.

    Args:
        toc (list[list[int  |  str  |  Any]]): Table of contents.
        i (int): Article index.
        article (str): Article text.
    """
    m = re.search("(?:Läs hela artikeln på|Se webartikeln på) (.*)", article)
    url = m.groups()[0] if m else None
    toc[i].append(url)
    logger.debug(f"Extracted url '{url}' from article {i}")


def extract_pages(toc: list[list[int | str | Any]], i: int, headers: list[str]) -> None:
    """Extract page numbers from headers.

    Args:
        toc (list[list[int  |  str  |  Any]]): Table of contents.
        i (int): Article index.
        headers (list[str]): Headers.
    """
    pages = headers[-2].split(" ")[1] if len(headers) >= 3 and headers[-2].startswith("Sida") else None
    toc[i].append(pages)
    logger.debug(f"Extracted pages '{pages}' from article {i}")


def extract_media(toc: list[list[int | str | Any]], i: int, headers: list[str]) -> None:
    """Extract media type from headers.

    Args:
        toc (list[list[int  |  str  |  Any]]): Table of contents.
        i (int): Article index.
        headers (list[str]): Headers.
    """
    media = headers[-1] if headers[-1].startswith("Publicerat") else None
    if media:
        media = "webb" if "webb" in media else "print"
    toc[i].append(media)
    logger.debug(f"Extracted media '{media}' from article {i}")


def fix_header(article: str) -> tuple[str, int]:
    """Fix header. Remove empty lines from article header.

    Args:
        article (str): Article text.

    Returns:
        tuple[str, int]: Article text and header length.
    """
    lines = article.split('\n')
    output_lines = []

    found_publicerat = False

    for line in lines:
        if line.strip().startswith('Publicerat'):
            found_publicerat = True

        if found_publicerat or line.strip() != '':
            output_lines.append(line)

    article = '\n'.join(output_lines)
    header_lenght = len(str(article).split("\n\n", maxsplit=1)[0].strip().split("\n"))

    return article, header_lenght


def log_diffs(duplicates: pd.DataFrame, output_folder: str, save_diffs: bool = True) -> None:
    """Log differences between duplicates.

    Args:
        duplicates (pd.DataFrame): DataFrame with duplicates.
        output_folder (str): Output folder.
        save_diffs (bool, optional): Save differences to file. Defaults to True.
    """
    diff_folder = f"{output_folder}/diff"
    if save_diffs and not os.path.exists(diff_folder):
        os.makedirs(diff_folder)

    # Identify duplicates
    diff_articles = duplicates[
        ~duplicates.duplicated(subset=['document_name', 'source', 'date', 'media', 'article_text'], keep=False)
    ]

    # Group the diff_articles
    grouped = diff_articles.groupby(['document_name', 'source', 'date', 'media'])

    # For each group of diff_articles, compare the 'article_text'
    for name, group in grouped:
        texts = group['article_text'].tolist()
        for i in range(1, len(texts)):
            diff = difflib.ndiff(texts[i - 1].splitlines(), texts[i].splitlines())
            diff_text = '\n'.join(diff)
            logger.info(f"Differences for {name}:\n{diff_text}")
            # save diff_text to file
            if save_diffs:
                with open(
                    f"{diff_folder}/{name[0]}_{name[1]}_{name[2]}_{name[3]}_{i-1}.diff", "w", encoding="utf-8"
                ) as f:
                    f.write(diff_text)


def main(
    input_folder: str,
    save_short_headers: bool = False,
    stop_words: str | None = "Bildtext|Image-text|Pressbild|Snabbversion",
    remove_captions: bool = True,
    remove_copyright: bool = True,
) -> None:
    """Main function.

    Args:
        input_folder (str): Input folder.
        save_short_headers (bool, optional): Save files with short headers. Defaults to False.
    """
    output_folder: str = f"{input_folder}/output"
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    logger.add(f"{output_folder}/run.log", level="WARNING", encoding="utf8")

    all_metadata = []  # List to store all metadata

    article_counts = {}
    for filename in os.listdir(input_folder):
        if not filename.endswith(".txt"):
            continue
        filepath: str = f"{input_folder}/{filename}"
        toc, offset = get_toc(filepath, "Innehållsförteckning:", 2)
        articles: list[str] = get_articles(filepath, offset)

        df: pd.DataFrame = create_corpus(toc, articles, stop_words, remove_captions, remove_copyright)

        df['document_name'] = df.source.fillna('').str.replace(r'\W+', '_', regex=True).str.lower().str.strip()
        df['document_name'] = (
            df.document_name
            + '_'
            + df.title.fillna('').str.replace(r'\W+', '_', regex=True).str.lower().str.strip().str.strip('_').str[:60]
        )
        df['document_name'] = (
            df.document_name
            + '_'
            + df.date.fillna('').str.replace(' ', '').str.replace(':', '').str.replace('-', '')
            + '_'
            + df.media.fillna('')
        )
        df['filename'] = df.document_name + ".txt"
        df['year'] = df.date.fillna(0).str[:4].astype(int)
        df["input_file"] = filename

        df['id'] = df.input_file.str.rsplit('_').str[-1].str.replace('.txt', '').apply(roman.fromRoman).astype(
            int
        ).astype(str).str.zfill(3) + df.index.astype(str).str.zfill(3)

        article_counts[filename] = len(df)
        all_metadata.append(df)

    logger.info(f'Found {sum(article_counts.values())} articles')

    # Combine metadata
    document_index: pd.DataFrame = pd.concat(all_metadata, ignore_index=True)
    document_index.reset_index(drop=True, inplace=True)
    document_index['document_id'] = document_index.index

    # Check for duplicates
    duplicates: pd.DataFrame = document_index[
        document_index.duplicated(subset=['document_name', 'source', 'date', 'media'], keep=False)
    ]
    logger.info(f"Found {len(duplicates)} non-unique articles")

    # Log differences between duplicates
    log_diffs(duplicates, output_folder, save_diffs=False)

    # Save duplicates to csv
    duplicates = (
        duplicates.groupby(['document_name', 'source', 'date', 'media'])
        .agg(urls=('url', lambda x: ', '.join(x)), count=('url', 'count'))  # pylint: disable=unnecessary-lambda
        .reset_index()
    )
    duplicates.to_csv(f"{output_folder}/duplicates.csv", index=False, sep=";", encoding="utf-8-sig")

    # Remove duplicates. Keep the last article.
    document_index.drop_duplicates(subset=['document_name', 'source', 'date', 'media'], keep='last', inplace=True)
    logger.info(f"Removed {len(duplicates)} duplicates")
    logger.info(f'Saving {len(document_index)} unique articles')

    # Save articles to txt files
    for _, metadata in document_index.iterrows():
        with open(f"{output_folder}/{metadata['filename']}", "w", encoding="utf-8") as f:
            f.write(metadata['title'] + "\n\n")
            f.write(metadata['article_text'])
    logger.success(
        f'Saved {len([f for f in os.listdir(output_folder) if f.endswith(".txt")])} articles to {output_folder}'
    )

    # save articles with header_lenght < 3 to txt files in subfolder "short_headers"
    if save_short_headers:
        short_headers_folder = f"{output_folder}/short_headers"
        if not os.path.exists(short_headers_folder):
            os.makedirs(short_headers_folder)
        short_headers = document_index[document_index.header_lenght < 3]
        for _, metadata in short_headers.iterrows():
            with open(f"{short_headers_folder}/{metadata['filename']}", "w", encoding="utf-8") as f:
                f.write(metadata['title'] + "\n\n")
                f.write(metadata['article_text'])
        logger.success(
            f'Saved {len([f for f in os.listdir(short_headers_folder) if f.endswith(".txt")])} articles with short headers to {short_headers_folder}'
        )

    # Save document_index to csv
    document_index = document_index.drop(
        columns=["article_text", "full_text", "header", "toc_line_number", "header_lenght"]
    )
    document_index.to_csv(f"{output_folder}/document_index.csv", index=False, sep=";", encoding="utf-8-sig")
    logger.success(f'Saved document_index to {output_folder}/document_index.csv')


if __name__ == "__main__":
    # typer.run(main)
    logger.remove()
    logger.add(sys.stderr, level='INFO')
    main(
        "./input",
        save_short_headers=False,
        stop_words="Bildtext|Image-text|Pressbild|Snabbversion",
        remove_captions=True,
        remove_copyright=True,
    )

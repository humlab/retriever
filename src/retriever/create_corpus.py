# pylint: disable=redefined-outer-name
import difflib
import os
import re
from typing import Any

import pandas as pd
import typer
from loguru import logger

MIN_HEADER_LENGTH = 3


def get_toc(filename: str, toc_name: str, max_not_matched_lines: int) -> tuple[list[list[int | str | Any]], int]:
    """Extracts the table of contents from a Retriever export file.

    Args:
        filename (str): Input file.
        toc_name (str): Name of the table of contents.
        max_not_matched_lines (int): Number of lines not starting with '>' to read before breaking.

    Returns:
        tuple[list[list[int | str | Any]], int]: Table of contents and offset.
    """
    data: list[list[int | str | Any]] = []
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
                logger.warning(f"Unparsed toc line {toc_line_number}: {line.strip()!r}")
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

    if articles and not articles[-1]:
        articles.pop()  # an export may end with the separator line, which is not an empty article

    return articles


def create_corpus(
    toc: list[list[int | str | Any]],
    articles: list[str],
    stop_words: str | None = None,
    remove_captions: bool = False,
    remove_copyright: bool = False,
) -> pd.DataFrame:
    """Create a corpus from the table of contents and articles.

    Args:
        toc (list[list[int  |  str  |  Any]]): Table of contents.
        articles (list[str]): List of articles.
        stop_words (str, optional): Stop words. A string with stop words separated by '|'. Defaults to None.
        remove_captions (bool, optional): Remove captions from article. Defaults to False.
        remove_copyright (bool, optional): Remove copyright string from article. Defaults to False.

    Returns:
        pd.DataFrame: Corpus.
    """
    if len(toc) != len(articles):
        raise ValueError(f"{len(toc)} table of contents entries but {len(articles)} articles — the export does not line up")

    rows: list[dict[str, object]] = []
    for i, article in enumerate(articles):
        title, source, date = toc[i][0], toc[i][1], toc[i][2]
        logger.debug(f"Processing article {i}: '{title}', {source}, {date}")

        cleaned_article, header_lenght = fix_header(article)
        headers = extract_headers(cleaned_article)
        check_title(title, cleaned_article)

        # the keys here are the corpus columns, in the order they are written
        row: dict[str, object] = {
            "title": title,
            "source": source,
            "date": date,
            "toc_line_number": toc[i][3],
            "full_text": article,
            "header_lenght": header_lenght,
            "header": "\n".join(headers),
            "media": extract_media(headers),
            "pages": extract_pages(headers),
            "url": extract_url(cleaned_article),
        }

        # Remove url from article
        cleaned_article = re.sub("(?:Läs hela artikeln på|Se webartikeln på) .*", "", cleaned_article).strip()

        if stop_words:
            cleaned_article = remove_stop_words_from_article(cleaned_article, stop_words)

        cleaned_article = remove_captions_from_article(cleaned_article) if remove_captions else cleaned_article

        cleaned_article = remove_copyright_string(cleaned_article) if remove_copyright else cleaned_article

        if remove_copyright and "Retriever" in cleaned_article:
            logger.warning(f"'Retriever' not removed from article {i}")

        logger.debug(f"Removing header from article {i}")

        cleaned_article = cleaned_article.partition("\n\n")[2].strip()
        if not cleaned_article:
            logger.warning(f"Empty article body: '{title}' ({source}, {date})")

        row["article_text"] = cleaned_article
        rows.append(row)

    corpus = pd.DataFrame(rows)
    corpus["date_time"] = pd.to_datetime(corpus["date"], format="mixed", errors="coerce")

    # Check for missing values
    if len(empty := corpus[corpus.drop(columns=["pages", "media"]).isnull().any(axis=1)]):
        logger.info(f"Missing values in df:\n{empty}")
    return corpus


def remove_copyright_string(article: str) -> str:
    """Remove the copyright string from the end of the article text.

    Args:
        article (str): Article text.

    Returns:
        str: The article text without the copyright string.
    """
    logger.debug("Removing copyright string")

    if "©" not in article:
        logger.warning("No copyright string in article")
    elif "©" not in article.rsplit("\n", maxsplit=1)[-1]:
        logger.warning("Copyright string not in last line")

    # the export glues the copyright text to the last line of the article, behind its " |" footer separator
    article = re.sub(r"(?:\s*\|)*\s*©.*$", "", article).strip()

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


def remove_stop_words_from_article(article: str, stop_words: str) -> str:
    """Remove "Stop word: " prefixes anywhere in the article text.

    Args:
        article (str): Article text.
        stop_words (str): Stop words. A string with stop words separated by '|'.

    Returns:
        str: Article text without the stop words.
    """
    logger.debug(f"Removing stop words: '{', '.join(stop_words.split('|'))}'")
    return re.sub(rf"({stop_words}):\s?", "", article).strip()


def check_title(title: object, article: str) -> None:
    """Log a warning if the title in the toc does not match the first line of the article.

    Args:
        title (object): Title from the table of contents.
        article (str): Article text.
    """
    first_line = article.split("\n", maxsplit=1)[0].strip()
    article_title = re.sub(r"\W+", " ", first_line).lower().strip()
    toc_title = re.sub(r"\W+", " ", str(title)).lower().strip()
    if not toc_title.startswith(article_title[:25]):
        logger.warning(f"Title mismatch: toc '{title}' but article starts with '{first_line}'")


def extract_headers(article: str) -> list[str]:
    """Extract the header lines of an article.

    Args:
        article (str): Article text.

    Returns:
        list[str]: Headers.
    """
    headers = article.split("\n\n", maxsplit=1)[0].strip().split("\n")

    # Log ERROR if headers length is less than MIN_HEADER_LENGTH
    if len(headers) < MIN_HEADER_LENGTH:
        logger.error(f"Headers length is less than {MIN_HEADER_LENGTH} in article starting with '{headers[0]}'")
    return headers


def extract_url(article: str) -> str | None:
    """Extract the url of an article.

    Args:
        article (str): Article text.

    Returns:
        str | None: The url, or None if the article has none.
    """
    match = re.search("(?:Läs hela artikeln på|Se webartikeln på) (.*)", article)
    return match.group(1) if match else None


def extract_pages(headers: list[str]) -> str | None:
    """Extract the page number from the headers.

    Args:
        headers (list[str]): Headers.

    Returns:
        str | None: The page number, or None if the article has none.
    """
    if len(headers) < MIN_HEADER_LENGTH or not headers[-2].startswith("Sida"):
        return None
    parts = headers[-2].split(" ")
    return parts[1] if len(parts) > 1 else None


def extract_media(headers: list[str]) -> str | None:
    """Tell if the article is from the web or in print, using the last header line.

    Args:
        headers (list[str]): Headers.

    Returns:
        str | None: 'webb', 'print', or None if the headers do not say.
    """
    if not headers[-1].startswith("Publicerat"):
        return None
    return "webb" if "webb" in headers[-1] else "print"


def fix_header(article: str) -> tuple[str, int]:
    """Fix header. Remove empty lines from article header.

    Args:
        article (str): Article text.

    Returns:
        tuple[str, int]: Article text and header length.
    """
    lines = article.split("\n")
    output_lines: list[str] = []

    found_publicerat = False

    for line in lines:
        if line.strip().startswith("Publicerat"):
            found_publicerat = True

        if found_publicerat or line.strip() != "":
            output_lines.append(line)

    article = "\n".join(output_lines)
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
    if save_diffs:
        os.makedirs(diff_folder, exist_ok=True)

    # Identify duplicates
    diff_articles = duplicates[~duplicates.duplicated(subset=["document_name", "source", "date", "media", "article_text"], keep=False)]

    # Group the diff_articles
    grouped = diff_articles.groupby(["document_name", "source", "date", "media"])

    # For each group of diff_articles, compare the 'article_text'
    for name, group in grouped:
        texts = group["article_text"].tolist()
        for i in range(1, len(texts)):
            diff = difflib.ndiff(texts[i - 1].splitlines(), texts[i].splitlines())
            diff_text = "\n".join(diff)
            logger.info(f"Differences for {name}:\n{diff_text}")
            # save diff_text to file
            if save_diffs:
                with open(f"{diff_folder}/{name[0]}_{name[1]}_{name[2]}_{name[3]}_{i - 1}.diff", "w", encoding="utf-8") as f:
                    f.write(diff_text)


def main(
    input_folder: str,
    save_short_headers: bool = False,
    stop_words: str | None = None,
    remove_captions: bool = False,
    remove_copyright: bool = True,
) -> None:
    """Main function.

    Args:
        input_folder (str): Input folder.
        save_short_headers (bool, optional): Save files with short headers. Defaults to False.
        stop_words (str, optional): Stop words. A string with stop words separated by '|'. Defaults to None.
        remove_captions (bool, optional): Remove captions from article. Defaults to False.
        remove_copyright (bool, optional): Remove copyright string from article. Defaults to False.
    """
    output_folder: str = f"{input_folder}/output"
    os.makedirs(output_folder, exist_ok=True)
    logger.add(f"{output_folder}/run.log", level="WARNING", encoding="utf8")

    all_metadata: list[pd.DataFrame] = []  # List to store all metadata

    article_counts: dict[str, int] = {}

    # sorted: the article order here fixes document_id and which copy of a duplicate survives
    for filename in sorted(os.listdir(input_folder)):
        if not filename.endswith(".txt"):
            continue
        filepath: str = f"{input_folder}/{filename}"
        toc, offset = get_toc(filepath, "Innehållsförteckning:", 2)
        articles: list[str] = get_articles(filepath, offset)

        df: pd.DataFrame = create_corpus(toc, articles, stop_words, remove_captions, remove_copyright)

        df["document_name"] = df.source.fillna("").str.replace(r"\W+", "_", regex=True).str.lower().str.strip()
        df["document_name"] = (
            df.document_name
            + "_"
            + df.title.fillna("").str.replace(r"\W+", "_", regex=True).str.lower().str.strip().str.strip("_").str[:60]
        )
        df["document_name"] = (
            df.document_name
            + "_"
            + df.date.fillna("").str.replace(" ", "").str.replace(":", "").str.replace("-", "")
            + "_"
            + df.media.fillna("")
        )
        df["filename"] = df.document_name + ".txt"
        df["year"] = df["date"].str[:4]
        df["input_file"] = filename

        article_counts[filename] = len(df)
        all_metadata.append(df)

    logger.info(f"Found {sum(article_counts.values())} articles")

    # Combine metadata
    document_index: pd.DataFrame = pd.concat(all_metadata, ignore_index=True)
    document_index["document_id"] = document_index.index

    # Check for duplicates
    duplicates: pd.DataFrame = document_index[document_index.duplicated(subset=["document_name", "source", "date", "media"], keep=False)]
    logger.info(f"Found {len(duplicates)} non-unique articles")

    # Log differences between duplicates
    log_diffs(duplicates, output_folder, save_diffs=False)

    # Save duplicates to csv
    def join_urls(urls: Any) -> str:
        return ", ".join(url for url in urls if url)  # url is None when the export has no url, so it is skipped

    duplicates = (
        duplicates.groupby(["document_name", "source", "date", "media"]).agg(urls=("url", join_urls), count=("url", "size")).reset_index()
    )
    duplicates.to_csv(f"{output_folder}/duplicates.csv", index=False, sep=";", encoding="utf-8-sig")

    # Punctuation is stripped from document_name, so two different articles can collide on one file: the lost text is not in duplicates.csv
    texts_per_name = document_index.groupby("document_name")["article_text"].nunique()
    for name in texts_per_name[texts_per_name > 1].index:
        logger.warning(f"Different articles share the file '{name}.txt'; only the last one was saved")

    # Remove duplicates. Keep the last article.
    document_index.drop_duplicates(subset=["document_name", "source", "date", "media"], keep="last", inplace=True)
    logger.info(f"Removed {duplicates['count'].sum() - len(duplicates)} duplicates")
    logger.info(f"Saving {len(document_index)} unique articles")

    # Save articles to txt files
    for _, metadata in document_index.iterrows():
        with open(f"{output_folder}/{metadata['filename']}", "w", encoding="utf-8") as f:
            f.write(metadata["title"] + "\n\n")
            f.write(metadata["article_text"])
    logger.success(f"Saved {len([f for f in os.listdir(output_folder) if f.endswith('.txt')])} articles to {output_folder}")

    # save articles with header_lenght < MIN_HEADER_LENGTH to txt files in subfolder "short_headers"
    if save_short_headers:
        short_headers_folder: str = f"{output_folder}/short_headers"
        os.makedirs(short_headers_folder, exist_ok=True)
        short_headers = document_index[document_index.header_lenght < MIN_HEADER_LENGTH]
        for _, metadata in short_headers.iterrows():
            with open(f"{short_headers_folder}/{metadata['filename']}", "w", encoding="utf-8") as f:
                f.write(metadata["title"] + "\n\n")
                f.write(metadata["article_text"])
        logger.success(
            f"Saved {len([f for f in os.listdir(short_headers_folder) if f.endswith('.txt')])} articles "
            f"with short headers to {short_headers_folder}"
        )

    # Save document_index to csv
    document_index = document_index.drop(columns=["article_text", "full_text", "header", "toc_line_number", "header_lenght"])
    document_index.to_csv(f"{output_folder}/document_index.csv", index=False, sep=";", encoding="utf-8-sig")
    logger.success(f"Saved document_index to {output_folder}/document_index.csv")


def cli() -> None:
    """Command line entry point, see main() for the arguments."""
    typer.run(main)


if __name__ == "__main__":
    cli()

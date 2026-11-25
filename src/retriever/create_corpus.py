import difflib
import os
import re
from datetime import datetime
from typing import Any

import pandas as pd
import typer
from loguru import logger

MIN_HEADER_LENGTH = 3
MAX_NOT_MATCHED_LINES = 2  # lines without a ">" in a row, after the toc, before the toc is taken to be over
MAX_TITLE_CHARS = 100  # longest name this makes is 159 characters, against 255 for a name and 260 for a path
PROCESSED_FOLDER = "data/processed"  # article text, one subfolder per run
LOGS_FOLDER = "data/logs"  # run.log and the csv files, one subfolder per run


def get_toc(filename: str, toc_name: str) -> tuple[list[list[int | str | Any]], int]:
    """Extracts the table of contents from a Retriever export file.

    Args:
        filename (str): Input file.
        toc_name (str): Name of the table of contents.

    Returns:
        tuple[list[list[int | str | Any]], int]: Table of contents and the line the articles start on,
            which is the line after the last toc entry — however many blank lines the export puts there.
    """
    data: list[list[int | str | Any]] = []
    with open(filename, "r", encoding="utf-8") as file:
        lines = file.readlines()

    # Set a flag to know when we've reached the relevant part of the file
    flag = False

    # Counter for lines not starting with '>'
    not_matched_lines = 0

    last_toc_line = 0  # the articles start below this one, wherever the toc really ends
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
                last_toc_line = toc_line_number
                # Reset not matched lines counter
                not_matched_lines = 0
            else:
                logger.warning(f"Unparsed toc line {toc_line_number}: {line.strip()!r}")
        else:
            not_matched_lines += 1
        # Break if more than MAX_NOT_MATCHED_LINES does not start with >
        if not_matched_lines > MAX_NOT_MATCHED_LINES:
            break

    if not data:
        raise ValueError(f"No table of contents entries found in {filename}")

    logger.debug(f"Found {len(data)} articles in {filename}")

    return data, last_toc_line + 1  # "title", "source", "date", "toc_line_number"


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

        cleaned_article = fix_header(article)
        headers = extract_headers(cleaned_article)
        check_title(title, cleaned_article)

        # the keys here are the corpus columns, in the order they are written
        row: dict[str, object] = {
            "title": title,
            "source": source,
            "date": date,
            "toc_line_number": toc[i][3],
            "full_text": article,
            "header": "\n".join(headers),
            "media": extract_media(headers),
            "pages": extract_pages(headers),
            "url": extract_url(cleaned_article),
        }

        # Remove url from article
        cleaned_article = re.sub("(?:Läs hela artikeln på|Se webartikeln på) .*", "", cleaned_article).strip()

        if stop_words:
            cleaned_article = remove_stop_words_from_article(cleaned_article, stop_words)

        credits = len(re.findall(CREDIT_LABEL, cleaned_article)) if remove_captions else 0
        cleaned_article = remove_captions_from_article(cleaned_article) if remove_captions else cleaned_article
        # counted on both sides of the cut, so main can say how many credits were too glued to separate
        credits_left = len(re.findall(CREDIT_LABEL, cleaned_article)) if remove_captions else 0

        cleaned_article = remove_copyright_string(cleaned_article) if remove_copyright else cleaned_article

        if remove_copyright and "Retriever" in cleaned_article:
            logger.warning(f"'Retriever' not removed from article {i}")

        logger.debug(f"Removing header from article {i}")

        cleaned_article = cleaned_article.partition("\n\n")[2].strip()
        if not cleaned_article:
            logger.warning(f"Empty article body: '{title}' ({source}, {date})")

        row["article_text"] = cleaned_article
        row["credits"] = credits
        row["credits_left"] = credits_left
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

    # line 1 is the title in an export article, so the two warnings below name the article they are about
    title = article.partition("\n")[0]
    if "©" not in article:
        logger.warning(f"No copyright string in article: {title!r}")
    elif "©" not in article.rsplit("\n", maxsplit=1)[-1]:
        logger.warning(f"Copyright string not in last line of article: {title!r}")

    # the export glues the copyright text to the last line of the article, behind its " |" footer separator
    return re.sub(r"(?:\s*\|)*\s*©.*$", "", article).strip()


CREDIT_LABEL = r"(?:Image-text:\s*)?Bild:\s*"
CREDIT_NAMES = r"(?:[A-ZÅÄÖ][^\W\d_]*(?:[ ,]+[A-ZÅÄÖ][^\W\d_]*){0,1})?"
CREDIT_AGENCIES = r"(?:\s*/+\s*[^\W\d_]+)+"
CREDIT = re.compile(CREDIT_LABEL + CREDIT_NAMES + r"(?:" + CREDIT_AGENCIES + r"|(?:\s\||$))")


def remove_captions_from_article(article: str) -> str:
    """Remove photo credits ("Bild: Name/TT") from the article text.

    Note: A photo credit is only cut where it ends at a marker the export really uses: an agency slash (/TT, /SvD),
    the " |" subject separator, or the end of the line. Anywhere else it is glued to the sentence that
    follows and those words would be cut with it, so such a credit stays put and is logged instead.
    Two words of name at most, each one capitalised the way the export writes it (Namn Efternamn, TT, Arkivbild).

    Args:
        article (str): Article text.

    Returns:
        str: Article text without the credits that can be told apart from the body.
    """
    lines: list[str] = []
    for line in article.split("\n"):
        match = CREDIT.search(line)
        if match is None and "Bild:" in line:
            logger.debug(f"Caption not recognised: {line}")
        cut = (line[: match.start()] + line[match.end() :].lstrip()).rstrip() if match else line
        lines.append(cut)
    return "\n".join(lines).strip()


def remove_stop_words_from_article(article: str, stop_words: str) -> str:
    """Remove "Stop word: " prefixes anywhere in the article text.

    Args:
        article (str): Article text.
        stop_words (str): Stop words. A string with stop words separated by '|'.

    Returns:
        str: Article text without the stop words.
    """
    logger.debug(f"Removing stop words: '{', '.join(stop_words.split('|'))}'")
    words = "|".join(re.escape(word) for word in stop_words.split("|") if word)
    return re.sub(rf"({words}):\s?", "", article).strip()


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


def fix_header(article: str) -> str:
    """Fix header. Remove empty lines from article header.

    Args:
        article (str): Article text.

    Returns:
        str: Article text without empty lines in the header.
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

    return article


def log_diffs(duplicates: pd.DataFrame, level: str = "DEBUG") -> None:
    """Log differences between duplicates.

    Args:
        duplicates (pd.DataFrame): DataFrame with duplicates.
        level (str, optional): Level to report the differences at. Defaults to "DEBUG".
    """
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
            logger.log(level, f"Differences for {name}:\n{diff_text}")


def add_document_names(corpus: pd.DataFrame, input_file: str) -> pd.DataFrame:
    """Add the columns that say where an article is saved and which copies count as one article.

    Args:
        corpus (pd.DataFrame): One export, as returned by create_corpus.
        input_file (str): Name of the export file the rows were read from.

    Returns:
        pd.DataFrame: The same corpus with document_name, filename, year and input_file added.
    """
    corpus["document_name"] = corpus.source.fillna("").str.replace(r"\W+", "_", regex=True).str.lower().str.strip()
    title_slug = corpus.title.fillna("").str.replace(r"\W+", "_", regex=True).str.lower().str.strip().str.strip("_")
    corpus["document_name"] = corpus.document_name + "_" + title_slug.str[:MAX_TITLE_CHARS]
    corpus["document_name"] = (
        corpus.document_name
        + "_"
        + corpus.date.fillna("").str.replace(" ", "").str.replace(":", "").str.replace("-", "")
        + "_"
        + corpus.media.fillna("")
    )
    corpus["filename"] = corpus.document_name + ".txt"
    for title in corpus.title[title_slug.str.len() > MAX_TITLE_CHARS]:
        logger.debug(f"Title cut to {MAX_TITLE_CHARS} characters: {title!r}")
    corpus["year"] = corpus["date"].str[:4]
    corpus["input_file"] = input_file
    return corpus


def save_articles(document_index: pd.DataFrame, folder: str) -> int:
    """Write one .txt file per article: the title, a blank line, the body.

    Args:
        document_index (pd.DataFrame): The articles to save, with their filename column.
        folder (str): Folder to write into.

    Returns:
        int: The number of .txt files in the folder afterwards.
    """
    for _, metadata in document_index.iterrows():
        with open(f"{folder}/{metadata['filename']}", "w", encoding="utf-8") as f:
            f.write(metadata["title"] + "\n\n")
            f.write(metadata["article_text"])
    return len([name for name in os.listdir(folder) if name.endswith(".txt")])


def run_summary(index: pd.DataFrame, read: int, saved: int, output_folder: str, remove_captions: bool) -> str:
    """One line about what the run did, at a level run.log keeps.

    Args:
        index (pd.DataFrame): The index while its internal columns are still there, so the credit counts can be read.
        read (int): Articles read, before duplicates were dropped.
        saved (int): Article files written.
        output_folder (str): Folder the articles were written to.
        remove_captions (bool): Whether captions were removed from the articles.

    Returns:
        str: The summary line.
    """
    credits_part = ""
    if remove_captions and (seen := int(index.credits.sum())) > 0:
        cut: int = seen - int(index.credits_left.sum())
        credits_part = f", {cut} of {seen} credits cut ({cut / seen * 100:.1f} %)"

    missing: int = int(index.drop(columns=["pages", "media"]).isnull().any(axis=1).sum())
    return f"Run summary: {read} articles read, {saved} saved to {output_folder}, {missing} with missing values{credits_part}"


def main(
    input_folder: str,
    toc_name: str = "Innehållsförteckning:",
    stop_words: str | None = None,
    remove_captions: bool = False,
    remove_copyright: bool = True,
) -> None:
    """Main function.

    Args:
        input_folder (str): Folder with the export files.
        toc_name (str, optional): The words the export writes above its table of contents. Defaults to
            "Innehållsförteckning:".
        stop_words (str, optional): Stop words. A string with stop words separated by '|'. Defaults to None.
        remove_captions (bool, optional): Remove captions from article. Defaults to False.
        remove_copyright (bool, optional): Remove copyright string from article. Defaults to True.
    """
    run: str = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    output_folder: str = f"{PROCESSED_FOLDER}/{run}"
    log_folder: str = f"{LOGS_FOLDER}/{run}"
    os.makedirs(output_folder, exist_ok=True)
    os.makedirs(log_folder, exist_ok=True)

    handler_id: int = logger.add(f"{log_folder}/run.log", level="INFO", encoding="utf8", mode="w")
    if remove_captions:
        logger.info("Removing captions should be done with care: a credit glued into the text is not easily separable.")
    try:
        all_metadata: list[pd.DataFrame] = []  # List to store all metadata

        # sorted: the article order here fixes document_id and which copy of a duplicate survives
        for filename in sorted(os.listdir(input_folder)):
            if not filename.endswith(".txt"):
                continue
            filepath: str = f"{input_folder}/{filename}"
            toc, offset = get_toc(filepath, toc_name)
            articles: list[str] = get_articles(filepath, offset)

            corpus = create_corpus(toc, articles, stop_words, remove_captions, remove_copyright)
            if remove_captions and (left := int(corpus.credits_left.sum())) > 0:
                logger.debug(f"Captions not recognised in {filename}: {left}")

            df: pd.DataFrame = add_document_names(corpus, filename)

            all_metadata.append(df)

        if not all_metadata:
            raise ValueError(f"No .txt exports in {input_folder}")

        # Combine metadata
        document_index: pd.DataFrame = pd.concat(all_metadata, ignore_index=True)
        logger.info(f"Found {len(document_index)} articles")
        document_index["document_id"] = document_index.index

        read: int = len(document_index)

        # Check for duplicates
        not_unique = document_index.duplicated(subset=["document_name", "source", "date", "media"], keep=False)
        duplicates: pd.DataFrame = document_index[not_unique]
        logger.info(f"Found {len(duplicates)} non-unique articles")

        # Log differences between duplicates
        log_diffs(duplicates)

        # Save duplicates to csv
        def join_urls(urls: Any) -> str:
            return ", ".join(url for url in urls if url)  # url is None when the export has no url, so it is skipped

        duplicates = (
            duplicates.groupby(["document_name", "source", "date", "media"])
            .agg(urls=("url", join_urls), count=("url", "size"))
            .reset_index()
        )
        duplicates.to_csv(f"{log_folder}/duplicates.csv", index=False, sep=";", encoding="utf-8-sig")

        # Punctuation is stripped from document_name, so two different articles can collide on one file;
        # duplicates.csv does not cover that
        texts_per_name = document_index.groupby("document_name")["article_text"].nunique()
        for name in texts_per_name[texts_per_name > 1].index:
            logger.warning(f"Different articles share the file '{name}.txt'; only the last one was saved")

        # Remove duplicates. Keep the last article.
        document_index = document_index.drop_duplicates(subset=["document_name", "source", "date", "media"], keep="last")
        logger.info(f"Removed {duplicates['count'].sum() - len(duplicates)} duplicates")
        logger.info(f"Saving {len(document_index)} unique articles")

        # Save articles to txt files
        saved: int = save_articles(document_index, output_folder)
        logger.success(f"Saved {saved} articles to {output_folder}")
        summary = run_summary(document_index, read, saved, output_folder, remove_captions)

        # Save document_index to csv
        document_index = document_index.drop(columns=["article_text", "full_text", "header", "toc_line_number", "credits", "credits_left"])
        document_index.to_csv(f"{log_folder}/document_index.csv", index=False, sep=";", encoding="utf-8-sig")
        logger.success(f"Saved document_index to {log_folder}/document_index.csv")
        # one line about the whole run, at the level run.log keeps
        logger.info(summary)
    finally:
        logger.remove(handler_id)


def cli() -> None:
    """Command line entry point, see main() for the arguments."""
    typer.run(main)


if __name__ == "__main__":  # pragma: no cover
    cli()

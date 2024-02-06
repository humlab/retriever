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
    with open(filename, "r", encoding="utf-8") as file:
        data = "".join(file.readlines()[offset - 1 :])
    articles = data.split("==============================================================================")
    articles = [article.strip() for article in articles]

    return articles


def create_corpus(toc: list[list[int | str | Any]], articles: list[str]) -> pd.DataFrame:
    for i, article in enumerate(articles):
        toc_entry = toc[i]
        title, source, date, _ = toc_entry
        logger.debug(f"Processing article {i}: '{title}', {source}, {date}")

        toc[i].append(article)  # append full text to toc entry

        if len(str(article).split("\n\n", maxsplit=1)[0].strip().split("\n")) < 3:
            article = article.replace("\n\n", "\n", 1)

        headers = str(article).split("\n\n", maxsplit=1)[0].strip().split("\n")
        toc[i].append("\n".join(headers))  # append header to toc entry

        # Log ERROR if headers length is less than 3
        if len(headers) < 3:
            logger.error(f"Headers length is less than 3 in article {i}: '{title}', {source}, {date}")

        # Strip non alphanumeric characters from title
        title = re.sub(r"\W+", " ", str(title)).lower().strip()
        toc_title = re.sub(r"\W+", " ", str(toc[i][0])).lower().strip()
        assert toc_title.startswith(title), f"toc_title: {toc_title}, title: {title}"

        media = headers[-1] if headers[-1].startswith("Publicerat") else None
        if media:
            media = "webb" if "webb" in media else "print"
        toc[i].append(media)
        logger.debug(f"Extracted media '{media}' from article {i}")

        pages = headers[-2].split(" ")[1] if len(headers) >= 3 and headers[-2].startswith("Sida") else None
        toc[i].append(pages)
        logger.debug(f"Extracted pages '{pages}' from article {i}")

        m = re.search("(?:Läs hela artikeln på|Se webartikeln på) (.*)", article)
        url = m.groups()[0] if m else None
        toc[i].append(url)
        logger.debug(f"Extracted url '{url}' from article {i}")
        article = re.sub("(?:Läs hela artikeln på|Se webartikeln på) .*", "", article).strip()

        # TODO: Save captions to separate column
        stop_words = "Bildtext|Image-text|Pressbild|Snabbversion"
        logger.debug(f"Removing stop words: '{', '.join(stop_words.split('|'))}' from article {i}")
        article = re.sub(r"(Bildtext|Image-text|Pressbild|Snabbversion):\s?", "", article).strip()

        # FIXME: Does not remove if there is a new line between Bild: and [Name]
        logger.debug(f"Removing captions like 'Bild: [Name [Name]]' from article {i}")
        article = re.sub(r"Bild: ([A-Z][a-z]+(?: [A-Z][a-z]+)*)(/TT)?\s?", "", article).strip()

        # TODO: Only look at the end of the article text
        logger.debug(f"Removing copyright string from article {i}")
        article = re.sub(r"©.*$", "", article).strip()

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
        "header",
        "media",
        "pages",
        "url",
        "article_text",
    ]
    df = pd.DataFrame(toc, columns=columns)
    df["date_time"] = pd.to_datetime(df["date"], format="mixed")

    # Check for missing values
    if len(empty := df[df.drop(columns=["pages", "media"]).isnull().any(axis=1)]):
        logger.info(f"Missing values in df:\n{empty}")
    return df


def log_diffs(duplicates: pd.DataFrame, output_folder: str) -> None:

    diff_folder = f"{output_folder}/diff"
    if not os.path.exists(diff_folder):
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
            with open(f"{diff_folder}/{name[0]}_{name[1]}_{name[2]}_{name[3]}_{i-1}.diff", "w", encoding="utf-8") as f:
                f.write(diff_text)


def main(input_folder: str) -> None:
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

        df: pd.DataFrame = create_corpus(toc, articles)

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

    # Save all_metadata
    document_index: pd.DataFrame = pd.concat(all_metadata, ignore_index=True)
    document_index.reset_index(drop=True, inplace=True)
    document_index['document_id'] = document_index.index

    duplicates: pd.DataFrame = document_index[
        document_index.duplicated(subset=['document_name', 'source', 'date', 'media'], keep=False)
    ]
    logger.info(f"Found {len(duplicates)} non-unique articles")

    log_diffs(duplicates, output_folder)

    duplicates = (
        duplicates.groupby(['document_name', 'source', 'date', 'media'])
        .agg(urls=('url', lambda x: ', '.join(x)), count=('url', 'count'))  # pylint: disable=unnecessary-lambda
        .reset_index()
    )
    duplicates.to_csv(f"{output_folder}/duplicates.csv", index=True, sep=";", encoding="utf-8-sig")

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

    # Save document_index to csv
    document_index = document_index.drop(columns=["article_text", "full_text", "header", "toc_line_number"])
    document_index.to_csv(f"{output_folder}/document_index.csv", index=False, sep=";", encoding="utf-8-sig")
    logger.success(f'Saved document_index to {output_folder}/document_index.csv')


if __name__ == "__main__":
    # typer.run(main)
    logger.remove()
    logger.add(sys.stderr, level='INFO')
    main("./input")

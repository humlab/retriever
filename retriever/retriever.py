# pylint: disable=redefined-outer-name
import os
import re
import typer

# import openpyxl
import pandas as pd
from loguru import logger


def get_toc(filename, toc_name, max_not_matched_lines):
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

    logger.info(f"Found {len(data)} articles in {filename}")

    return data, toc_line_number  # "title", "source", "date", "toc_line_number"


def get_articles(filename: str, offset) -> list[str]:
    with open(filename, "r", encoding="utf-8") as file:
        data = "".join(file.readlines()[offset - 1 :])
    articles = data.split("==============================================================================")
    articles = [article.strip() for article in articles]

    return articles


def create_db(toc, articles, filename):
    for i, article in enumerate(articles):
        toc_entry = toc[i]
        title, source, date, _ = toc_entry
        logger.info(f"Processing article {i} in {filename}: '{title}', {source}, {date}")

        toc[i].append(article)  # append full text to toc entry

        headers = str(article).split("\n\n", maxsplit=1)[0].strip().split("\n")
        toc[i].append("\n".join(headers))  # append header to toc entry

        # Log ERROR if headers length is less than 3
        if len(headers) < 3:
            logger.error(f"Header less than 3 rows: {filename.replace('.txt', '')}:{i} '{title}', {source}, {date}")

        # Strip non alphanumeric characters from title
        title = re.sub(r"\W+", " ", title).lower().strip()
        toc_title = re.sub(r"\W+", " ", toc[i][0]).lower().strip()
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
    columns = [
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
    df["date_time"] = pd.to_datetime(df["date"], format="ISO8601")

    # Check for missing values
    if len(empty := df[df.drop(columns=["pages", "media"]).isnull().any(axis=1)]):
        logger.info(f"Missing values in df:\n{empty}")
    return df


def save_articles(output_folder, filename, df):
    for i, article_text in enumerate(df["article_text"]):
        title = df["title"][i]
        source = df["source"][i]
        date = df["date_time"][i].strftime("%Y%m%d")
        media = df["media"][i]
        article_filename = f"{source}_{date}_{media}"
        article_filename = re.sub(r"\W+", "_", article_filename).lower().strip()
        # FIXME: Add extension to filename after removing non-alphanumeric characters
        article_filename = f"{output_folder}/{article_filename}.txt"
        with open(article_filename, "w", encoding="utf-8") as f:
            f.write(title + "\n\n")
            f.write(article_text)
            # logger.debug(f"Saved article {i} to {article_filename}")

    logger.success(f"Saved {len(df)} articles from '{filename}' to {output_folder}")


def main(input_folder: str) -> None:
    output_folder = f"{input_folder}/output"
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    logger.add(f"{output_folder}/extract.log", level="WARNING", encoding="utf8", format="{message}")

    all_metadata = []  # List to store all metadata

    for filename in os.listdir(input_folder):
        if not filename.endswith(".txt"):
            continue
        # logger.add(f"{output_folder}/{os.path.basename(filename)}.log", level="WARNING", encoding="utf8", format="{message}"    )

        filepath = f"{input_folder}/{filename}"
        toc, offset = get_toc(filepath, "Innehållsförteckning:", 2)
        articles = get_articles(filepath, offset)
        df = create_db(toc, articles, filename)

        # Add 'input_file' column to the metadata DataFrame.
        # FIXME: Add 'input_file' in create_db to be able to use when logging errors
        df["input_file"] = filename

        metadata = df.drop(columns=["article_text", "full_text", "header", "toc_line_number"])

        # metadata_filename = filename.replace(".txt", "_metadata.xlsx")
        # metadata.to_excel(f"{output_folder}/{metadata_filename}", index=False)

        # Append metadata to the all_metadata list
        all_metadata.append(metadata)

        # Save articles to txt files
        save_articles(output_folder, filename, df)

    # Save all_metadata to excel
    all_metadata_df = pd.concat(all_metadata, ignore_index=True)
    all_metadata_df.to_excel(f"{output_folder}/metadata.xlsx", index=False)
    # save to csv
    all_metadata_df.to_csv(f"{output_folder}/metadata.csv", index=False, sep=";", encoding="utf-8-sig")


if __name__ == "__main__":
    typer.run(main)
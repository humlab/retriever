
A data processing tool designed to extract and process articles from text files exported by a system called "Retriever". The main functionalities include:

1. **Extracting Table of Contents**: The function `get_toc` extracts the table of contents from a given file, identifying the relevant section and parsing the lines to collect metadata such as titles, sources, and dates.

2. **Extracting Articles**: The function `get_articles` reads the articles from the file starting from a specified offset, splitting the content into individual articles.

3. **Processing Articles**: Various functions are provided to clean and process the articles, such as removing captions, stop words, and copyright strings.

4. **Creating a Corpus**: The function `create_corpus` combines the table of contents and articles into a structured format, creating a DataFrame that can be further analyzed or exported.

5. **Handling Duplicates**: The main function identifies and logs duplicate articles, saving unique articles to text files and generating a CSV file with metadata.

6. **Logging and Output**: The project uses the `loguru`  library for logging and saves the processed articles and metadata to specified output folders.

The project is structured to handle multiple text files, process them, and save the results in a systematic and organized manner.

## Instructions

To use `retriever.py`, follow these steps:

1. **Install Dependencies**: Ensure you have all the required dependencies installed. You can do this using Poetry:
    ```sh
    poetry install
    ```

2. **Prepare Input Files**: Place your text files exported by the Retriever system into the input folder.

3. **Run the Script**: Execute the script using the following command:
    ```sh
    poetry run python retriever/retriever.py input
    ```

    You can also pass additional options:
    - `--save-short-headers`: Save files with short headers.
    - `--stop-words`: Provide a string with stop words separated by '|'.
    - `--remove-captions`: Remove captions from articles.
    - `--remove-copyright`: Remove copyright strings from articles.

    Example:
    ```sh
    poetry run python retriever/retriever.py input --save-short-headers --remove-captions --remove-copyright
    ```

4. **Output**: The processed articles and metadata will be saved in the `output` folder within the `input` directory. The metadata will be saved as *document_index.csv*.


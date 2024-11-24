from rank_bm25 import BM25Okapi
import re
import os


def tokenize_with_re(text):
    """
    Tokenize text using regular expressions to handle punctuation.
    Splits on non-alphanumeric characters and converts to lowercase.

    :param text: The input text to tokenize.
    :return: A list of tokens.
    """
    return re.findall(r'\b\w+\b', text.lower())


class BM25FileSearch:
    def __init__(self, file_paths):
        """
        Initialize the BM25FileSearch class with a list of text file paths.
        Tokenizes the files and prepares the BM25 model.

        :param file_paths: List of paths to text files.
        """
        self.file_paths = file_paths
        self.documents = []
        self.file_names = []
        self.file_name_to_content = {}
        self.bm25 = None
        self.tokenize_files()
        if self.documents:
            self.bm25 = BM25Okapi(self.documents)
        else:
            raise ValueError("No valid documents provided.")

    def tokenize_files(self):
        # Tokenize and load documents
        for file_path in self.file_paths:
            if os.path.exists(file_path):
                with open(file_path, 'r', encoding='utf-8') as file:
                    content = file.read()
                    tokens = self.tokenize(content.lower())
                    self.documents.append(tokens)
                    self.file_names.append(file_path)
                    self.file_name_to_content[file_path] = content
            else:
                print(f"File not found: {file_path}")

    def tokenize(self, text):
        """
        Tokenize the input text using a simple split method.

        :param text: The text to tokenize.
        :return: A list of tokens.
        """
        return tokenize_with_re(text)

    def search(self, query, top_n=10):
        """
        Search for the most relevant files based on the query.

        :param query: A string representing the query.
        :param top_n: Number of top relevant files to return.
        :return: A list of tuples (file_name, score), sorted by relevance.
        """
        if self.bm25 is None:
            raise ValueError("BM25 model is not initialized.")

        query_tokens = self.tokenize(query.lower())
        scores = self.bm25.get_scores(query_tokens)
        ranked_indices = sorted(
            range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_n]
        return [(self.file_names[i], self.file_name_to_content[self.file_names[i]], scores[i]) for i in ranked_indices]


def extract_js_definitions(files):
    """
    Extracts all function and class definitions from a list of JS files.

    Args:
        files (list[str]): Paths to the JS files.

    Returns:
        dict: A dictionary where keys are filenames and values are lists of extracted definitions.
    """
    # Regular expressions to match functions and classes
    function_pattern = re.compile(r'function\s+\w+\s*\([^)]*\)\s*{[^}]*}')
    arrow_function_pattern = re.compile(r'\w+\s*=\s*\([^)]*\)\s*=>\s*{[^}]*}')
    class_pattern = re.compile(r'class\s+\w+\s*{[^}]*}')

    definitions = {}

    for file in files:
        try:
            with open(file, 'r', encoding='utf-8') as f:
                content = f.read()

            # Extract function and class definitions
            functions = function_pattern.findall(content)
            arrow_functions = arrow_function_pattern.findall(content)
            classes = class_pattern.findall(content)

            # Combine all extracted definitions
            definitions[file] = functions + arrow_functions + classes

        except Exception as e:
            print(f"Error processing file {file}: {e}")
            definitions[file] = []
    returned_definitions = {}
    for key, value in definitions.items():
        if len(value) > 0:
            returned_definitions[key] = value
    return returned_definitions


class BM25FileWithClassesAndFunctionsSearch(BM25FileSearch):
    def tokenize_files(self):
        super().tokenize_files()
        js_definitions = extract_js_definitions(self.file_paths)
        for name, definitions in js_definitions.items():
            for idx, definition in enumerate(definitions):
                tokens = self.tokenize(definition)
                self.documents.append(tokens)
                self.file_names.append(f'{name} - item {idx}')
                self.file_name_to_content[f'{name} - item {idx}'] = definition

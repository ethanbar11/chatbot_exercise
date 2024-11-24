
import os

from openai_api import OpenAIAPI
from prompt_creator import convert_files_list_and_query_to_messages
from retrievers.tf_idf_retriever import BM25FileSearch, BM25FileWithClassesAndFunctionsSearch
from datetime import datetime


def find_code_files(directory):
    """
    Recursively finds all code files in a directory.

    Parameters:
        directory (str): Path to the directory to search.

    Returns:
        list: A list of file paths for all code files found.
    """
    code_files = []

    # Common code file extensions
    code_extensions = {
        '.py', '.java', '.c', '.cpp', '.h', '.hpp', '.js', '.ts',
        '.html', '.css', '.php', '.rb', '.go', '.rs', '.sh', '.bat',
        '.pl', '.swift', '.kt', '.cs', '.json', '.xml', '.sql', '.asm'
    }

    for root, _, files in os.walk(directory):
        for file in files:
            filepath = os.path.join(root, file)
            _, ext = os.path.splitext(file)
            if ext.lower() in code_extensions:
                code_files.append(filepath)
    return code_files


def main(repo_path, questions):
    print('Searching for code files...')
    code_files = find_code_files(repo_path)
    print(f'Found {len(code_files)} code files.')
    retriever = BM25FileWithClassesAndFunctionsSearch(code_files)
    openai_api = OpenAIAPI()
    for question in questions:
        start = datetime.now()
        print(
            f'Searching for the most relevant files for the question: {question}')
        results = retriever.search(question, top_n=20)
        print('Results:')
        for file, content, score in results:
            print(f'{file} - Score: {score}')

        messages = convert_files_list_and_query_to_messages(
            results, question)
        openai_response = openai_api.create_chat_completion(messages)
        print('For the following question:')
        print(question)
        print('The response from OpenAI is:')
        print(openai_response)
        print(f'Time taken: {datetime.now() - start}s')
        print('-----------------------------------')


if __name__ == '__main__':
    repo_path = '/root/code/tabnine_code/repo_examples'
    questions = ['How do I send a request to the tabnine binary?',
                 'Where are we registering all the handlers for the chat application?']
    main(repo_path, questions)

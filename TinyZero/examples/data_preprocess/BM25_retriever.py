from rank_bm25 import BM25Okapi
from tqdm import tqdm
import torch
import os
import json
from typing import List, Dict
def create_bm(texts: List[str]):
    tokenized_texts = [tokenize(text) for text in tqdm(texts, desc="Tokenizing texts")]

    # Create BM25 object
    bm25 = BM25Okapi(tokenized_texts)
    return bm25
def search_relevant_texts_case(texts: List[str], bm25, query: str, top_k: int=10000)-> List[str]:
    tokenized_query = tokenize(" ".join(query))
    scores = bm25.get_scores(tokenized_query)

    # Sort documents by BM25 scores
    sorted_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    sorted_documents = [texts[i] for i in sorted_indices[:top_k]]

    return sorted_documents

def tokenize(text:str) -> List[str]:
    """
    Tokenize a given text into lowercase words split by spaces.
    
    Args:
        text (str): The text to tokenize.
    
    Returns:
        list of str: Tokenized words.
    """
    return text.lower().split()


def search_relevant_texts(texts: List[str], keywords: List[str], top_k=10000) -> List[str]:
    """
    Search for relevant texts based on keywords using BM25.

    Args:
        texts (list of str): List of documents to search.
        keywords (list of str): List of keywords to match against the documents.
        top_k (int): Maximum number of top documents to retrieve.

    Returns:
        list of str: List of top-k relevant documents sorted by relevance score.
    """
    # Tokenize the texts and keywords
    tokenized_texts = [tokenize(text) for text in tqdm(texts, desc="Tokenizing texts")]
    tokenized_keywords = tokenize(" ".join(keywords))

    # Create BM25 object
    bm25 = BM25Okapi(tokenized_texts)

    # Calculate BM25 scores for the keywords
    scores = bm25.get_scores(tokenized_keywords)

    # Sort documents by BM25 scores
    sorted_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    sorted_documents = [texts[i] for i in sorted_indices[:top_k]]

    return sorted_documents


def collect_relevant_texts(paths: List[str]) -> List[str]:
    """
    Collect all text data from JSONL files within the specified directories.

    Args:
        paths (list of str): List of directory paths containing JSONL files.

    Returns:
        list of str: Collected texts from all specified directories.
    """
    collected_texts = []

    for path in tqdm(paths, desc="Processing directories"):
        try:
            jsonl_files = [f for f in os.listdir(path) if f.endswith('.jsonl')]
        except FileNotFoundError:
            tqdm.write(f"Directory not found: {path}")
            continue

        for jsonl_file in tqdm(jsonl_files, desc=f"Processing files in {path}", leave=False):
            file_path = os.path.join(path, jsonl_file)
            with open(file_path, 'r', encoding='utf-8') as file:
                for line in file:
                    try:
                        data = json.loads(line)
                        collected_texts.append(data.get('text', ' '.join(data.values())))
                    except json.JSONDecodeError:
                        tqdm.write(f"Invalid JSON in file: {file_path}")
    
    return collected_texts


def extract_keywords(domain_name: str) -> List[str]:
    """
    Extract a list of synonyms for a given domain name using an AI model.

    Args:
        domain_name (str): The domain name to generate synonyms for.

    Returns:
        list of str: List of synonyms for the given domain name.
    """
    def generate_synonyms_prompt(domain: str) -> str:
        return (f"You can generate synonyms about this domain: {domain}. Each synonym must have the same meaning "
                f"as the domain. Your output must only be a list of all synonyms like [\"xxx\", \"xxx\"].")

    def parse_synonyms_from_response(response: str) -> List[str]:
        try:
            start_idx = response.find('[')
            end_idx = response.find(']')
            return eval(response[start_idx:end_idx + 1])
        except (ValueError, SyntaxError):
            raise ValueError("Invalid response format for extracting synonyms.")

    prompt = generate_synonyms_prompt(domain_name)


def search_relevant_documents(domain_name: str, top_k: int, paths: List[str])->List[str]:
    """
    Search for documents relevant to a specified domain name.

    Args:
        domain_name (str): The domain name to search for.
        top_k (int): Number of top relevant documents to return.
        paths (list of str): List of directory paths containing JSONL files.

    Returns:
        list of str: List of top-k relevant documents.
    """
    # Extract domain-related keywords
    domain_keywords = extract_keywords(domain_name)
    print("Extracted Keywords:", domain_keywords)

    # Collect texts from specified paths
    all_texts = collect_relevant_texts(paths)
    print("Total Texts Collected:", len(all_texts))

    # Search for relevant texts
    relevant_texts = search_relevant_texts(all_texts, domain_keywords, top_k=top_k)
    return relevant_texts

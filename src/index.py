
import numpy as np
import re

from sentence_transformers import SentenceTransformer, util
from bs4 import BeautifulSoup
from tqdm import tqdm

from .preprocess import read_txt_file


def remove_html_tags(split):
    soup = BeautifulSoup(split)
    text = soup.get_text()
    return text


def sentence_splitter(text, chunk_size, chunk_overlap, model):
    """Split text to chunks of chunk_size with chunk_overlap. Also make sure it does not reach model max sequence length."""
    tokens = model.tokenizer(text)['input_ids']
    n_chunks = int(np.ceil(len(tokens) / chunk_size))
    start = 0
    end = chunk_size
    chunks = []
    for n in range(n_chunks):
        assert len(tokens[start:end]) <= chunk_size
        chunk_str = model.tokenizer.decode(
            tokens[start:end], skip_special_tokens=True)
        chunks.append(chunk_str)
        start = end - chunk_overlap
        end = start + chunk_size
    return chunks


def split_and_embed_corpus(corpus_files, model='distiluse-base-multilingual-cased-v2'):
    model = SentenceTransformer('distiluse-base-multilingual-cased-v2')
    max_seq_length = model.max_seq_length
    chunk_size = max_seq_length - 2  # For 2 special tokens.
    chunk_overlap = 10

    paragraph_split_pattern = r"\<h2\>"
    doc_chunks = {}
    doc_chunks_emb = {}

    chunk_counter = 0

    for file_path in tqdm(corpus_files):
        data = read_txt_file(file_path)
        # First, split content by <h2>.
        re_splits = re.split(paragraph_split_pattern, data)

        # Remove HTML tags.
        re_splits = list(map(remove_html_tags, re_splits))
        if len(re_splits) > 1:
            # Remove first paragraph since it only contains headers and stuffs.
            re_splits = re_splits[1:]

        # Simple split with fixed chunk size and overlap.
        all_chunks = []
        for re_split in re_splits:
            chunks = sentence_splitter(
                re_split, chunk_size=chunk_size, chunk_overlap=chunk_overlap, model=model)
            all_chunks.extend(chunks)

        file_name = file_path.split('/')[-1]
        # Save chunk content with its embedding.
        doc_chunks[file_name] = all_chunks
        doc_chunks_emb[file_name] = model.encode(
            all_chunks, show_progress_bar=False)

        chunk_counter += len(all_chunks)

    print('Total chunks:', chunk_counter)

    return doc_chunks, doc_chunks_emb

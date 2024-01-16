import re
import pandas as pd

from tqdm import tqdm
from glob import glob
from pathlib import Path
from string import punctuation
from sentence_transformers import SentenceTransformer

from src.preprocess import hybrid_search_topn_diseases, preprocess_corpus_files, preprocess_option, merge_options
from src.index import split_and_embed_corpus
from src.retrieval import retrieve_context
from src.llm import load_llm, predict, is_the_answer_correct, instruct_prompt

corpus_path = Path('/kaggle/input/kalapa-vietmedqa/corpus')
corpus_preproc_path = Path('/kaggle/working/output')
test_path = Path('/kaggle/input/kalapa-vietmedqa/public_test.csv')

if __name__ == '__main__':
    # 0. Preprocessing steps.
    preprocess_corpus_files(corpus_path, corpus_preproc_path)

    test = pd.read_csv(test_path)
    for i in range(1, 7):
        test[f'option_{i}_pp'] = test[f'option_{i}'].apply(preprocess_option)
        assert len(test[test[f'option_{i}'] == '']) == 0, i
        test.drop(columns=[f'option_{i}'], inplace=True)
    test['options'] = test.apply(merge_options, axis=1)

    # 1. Indexing (split and embed texts).
    corpus_glob = str(corpus_preproc_path/'*')
    disease_files = glob(corpus_glob)
    diseases = [Path(p).stem.lower().replace('hội chứng', '').replace(
        'bệnh', '').strip().split(' ') for p in glob(corpus_glob)]
    questions = [re.sub(f"[{punctuation}]", "", q).lower().split()
                 for q in test['question'].tolist()]
    answers = [re.sub(f"[{punctuation}]", "", q).lower().split()
               for q in test['options'].tolist()]
    ids = test['id'].tolist()

    # Text => chunks of text => embeddings.
    doc_chunks, doc_chunks_emb = split_and_embed_corpus(
        disease_files,
        model='distiluse-base-multilingual-cased-v2'
    )

    # 2. Retrieval steps.
    # Reduce the number of disease files to look at for each question.
    topn_diseases_dict = hybrid_search_topn_diseases(
        disease_files, diseases, questions, answers, ids, n=5
    )

    # Start retrieving.
    embedding_model = SentenceTransformer(
        'distiluse-base-multilingual-cased-v2')
    retrieved_context_dict = {}
    for index, row in tqdm(test.iterrows(), total=test.shape[0]):
        doc_id = row.id
        query = row.question + '\n' + row.options

        files_to_look_at = topn_diseases_dict[doc_id]['topn_disease']

        # print('index=', doc_id)
        # print(query, '\n')
        # print(files_to_look_at, '\n')
        # print('#' * 5)
        retrieved_context = retrieve_context(
            query, files_to_look_at, doc_chunks, doc_chunks_emb, embedding_model)
        retrieved_context_dict[doc_id] = retrieved_context

    # 3. Augmented LLM.
    llm, tokenizer = load_llm()

    # Testing on only 1 document.
    print(predict(test.iloc[0], llm, tokenizer,
          retrieved_context_dict, correct_str='1', incorrect_str='0'))

    # You can run the rest here if you want.

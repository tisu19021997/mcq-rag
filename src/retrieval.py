
from sentence_transformers import util


def retrieve_context(query, diseases_to_search, context_chunks, context_chunks_emb, model, n_doc=2, k_context=3):
    # Total chunk of context = topn_doc * topn_context.
    query_emb = model.encode(query, show_progress_bar=False)
    retrieved_context = []
    for disease in diseases_to_search[:n_doc]:
        hits = util.semantic_search(
            query_emb, context_chunks_emb[disease], top_k=k_context, score_function=util.dot_score)
        topk_context_idx = [hit['corpus_id'] for hit in hits[0]]
        topk_context = [context_chunks[disease][i] for i in topk_context_idx]
        retrieved_context.append(topk_context)
    return retrieved_context

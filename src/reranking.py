from sentence_transformers import CrossEncoder


reranker = CrossEncoder(
    "cross-encoder/ms-marco-MiniLM-L-6-v2"
)


def rerank(query, documents):

    pairs = [
        (
            query,
            row["document"]
        )
        for _, row in documents.iterrows()
    ]


    scores = reranker.predict(
        pairs
    )


    documents["rerank_score"] = scores


    return documents.sort_values(
        "rerank_score",
        ascending=False
    )
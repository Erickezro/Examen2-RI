import faiss
import pandas as pd
from sentence_transformers import SentenceTransformer


df = pd.read_csv(
    "data/arxiv_cleaned.csv"
)


embedding_model = SentenceTransformer(
    "BAAI/bge-small-en-v1.5"
)


index = faiss.read_index(
    "data/arxiv_index.faiss"
)


def buscar(query, k=25):

    q_emb = embedding_model.encode(
        [query],
        normalize_embeddings=True,
        convert_to_numpy=True
    ).astype("float32")


    scores, indices = index.search(
        q_emb,
        k
    )


    resultados = df.iloc[
        indices[0]
    ].copy()


    resultados["score"] = scores[0]


    return resultados[
        [
            "titles",
            "terms",
            "score",
            "document"
        ]
    ]
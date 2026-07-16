from src.retrieval import buscar
from src.reranking import reranker
from src.llm import generar_respuesta


def rag_pipeline(query, top_k=5):

    retrieved = buscar(query, k=25)

    pairs = [
        (query, row["document"])
        for _, row in retrieved.iterrows()
    ]

    scores = reranker.predict(pairs)

    retrieved["rerank_score"] = scores

    retrieved = (
        retrieved
        .sort_values(
            "rerank_score",
            ascending=False
        )
        .head(top_k)
    )


    contexto = "\n\n".join(
        row["document"]
        for _, row in retrieved.iterrows()
    )


    respuesta = generar_respuesta(
        query,
        contexto
    )

    return respuesta, retrieved
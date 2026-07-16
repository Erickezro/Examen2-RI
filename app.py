import streamlit as st

from src.rag_pipeline import rag_pipeline


st.title(
    "🔎 Scientific RAG Assistant"
)


query = st.text_input(
    "Ask a scientific question"
)



if st.button("Generate"):

    if query:

        respuesta, documentos = rag_pipeline(
            query
        )


        st.subheader(
            "Answer"
        )

        st.write(
            respuesta
        )

        if documentos is not None:

            st.subheader("Retrieved evidence")

            for _, row in documentos.iterrows():
                st.write(row["titles"])
                st.write(
                    f"Similarity: {row['score']:.3f} | "
                    f"Rerank: {row['rerank_score']:.3f}"
                )
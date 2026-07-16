from openai import OpenAI
from dotenv import load_dotenv
import os


load_dotenv()


client = OpenAI(
    api_key=os.getenv(
        "OPENROUTER_API_KEY"
    ),
    base_url=
    "https://openrouter.ai/api/v1"
)


def generar_respuesta(query, contexto):

    prompt = f"""
You are a scientific assistant.
Answer ONLY using the provided context.

Context:
{contexto}

Question:
{query}

Answer:
"""


    response = client.chat.completions.create(
        model="openai/gpt-oss-20b:free",
        messages=[
            {
                "role": "system",
                "content": "Answer scientific questions only using the context."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0.2
    )


    return response.choices[0].message.content
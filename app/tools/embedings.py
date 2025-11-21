from app.deps.litellm_client import client

EMBEDDING_MODEL = "codemie-text-embedding-ada-002"

def generate_embeddings(text_chunks):
    """Генерує вектори для списку текстів за допомогою OpenAI API."""
    response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=text_chunks,
    )

    return [embedding.embedding for embedding in response.data]


def generate_embedding(text):
    """Генерує вектор для одного текстового запиту."""
    try:
        response = client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=text,
        )
        return response.data[0].embedding
    except Exception as e:
        print(f"Помилка під час генерації одного embedding: {e}")
        return None


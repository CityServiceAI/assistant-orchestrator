import faiss
import numpy as np
import pandas as pd
import os
from app.tools.embedings import generate_embeddings, generate_embedding


def setup_faiss_database_from_csv(csv_file_path):
    """
    Читає CSV, генерує embeddings і зберігає їх у FAISS індекс.
    """
    print(f"Читання даних із {csv_file_path}...")
    df = pd.read_csv(csv_file_path)

    # Підготовка даних для векторизації
    df['full_description'] = df['description'] + " " + df['category_name'] + " " + df['responsible_entity']

    # Генерація векторів
    print("Генерація embeddings...")
    embeddings = generate_embeddings(df['full_description'].tolist())

    if not embeddings:
        print("База даних не створена через помилку embeddings.")
        return None, None

    # Конвертація embeddings до формату numpy (потрібно для FAISS)
    embeddings_np = np.array(embeddings).astype('float32')

    # Створення індексу FAISS (IndexFlatL2 - простий індекс, L2 - евклідова відстань)
    dimension = embeddings_np.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings_np)

    print(f"✅ FAISS індекс успішно створений та заповнений. Кількість записів: {index.ntotal}")

    # Зберігаємо DataFrame з метаданими окремо
    # Можна зберегти як pickle або просто повернути об'єкт df
    return index, df

CATEGORIES_2_CSV = os.getenv("CATEGORIES_2_CSV", 'app/data/categories_2.csv')
FAISS_INDEX, METADATA_DF = setup_faiss_database_from_csv(CATEGORIES_2_CSV)


def search_categories(tags, n_results=5):
    query_text = " ".join(tags)
    query_embedding = generate_embedding(query_text)
    if query_embedding is None:
        return None

    query_embedding_np = np.array(query_embedding).astype('float32').reshape(1, -1)

    # D - відстані, I - індекси (позиції в DataFrame)
    distances, indices = FAISS_INDEX.search(query_embedding_np, n_results)

    formatted_results = []
    for i in range(n_results):
        idx = indices[0][i]
        distance = distances[0][i]

        # Отримуємо рядок метаданих з DataFrame за індексом idx
        metadata = METADATA_DF.iloc[idx]

        formatted_results.append({
            "Код": metadata['code'],
            "Опис": metadata['description'],
            "Відповідальний": metadata['responsible_entity'],
            "Категорія": metadata['category_name'],
            "Релевантність": round(float(distance), 4)
        })

    return formatted_results
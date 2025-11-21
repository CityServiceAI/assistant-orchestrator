import faiss
import numpy as np
import pandas as pd

from app.tools.embedings import generate_embeddings


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
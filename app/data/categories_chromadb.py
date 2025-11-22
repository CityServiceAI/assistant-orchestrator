import chromadb
import pandas as pd
import os
from app.tools.embedings import generate_embeddings, generate_embedding
import logging

chroma_client = chromadb.Client()

def setup_chromadb_database_from_csv(csv_file_path):
    """
    Читає CSV, генерує embeddings і зберігає їх у векторну БД ChromaDB.
    """
    print(f"Читання даних із {csv_file_path}...")
    df = pd.read_csv(csv_file_path)

    # Підготовка даних для векторизації (повний опис)
    df['full_description'] = df['description'] + " " + df['category_name'] + " " + df['responsible_entity']

    # Генерація векторів
    print("Генерація embeddings...")
    embeddings = generate_embeddings(df['full_description'].tolist())

    if not embeddings:
        print("База даних не створена через помилку embeddings.")
        return

    # Налаштування та заповнення векторної БД (ChromaDB)
    collection_name = "problem_categories_ukr_2"
    # Якщо колекція існує, видаляємо її, щоб створити нову
    try:
        chroma_client.delete_collection(name=collection_name)
    except Exception:
        logging.error("CANNOT DELETE COLLECTION")
        pass

    collection = chroma_client.create_collection(name=collection_name)

    # Підготовка метаданих для зберігання
    metadata_dicts = df.drop(columns=['full_description']).to_dict(orient='records')

    collection.add(
        embeddings=embeddings,
        documents=df['full_description'].tolist(),
        metadatas=metadata_dicts,
        ids=[str(i) for i in range(len(df))] # Використовуємо індекси як ID
    )

    print(f"✅ Векторна база даних '{collection_name}' успішно створена та заповнена.")
    return collection

CATEGORIES_2_CSV = os.getenv("CATEGORIES_2_CSV", 'app/data/categories_2.csv')
COLLECTION = setup_chromadb_database_from_csv(CATEGORIES_2_CSV)

def search_categories(tags_list, n_results=3):
    query_text = " ".join(tags_list)
    query_embedding = generate_embedding(query_text)

    if query_embedding is None:
        return None

    # 3. Виконуємо пошук (запит) по векторах
    results = COLLECTION.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
        include=['metadatas', 'distances']
    )

    # Форматуємо результати для зручності використання в Кроці 3
    formatted_results = []
    for i in range(len(results['metadatas'][0])):
        metadata = results['metadatas'][0][i]
        distance = results['distances'][0][i]
        formatted_results.append({
            "Код": metadata['code'],
            "Опис": metadata['description'],
            "Відповідальний": metadata['responsible_entity'],
            "Категорія": metadata['category_name'],
            "Релевантність": round(distance, 4)
        })

    return formatted_results

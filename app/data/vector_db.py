import chromadb
import pandas as pd

from app.tools.embedings import generate_embeddings
import logging

chroma_client = chromadb.Client()

def setup_vector_database_from_csv(csv_file_path):
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
    collection_name = "problem_categories_ukr"
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

import os
import chromadb
from chromadb.utils.embedding_functions import OllamaEmbeddingFunction

# Configuración
BASE_DIR = os.path.dirname(__file__)
DOCUMENTS_PATH = os.path.join(BASE_DIR, "documents")
CHROMA_PATH = os.path.join(BASE_DIR, "chroma_db")
COLLECTION_NAME = "finbot"
CHUNK_SIZE = 400
CHUNK_OVERLAP = 50
OLLAMA_URL = "http://localhost:11434"
EMBEDDING_MODEL = "nomic-embed-text"


def load_documents(path: str) -> list[dict]:
    """Carga todos los ficheros .txt del directorio de documentos."""
    documents = []
    for filename in os.listdir(path):
        if filename.endswith(".txt"):
            filepath = os.path.join(path, filename)
            with open(filepath, "r", encoding="utf-8") as f:
                content = f.read()
            documents.append({
                "filename": filename,
                "content": content
            })
            print(f"  Cargado: {filename} ({len(content)} caracteres)")
    return documents


def split_into_chunks(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Divide el texto en chunks con solapamiento."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start += chunk_size - overlap
    return chunks


def ingest():
    print("=== FinBot RAG — Ingestión de documentos ===\n")

    # 1. Cargar documentos
    print("1. Cargando documentos...")
    documents = load_documents(DOCUMENTS_PATH)
    print(f"   {len(documents)} documentos cargados.\n")

    # 2. Crear cliente ChromaDB
    print("2. Conectando con ChromaDB...")
    client = chromadb.PersistentClient(path=CHROMA_PATH)

    # 3. Configurar función de embeddings con Ollama
    embedding_function = OllamaEmbeddingFunction(
        url=f"{OLLAMA_URL}/api/embeddings",
        model_name=EMBEDDING_MODEL
    )

    # 4. Crear o limpiar la colección
    if COLLECTION_NAME in [c.name for c in client.list_collections()]:
        client.delete_collection(COLLECTION_NAME)
        print(f"   Colección '{COLLECTION_NAME}' eliminada para reindexar.")

    collection = client.create_collection(
        name=COLLECTION_NAME,
        embedding_function=embedding_function
    )
    print(f"   Colección '{COLLECTION_NAME}' creada.\n")

    # 5. Dividir en chunks e indexar
    print("3. Indexando chunks...")
    total_chunks = 0

    for doc in documents:
        chunks = split_into_chunks(doc["content"], CHUNK_SIZE, CHUNK_OVERLAP)
        ids = [f"{doc['filename']}_chunk_{i}" for i in range(len(chunks))]
        metadatas = [{"source": doc["filename"]} for _ in chunks]

        collection.add(
            documents=chunks,
            ids=ids,
            metadatas=metadatas
        )

        print(f"   {doc['filename']}: {len(chunks)} chunks indexados")
        total_chunks += len(chunks)

    print(f"\n=== Ingestión completada: {total_chunks} chunks totales ===")


if __name__ == "__main__":
    ingest()
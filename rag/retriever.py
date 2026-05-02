import chromadb
from chromadb.utils.embedding_functions import OllamaEmbeddingFunction
import os

# Configuración
CHROMA_PATH = os.path.join(os.path.dirname(__file__), "chroma_db")
COLLECTION_NAME = "finbot"
OLLAMA_URL = "http://localhost:11434"
EMBEDDING_MODEL = "nomic-embed-text"
N_RESULTS = 3


def get_collection():
    """Conecta con ChromaDB y devuelve la colección de FinBot."""
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    embedding_function = OllamaEmbeddingFunction(
        url=f"{OLLAMA_URL}/api/embeddings",
        model_name=EMBEDDING_MODEL
    )
    return client.get_collection(
        name=COLLECTION_NAME,
        embedding_function=embedding_function
    )


def retrieve(query: str, n_results: int = N_RESULTS) -> list[dict]:
    """
    Dado una pregunta, recupera los chunks más relevantes de ChromaDB.
    Devuelve una lista de dicts con 'text', 'source' y 'distance'.
    """
    collection = get_collection()

    results = collection.query(
        query_texts=[query],
        n_results=n_results
    )

    chunks = []
    for i in range(len(results["documents"][0])):
        chunks.append({
            "text": results["documents"][0][i],
            "source": results["metadatas"][0][i]["source"],
            "distance": round(results["distances"][0][i], 4)
        })

    return chunks


def retrieve_and_print(query: str):
    """Versión de diagnóstico: imprime los chunks recuperados."""
    print(f"\n PREGUNTA: {query}")
    print("-" * 60)

    chunks = retrieve(query)

    for i, chunk in enumerate(chunks, 1):
        print(f"\n[Chunk {i}] Fuente: {chunk['source']} | Distancia: {chunk['distance']}")
        print(chunk["text"])

    print("\n" + "=" * 60)


if __name__ == "__main__":
    preguntas_test = [
        "¿Cuál es la rentabilidad de la cuenta de ahorro?",
        "¿Puedo recuperar mi dinero antes de que venza el depósito?",
        "¿Qué documentos necesito para contratar la tarjeta de crédito?",
        "¿El plan de pensiones garantiza el capital?",
        "¿Cuántos productos de ahorro puedo tener a la vez?"
    ]

    for pregunta in preguntas_test:
        retrieve_and_print(pregunta)
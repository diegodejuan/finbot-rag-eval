import ollama
from retriever import retrieve

# Configuración
LLM_MODEL = "llama3.2"
N_RESULTS = 3

SYSTEM_PROMPT = """Eres FinBot, el asistente virtual bancario de FinBank.

REGLAS ESTRICTAS:
- Responde SIEMPRE en español
- Usa SOLO la información del contexto proporcionado
- Si la información no está en el contexto, di exactamente: "No dispongo de información sobre ese tema. Por favor, contacta con nuestro Servicio de Atención al Cliente."
- NUNCA recomiendes productos específicos a clientes individuales
- NUNCA accedas ni menciones datos de cuentas, saldos ni movimientos
- NUNCA prestes asesoramiento financiero personalizado
- Sé conciso y profesional"""


def build_context(chunks: list[dict]) -> str:
    """Construye el bloque de contexto a partir de los chunks recuperados."""
    context_parts = []
    for i, chunk in enumerate(chunks, 1):
        context_parts.append(f"[Fuente {i} — {chunk['source']}]\n{chunk['text']}")
    return "\n\n".join(context_parts)


def ask_finbot(question: str, verbose: bool = False) -> dict:
    """
    Pipeline RAG completo:
    1. Recupera chunks relevantes
    2. Construye el contexto
    3. Llama al LLM con el contexto
    4. Devuelve respuesta + metadata
    """
    # 1. Recuperar chunks
    chunks = retrieve(question, n_results=N_RESULTS)
    context = build_context(chunks)

    if verbose:
        print(f"\n{'='*60}")
        print(f"PREGUNTA: {question}")
        print(f"\nCONTEXTO RECUPERADO:")
        for i, chunk in enumerate(chunks, 1):
            print(f"  [{i}] {chunk['source']} (distancia: {chunk['distance']})")
        print()

    # 2. Construir mensajes
    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT
        },
        {
            "role": "user",
            "content": f"""Usa el siguiente contexto para responder la pregunta del cliente.

CONTEXTO:
{context}

PREGUNTA DEL CLIENTE:
{question}"""
        }
    ]

    # 3. Llamar al LLM
    response = ollama.chat(
        model=LLM_MODEL,
        messages=messages
    )

    answer = response["message"]["content"]

    if verbose:
        print(f"RESPUESTA DE FINBOT:")
        print(answer)
        print(f"{'='*60}\n")

    # 4. Devolver resultado completo
    return {
        "question": question,
        "answer": answer,
        "context": [chunk["text"] for chunk in chunks],
        "sources": [chunk["source"] for chunk in chunks]
    }


if __name__ == "__main__":
    preguntas_test = [
        "¿Cuál es la rentabilidad de la cuenta de ahorro?",
        "¿Puedo recuperar mi dinero antes de que venza el depósito?",
        "¿Qué documentos necesito para contratar la tarjeta de crédito?",
        "¿El plan de pensiones garantiza el capital?",
        "¿Cuántos productos de ahorro puedo tener a la vez?"
    ]

    for pregunta in preguntas_test:
        ask_finbot(pregunta, verbose=True)
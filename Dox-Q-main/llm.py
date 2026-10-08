import requests

# --- Ollama Configuration ---
OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama3.2"  # Change to any model you have pulled, e.g. "mistral", "gemma2"


def ask_llm(user_question: str, context: str) -> str:
    """
    Send a question + retrieved context to the local Ollama LLM and return its answer.

    Parameters:
        user_question (str): The user's original question.
        context (str): Retrieved document chunks from ChromaDB.

    Returns:
        str: The LLM's answer as a plain string.
    """
    prompt = f"""You are a helpful assistant. Answer the user's question using ONLY the context provided below.
If the answer is not in the context, say "I couldn't find that in the document."

--- CONTEXT START ---
{context}
--- CONTEXT END ---

User Question: {user_question}

Answer:"""

    print(f"Sending prompt to Ollama model '{OLLAMA_MODEL}'...")

    try:
        response = requests.post(
            OLLAMA_URL,
            json={
                "model": OLLAMA_MODEL,
                "prompt": prompt,
                "stream": False
            },
            timeout=60
        )
        response.raise_for_status()
        answer = response.json().get("response", "").strip()
        print(f"Ollama response received.")
        return answer

    except requests.exceptions.ConnectionError:
        print("Ollama is not running. Start it with: ollama serve")
        return "LLM is offline. Please start Ollama with `ollama serve`."

    except requests.exceptions.Timeout:
        print("Ollama request timed out.")
        return "The LLM took too long to respond. Please try again."

    except Exception as e:
        print(f"Ollama error: {e}")
        return f"LLM error: {str(e)}"
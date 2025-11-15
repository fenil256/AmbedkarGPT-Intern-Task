# AmbedkarGPT-Intern-Task
Required to build a simple command-line Q&amp;A system. The system will ingest the text from a provided short speech by Dr. B.R. Ambedkar and answer questions based solely on that content.
A simple command-line Q&A system using LangChain, Ollama (Llama2/Mistral), ChromaDB, and HuggingFace sentence embeddings.

---

## Setup Instructions

1. **Create and activate a virtual environment:**
    ```
    python -m venv env
    source env/bin/activate          # Windows: .\env\Scripts\activate
    ```

2. **Install Python dependencies:**
    ```
    pip install -r requirements.txt
    ```

3. **Install and start Ollama:**
    ```
    curl -fsSL https://ollama.ai/install.sh | sh
    ollama pull llama2      # Use 'llama2' for low RAM or 'mistral' if you have >4GB RAM
    ollama serve
    ```

4. **Add the provided files:**  
    - `main.py` (your Python code)
    - `speech.txt` (the given Ambedkar excerpt)
    - `questions.txt` (sample questions for the demo)

5. **Run the Q&A prototype:**
    ```
    python main.py
    ```

    Answer questions interactively, or use the example questions.

---

## Notes

- If you see deprecation warnings about `HuggingFaceEmbeddings` or `Ollama`, you can use updated imports:
    - Embeddings: `from langchain_huggingface import HuggingFaceEmbeddings`
    - Ollama: `from langchain_ollama import OllamaLLM`
    - Replace `qa_chain(query)` with `qa_chain.invoke(query)` where needed.
- If memory is low, use Ollama's smaller models like `llama2`.
- The system retrieves answers strictly from the provided speech excerpt.
- All dependencies are free, local, and require no API key.

---

## Demo Questions

See `questions.txt` for suggested demo queries to showcase your system.



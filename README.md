# Document RAG Chatbot

Retrieval-Augmented Generation (RAG) chatbot that answers questions from your own documents. Supports **PDF, DOCX, PPTX, PNG and JPG** (images are read with OCR).

## How it works
1. **Load**: text is extracted from each file (images via OCR)
2. **Chunk**: LangChain splits text into overlapping 800-character chunks
3. **Embed**: Hugging Face `all-MiniLM-L6-v2` turns each chunk into a vector
4. **Retrieve**: FAISS semantic search finds the 4 most relevant chunks for a question
5. **Answer**: Gemini answers using only those chunks and cites the source file and page

## Tech stack
Python, LangChain, Hugging Face Transformers (sentence-transformers), FAISS, Streamlit, Gemini API

## Run locally
```bash
python -m venv venv
venv\Scripts\activate           # Mac/Linux: source venv/bin/activate
pip install -r requirements.txt
copy .env.example .env          # Mac/Linux: cp .env.example .env, then add GEMINI_API_KEY
streamlit run app.py
```
The first run downloads the embedding model (~90 MB).

## Project structure
- `app.py` Streamlit chat interface
- `loaders.py` file readers for each format
- `rag.py` chunking, embeddings, retrieval, answer generation

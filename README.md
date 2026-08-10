# RAG Demo

A Django-based retrieval augmented generation (RAG) app for asking questions about mythology and getting answers grounded in the project knowledge base.

The app uses a simple web UI, retrieves the most relevant passages from a FAISS vector store, sends those passages to an LLM through OpenRouter, and renders the response back into the page.

## What this project does

- Accepts a question from a web form
- Finds relevant evidence chunks from the knowledge base
- Sends the evidence and question to a language model
- Returns a concise answer with short supporting explanation
- Renders the answer in the browser with Markdown converted to HTML

## How it works

1. The browser loads the main page from Django.
2. The prompt form sends the question to `POST /ask/` using HTMX.
3. `chat/views.py` calls `generate_answer()`.
4. `chat/services/retriever.py` retrieves the most relevant passages from FAISS.
5. The retriever embeds the user question with OpenRouter embeddings, scores candidates with a mix of semantic, lexical, and phrase matching, and then diversifies the final set so one source does not dominate the context.
6. `chat/services/llm.py` formats the retrieved passages into a prompt and sends them to the chat model through OpenRouter.
7. The response comes back as Markdown.
8. `chat/services/markdown.py` converts the Markdown to HTML.
9. `chat/templates/partials/answer.html` renders the answer fragment back into the page.

## Project structure

- `config/` - Django project settings, root URLs, WSGI/ASGI config
- `chat/` - Main app, views, templates, services, and knowledge assets
- `chat/services/` - RAG pipeline helpers
- `chat/templates/` - Main page and partial HTML fragments
- `chat/static/` - CSS and other static assets
- `chat/knowledge/` - Raw, processed, chunked, embedded, and vector-store data
- `manage.py` - Django command entry point

## Requirements

You need:

- Python 3.12 or compatible
- A virtual environment
- An `OPENROUTER_API_KEY`
- A `SECRET_KEY`
- Internet access the first time the app loads the FAISS index from Hugging Face

Optional:

- `HF_TOKEN` if the Hugging Face dataset is private

## Environment variables

Create a `.env` file in the project root and set at least:

```env
SECRET_KEY=your-django-secret-key
DEBUG=True
OPENROUTER_API_KEY=your-openrouter-api-key
```

Optional values:

```env
HF_TOKEN=your-hugging-face-token
RENDER_EXTERNAL_HOSTNAME=your-hostname
```

## Install and run

From a fresh clone:

```bash
cd rag-demo
python3 -m venv .venv
source .venv/bin/activate
pip install django python-dotenv openai numpy huggingface_hub markdown-it-py whitenoise pypdf sentence-transformers torch scikit-learn scipy faiss-gpu
```

If `faiss-gpu` is not available on your machine, install the CPU build instead and keep the rest of the project the same.

Then run Django:

```bash
python manage.py migrate
python manage.py runserver
```

Open the app at:

```text
http://127.0.0.1:8000/
```

## How to make changes

### UI and styling

- `chat/templates/index.html` controls the main page shell, the bottom input bar, and the theme toggle.
- `chat/templates/partials/answer.html` controls how the question and answer are rendered.
- `chat/static/chat/css/theme.css` contains the theme variables and all visual styling.

If you want to change the light/dark mode colors, update `chat/static/chat/css/theme.css`.

### Request flow

- `chat/views.py` handles the `index` page and the `/ask/` request.
- `chat/services/llm.py` builds the final prompt and calls the LLM.
- `chat/services/markdown.py` turns Markdown into HTML before the response is rendered.

If you want to change how answers are generated, start in `chat/services/llm.py`.

### Retrieval and knowledge base

- `chat/services/retriever.py` controls ranking and source diversification.
- `chat/services/vector_store.py` rebuilds the FAISS index from the embedded chunks.
- `chat/services/embedder.py` generates embeddings for chunked text.
- `chat/services/chunk_maker.py` splits processed text into chunks.
- `chat/services/pdf_loader.py` converts PDFs into plain text.

The knowledge files live under `chat/knowledge/`:

- `raw/` - source PDFs
- `processed/` - extracted text files
- `chunks/` - chunked JSON files
- `embeddings/` - chunk JSON with embedding vectors
- `vector_store/` - FAISS index and metadata

## Rebuilding the knowledge base

If you add or replace source texts, rebuild the pipeline in this order:

1. Put PDFs in `chat/knowledge/raw/<category>/`
2. Convert PDFs to text with `chat/services/pdf_loader.py`
3. Chunk the text with `chat/services/chunk_maker.py`
4. Embed the chunks with `chat/services/embedder.py`
5. Build the FAISS index with `chat/services/vector_store.py`

Example:

```bash
python chat/services/pdf_loader.py
python chat/services/chunk_maker.py
python chat/services/embedder.py
python chat/services/vector_store.py
```

## Important note about the vector store

The live app loads the FAISS index and metadata through `chat/services/retriever.py`. That means the deployed runtime expects the vector store files to be available from the configured Hugging Face dataset and the embedding model used to build the index must match the model used at query time.

If you change the embedding model, rebuild both the chunk embeddings and the FAISS index together.

## Useful commands

```bash
python manage.py check
python manage.py runserver
python manage.py migrate
```

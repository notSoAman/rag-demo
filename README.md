# RAG Demo

A Django-based Retrieval-Augmented Generation (RAG) demo app that answers questions about mythology by searching a FAISS vector store and sending the retrieved context to a language model.

This README covers quick setup, environment variables, development notes, and where to find the important pieces of the codebase.

Highlights
- Django backend with HTMX for progressive updates and Alpine.js for small UI state.
- FAISS vector store for retrieval, built from chunked/embedded knowledge files in `chat/knowledge/`.
- LLM calls are performed through an API client (configured via `OPENROUTER_API_KEY` by default).

## Quick start (local)

1. Clone the repo and enter it:

```bash
git clone <repo-url>
cd rag-demo
```

2. Create and activate a Python virtualenv (recommended):

```bash
python3 -m venv .venv
source .venv/bin/activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Create a `.env` file in the project root with at least the following values:

```env
SECRET_KEY=replace-with-your-secret
DEBUG=True
OPENROUTER_API_KEY=your-openrouter-api-key
# optional
HF_TOKEN=your-hf-token-if-needed
```

5. Run migrations and start the development server:

```bash
python manage.py migrate
python manage.py runserver
```

6. Open http://127.0.0.1:8000 in the browser.

Notes:
- If you plan to use FAISS locally, `faiss-cpu` is listed in `requirements.txt`. If you have a GPU and want the GPU build, install `faiss-gpu` separately.

## Environment variables
- `SECRET_KEY` — Django secret key (required)
- `DEBUG` — `True` or `False` for dev/prod behavior
- `OPENROUTER_API_KEY` — API key used for embeddings/LLM calls (project uses OpenRouter-compatible clients)
- `HF_TOKEN` — optional token if you fetch vector store artifacts from a private Hugging Face repo

Put these into a `.env` file or export them in your shell before running the server.

## Project layout (important files)
- `config/` — Django project settings and entrypoints
- `chat/` — main app (views, models, templates, services)
	- `chat/services/` — RAG helpers: `retriever.py`, `vector_store.py`, `embedder.py`, `chunk_maker.py`, `llm.py`, `markdown.py`
	- `chat/templates/` — base templates, partials (answers, sidebar, etc.)
	- `chat/static/` — styling (Tailwind via CDN + `chat/css/theme.css`)
- `chat/knowledge/` — raw, processed, chunked, embeddings, and FAISS index data (not all files are included in the repo)
- `manage.py` — Django management CLI

## How the request flow works (high level)

1. The browser sends the user's question to `POST /ask/` (HTMX is used for in-place updates).
2. `chat.views.ask()` calls the RAG pipeline to retrieve evidence and generate an answer.
3. `chat/services/retriever.py` finds relevant passages from the FAISS index.
4. `chat/services/llm.py` formats the prompt (including retrieved passages) and calls the LLM provider.
5. The response (Markdown) is converted to HTML and returned; partials render the new answer in the chat UI.

## UI notes
- The project uses Tailwind via CDN for utilities and small custom CSS in `chat/static/chat/css/theme.css`.
- Alpine.js handles small UI state (sidebar toggles, theme state) and HTMX handles the ask flow and partial replacement.
- If you want to change the theme variables, edit `chat/static/chat/css/theme.css`.

## Rebuilding the knowledge base

If you add source texts or PDFs, rebuild the pipeline in the following order:

1. Place PDFs or source files in `chat/knowledge/raw/<category>/`.
2. Run the text extraction (`chat/services/pdf_loader.py`).
3. Chunk the processed text (`chat/services/chunk_maker.py`).
4. Create embeddings for the chunks (`chat/services/embedder.py`).
5. Build the FAISS index (`chat/services/vector_store.py`).

Example:

```bash
python chat/services/pdf_loader.py
python chat/services/chunk_maker.py
python chat/services/embedder.py
python chat/services/vector_store.py
```

## Development & debugging tips
- Create a Django superuser to access the admin: `python manage.py createsuperuser`.
- Use `python manage.py check` and `python manage.py test` when adding functionality.
- To inspect the chat rendering, open `chat/templates/partials/answer.html` and `chat/templates/base.html`.

## Production notes
- Use `gunicorn` with `whitenoise` to serve static files, and run migrations before startup.
- Collect static files with `python manage.py collectstatic` and configure environment variables appropriately.

## Useful commands

```bash
python manage.py check
python manage.py migrate
python manage.py createsuperuser
python manage.py runserver
```



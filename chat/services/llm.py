from os import getenv
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI


load_dotenv()


client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=getenv("OPENROUTER_API_KEY"),
)


def _format_context(context: Any) -> str:
    """
    Convert retrieved FAISS results into readable context for the LLM.

    Each retrieved item can contain:
        - chunk_id
        - book
        - mythology
        - source_file
        - score
        - chunk_text
    """

    if not context:
        return ""

    if isinstance(context, str):
        return context

    formatted_parts = []

    for rank, item in enumerate(context, start=1):

        if isinstance(item, dict):

            metadata = []

            if item.get("chunk_id") is not None:
                metadata.append(
                    f"chunk_id: {item['chunk_id']}"
                )

            if item.get("book"):
                metadata.append(
                    f"source: {item['book']}"
                )

            if item.get("mythology"):
                metadata.append(
                    f"mythology: {item['mythology']}"
                )

            if item.get("source_file"):
                metadata.append(
                    f"source_file: {item['source_file']}"
                )

            if item.get("score") is not None:
                metadata.append(
                    f"retrieval_score: {item['score']:.4f}"
                )

            chunk_text = item.get("chunk_text", "")

            if chunk_text:
                formatted_parts.append(
                    f"--- Retrieved passage {rank} ---\n"
                    + "\n".join(metadata)
                    + "\n\n"
                    + chunk_text
                )

        else:
            formatted_parts.append(
                f"--- Retrieved passage {rank} ---\n"
                f"{str(item)}"
            )

    return "\n\n".join(formatted_parts)


def generate_answer(question: str, context: Any = None) -> str:
    """
    Generate an answer using retrieved RAG context.

    If context isn't supplied, retrieve it from FAISS.
    """

    # Retrieve from FAISS when this function is called normally.
    if context is None:
        try:
            from .retriever import retrieve_context
        except ImportError:
            from retriever import retrieve_context

        context = retrieve_context(
            question,
            top_k=8,
        )

    formatted_context = _format_context(context)

    if not formatted_context.strip():
        return "I don't know based on the provided context."

    system_prompt = """
You are a careful mythology research assistant.

You answer questions about the mythology contained in the
retrieved passages provided by the application.

IMPORTANT RULES:

1. Use ONLY the retrieved passages as your factual source.
   Do not use your own outside knowledge to fill missing information.

2. The retrieved passages come from different books and traditions.
   Do NOT assume that all passages describe the same tradition,
   interpretation, chronology, or classification.

3. If different retrieved passages give different answers,
   DO NOT silently choose one.

   Instead:
   - identify the relevant source(s)
   - explain what each source says
   - clearly state that the sources differ if they do.

4. Pay close attention to wording such as:
   - first incarnation
   - first avatar
   - first manifestation
   - first avatar in a particular sequence
   - incarnation mentioned in a particular scripture

   These terms may not mean the same thing.

5. If the question is ambiguous and the context supports
   multiple interpretations, explain the ambiguity rather than
   pretending there is one universally correct answer.

6. If the retrieved passages genuinely do not contain enough
   information to answer the question, say:

   "I don't know based on the provided context."

7. Never invent a citation, verse number, chunk, book, or quotation.

8. When possible, mention the source book when making an important
   claim.

9. Give a concise answer first, followed by a short explanation
   based on the retrieved evidence.

10. Do not mention FAISS, embeddings, retrieval, prompts,
    or the internal implementation unless the user explicitly
    asks about the system.
"""

    user_prompt = f"""
Question:
{question}

Retrieved evidence:
{formatted_context}

Using ONLY the retrieved evidence, answer the question.

If the evidence contains conflicting or different traditions,
explicitly explain the difference rather than merging them into
one answer.
"""

    completion = client.chat.completions.create(
        model="poolside/laguna-xs-2.1:free",
        messages=[
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": user_prompt,
            },
        ],
    )

    return completion.choices[0].message.content


if __name__ == "__main__":

    test_questions = [
        "What was Lord Vishnu's first avatar?",
        "What was the first incarnation of Vishnu?",
        "Which was the first avatar of Vishnu?",
        "What is the first incarnation of Lord Vishnu?",
        "Vishnu first incarnation Matsya",
        "Matsya Vishnu avatar",
    ]

    for question in test_questions:
        print(f"\nQuestion: {question}")
        print("-" * 80)

        answer = generate_answer(question)

        print(answer)
        print("-" * 80)
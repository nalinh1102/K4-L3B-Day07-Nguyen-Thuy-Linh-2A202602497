from typing import Callable

from .store import EmbeddingStore


class KnowledgeBaseAgent:
    """
    An agent that answers questions using a vector knowledge base.

    Retrieval-augmented generation (RAG) pattern:
        1. Retrieve top-k relevant chunks from the store.
        2. Build a prompt with the chunks as context.
        3. Call the LLM to generate an answer.
    """

    def __init__(self, store: EmbeddingStore, llm_fn: Callable[[str], str]) -> None:
        self.store = store
        self.llm_fn = llm_fn

    def answer(self, question: str, top_k: int = 3) -> str:
        context_chunks = self.store.search(question, top_k=top_k)
        if not context_chunks:
            return "I could not find relevant context in the knowledge base."

        context_text = "\n\n".join(
            f"[{index + 1}] {chunk['content']}"
            for index, chunk in enumerate(context_chunks)
        )
        prompt = (
            "Answer the question using only the context below. "
            "If the information is not present, say so clearly.\n\n"
            f"Question: {question}\n\nContext:\n{context_text}\n\nAnswer:"
        )
        return self.llm_fn(prompt)

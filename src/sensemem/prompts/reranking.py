"""Retrieval instructions and the publisher's Qwen3-VL reranker template."""

QWEN3_VL_INSTRUCTION = "Given a search query, retrieve relevant candidates that answer the query."
QWEN3_VL_SYSTEM = (
    "Judge whether the Document meets the requirements based on the Query and the Instruct "
    'provided. Note that the answer can only be "yes" or "no".'
)
MEDIA_QUERY_FALLBACK = "Retrieve candidates relevant to the query media."
MEDIA_DOCUMENT_FALLBACK = "Media evidence"


def qwen3_vl_prompt(query: str, document: str) -> str:
    """Render the publisher's template without interpreting source data as a template."""
    return (
        f"<|im_start|>system\n{QWEN3_VL_SYSTEM}<|im_end|>\n"
        f"<|im_start|>user\n<Instruct>: {QWEN3_VL_INSTRUCTION}<Query>:{query}\n"
        f"<Document>:{document}<|im_end|>\n<|im_start|>assistant\n"
    )

"""
generator.py
============

This file contains the GENERATION half of RAG (the "G" in RAG).

Generation is the final step of the pipeline: once retriever.py has found
the most relevant chunks of text, this file's job is to hand those chunks
to a local language model along with the user's question, and produce a
natural-language answer.

It is worth being very clear about the difference between the TWO models
used in this whole project, because beginners often mix them up:

    1) Sentence Transformer (used in ingestion.py and retriever.py)
       Text --> Embedding (a list of numbers representing meaning)
       Purpose: SEARCH - "which stored text is closest in meaning to this?"
       It CANNOT write sentences or answer questions.

    2) Generative Transformer LLM (used ONLY in this file)
       Context + Question --> Answer (natural language text)
       Purpose: WRITE - "given this information, compose an answer."
       It does NOT do similarity search.

RAG combines both: the Sentence Transformer finds WHAT information is
relevant, and the Generative LLM turns that information into a readable
answer.
"""

from typing import List, Dict

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


# ---------------------------------------------------------------------------
# LOCAL LLM SELECTION
# ---------------------------------------------------------------------------
# We use "google/flan-t5-base" as the default local generative model.
#
# Why this model for a beginner project?
#   - It is a "text-to-text" model, meaning it was trained to follow
#     instructions like "Answer the question using this context: ...",
#     which fits our RAG prompt perfectly.
#   - It is ~250 million parameters (~1 GB download). This is small
#     enough to comfortably run on a normal laptop CPU - no GPU required -
#     while still giving reasonably coherent answers.
#   - It runs fully offline after the first download, so no API key and
#     no per-request cost.
#
# If your machine is slower, or the download/RAM usage feels too heavy,
# switch to the smaller alternative below by changing this one line:
#
#   LLM_MODEL_NAME = "google/flan-t5-small"   # ~80M params, ~300MB, faster
#                                              # but noticeably less capable
#                                              # answers on harder questions.
#
# Expected limitations either way: these are small models compared to
# something like GPT-4. They are good for straightforward, well-supported
# questions about your documents, but may occasionally give short or
# imperfect answers on very complex or ambiguous questions. That trade-off
# is what makes them realistic to run on a beginner's computer.
LLM_MODEL_NAME = "google/flan-t5-base"

MAX_ANSWER_TOKENS = 256

# Module-level cache so the (fairly large) model and tokenizer are only
# loaded into memory once per app run, not on every single question.
_tokenizer = None
_model = None


def _load_llm():
    """Load the tokenizer and model once, and reuse them across calls."""
    global _tokenizer, _model
    if _model is None:
        _tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_NAME)
        _model = AutoModelForSeq2SeqLM.from_pretrained(LLM_MODEL_NAME)
    return _tokenizer, _model


# ---------------------------------------------------------------------------
# PROMPT CONSTRUCTION
# ---------------------------------------------------------------------------
def build_prompt(question: str, retrieved_chunks: List[Dict]) -> str:
    """
    Combine the retrieved context and the user's question into a single
    prompt for the LLM.

    Args:
        question: The user's original question.
        retrieved_chunks: Output of retriever.retrieve_relevant_chunks().

    Returns:
        A single formatted prompt string.
    """
    # This is the central idea of Retrieval-Augmented Generation: instead
    # of asking the LLM to answer purely from what it memorized during
    # training (which may be outdated, generic, or simply wrong about YOUR
    # documents), we hand it the actual relevant text we just retrieved and
    # ask it to answer USING that text. This grounds the answer in real
    # source material instead of the model's imagination.
    context_text = "\n\n".join(
        f"[Source: {chunk['filename']}]\n{chunk['text']}" for chunk in retrieved_chunks
    )

    # These instructions matter as much as the context itself. Without
    # them, a small local LLM is more likely to "hallucinate" (confidently
    # invent an answer) instead of admitting the documents don't cover it.
    prompt = f"""Answer the question using ONLY the context below.
If the answer is not contained in the context, say "I could not find this information in the provided documents."
Be concise and do not make up information that isn't in the context.

Context:
{context_text}

Question: {question}

Answer:"""

    return prompt


# ---------------------------------------------------------------------------
# ANSWER GENERATION
# ---------------------------------------------------------------------------
def generate_answer(question: str, retrieved_chunks: List[Dict]) -> Dict:
    """
    Generate a final answer using a local generative LLM, grounded in the
    retrieved document chunks.

    Args:
        question: The user's original question.
        retrieved_chunks: Output of retriever.retrieve_relevant_chunks().

    Returns:
        A dictionary: {"answer": <generated text>, "sources": [<filenames>]}
    """
    if not retrieved_chunks:
        # This should normally be caught earlier by the retriever, but we
        # guard against it here too, since generation without any context
        # would defeat the entire point of RAG (it would just be a plain,
        # ungrounded chatbot at that point).
        return {
            "answer": "I could not find any relevant information in the documents to answer this question.",
            "sources": [],
        }

    prompt = build_prompt(question, retrieved_chunks)

    tokenizer, model = _load_llm()

    # Convert the text prompt into token IDs the model understands.
    # truncation=True protects us in case the combined context + question
    # is longer than the model's maximum input length.
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)

    # This is the actual "generation" step: the model reads the prompt
    # (context + question) and produces a new sequence of tokens that
    # represents its answer.
    output_tokens = model.generate(
        **inputs,
        max_new_tokens=MAX_ANSWER_TOKENS,
    )

    # Convert the generated token IDs back into human-readable text.
    answer_text = tokenizer.decode(output_tokens[0], skip_special_tokens=True)

    # We also surface WHICH source documents were used, so the user can
    # verify the answer against the original PDF themselves rather than
    # blindly trusting the model - an important habit when working with
    # any LLM output.
    unique_sources = sorted({chunk["filename"] for chunk in retrieved_chunks})

    return {"answer": answer_text, "sources": unique_sources}

"""Standardized prompt templates for grounded citation synthesis and self-correction."""

SYSTEM_PROMPT = """You are an enterprise AI assistant with access to verified company documentation.
Your mission is to provide accurate, concise, and strictly grounded answers to user questions based ONLY on the provided Reference Documents.

MANDATORY RULES:
1. Grounding: Answer ONLY based on the facts directly stated in the Reference Documents. Do not use external assumptions or ungrounded knowledge.
2. Citations: Every factual claim or metric must cite its source using the reference identifier in brackets, e.g., [Ref-1] or [Ref-2].
3. Missing Information: If the reference documents do not contain sufficient evidence to answer the question, state:
   "Based on the provided documentation, I cannot verify this information."
4. Tone & Style: Be professional, objective, direct, and clear. Avoid filler words.
"""

USER_PROMPT_TEMPLATE = """Reference Documents:
{context}

User Question: {query}

Grounded Answer:"""

CORRECTION_PROMPT_TEMPLATE = """The previous generated response contained claims that could not be verified against the Reference Documents.
Please rewrite the answer to ONLY include statements that are directly supported by the Reference Documents.
Remove any speculative, extrapolated, or ungrounded claims.

Reference Documents:
{context}

Original Question: {query}
Unverified Response: {draft_response}

Corrected Grounded Answer:"""


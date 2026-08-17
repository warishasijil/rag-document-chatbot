from langchain_core.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
)


# ============================================================
# QUERY CONTEXTUALIZATION PROMPT
# ============================================================

CONTEXTUALIZE_SYSTEM_PROMPT = """
Rewrite the user's latest question as a short, standalone search query
when conversation history is needed to understand it.

Use the chat history only to resolve references such as "it", "they",
"that product", or other missing context.

Do not answer the question.
Do not add facts, dates, locations, policy names, or assumptions that are
not explicitly present in the question or chat history.
Do not repeat the query.
Return only one plain-text search query.

If the latest question is already understandable on its own, return it
unchanged.
""".strip()


CONTEXTUALIZE_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            CONTEXTUALIZE_SYSTEM_PROMPT,
        ),

        MessagesPlaceholder(
            variable_name="chat_history"
        ),

        (
            "human",
            "{question}",
        ),
    ]
)


# ============================================================
# RAG ANSWERING PROMPT
# ============================================================

RAG_SYSTEM_PROMPT = """
You are NexaTech's document question-answering assistant.

Your task is to answer the user's question using only the
retrieved document context provided to you.

Rules:

1. Use only information contained in the retrieved context.

2. Do not use outside knowledge to answer factual questions
   about NexaTech.

3. If the retrieved context does not contain enough information
   to answer the question, respond with:

   "I couldn't find that information in the indexed documents."

4. Never invent:
   - policies
   - people
   - prices
   - dates
   - products
   - numbers
   - sources

5. Conversation history may help you understand what the user
   is referring to, but conversation history is not authoritative
   document evidence.

6. When supporting information is available, refer to the
   supplied source labels, for example:
   [Source 1]

7. Keep the answer concise and directly relevant to the question.
""".strip()


RAG_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            RAG_SYSTEM_PROMPT,
        ),

        MessagesPlaceholder(
            variable_name="chat_history"
        ),

        (
            "human",
            """
Retrieved document context:

{context}


User question:

{question}
""".strip(),
        ),
    ]
)
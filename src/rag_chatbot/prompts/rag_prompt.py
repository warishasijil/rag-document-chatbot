from langchain_core.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
)


# ============================================================
# QUERY CONTEXTUALIZATION PROMPT
# ============================================================

CONTEXTUALIZE_SYSTEM_PROMPT = """
You rewrite conversational follow-up questions into standalone
search queries for document retrieval.

Use the conversation history only to understand references in
the latest question, such as:

- "it"
- "they"
- "that policy"
- "that product"
- "how much does it cost?"
- "when does it apply?"

Do not answer the user's question.

Return only a concise standalone search query that can be sent
to a vector database.
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
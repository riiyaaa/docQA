"""Step 3: answer with citations, or say the answer isn't there."""
import re

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

NOT_FOUND = "I couldn't find that in the documents."

SYSTEM = """You answer questions using ONLY the numbered passages below.

Rules:
- After every sentence, cite the passage(s) it came from, like [1] or [2][3].
- If the passages do not contain the answer, reply with exactly: {not_found}
- Never use outside knowledge. Keep answers short and direct.

Passages:
{context}"""

PROMPT = ChatPromptTemplate.from_messages([("system", SYSTEM), ("human", "{question}")])


def source_label(doc) -> str:
    page = doc.metadata.get("page")
    return f"{doc.metadata['source']}, p. {page}" if page else doc.metadata["source"]


def format_context(docs) -> str:
    return "\n\n".join(f"[{i}] ({source_label(d)})\n{d.page_content}"
                       for i, d in enumerate(docs, start=1))


class Answerer:
    def __init__(self, llm, retriever):
        self.retriever = retriever
        # LangChain Expression Language: prompt -> model -> plain string
        self.chain = PROMPT | llm | StrOutputParser()

    def ask(self, question: str) -> dict:
        return self.answer(question, self.retriever.retrieve(question))

    def answer(self, question: str, docs) -> dict:
        """Answer from passages that were already retrieved. Split out from ask()
        so the evaluation can score retrieval and the answer on the same passages."""
        if not docs:
            return {"answer": NOT_FOUND, "sources": []}

        answer = self.chain.invoke({
            "question": question,
            "context": format_context(docs),
            "not_found": NOT_FOUND,
        }).strip()

        # Keep only citations that point at real passages; small models
        # sometimes invent [7] when only 4 passages were given.
        cited = sorted({int(n) for n in re.findall(r"\[(\d+)\]", answer)
                        if 1 <= int(n) <= len(docs)})
        sources = [{"n": n, "source": source_label(docs[n - 1]),
                    "file": docs[n - 1].metadata["source"],
                    "page": docs[n - 1].metadata.get("page"),
                    "snippet": docs[n - 1].page_content[:200]} for n in cited]
        return {"answer": answer, "sources": sources}

"""The knowledge layer: agency policy documents, chunked and searchable.

A general model knows travel norms from the internet. It does not know *this*
agency's refund windows, change-fee schedule, or manager-review thresholds — and if
it doesn't know, it invents something plausible. This module is what the answers
are grounded in, and it is what makes citations possible: you can only cite a
source you actually retrieved.

Retrieval is TF-IDF cosine similarity (scikit-learn), not neural embeddings. That is
a deliberate choice, not a shortcut: at this corpus size (a few dozen chunks drawn
from five short policy documents) a neural embedding model buys negligible recall
improvement over term-weighted lexical matching, while sentence-transformers pulls
in PyTorch — a dependency that does not fit the free tier this project deploys to.
If the corpus grew to hundreds of documents with real paraphrase gaps between how
clients write and how policy is worded, that tradeoff would flip; see the README.
"""

from __future__ import annotations

import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from app.config import KNOWLEDGE_DIR, RETRIEVAL_TOP_K
from app.models import KnowledgeChunk

MAX_CHUNK_CHARS = 900


def chunk_markdown(text: str) -> list[tuple[str, str]]:
    """Split a document into (section, passage) pairs on its `##` headings.

    Chunking at all is the point: you want to retrieve the relevant paragraph, not a
    whole document. Splitting on headings rather than a fixed character count keeps
    each chunk about one topic, so a retrieved passage reads as a complete thought
    and the section name is a citation a human can actually check.
    """
    chunks: list[tuple[str, str]] = []
    section = "Introduction"
    buffer: list[str] = []

    def flush() -> None:
        body = "\n".join(buffer).strip()
        if not body:
            return
        # Long sections are split further on blank lines so no single chunk
        # dominates the retrieved context.
        if len(body) <= MAX_CHUNK_CHARS:
            chunks.append((section, body))
            return
        current = ""
        for paragraph in re.split(r"\n\s*\n", body):
            if len(current) + len(paragraph) > MAX_CHUNK_CHARS and current:
                chunks.append((section, current.strip()))
                current = paragraph
            else:
                current = f"{current}\n\n{paragraph}" if current else paragraph
        if current.strip():
            chunks.append((section, current.strip()))

    for line in text.splitlines():
        heading = re.match(r"^##\s+(.*)", line)
        if heading:
            flush()
            buffer = []
            section = heading.group(1).strip()
        elif line.startswith("# "):
            continue
        else:
            buffer.append(line)
    flush()
    return chunks


def index_knowledge(db: Session) -> int:
    """Re-read every document in knowledge/ and rebuild the searchable index.

    Destructive and idempotent on purpose: the documents on disk are the source of
    truth, so re-running this after editing one is always safe.
    """
    db.execute(delete(KnowledgeChunk))

    pending: list[tuple[str, str, str]] = []
    for path in sorted(KNOWLEDGE_DIR.glob("*.md")):
        for section, content in chunk_markdown(path.read_text(encoding="utf-8")):
            pending.append((path.name, section, content))

    for document, section, content in pending:
        db.add(KnowledgeChunk(document=document, section=section, content=content))
    db.commit()
    return len(pending)


def retrieve(db: Session, query: str, top_k: int = RETRIEVAL_TOP_K) -> list[KnowledgeChunk]:
    """Return the passages closest to the query by TF-IDF cosine similarity, best first.

    The vectorizer is fit fresh on every call. At tens of chunks that is a
    sub-millisecond operation, and it sidesteps any question of keeping a persisted
    vocabulary in sync with the documents on disk — the documents are re-read and
    re-fit every time, so there is nothing to go stale.
    """
    chunks = list(db.scalars(select(KnowledgeChunk)))
    if not chunks:
        return []

    corpus = [f"{c.document} {c.section} {c.content}" for c in chunks]
    vectorizer = TfidfVectorizer(stop_words="english")
    matrix = vectorizer.fit_transform(corpus)
    query_vector = vectorizer.transform([query])

    scores = cosine_similarity(query_vector, matrix)[0]
    best = scores.argsort()[::-1][:top_k]
    return [chunks[i] for i in best if scores[i] > 0]


def format_for_prompt(chunks: list[KnowledgeChunk]) -> str:
    """Render retrieved passages for the prompt, each labelled with its source.

    The label is not decoration: the model is told to cite using exactly these
    document and section names, which is what makes a fabricated citation detectable.
    """
    if not chunks:
        return "(no policy documents retrieved)"
    return "\n\n".join(
        f"[document: {chunk.document} | section: {chunk.section}]\n{chunk.content}"
        for chunk in chunks
    )

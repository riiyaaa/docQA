"""Step 1: turn files into searchable chunks and store them in Chroma."""
from pathlib import Path

import docx2txt
from pypdf import PdfReader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

SUPPORTED = {".pdf", ".docx", ".txt", ".md"}


def load_file(path: Path) -> list[Document]:
    """Read one file into Documents. PDFs produce one Document per page so we
    can cite page numbers later; other formats produce a single Document."""
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        reader = PdfReader(str(path))
        return [
            Document(page_content=page.extract_text() or "",
                     metadata={"source": path.name, "page": i + 1})
            for i, page in enumerate(reader.pages)
        ]
    if suffix == ".docx":
        text = docx2txt.process(str(path))
    else:
        text = path.read_text(encoding="utf-8", errors="ignore")
    return [Document(page_content=text, metadata={"source": path.name})]


def collect_files(paths: list[str]) -> list[Path]:
    """Expand directories into the supported files inside them."""
    files = []
    for p in map(Path, paths):
        candidates = sorted(p.rglob("*")) if p.is_dir() else [p]
        files += [f for f in candidates if f.suffix.lower() in SUPPORTED]
    return files


def split_documents(docs: list[Document], chunk_size: int, chunk_overlap: int) -> list[Document]:
    """Split into overlapping chunks. The splitter tries paragraph breaks first,
    then sentences, then words, so chunks end at natural boundaries."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap, add_start_index=True)
    chunks = [c for c in splitter.split_documents(docs) if c.page_content.strip()]
    for c in chunks:
        m = c.metadata
        # A stable ID means re-ingesting the same file overwrites instead of duplicating
        m["chunk_id"] = f"{m['source']}:{m.get('page', 0)}:{m['start_index']}"
    return chunks


def ingest(paths: list[str], vectorstore, chunk_size: int, chunk_overlap: int) -> int:
    """Load, split, embed, and store. Returns the number of chunks written."""
    docs = [d for f in collect_files(paths) for d in load_file(f)]
    chunks = split_documents(docs, chunk_size, chunk_overlap)
    if chunks:
        vectorstore.add_documents(chunks, ids=[c.metadata["chunk_id"] for c in chunks])
    return len(chunks)

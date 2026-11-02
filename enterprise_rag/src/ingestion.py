import os
import io
import re
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

import pymupdf
import winocr
from PIL import Image

try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:
    from langchain.text_splitter import RecursiveCharacterTextSplitter

from .config import ROOT_DIR, DEFAULT_CHUNK_SIZE, DEFAULT_CHUNK_OVERLAP

logger = logging.getLogger(__name__)


def clean_extracted_text(text: str) -> str:
    """
    Cleans raw text extracted from PDFs/OCR:
    - Preserves headings and structural section markers.
    - Removes non-printable control characters and OCR artifacts.
    - Normalizes unicode quotes, hyphens, and whitespace.
    - Collapses multiple blank lines into double newlines.
    """
    if not text:
        return ""

    # Replace zero-width spaces and control chars
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]", "", text)

    # Normalize typographical punctuation
    text = text.replace("“", '"').replace("”", '"').replace("’", "'").replace("‘", "'")
    text = text.replace("—", " - ").replace("–", " - ")

    # Remove soft hyphens or words broken across line wraps (e.g. "com-\nputer")
    text = re.sub(r"(\w+)-\n(\w+)", r"\1\2", text)

    # Clean lines individually while preserving headings
    cleaned_lines = []
    for line in text.split("\n"):
        line_stripped = " ".join(line.split())
        if line_stripped:
            cleaned_lines.append(line_stripped)
        elif cleaned_lines and cleaned_lines[-1] != "":
            # Preserve single empty line between paragraphs
            cleaned_lines.append("")

    cleaned = "\n".join(cleaned_lines).strip()
    return cleaned


def extract_generic_metadata(filename: str, sample_text: str = "") -> Dict[str, str]:
    """
    Extracts generic enterprise metadata from filename and document content.
    Designed to work seamlessly across:
    - Enterprise policies (HR, Finance, Security, Operations)
    - Standard Operating Procedures (SOPs) and Technical Manuals
    - Institutional & Departmental publications
    """
    name_clean = Path(filename).stem.replace("_", " ").replace("-", " ")
    name_lower = filename.lower()

    # Generic Enterprise Category Detection
    if any(k in name_lower for k in ["policy", "guideline", "rule", "leave", "hr", "conduct"]):
        category = "Policy & Compliance"
    elif any(k in name_lower for k in ["security", "cyber", "malware", "forensics"]):
        category = "Security & IT"
    elif any(k in name_lower for k in ["syllabus", "curriculum", "scheme", "degree", "academic", "course"]):
        category = "Academic Curriculum"
    elif any(k in name_lower for k in ["newsletter", "bytes", "reflection", "report", "update"]):
        category = "Institutional Newsletter"
    elif any(k in name_lower for k in ["admission", "approval", "intake", "ara", "dte", "acap"]):
        category = "Admissions & Regulatory"
    elif any(k in name_lower for k in ["finance", "fee", "budget", "cost", "expense"]):
        category = "Finance & Fees"
    else:
        category = "General Enterprise Documentation"

    # Academic Year / Date Tagging if present
    ay_match = re.search(r"(20\d\d\s*[-–]\s*\d\d|\b20\d\d\b)", name_clean)
    academic_year = ay_match.group(1).replace(" ", "") if ay_match else "-"

    # Document Title derivation
    title = name_clean.title()
    # Try finding title in first line of text
    if sample_text:
        first_line = sample_text.split("\n")[0].strip()
        if 4 < len(first_line) < 100 and not first_line.startswith(("http", "www", "page")):
            title = first_line

    return {
        "document_title": title,
        "category": category,
        "academic_year": academic_year,
        "source_file": Path(filename).name,
        "source_pdf": Path(filename).name
    }


class DocumentParser:
    """
    Extracts clean text from digital and scanned PDFs.
    Uses PyMuPDF (digital layers) with native Windows OCR (winocr) fallback.
    Caches parsed text on disk in data/parsed_cache/.
    """

    def __init__(self, cache_dir: Optional[Path] = None):
        self.cache_dir = cache_dir or (ROOT_DIR / "data" / "parsed_cache")
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def parse_pdf(self, pdf_path: str) -> List[Dict[str, Any]]:
        """
        Parses a PDF file into list of clean pages: [{'page_num': 1, 'text': '...'}]
        """
        path = Path(pdf_path)
        cache_file = self.cache_dir / f"{path.stem}.json"

        # Check disk cache
        if cache_file.exists():
            try:
                with open(cache_file, "r", encoding="utf-8") as f:
                    cached_data = json.load(f)
                    # Ensure cached text is cleaned
                    for p in cached_data:
                        p["text"] = clean_extracted_text(p.get("text", ""))
                    return cached_data
            except Exception:
                pass

        if not path.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")

        pages_data = []
        doc = pymupdf.open(str(path))
        print(f"  Parsing '{path.name}' ({len(doc)} pages)...")

        for idx, page in enumerate(doc):
            raw_text = (page.get_text() or "").strip()

            # If page is scanned/image-only, use Windows OCR
            if len(raw_text) < 30:
                try:
                    pix = page.get_pixmap(dpi=150)
                    img = Image.open(io.BytesIO(pix.tobytes("png")))
                    ocr_res = winocr.recognize_pil_sync(img, lang="en")
                    raw_text = ocr_res.get("text", "").strip()
                except Exception as e:
                    logger.warning(f"OCR fallback failed on page {idx+1} of {path.name}: {e}")

            cleaned = clean_extracted_text(raw_text)
            pages_data.append({
                "page_num": idx + 1,
                "text": cleaned
            })

        # Save to disk cache
        try:
            with open(cache_file, "w", encoding="utf-8") as f:
                json.dump(pages_data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.warning(f"Could not write cache file {cache_file}: {e}")

        return pages_data


class DocumentChunker:
    """
    Enterprise-grade Semantic and Character Chunker using LangChain's RecursiveCharacterTextSplitter.
    - Preserves headings and structured boundaries.
    - Generates overlapping chunks with configurable chunk size & overlap.
    - Attaches rich enterprise metadata (source PDF, page number, chunk ID, title, category).
    - Prevents duplicates via unique deterministic chunk IDs.
    """

    def __init__(
        self,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        chunk_overlap: int = DEFAULT_CHUNK_OVERLAP
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n## ", "\n### ", "\n#### ", "\n\n", "\n", ". ", " ", ""],
            keep_separator=True
        )

    def process_document(
        self,
        filepath: str,
        pages_data: List[Dict[str, Any]],
        custom_metadata: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Converts extracted pages into structured chunks with full metadata headers.
        """
        filename = Path(filepath).name
        stem = Path(filepath).stem

        # Combine first few pages to detect overall document title
        sample_text = "\n".join(p.get("text", "") for p in pages_data[:2])
        meta = extract_generic_metadata(filename, sample_text=sample_text)
        if custom_metadata:
            meta.update(custom_metadata)

        document_chunks = []
        chunk_idx = 1
        seen_hashes = set()

        for page in pages_data:
            page_num = page.get("page_num", 1)
            raw_text = page.get("text", "").strip()
            if not raw_text:
                continue

            split_texts = self.text_splitter.split_text(raw_text)

            for sub_text in split_texts:
                sub_clean = clean_extracted_text(sub_text)
                if len(sub_clean) < 20:
                    continue

                # Deduplication within document
                content_hash = hash(sub_clean)
                if content_hash in seen_hashes:
                    continue
                seen_hashes.add(content_hash)

                chunk_id = f"{stem}_p{page_num}_c{chunk_idx}"

                # Contextual header prepended for high-precision semantic retrieval
                header = (
                    f"[{meta['category']} | Document: {meta['document_title']} | "
                    f"File: {filename} | Page: {page_num}]\n"
                )
                contextual_text = header + sub_clean

                chunk_obj = {
                    "id": chunk_id,
                    "chunk_id": chunk_id,
                    "text": contextual_text,
                    "raw_text": sub_clean,
                    "source_pdf": filename,
                    "source_file": filename,
                    "page_number": page_num,
                    "page": page_num,
                    "document_title": meta["document_title"],
                    "category": meta["category"],
                    "academic_year": meta.get("academic_year", "-"),
                    "program": meta.get("program", meta["document_title"]),
                    "payload": {
                        "chunk_id": chunk_id,
                        "source_pdf": filename,
                        "source_file": filename,
                        "page_number": page_num,
                        "page": page_num,
                        "document_title": meta["document_title"],
                        "category": meta["category"],
                        "academic_year": meta.get("academic_year", "-"),
                        "raw_text": sub_clean
                    }
                }
                document_chunks.append(chunk_obj)
                chunk_idx += 1

        return document_chunks


class IngestionPipeline:
    """
    Person 1 Orchestrator:
    - Scans directory or cache for PDF documents.
    - Cleans and parses text using OCR fallback where needed.
    - Chunks with RecursiveCharacterTextSplitter.
    - Deduplicates and structures chunks ready for FAISS and Qdrant indexing.
    """

    def __init__(
        self,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        chunk_overlap: int = DEFAULT_CHUNK_OVERLAP
    ):
        self.parser = DocumentParser()
        self.chunker = DocumentChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    def ingest_directory(
        self,
        directory_path: str,
        output_json_path: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Parses all PDFs in directory and returns all generated chunks."""
        p = Path(directory_path)
        if not p.exists():
            raise FileNotFoundError(f"Directory not found: {directory_path}")

        pdf_files = list(p.glob("*.pdf"))
        print(f"\n[Ingestion] Found {len(pdf_files)} PDF file(s) in {directory_path}")

        all_chunks = []
        for pdf_path in pdf_files:
            pages = self.parser.parse_pdf(str(pdf_path))
            chunks = self.chunker.process_document(str(pdf_path), pages)
            all_chunks.extend(chunks)
            print(f"    Extracted {len(pages)} pages -> {len(chunks)} chunks from '{pdf_path.name}'.")

        print(f"[Ingestion] Complete! Total chunks generated: {len(all_chunks)}")

        if output_json_path:
            out_p = Path(output_json_path)
            out_p.parent.mkdir(parents=True, exist_ok=True)
            with open(out_p, "w", encoding="utf-8") as f:
                json.dump(all_chunks, f, ensure_ascii=False, indent=2)
            print(f"[Ingestion] Chunks saved to: {out_p}")

        return all_chunks

    def ingest_from_parsed_cache(
        self,
        cache_dir: Optional[str] = None,
        output_json_path: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Loads all cached JSON files from data/parsed_cache/ and builds the complete chunk dataset.
        Essential for environments where PDFs have already been OCR-scanned or provided in cache.
        """
        c_dir = Path(cache_dir or (ROOT_DIR / "data" / "parsed_cache"))
        if not c_dir.exists():
            raise FileNotFoundError(f"Cache directory not found: {c_dir}")

        json_files = list(c_dir.glob("*.json"))
        print(f"\n[Ingestion] Ingesting from {len(json_files)} cached document(s) in {c_dir}...")

        all_chunks = []
        seen_chunk_ids = set()

        for jf in json_files:
            try:
                with open(jf, "r", encoding="utf-8") as f:
                    pages = json.load(f)
                pdf_filename = f"{jf.stem}.pdf"
                chunks = self.chunker.process_document(pdf_filename, pages)
                for c in chunks:
                    if c["chunk_id"] not in seen_chunk_ids:
                        seen_chunk_ids.add(c["chunk_id"])
                        all_chunks.append(c)
                print(f"    Loaded '{jf.stem}' ({len(pages)} pages) -> {len(chunks)} chunks.")
            except Exception as e:
                logger.error(f"Error processing cached file {jf}: {e}")

        print(f"[Ingestion] Total unified chunks generated: {len(all_chunks)}")

        if output_json_path:
            out_p = Path(output_json_path)
            out_p.parent.mkdir(parents=True, exist_ok=True)
            with open(out_p, "w", encoding="utf-8") as f:
                json.dump(all_chunks, f, ensure_ascii=False, indent=2)
            print(f"[Ingestion] Saved all chunks to: {out_p}")

        return all_chunks

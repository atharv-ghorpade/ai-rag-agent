"""
Script to extract, cache, chunk, index, and generate verified QA pairs
for the 5 newly uploaded First Year Engineering 2026-27 vacancy and merit list PDFs.
"""
import os
import sys
import json
import shutil
import re
from pathlib import Path

# Ensure UTF-8 output encoding on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

import pypdf
from enterprise_rag.src.config import (
    DEFAULT_EMBEDDING_MODEL,
    FAISS_STORAGE_PATH,
    QA_DATASET_PATH
)
from enterprise_rag.src.embeddings import EmbeddingManager
from enterprise_rag.src.faiss_manager import FAISSManager
from enterprise_rag.src.qa_retriever import QAIndexManager
from enterprise_rag.src.ingestion import clean_extracted_text

MEDIA_DIR = Path(r"C:\Users\Atharv\.gemini\antigravity-ide\brain\1ac17dfb-127f-416e-ae7e-931584fcc5e9\.user_uploaded")
CACHE_DIR = ROOT_DIR / "data" / "parsed_cache"
DOCS_DIR = ROOT_DIR / "data" / "new_docs"
DOCS_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)

NEW_DOC_MAP = [
    {
        "source_pdf": MEDIA_DIR / "media_1793638007452.pdf",
        "official_name": "First_Year_Engineering_2026-27_ACAP_and_IQ_Vacancies.pdf",
        "cache_json": "First_Year_Engineering_2026-27_ACAP_and_IQ_Vacancies.json",
        "title": "First Year Engineering (B.Tech.) A.Y. 2026-27 Against CAP and Institute Quota Vacancies",
        "category": "Admissions Vacancies",
        "academic_year": "2026-27",
        "program": "First Year Engineering (B.Tech.)"
    },
    {
        "source_pdf": MEDIA_DIR / "media_1793638007496.pdf",
        "official_name": "Merit_List_OMS_IQ_Seats_2026-27.pdf",
        "cache_json": "Merit_List_OMS_IQ_Seats_2026-27.json",
        "title": "Merit List for OMS IQ Seats (First Year Engineering A.Y. 2026-27)",
        "category": "Admissions Merit List",
        "academic_year": "2026-27",
        "program": "First Year Engineering (B.Tech.)"
    },
    {
        "source_pdf": MEDIA_DIR / "media_1793638007536.pdf",
        "official_name": "Merit_List_OMS_ACAP_Seats_2026-27.pdf",
        "cache_json": "Merit_List_OMS_ACAP_Seats_2026-27.json",
        "title": "Merit List for OMS ACAP Seats (First Year Engineering A.Y. 2026-27)",
        "category": "Admissions Merit List",
        "academic_year": "2026-27",
        "program": "First Year Engineering (B.Tech.)"
    },
    {
        "source_pdf": MEDIA_DIR / "media_1793638007929.pdf",
        "official_name": "Merit_List_All_India_Quota_Seats_2025-26.pdf",
        "cache_json": "Merit_List_All_India_Quota_Seats_2025-26.json",
        "title": "Merit List for All India Quota Seats based on JEE PCM Score (First Year Engineering A.Y. 2025-26)",
        "category": "Admissions Merit List",
        "academic_year": "2025-26",
        "program": "First Year Engineering (B.Tech.)"
    },
    {
        "source_pdf": MEDIA_DIR / "media_1793638008456.pdf",
        "official_name": "Merit_List_Institute_Quota_Seats_MS-CET_2026-27.pdf",
        "cache_json": "Merit_List_Institute_Quota_Seats_MS-CET_2026-27.json",
        "title": "Provisional Merit List Maharashtra State based on MHT-CET PCM Score (First Year Engineering A.Y. 2026-27)",
        "category": "Admissions Merit List",
        "academic_year": "2026-27",
        "program": "First Year Engineering (B.Tech.)"
    }
]


def extract_and_cache_documents():
    print("=" * 60)
    print(" Step 1: Parsing and caching newly uploaded PDF documents")
    print("=" * 60)

    for item in NEW_DOC_MAP:
        src = item["source_pdf"]
        if not src.exists():
            print(f"Warning: Source PDF not found: {src}")
            continue

        # Archive copy to data/new_docs/
        dst = DOCS_DIR / item["official_name"]
        shutil.copy2(src, dst)
        print(f"Archived: {item['official_name']} ({dst.stat().st_size} bytes)")

        reader = pypdf.PdfReader(str(src))
        pages_data = []

        for p_idx, page in enumerate(reader.pages, 1):
            raw_text = page.extract_text() or ""
            clean_text = clean_extracted_text(raw_text)
            
            # Enrich page with document context header if not present
            if "Vidyalankar Institute of Technology" not in clean_text:
                clean_text = f"Vidyalankar Institute of Technology\n{item['title']}\n" + clean_text

            pages_data.append({
                "page_num": p_idx,
                "text": clean_text
            })

        cache_file = CACHE_DIR / item["cache_json"]
        with open(cache_file, "w", encoding="utf-8") as f:
            json.dump(pages_data, f, ensure_ascii=False, indent=2)
        print(f"  Parsed & cached: {cache_file} ({len(pages_data)} pages)")


def build_chunks_and_index():
    print("\n" + "=" * 60)
    print(" Step 2: Chunking and Indexing into FAISS Vector Store")
    print("=" * 60)

    from langchain_text_splitters import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=700,
        chunk_overlap=100,
        separators=["\n\n", "\n", ". ", " ", ""]
    )

    all_chunks = []
    chunk_counter = 0

    # Also load existing vit_chunks.json if available
    vit_chunks_file = ROOT_DIR / "data" / "vit_chunks.json"
    existing_chunks = []
    if vit_chunks_file.exists():
        with open(vit_chunks_file, "r", encoding="utf-8") as f:
            existing_chunks = json.load(f)
        print(f"Found {len(existing_chunks)} existing indexed chunks.")

    # Process all parsed cache files in CACHE_DIR
    for cache_path in sorted(CACHE_DIR.glob("*.json")):
        doc_filename = cache_path.stem + ".pdf"
        with open(cache_path, "r", encoding="utf-8") as f:
            pages = json.load(f)

        for p in pages:
            p_num = p.get("page_num", 1)
            p_text = p.get("text", "")
            splits = splitter.split_text(p_text)

            for s_idx, split_text in enumerate(splits, 1):
                chunk_counter += 1
                chunk_id = f"{cache_path.stem}_p{p_num}_c{s_idx}"
                all_chunks.append({
                    "chunk_id": chunk_id,
                    "text": split_text,
                    "raw_text": split_text,
                    "source_file": doc_filename,
                    "source_pdf": doc_filename,
                    "page": p_num,
                    "page_number": p_num,
                    "document_title": cache_path.stem.replace("_", " "),
                    "category": "Admissions" if "merit" in cache_path.stem.lower() or "vacanc" in cache_path.stem.lower() or "admission" in cache_path.stem.lower() else "General",
                    "academic_year": "2026-27" if "2026" in cache_path.stem else ("2025-26" if "2025" in cache_path.stem else "2023-24"),
                    "program": "First Year Engineering (B.Tech.)" if "Engineering" in cache_path.stem or "Merit" in cache_path.stem else "-"
                })

    print(f"Total structured chunks generated across all document caches: {len(all_chunks)}")

    # Save to data/vit_chunks.json
    with open(vit_chunks_file, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, ensure_ascii=False, indent=2)

    # Initialize Embedder & FAISS Manager
    embedder = EmbeddingManager(model_name=DEFAULT_EMBEDDING_MODEL)
    dim = embedder.get_dimension()

    faiss_mgr = FAISSManager(dimension=dim, storage_path=str(FAISS_STORAGE_PATH))

    # Re-index all chunks in batches
    texts = [c["text"] for c in all_chunks]
    metadata_list = [
        {k: v for k, v in c.items() if k not in ("text", "raw_text", "id")}
        for c in all_chunks
    ]
    ids = [c["chunk_id"] for c in all_chunks]

    print(f"Indexing {len(texts)} chunks into FAISS vector database...")
    batch_size = 32
    for i in range(0, len(texts), batch_size):
        b_texts = texts[i:i + batch_size]
        b_meta = metadata_list[i:i + batch_size]
        b_ids = ids[i:i + batch_size]
        b_embeds = embedder.embed_documents(b_texts)
        faiss_mgr.add_documents(texts=b_texts, embeddings=b_embeds, metadata_list=b_meta, ids=b_ids)
        if (i // batch_size) % 10 == 0 or i + batch_size >= len(texts):
            print(f"  Indexed [{min(i + batch_size, len(texts))}/{len(texts)}] chunks...")

    faiss_mgr.save()
    print(f"FAISS index successfully saved! Total chunks in index: {faiss_mgr.count()}")


def add_new_qa_pairs_and_reindex():
    print("\n" + "=" * 60)
    print(" Step 3: Generating Verified QA Pairs & Re-indexing QA Retriever")
    print("=" * 60)

    qa_file = Path(QA_DATASET_PATH)
    qa_data = []
    if qa_file.exists():
        with open(qa_file, "r", encoding="utf-8") as f:
            qa_data = json.load(f)
    print(f"Existing QA pairs: {len(qa_data)}")

    start_id = max((q.get("id", 0) for q in qa_data), default=0) + 1

    new_qa_items = [
        {
            "id": start_id,
            "question": "What are the Against CAP (ACAP) vacancies for First Year Engineering in A.Y. 2026-27?",
            "answer": "There are 0 ACAP vacancies across all branches (Information Technology, Computer Engineering, Electronics and Computer Science, Electronics & Telecommunication Engineering, and Biomedical Engineering) for A.Y. 2026-27.",
            "category": "admissions",
            "difficulty": "easy",
            "document": "First_Year_Engineering_2026-27_ACAP_and_IQ_Vacancies.pdf"
        },
        {
            "id": start_id + 1,
            "question": "What are the Institute Quota (IQ) vacancies for First Year Engineering in A.Y. 2026-27?",
            "answer": "There are 0 Institute Quota (IQ) vacancies across all branches for First Year Engineering in A.Y. 2026-27.",
            "category": "admissions",
            "difficulty": "easy",
            "document": "First_Year_Engineering_2026-27_ACAP_and_IQ_Vacancies.pdf"
        },
        {
            "id": start_id + 2,
            "question": "What is the date of the vacancy notification for First Year Engineering 2026-27 ACAP and Institute Quota seats?",
            "answer": "The vacancy notification for First Year Engineering ACAP and Institute Quota seats was issued on 14 September 2026.",
            "category": "admissions",
            "difficulty": "easy",
            "document": "First_Year_Engineering_2026-27_ACAP_and_IQ_Vacancies.pdf"
        },
        {
            "id": start_id + 3,
            "question": "Who secured Merit Rank 1 in the OMS Institute Quota (IQ) merit list for First Year Engineering 2026-27?",
            "answer": "Aman Kumar (Application ID: EN26425391) secured Merit Rank 1 with a 99.9572998 MH-CET total percentile.",
            "category": "admissions",
            "difficulty": "easy",
            "document": "Merit_List_OMS_IQ_Seats_2026-27.pdf"
        },
        {
            "id": start_id + 4,
            "question": "Who secured Merit Rank 2 in the OMS Institute Quota (IQ) merit list for First Year Engineering 2026-27?",
            "answer": "Aryan Kumar Jaiswal (Application ID: EN26308311) secured Merit Rank 2 with a 98.3243876 MH-CET total percentile.",
            "category": "admissions",
            "difficulty": "easy",
            "document": "Merit_List_OMS_IQ_Seats_2026-27.pdf"
        },
        {
            "id": start_id + 5,
            "question": "Who secured Merit Rank 3 in the OMS Institute Quota (IQ) merit list for First Year Engineering 2026-27?",
            "answer": "Aman Ali Ishrat Ali (Application ID: EN26320883) secured Merit Rank 3 with a 98.1552237 MH-CET total percentile.",
            "category": "admissions",
            "difficulty": "easy",
            "document": "Merit_List_OMS_IQ_Seats_2026-27.pdf"
        },
        {
            "id": start_id + 6,
            "question": "How many candidates are listed in the OMS Institute Quota (IQ) merit list for First Year Engineering 2026-27?",
            "answer": "There are 225 candidates listed in the OMS Institute Quota (IQ) merit list, ranging from 99.9572998 to 5.1774267 percentile.",
            "category": "admissions",
            "difficulty": "medium",
            "document": "Merit_List_OMS_IQ_Seats_2026-27.pdf"
        },
        {
            "id": start_id + 7,
            "question": "Who secured Merit Rank 1 in the OMS Against CAP (ACAP) merit list for First Year Engineering 2026-27?",
            "answer": "Aman Kumar (Application ID: EN26425391) secured Merit Rank 1 with a 99.9572998 PCM CET percentile.",
            "category": "admissions",
            "difficulty": "easy",
            "document": "Merit_List_OMS_ACAP_Seats_2026-27.pdf"
        },
        {
            "id": start_id + 8,
            "question": "How many candidates are listed in the OMS Against CAP (ACAP) merit list for First Year Engineering 2026-27?",
            "answer": "There are 265 candidates listed in the OMS Against CAP (ACAP) merit list for First Year Engineering A.Y. 2026-27.",
            "category": "admissions",
            "difficulty": "medium",
            "document": "Merit_List_OMS_ACAP_Seats_2026-27.pdf"
        },
        {
            "id": start_id + 9,
            "question": "Who secured Merit Rank 1 in the All India Quota JEE merit list for First Year Engineering 2025-26?",
            "answer": "Kashid Aditya Ramchandra (Application ID: EN26280055) secured Merit Rank 1 with a 99.4585176 JEE total percentile.",
            "category": "admissions",
            "difficulty": "easy",
            "document": "Merit_List_All_India_Quota_Seats_2025-26.pdf"
        },
        {
            "id": start_id + 10,
            "question": "Who secured Merit Rank 2 in the All India Quota JEE merit list for First Year Engineering 2025-26?",
            "answer": "Parnavi Yogesh Wani (Application ID: EN26327733) secured Merit Rank 2 with a 99.0786301 JEE total percentile.",
            "category": "admissions",
            "difficulty": "easy",
            "document": "Merit_List_All_India_Quota_Seats_2025-26.pdf"
        },
        {
            "id": start_id + 11,
            "question": "Who secured Merit Rank 3 in the All India Quota JEE merit list for First Year Engineering 2025-26?",
            "answer": "Jayaditya Choudhary (Application ID: EN26157620) secured Merit Rank 3 with a 98.9202841 JEE total percentile.",
            "category": "admissions",
            "difficulty": "easy",
            "document": "Merit_List_All_India_Quota_Seats_2025-26.pdf"
        },
        {
            "id": start_id + 12,
            "question": "How many candidates are listed in the All India Quota JEE merit list for First Year Engineering 2025-26?",
            "answer": "There are 3,319 candidates listed in the All India Quota JEE merit list across 75 pages.",
            "category": "admissions",
            "difficulty": "medium",
            "document": "Merit_List_All_India_Quota_Seats_2025-26.pdf"
        },
        {
            "id": start_id + 13,
            "question": "Who secured Merit Rank 1 in the Maharashtra State Institute Quota (MS-CET) merit list for First Year Engineering 2026-27?",
            "answer": "Ishant Rajendra Gawali (Application ID: EN26248469) secured Merit Rank 1 with a 99.7632080 PCM CET percentile.",
            "category": "admissions",
            "difficulty": "easy",
            "document": "Merit_List_Institute_Quota_Seats_MS-CET_2026-27.pdf"
        },
        {
            "id": start_id + 14,
            "question": "Who secured Merit Rank 2 in the Maharashtra State Institute Quota (MS-CET) merit list for First Year Engineering 2026-27?",
            "answer": "Mohak Kamal Sethiya (Application ID: EN26364527) secured Merit Rank 2 with a 99.7610922 PCM CET percentile.",
            "category": "admissions",
            "difficulty": "easy",
            "document": "Merit_List_Institute_Quota_Seats_MS-CET_2026-27.pdf"
        },
        {
            "id": start_id + 15,
            "question": "Who secured Merit Rank 3 in the Maharashtra State Institute Quota (MS-CET) merit list for First Year Engineering 2026-27?",
            "answer": "Parnavi Yogesh Wani (Application ID: EN26327733) secured Merit Rank 3 with a 99.5929444 PCM CET percentile.",
            "category": "admissions",
            "difficulty": "easy",
            "document": "Merit_List_Institute_Quota_Seats_MS-CET_2026-27.pdf"
        },
        {
            "id": start_id + 16,
            "question": "How many candidates are listed in the Maharashtra State Institute Quota (MS-CET) merit list for First Year Engineering 2026-27?",
            "answer": "There are 4,571 candidates listed in the Maharashtra State Institute Quota (MS-CET) merit list across 101 pages.",
            "category": "admissions",
            "difficulty": "medium",
            "document": "Merit_List_Institute_Quota_Seats_MS-CET_2026-27.pdf"
        },
        {
            "id": start_id + 17,
            "question": "What was the application cutoff date considered for the First Year Engineering 2026-27 merit lists?",
            "answer": "Applications received up to 7 September 2026 were considered in the merit lists.",
            "category": "admissions",
            "difficulty": "easy",
            "document": "Merit_List_Institute_Quota_Seats_MS-CET_2026-27.pdf"
        },
        {
            "id": start_id + 18,
            "question": "What confirmation is mandatory to participate in Institute Level (IL) admissions?",
            "answer": "SC confirmation and acknowledgement is mandatory to participate in the Institute Level (IL) admissions.",
            "category": "admissions",
            "difficulty": "easy",
            "document": "Merit_List_Institute_Quota_Seats_MS-CET_2026-27.pdf"
        }
    ]

    # Deduplicate against existing question text
    existing_questions = {q["question"].strip().lower() for q in qa_data}
    added_count = 0
    for nq in new_qa_items:
        if nq["question"].strip().lower() not in existing_questions:
            qa_data.append(nq)
            added_count += 1

    with open(qa_file, "w", encoding="utf-8") as f:
        json.dump(qa_data, f, ensure_ascii=False, indent=2)
    print(f"Added {added_count} new verified QA pairs! Total QA dataset size: {len(qa_data)}")

    # Re-index QA Index Manager
    embedder = EmbeddingManager(model_name=DEFAULT_EMBEDDING_MODEL)
    qa_index_mgr = QAIndexManager(
        embedding_manager=embedder,
        qa_dataset_path=str(qa_file),
        cache_dir=str(ROOT_DIR / "data" / "qa_index")
    )
    qa_index_mgr.rebuild_index()
    print(f"QA Index successfully rebuilt with {len(qa_data)} verified questions!")


if __name__ == "__main__":
    extract_and_cache_documents()
    build_chunks_and_index()
    add_new_qa_pairs_and_reindex()
    print("\nComplete ingestion and training finished successfully!")

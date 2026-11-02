"""
Comprehensive Ingestion, Chunking, Indexing & QA Expansion Script
Indexes all 20 parsed documents including:
- BTech_Computer_Engineering_Honours_Minor_R2022 (87 pages)
- BYTES_Oct2025_Vol17_Issue01 (8 pages)
- BYTES_Jan2026_Vol17_Issue02 (8 pages)
- BYTES_July2026_Vol17_Issue04 (8 pages)
- All Merit Lists, Vacancies, and Admissions policies
"""

import sys
import json
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from enterprise_rag.src.ingestion import IngestionPipeline
from enterprise_rag.src.embeddings import EmbeddingManager
from enterprise_rag.src.faiss_manager import FAISSManager
from enterprise_rag.src.qa_retriever import QAIndexManager

def main():
    print("=" * 60)
    print("1. Ingesting and chunking all documents from data/parsed_cache/...")
    print("=" * 60)
    
    ingest = IngestionPipeline()
    chunks = ingest.ingest_from_parsed_cache(
        cache_dir=str(ROOT_DIR / "data" / "parsed_cache"),
        output_json_path=str(ROOT_DIR / "data" / "vit_chunks.json")
    )
    print(f"Total unified chunks created: {len(chunks)}")
    
    print("\n" + "=" * 60)
    print("2. Expanding QA Dataset with verified pairs for newly added documents...")
    print("=" * 60)
    
    qa_path = ROOT_DIR / "data" / "qa_dataset.json"
    with open(qa_path, "r", encoding="utf-8") as f:
        qa_data = json.load(f)
        
    existing_questions = {q["question"].lower().strip() for q in qa_data}
    start_id = max((q["id"] for q in qa_data), default=135) + 1
    
    new_qa_entries = [
        # BTech Honours/Minor (R-2022)
        {
            "id": start_id,
            "question": "What is the basic eligibility criteria for opting for Honours or Minor Degree in Computer Engineering (R-2022)?",
            "answer": "Basic eligibility for opting for Honours or Minor is a minimum CGPA of 6.75 at the end of the 4th semester and earning 80 credits from Semester 1 to Semester 4 (41 credits for Direct Second Year students).",
            "category": "curriculum",
            "difficulty": "easy",
            "document": "BTech_Computer_Engineering_Honours_Minor_R2022.pdf"
        },
        {
            "id": start_id + 1,
            "question": "How many additional credits are required to obtain an Honours or Minor Degree at VIT?",
            "answer": "Learners can obtain an Honours or Minor Degree by completing 18 additional credits over and above the minimum credits required for the B.Tech degree.",
            "category": "curriculum",
            "difficulty": "easy",
            "document": "BTech_Computer_Engineering_Honours_Minor_R2022.pdf"
        },
        {
            "id": start_id + 2,
            "question": "When can students opt for Honours or Minor specialisations in Computer Engineering?",
            "answer": "Learners opt for Honours or Minor specialisations during the break of Semester 5 and Semester 6.",
            "category": "curriculum",
            "difficulty": "easy",
            "document": "BTech_Computer_Engineering_Honours_Minor_R2022.pdf"
        },
        {
            "id": start_id + 3,
            "question": "What are the courses under Next-Gen Artificial Intelligence and Machine Learning Honours track?",
            "answer": "The courses under Next-Gen AI & ML are: Ethics in AI (HMCE01T/P, Semester 6), Scalable ML and BDA (HMCE02T/P, Semester 7), and Generative AI Models (HMCE03T/P, Semester 8).",
            "category": "curriculum",
            "difficulty": "medium",
            "document": "BTech_Computer_Engineering_Honours_Minor_R2022.pdf"
        },
        {
            "id": start_id + 4,
            "question": "What are the courses under Next-Gen Data Science Honours track?",
            "answer": "The courses under Next-Gen Data Science are: Responsible AI and Data Ethics (HMCE04T/P, Semester 6), Cloud native & BDA (HMCE05T/P, Semester 7), and Realtime Data Analytics (HMCE06T/P, Semester 8).",
            "category": "curriculum",
            "difficulty": "medium",
            "document": "BTech_Computer_Engineering_Honours_Minor_R2022.pdf"
        },
        {
            "id": start_id + 5,
            "question": "What is the course code and syllabus structure for Ethics in AI?",
            "answer": "Ethics in AI has course code HMCE01T (theory, 2 credits, 75 marks) and HMCE01P (lab, 1 credit, 50 marks). It covers bias and fairness, privacy challenges, explainable AI (XAI), and legal frameworks like GDPR.",
            "category": "curriculum",
            "difficulty": "medium",
            "document": "BTech_Computer_Engineering_Honours_Minor_R2022.pdf"
        },
        {
            "id": start_id + 6,
            "question": "What is the course code and syllabus structure for Generative AI Models?",
            "answer": "Generative AI Models has course code HMCE03T (theory, 2 credits, 75 marks) and HMCE03P (lab, 1 credit, 50 marks). It covers GANs, VAEs, Diffusion Models (DDPM, Stable Diffusion), and LLMs (Transformers, LoRA, RLHF).",
            "category": "curriculum",
            "difficulty": "medium",
            "document": "BTech_Computer_Engineering_Honours_Minor_R2022.pdf"
        },
        {
            "id": start_id + 7,
            "question": "What courses are included in the Cyber Security Honours/Minor track in Computer Engineering?",
            "answer": "The Cyber Security track includes Cryptography and Network Security (HMCE13T/P, Sem 6), Web Application Security (HMCE14T/P, Sem 7), and Digital Forensics (HMCE15T/P, Sem 8).",
            "category": "curriculum",
            "difficulty": "medium",
            "document": "BTech_Computer_Engineering_Honours_Minor_R2022.pdf"
        },
        {
            "id": start_id + 8,
            "question": "What is the syllabus scheme and credit distribution for the Honours/Minor Degree?",
            "answer": "The 18 credits consist of: HM01 Industry Interaction (1 credit), Course 1 Theory & Lab (3 credits), HM02 Survey Report/Paper (2 credits), Course 2 Theory & Lab (3 credits), HM03 Seminar (2 credits), Course 3 Theory & Lab (3 credits), and HM04 Capstone Project (4 credits).",
            "category": "curriculum",
            "difficulty": "medium",
            "document": "BTech_Computer_Engineering_Honours_Minor_R2022.pdf"
        },
        {
            "id": start_id + 9,
            "question": "What are the minor programmes offered by other departments to Computer Engineering students?",
            "answer": "Minor programmes offered include Next-Gen Cyber Security (Information Technology), UI/UX (Information Technology), Blockchain (Information Technology), and Next-Gen Internet of Things (Electronics and Telecommunication Engineering).",
            "category": "curriculum",
            "difficulty": "medium",
            "document": "BTech_Computer_Engineering_Honours_Minor_R2022.pdf"
        },
        # BYTES October 2025
        {
            "id": start_id + 10,
            "question": "Who completed their Ph.D. in Technology in the October 2025 issue of BYTES?",
            "answer": "Prof. Snehal Andhare successfully completed her Ph.D. in Technology as celebrated under Staff Appreciation in the October 2025 issue of BYTES.",
            "category": "faculty_achievement",
            "difficulty": "easy",
            "document": "BYTES_Oct2025_Vol17_Issue01.pdf"
        },
        {
            "id": start_id + 11,
            "question": "When was the Dr. Bhau Daji Lad Museum visit conducted and who organized it?",
            "answer": "The visit to Dr. Bhau Daji Lad Museum in Byculla, Mumbai was organized on 25th August 2025 by HCC in association with GDG.",
            "category": "events",
            "difficulty": "easy",
            "document": "BYTES_Oct2025_Vol17_Issue01.pdf"
        },
        {
            "id": start_id + 12,
            "question": "Who are the department staff featured in the October 2025 issue of BYTES?",
            "answer": "Prof. Kshitija Kulkarni (Assistant Professor, MS, 2 years experience, specialization in Cyber Security) and Prof. Devendra Pandit (Assistant Professor, BE, 30 years experience, specialization in Programming and Data Structures).",
            "category": "faculty",
            "difficulty": "easy",
            "document": "BYTES_Oct2025_Vol17_Issue01.pdf"
        },
        {
            "id": start_id + 13,
            "question": "Who served as General Secretary of the Student Council in October 2025?",
            "answer": "Samudragupta Jejurkar (TE-C) served as General Secretary of the Student Council under the mentorship of Prof. Sachin Deshpande.",
            "category": "student",
            "difficulty": "easy",
            "document": "BYTES_Oct2025_Vol17_Issue01.pdf"
        },
        {
            "id": start_id + 14,
            "question": "What were the major events organized under GDG Flagship Spectrum 25 in August 2025?",
            "answer": "GDG Flagship Spectrum 25 included Launchpad (18th-19th August 2025, prize pool ₹8,000), GSoC Guidance Session (21st August 2025 with mentors Sanjay Sargam and Chaitanya Shahare), and Hackbuild Hackathon (23rd August 2025, prize pool ₹40,000+).",
            "category": "events",
            "difficulty": "medium",
            "document": "BYTES_Oct2025_Vol17_Issue01.pdf"
        },
        {
            "id": start_id + 15,
            "question": "What movie was screened during CSI-VIT Flagship FilmFiesta on 24th September 2025?",
            "answer": "The critically acclaimed film Ford vs Ferrari was screened during CSI-VIT Flagship FilmFiesta on 24th September 2025 in the auditorium.",
            "category": "events",
            "difficulty": "easy",
            "document": "BYTES_Oct2025_Vol17_Issue01.pdf"
        },
        {
            "id": start_id + 16,
            "question": "Who wrote the article 'Cybersecurity in the Age of AI: Attacks and Defences' in BYTES?",
            "answer": "Poorva Kale (TE-C) authored the student article 'Cybersecurity in the Age of AI: Attacks and Defences' in the October 2025 issue of BYTES.",
            "category": "student",
            "difficulty": "easy",
            "document": "BYTES_Oct2025_Vol17_Issue01.pdf"
        },
        # BYTES January 2026
        {
            "id": start_id + 17,
            "question": "Who secured Second Rank at the 20th Aavishkar Inter-Collegiate Research Convention?",
            "answer": "Mr. Tejas Pradip Pawar from second year Computer Engineering secured Second Rank at the 20th Aavishkar Inter-Collegiate Research Convention (AY 2025-26) under the mentorship of Dr. Amit K. Nerurkar.",
            "category": "student_achievement",
            "difficulty": "easy",
            "document": "BYTES_Jan2026_Vol17_Issue02.pdf"
        },
        {
            "id": start_id + 18,
            "question": "Which students were selected for the CHANAKYA Fellowship Program by TIH Foundation at IIT Bombay?",
            "answer": "Ms. Devanshi Mahajan, Ms. Sakshi Patil (guided by Dr. Amit K. Nerurkar), and Mr. Sheshank Singh (guided by Dr. Sheetal M.) were selected for the CHANAKYA Fellowship, each receiving ₹10,000 per month (₹1,00,000 over 10 months).",
            "category": "student_achievement",
            "difficulty": "medium",
            "document": "BYTES_Jan2026_Vol17_Issue02.pdf"
        },
        {
            "id": start_id + 19,
            "question": "What was the project topic for the CHANAKYA Fellowship recipients at VIT?",
            "answer": "The project focuses on the development of an image-based crop disease detection system leveraging AI/ML techniques and computer vision.",
            "category": "research",
            "difficulty": "easy",
            "document": "BYTES_Jan2026_Vol17_Issue02.pdf"
        },
        {
            "id": start_id + 20,
            "question": "Who is Deep Shahane and where does he currently work?",
            "answer": "Deep Shahane is an alumnus of Computer Engineering from VIT who completed his Master's in Computer Science at University at Buffalo and currently works as a Research Assistant at the SAIR Lab.",
            "category": "alumni",
            "difficulty": "easy",
            "document": "BYTES_Jan2026_Vol17_Issue02.pdf"
        },
        {
            "id": start_id + 21,
            "question": "Who was the guest speaker for the CSI-VIT expert lecture on Scalable Anomaly Detection Across Distributed Architectures?",
            "answer": "Ms. Ridhima Tambde, a Data Scientist - Bot & Abuse, delivered the expert lecture on 17th October 2025 at the VIT Auditorium.",
            "category": "events",
            "difficulty": "easy",
            "document": "BYTES_Jan2026_Vol17_Issue02.pdf"
        },
        # BYTES July 2026
        {
            "id": start_id + 22,
            "question": "Which research paper won the Best Paper Presentation Award at the 9th ICMDRSE-2026 in Kuala Lumpur, Malaysia?",
            "answer": "The paper 'A Review of Secure Health Tracking Using Smart Technology' co-authored by Dr. Ravindra Sangle, Dr. Amit K. Nerurkar, Dr. Swapnil Sonawane, and Dr. Sanjeev Dwivedi received the Best Paper Presentation (Virtual) Award.",
            "category": "faculty_achievement",
            "difficulty": "medium",
            "document": "BYTES_July2026_Vol17_Issue04.pdf"
        },
        {
            "id": start_id + 23,
            "question": "Who is Srishti Agrawal and what are her academic achievements?",
            "answer": "Srishti Agrawal is a 2021 Computer Engineering alumna from VIT, currently a PhD student in Human-Computer Interaction (HCI) at Purdue University and Doctoral Researcher at the DVC Lab, with publications at CHI 2025, NeurIPS 2024, and ASSETS 2023.",
            "category": "alumni",
            "difficulty": "medium",
            "document": "BYTES_July2026_Vol17_Issue04.pdf"
        },
        {
            "id": start_id + 24,
            "question": "Who are the department executive staff featured in the July 2026 issue of BYTES?",
            "answer": "Mrs. Sonali Surve (Department Executive, B.Com, 9 years experience, Excel & Google Workspace) and Ms. Sayli Patil (Department Executive, BMS, 1 year experience, Tally ERP 9 with GST).",
            "category": "staff",
            "difficulty": "easy",
            "document": "BYTES_July2026_Vol17_Issue04.pdf"
        },
        {
            "id": start_id + 25,
            "question": "What were the flagship competitions organized under CSI-VIT ENTHUSIA in March-April 2026?",
            "answer": "CSI-VIT ENTHUSIA included Design Paradox (UI/UX design), MindMare (strategic decision-making), Parallax (hacker-escape puzzle rooms), and Grid Rush (Sudoku-based algorithmic coding).",
            "category": "events",
            "difficulty": "medium",
            "document": "BYTES_July2026_Vol17_Issue04.pdf"
        }
    ]
    
    added_count = 0
    for entry in new_qa_entries:
        if entry["question"].lower().strip() not in existing_questions:
            qa_data.append(entry)
            existing_questions.add(entry["question"].lower().strip())
            added_count += 1
            
    with open(qa_path, "w", encoding="utf-8") as f:
        json.dump(qa_data, f, indent=2, ensure_ascii=False)
    print(f"Added {added_count} new QA pairs (Total: {len(qa_data)})")
    
    print("\n" + "=" * 60)
    print("3. Embedding & Rebuilding Vector Index (FAISS)...")
    print("=" * 60)
    
    embedder = EmbeddingManager()
    storage_dir = ROOT_DIR / "data" / "faiss_index"
    for f in [storage_dir / "index.faiss", storage_dir / "docstore.json"]:
        if f.exists():
            f.unlink()
    faiss_mgr = FAISSManager(dimension=embedder.get_dimension(), storage_path=str(storage_dir))
    
    texts = [c["text"] for c in chunks]
    metadata_list = [
        {k: v for k, v in c.items() if k not in ("text", "id")}
        for c in chunks
    ]
    ids = [c["chunk_id"] for c in chunks]
    
    batch_size = 64
    total_indexed = 0
    for i in range(0, len(texts), batch_size):
        b_texts = texts[i:i + batch_size]
        b_meta = metadata_list[i:i + batch_size]
        b_ids = ids[i:i + batch_size]
        embeddings = embedder.embed_documents(b_texts)
        count = faiss_mgr.add_documents(
            texts=b_texts,
            embeddings=embeddings,
            metadata_list=b_meta,
            ids=b_ids
        )
        total_indexed += count
        print(f"  Indexed [{total_indexed}/{len(texts)}] chunks...")
        
    print(f"Successfully rebuilt FAISS vector index with {total_indexed} chunks!")
    
    print("\n" + "=" * 60)
    print("4. Rebuilding QA FAISS Index...")
    print("=" * 60)
    
    qa_mgr = QAIndexManager(embedder, qa_dataset_path=str(qa_path), cache_dir=str(ROOT_DIR / "data" / "qa_index"))
    qa_count = qa_mgr.rebuild_index()
    print(f"Successfully rebuilt QA Index with {qa_count} question pairs!")
    print("\n[SUCCESS] All documents and QA pairs ingested and indexed!")

if __name__ == "__main__":
    main()

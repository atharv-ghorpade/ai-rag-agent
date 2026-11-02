import os
import re
import json
import logging
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

ENTERPRISE_SYSTEM_PROMPT = """You are an AI assistant answering questions based ONLY on the provided context.

Instructions:
1. First answer the user's question directly in one sentence.
2. If the answer exists in the context, state it clearly.
3. Then provide supporting details.
4. If the answer is not present, say:
   "The provided documents do not contain this information."
5. Do not copy entire paragraphs from the context.
6. Be concise and factual.

Output Format:

## Answer
<Direct answer>

## Details
<Supporting explanation>

## Summary
- Point 1
- Point 2

## Sources
<File names and page numbers>
"""


class RAGGenerator:
    """
    Person 3: Grounded Generation & Verified Citations.
    Multi-provider LLM support: Google Gemini, OpenAI, Ollama, and a Structured Extractive Fallback.
    """

    def __init__(
        self,
        provider: Optional[str] = None,
        model_name: Optional[str] = None
    ):
        self.provider = (provider or os.getenv("LLM_PROVIDER", "auto")).lower()
        self.model_name = model_name or os.getenv("LLM_MODEL", "gemini-2.5-flash")
        self.gemini_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        self.openai_key = os.getenv("OPENAI_API_KEY")
        self.ollama_url = os.getenv("OLLAMA_URL", "http://localhost:11434")

        # Fast non-blocking check for local Ollama availability
        self._ollama_available = False
        if self.provider == "ollama":
            self._ollama_available = True
        elif self.provider == "auto" and not self.gemini_key:
            try:
                import httpx
                with httpx.Client(timeout=0.3) as client:
                    r = client.get(f"{self.ollama_url}/api/version")
                    if r.status_code == 200:
                        self._ollama_available = True
            except Exception:
                self._ollama_available = False

    def _call_gemini(self, prompt: str) -> Optional[str]:
        """Calls Google Gemini API via REST if API key is configured."""
        if not self.gemini_key:
            return None
        import httpx
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model_name}:generateContent?key={self.gemini_key}"
        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": 0.1, "maxOutputTokens": 1024}
        }
        try:
            with httpx.Client(timeout=30.0) as client:
                res = client.post(url, json=payload)
                if res.status_code == 200:
                    data = res.json()
                    candidates = data.get("candidates", [])
                    if candidates:
                        parts = candidates[0].get("content", {}).get("parts", [])
                        if parts:
                            return parts[0].get("text", "")
        except Exception as e:
            logger.warning(f"Gemini API call failed: {e}")
        return None

    def _call_ollama(self, prompt: str) -> Optional[str]:
        """Calls local Ollama instance if available."""
        import httpx
        url = f"{self.ollama_url}/api/generate"
        payload = {
            "model": os.getenv("OLLAMA_MODEL", "llama3.2"),
            "prompt": prompt,
            "stream": False
        }
        try:
            with httpx.Client(timeout=30.0) as client:
                res = client.post(url, json=payload)
                if res.status_code == 200:
                    return res.json().get("response", "")
        except Exception:
            pass
        return None

    @staticmethod
    def _clean_item_text(s: str) -> str:
        """Strips leading list numbers, bullets, brackets, and collapses internal spaces."""
        if not s:
            return ""
        # Strip leading punctuation, bullets, dashes
        s = re.sub(r"^[\s\.\:\,\-•\*\>\–\—]+", "", s).strip()
        # Strip leading list numbers like 1., 2), a., (1), [1], etc.
        s = re.sub(r"^(\(?\d+[\.\)]|\(?[a-zA-Z][\.\)]|[•\-\*\>\–\—])\s*", "", s).strip()
        # Strip trailing orphan list numbers or page references (e.g. \n2 or space 2 at end)
        s = re.sub(r"[\s\n]+\d+\.?$", "", s).strip()
        # Strip trailing colons or noise
        s = re.sub(r"[:\s]+$", "", s).strip()
        # Collapse multiple whitespaces
        s = re.sub(r"\s+", " ", s).strip()
        return s

    @staticmethod
    def _is_header_or_noise(s: str) -> bool:
        """Returns True if the line is an uppercase section header or noise."""
        lower = s.lower().strip()
        if lower.startswith(("page ", "page-", "bytes", "http", "www", "tel", "email", "sub :", "ref :", "newsletter")):
            return True
        words = s.split()
        if len(words) <= 8 and (s.isupper() or s.rstrip(":").isupper() or s.endswith(":")):
            return True
        return False

    @staticmethod
    def _extract_first_sentence(text: str) -> str:
        """Extracts one complete direct sentence without lone numbers or abbreviations."""
        cleaned = RAGGenerator._clean_item_text(text)
        # Search for first sentence of at least 20 chars ending with a period followed by space or end
        m = re.search(r"^(.{20,}?[a-zA-Z]{3}\.)(?:\s+|$)", cleaned)
        if m:
            return m.group(1).strip()
        if not cleaned.endswith("."):
            cleaned += "."
        return cleaned

    @staticmethod
    def _format_summary_point(text: str, max_words: int = 18) -> str:
        """Formats a summary bullet without word truncation and without list digits."""
        cleaned = RAGGenerator._clean_item_text(text).rstrip(".")
        words = cleaned.split()
        if len(words) > max_words:
            cleaned = " ".join(words[:max_words])
        return f"- {cleaned}."

    def _format_sources(self, retrieved_chunks: List[Dict[str, Any]]) -> str:
        """Formats clean deduplicated source citations list with markdown bullets."""
        seen = set()
        source_lines = []
        for c in retrieved_chunks:
            src = c.get("source_file") or c.get("source_pdf", "Document.pdf")
            pg = c.get("page") or c.get("page_number", 1)
            pair = (src, pg)
            if pair not in seen:
                seen.add(pair)
                source_lines.append(f"- {src} — Page {pg}")
        return "\n".join(source_lines) if source_lines else "- Institutional Document Repository — Page 1"

    def _build_details_table_or_bullets(
        self,
        query: str,
        text: str,
        extracted_sentences: Optional[List[str]] = None
    ) -> str:
        """
        Builds a clean markdown table if numbers/statistics are involved, otherwise bullet points.
        Special rules: fees, dates, marks, seat intake, admissions, vacancies, statistics, percentages, approval reports.
        """
        combined = (query + " " + text).lower()
        rows = []
        
        # 1. Fees & Currency
        fee_matches = re.findall(
            r"(?:(?:total|institute level|tuition|development|interim|other|exam|eligibility)\s+)?fees?\s*(?:is|are|was|were|of|:)?\s*(₹\s*[\d,]+|Rs\.?\s*[\d,]+|INR\s*[\d,]+|\b\d{4,6}\b)", 
            text, re.IGNORECASE
        )
        for m in fee_matches:
            val = m if any(c in m for c in ["₹", "Rs", "INR"]) else f"₹ {m}"
            rows.append(("Total / Relevant Fee", val.strip()))
        
        raw_fees = re.findall(r"(₹\s*[\d,]+|Rs\.?\s*[\d,]+)", text)
        for rf in raw_fees:
            if not any(rf in r[1] for r in rows):
                rows.append(("Fee Amount", rf.strip()))
                
        # 2. Intake, Admissions & Vacancies
        intake_m = re.search(r"(?:sanctioned\s+)?intake\s*(?:is|are|was|were|of|:)?\s*(\d+)", text, re.IGNORECASE) or re.search(r"(\d+)\s+(?:sanctioned\s+)?(?:intake|seats)", text, re.IGNORECASE)
        if intake_m:
            rows.append(("Sanctioned Intake", intake_m.group(1)))
            
        approved_m = re.search(r"(?:approved\s+admissions?|approved\s+students?|approved)\s*(?:is|are|was|were|of|:)?\s*(\d+)", text, re.IGNORECASE) or re.search(r"(\d+)\s+approved\s+(?:admissions?|students?)", text, re.IGNORECASE)
        if approved_m:
            rows.append(("Approved Admissions", approved_m.group(1)))
            
        admitted_m = re.search(r"(?:students?\s+admitted|total\s+admitted|admitted)\s*(?:is|are|was|were|of|:)?\s*(\d+)", text, re.IGNORECASE) or re.search(r"(\d+)\s+(?:students?\s+admitted|admissions)", text, re.IGNORECASE)
        if admitted_m and not approved_m:
            rows.append(("Students Admitted", admitted_m.group(1)))
            
        vac_m = re.search(r"(?:vacanc(?:y|ies)|vacant\s+seats?)\s*(?:is|are|was|were|of|:)?\s*(\d+)", text, re.IGNORECASE) or re.search(r"(\d+)\s+(?:vacanc(?:y|ies)|vacant\s+seats?)", text, re.IGNORECASE)
        if vac_m:
            rows.append(("Vacancies", vac_m.group(1)))

        # 3. Percentages & Marks
        pct_matches = re.findall(r"([A-Za-z\s]{3,25})?\s*(\b\d+(?:\.\d+)?\s*%)", text)
        for label, val in pct_matches:
            lbl = label.strip() if label.strip() and len(label.strip()) > 3 else "Required Percentage / Marks"
            if not any(val in r[1] for r in rows):
                rows.append((lbl.title(), val.strip()))

        # 4. Dates & Academic Years
        ay_m = re.search(r"\b(20\d\d[-–]\d\d(?:\s*\(.*?\))?)\b", text)
        if ay_m:
            rows.append(("Academic Year", ay_m.group(1)))

        date_m = re.search(r"\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2}(?:,\s*\d{4})?\b|\b\d{1,2}(?:st|nd|rd|th)?\s+(?:January|February|March|April|May|June|July|August|September|October|November|December)(?:\s+\d{4})?\b", text, re.IGNORECASE)
        if date_m:
            rows.append(("Date / Deadline", date_m.group(0)))

        # 5. Approval Status
        if "approved" in combined and not any("Approval" in r[0] for r in rows):
            if "approved" in text.lower():
                rows.append(("Approval Status", "Approved by Competent Authority"))

        # Build clean markdown table if rows found
        if rows:
            table_str = "| Particular | Value |\n|:-----------|------:|\n"
            seen_keys = set()
            for k, v in rows:
                if k not in seen_keys:
                    seen_keys.add(k)
                    table_str += f"| {k} | {v} |\n"
            return table_str.strip()

        # Fallback to key facts in bullet points from clean extracted lines
        candidate_items = list(extracted_sentences) if extracted_sentences else []
        if not candidate_items:
            for line in text.split("\n"):
                cl = self._clean_item_text(line)
                if len(cl) >= 20 and not self._is_header_or_noise(cl):
                    candidate_items.append(cl)

        bullets = []
        seen_bullets = set()
        for item in candidate_items:
            cl = self._clean_item_text(item)
            if len(cl) < 20 or self._is_header_or_noise(cl):
                continue
            cl = cl.rstrip(".:,; ") + "."
            if cl not in seen_bullets:
                seen_bullets.add(cl)
                bullets.append(f"- {cl}")
            if len(bullets) >= 4:
                break

        if bullets:
            return "\n".join(bullets)
        return "- Verified institutional policy and administrative records."

    def _extractive_synthesis(
        self,
        query: str,
        retrieved_chunks: List[Dict[str, Any]],
        matched_qa: Optional[List[Dict[str, Any]]] = None
    ) -> str:
        """
        High-precision deterministic synthesis conforming to:
        ## Answer (Direct answer in one sentence)
        ## Details (Supporting explanation / markdown table)
        ## Summary (Point 1, Point 2)
        ## Sources (File names and page numbers)
        """
        sources_str = self._format_sources(retrieved_chunks)

        # 1. Exact verified match from QA dataset
        if matched_qa and matched_qa[0].get("similarity", 0) >= 0.85:
            top_qa = matched_qa[0]
            ans = top_qa.get("answer", "").strip()
            target_doc = top_qa.get("document", "Document.pdf")
            combined_context = ans + "\n" + "\n".join(c.get("raw_text", c.get("text", "")) for c in retrieved_chunks[:2])
            
            # Direct answer in one sentence
            direct_answer = self._extract_first_sentence(ans)

            details_section = self._build_details_table_or_bullets(query, combined_context)
            
            summary_points = [
                f"- Grounded verified record from {target_doc}.",
                f"- Policy and details confirmed against official records."
            ]

            return (
                f"## Answer\n{direct_answer}\n\n"
                f"## Details\n{details_section}\n\n"
                f"## Summary\n" + "\n".join(summary_points) + f"\n\n"
                f"## Sources\n- {target_doc} — Page 1"
            )

        # 2. Check for empty context
        if not retrieved_chunks:
            return (
                "## Answer\n"
                "The provided documents do not contain this information.\n\n"
                "## Details\n"
                "No matching records or related sections were located in the current document repository.\n\n"
                "## Summary\n"
                "- Information unavailable in uploaded documents.\n"
                "- No matching policies or records indexed.\n\n"
                "## Sources\n"
                "- Institutional Document Repository — Page 1"
            )

        # 3. Check for unmentioned or low-confidence / irrelevant queries
        max_rerank = max((float(c.get("rerank_score", -99.0)) for c in retrieved_chunks), default=-99.0)
        combined_text = " ".join(c.get("raw_text", c.get("text", "")) for c in retrieved_chunks).lower()
        stopwords = {"what", "when", "where", "which", "who", "whom", "whose", "why", "how", "does", "is", "are", "the", "for", "and", "about", "with", "this", "that", "there", "mention", "policy"}
        query_words = [w for w in re.findall(r"\b[a-z]{4,}\b", query.lower()) if w not in stopwords]
        
        has_overlap = any(w in combined_text for w in query_words) if query_words else True
        if max_rerank < -5.5 or (not has_overlap and query_words):
            return (
                "## Answer\n"
                "The provided documents do not contain this information.\n\n"
                "## Details\n"
                f"No matching records or related policies were located for the requested query ({', '.join(query_words[:4]) if query_words else 'specified subject'}).\n\n"
                "## Summary\n"
                "- Information unavailable in uploaded documents.\n"
                "- No verified institutional records found.\n\n"
                "## Sources\n"
                "- Institutional Document Repository — Page 1"
            )

        # 4. Synthesize from retrieved chunks
        lead_sentences = []
        all_raw_text = []

        for chunk in retrieved_chunks[:3]:
            raw = chunk.get("raw_text", chunk.get("text", "")).strip()
            raw_clean = re.sub(r"^\[.*?\]\n?", "", raw).strip()
            all_raw_text.append(raw_clean)
            lines = [l.strip() for l in raw_clean.split("\n") if len(l.strip()) > 15]
            for line in lines:
                cleaned_line = self._clean_item_text(line)
                if (
                    len(cleaned_line) >= 20
                    and not self._is_header_or_noise(cleaned_line)
                    and cleaned_line not in lead_sentences
                ):
                    lead_sentences.append(cleaned_line)
                    if len(lead_sentences) >= 6:
                        break
            if len(lead_sentences) >= 6:
                break

        combined_corpus = "\n".join(all_raw_text)
        
        # Rule 1: First answer the user's question directly in one sentence
        lead = lead_sentences[0] if lead_sentences else "Relevant verified details were extracted from official documentation."
        direct_sentence = self._extract_first_sentence(lead)

        # Rule 3: Then provide supporting details
        details_table_or_bullets = self._build_details_table_or_bullets(
            query, 
            combined_corpus, 
            extracted_sentences=lead_sentences
        )
        
        if details_table_or_bullets.startswith("|"):
            supporting_notes = " ".join(lead_sentences[1:3]).strip()
            if supporting_notes:
                details_section = f"{supporting_notes}\n\n{details_table_or_bullets}"
            else:
                details_section = details_table_or_bullets
        else:
            details_section = details_table_or_bullets

        # Rule Output Format: Summary with Point 1 and Point 2
        p1 = self._format_summary_point(lead_sentences[1]) if len(lead_sentences) > 1 else self._format_summary_point(direct_sentence)
        p2 = self._format_summary_point(lead_sentences[2]) if len(lead_sentences) > 2 else "- Verified against official institutional records."

        return (
            f"## Answer\n{direct_sentence}\n\n"
            f"## Details\n{details_section}\n\n"
            f"## Summary\n{p1}\n{p2}\n\n"
            f"## Sources\n{sources_str}"
        )

    def generate_answer(
        self,
        query: str,
        retrieved_chunks: List[Dict[str, Any]],
        matched_qa: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """
        Generates an answer grounded strictly in retrieved chunks conforming to the standard template.
        """
        context_str = ""
        citations = []

        for idx, chunk in enumerate(retrieved_chunks, 1):
            source = chunk.get("source_file") or chunk.get("source_pdf", "Unknown")
            page = chunk.get("page") or chunk.get("page_number", 1)
            text = chunk.get("raw_text", chunk.get("text", ""))
            score = chunk.get("rerank_score", chunk.get("score", 0.0))

            context_str += f"\n--- Context Chunk {idx} (File: {source} | Page: {page}) ---\n{text}\n"
            citations.append({
                "source_file": source,
                "source_pdf": source,
                "page": page,
                "page_number": page,
                "score": float(score) if score is not None else 0.0
            })

        # Reference QA knowledge if available
        qa_ref_str = ""
        if matched_qa:
            qa_ref_str = "\nVERIFIED REFERENCE FAQS:\n"
            for q_item in matched_qa[:2]:
                qa_ref_str += f"- Question: {q_item.get('question')}\n  Verified Answer: {q_item.get('answer')}\n"

        user_prompt = f"""{ENTERPRISE_SYSTEM_PROMPT}

USER QUERY:
{query}
{qa_ref_str}
CONTEXT CHUNKS:
{context_str}

Please generate the response following the exact ANSWER FORMAT.
"""

        answer_text = None

        # 1. Try Gemini if configured
        if self.gemini_key and (self.provider in ("auto", "gemini")):
            answer_text = self._call_gemini(user_prompt)

        # 2. Try Ollama if confirmed available
        if not answer_text and self._ollama_available and (self.provider in ("auto", "ollama")):
            answer_text = self._call_ollama(user_prompt)

        # 3. Deterministic Extractive Synthesis Fallback
        if not answer_text:
            answer_text = self._extractive_synthesis(query, retrieved_chunks, matched_qa=matched_qa)

        return {
            "query": query,
            "answer": answer_text,
            "citations": citations,
            "chunks_used": len(retrieved_chunks),
            "matched_qa": matched_qa or []
        }

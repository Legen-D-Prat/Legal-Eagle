"""
engine.py — RAG pipeline for Legal Eagle.
Uses PyMuPDF for PDF parsing, a pure-numpy vector store, and
the official Groq SDK (no redirect issues) with LLaMA 3.1 as the LLM.
"""

import re, json, hashlib, time
import numpy as np
import fitz  # PyMuPDF
from groq import Groq
from prompts import SYSTEM_PROMPT, CLAUSE_PROMPT, SUMMARY_PROMPT

GROQ_MODEL = "llama-3.1-8b-instant"

CATEGORIES = [
    "Indemnity",
    "Termination",
    "Governing Law",
    "Confidentiality",
    "Intellectual Property",
    "Limitation of Liability",
    "Auto-Renewal",
    "Dispute Resolution",
]

KEYWORDS = {
    "Indemnity":               ["indemnif", "hold harmless", "defend", "losses", "claims"],
    "Termination":             ["terminat", "notice period", "cancel", "expire", "breach"],
    "Governing Law":           ["governing law", "jurisdiction", "courts of", "applicable law"],
    "Confidentiality":         ["confidential", "non-disclosure", "proprietary", "secret"],
    "Intellectual Property":   ["intellectual property", "patent", "copyright", "assignment", "ownership"],
    "Limitation of Liability": ["limitation of liability", "liability cap", "consequential", "indirect damages"],
    "Auto-Renewal":            ["renew", "auto-renew", "automatically extend", "successive term"],
    "Dispute Resolution":      ["arbitration", "mediation", "dispute resolution", "binding arbitration"],
}


# ── Pure-numpy vector store (no ChromaDB needed) ───────────────────────────────
class VectorStore:
    def __init__(self):
        self.docs = []
        self.matrix = None

    def add(self, documents, embeddings):
        self.docs = documents
        self.matrix = np.array(embeddings, dtype=np.float32)

    def query(self, query_embedding, n=5):
        if self.matrix is None or not self.docs:
            return []
        qvec = np.array(query_embedding, dtype=np.float32)
        scores = self.matrix @ qvec
        top_k = min(n, len(self.docs))
        top_idx = np.argpartition(scores, -top_k)[-top_k:]
        top_idx = top_idx[np.argsort(scores[top_idx])[::-1]]
        return [self.docs[i] for i in top_idx]


# ── Hash-based embedding (no GPU / downloads needed) ──────────────────────────
def embed(texts):
    DIM = 384
    out = []
    for text in texts:
        words = re.findall(r'\b[a-z]{3,}\b', text.lower())
        vec = np.zeros(DIM, dtype=np.float32)
        for w in words:
            vec[int(hashlib.md5(w.encode()).hexdigest(), 16) % DIM] += 1.0
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec /= norm
        out.append(vec.tolist())
    return out


# ── Groq LLM call via official SDK (handles auth + HTTP correctly) ─────────────
def call_llm(client, system, user, max_tokens=600):
    """Call Groq with automatic retry on rate-limit (429) errors."""
    for attempt in range(4):
        try:
            resp = client.chat.completions.create(
                model=GROQ_MODEL,
                max_tokens=max_tokens,
                temperature=0.2,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user",   "content": user},
                ],
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            err = str(e)
            if "429" in err or "rate_limit" in err.lower() or "too many" in err.lower():
                wait = 20 * (attempt + 1)   # 20s → 40s → 60s → 80s
                time.sleep(wait)
            else:
                raise
    raise RuntimeError(
        "Groq rate limit hit after 4 retries. "
        "Wait ~1 minute, then try again. "
        "Free tier allows ~30 requests/min."
    )


# ── Sentence Window Parser ─────────────────────────────────────────────────────
def parse_windows(text, window=4, stride=2):
    sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text.strip())
    sentences = [s.strip() for s in sentences if len(s.strip()) > 30]
    chunks = []
    for i in range(0, len(sentences), stride):
        chunk = " ".join(sentences[i:i + window])
        if len(chunk) > 80:
            chunks.append(chunk)
    return chunks


# ── Main pipeline ──────────────────────────────────────────────────────────────
def analyse_contract(pdf_path, api_key, progress=None):
    client = Groq(api_key=api_key.strip())

    # 1. Extract PDF text
    if progress: progress(0.05, "Extracting text from PDF...")
    doc = fitz.open(pdf_path)
    full_text = "\n".join(page.get_text() for page in doc)
    page_count = len(doc)

    # 2. Detect document type
    if progress: progress(0.15, "Identifying document type...")
    doc_type = call_llm(
        client,
        system="You are a legal document classifier.",
        user=(
            "What type of legal document is this? "
            "Reply with ONLY one of: NDA, Lease, Service Agreement, Employment Contract, Other.\n\n"
            + full_text[:800]
        ),
        max_tokens=10,
    )

    # 3. Build vector index
    if progress: progress(0.25, "Building vector index...")
    chunks = parse_windows(full_text)
    vs = VectorStore()
    vs.add(chunks, embed(chunks))

    # 4. Agentic loop — analyse each category
    analyses = []
    for i, cat in enumerate(CATEGORIES):
        if progress:
            progress(0.35 + i / len(CATEGORIES) * 0.52, f"Analysing: {cat}...")

        query_text = " ".join(KEYWORDS.get(cat, [cat.lower()])[:3])
        top_chunks = vs.query(embed([query_text])[0], n=5)

        if not top_chunks:
            analyses.append({
                "category": cat, "risk_level": "LOW", "risk_score": 1,
                "concern": "Clause not found in document.",
                "problematic_text": None,
                "safer_alternative": "N/A — not present in contract.",
                "explanation": "This category is not addressed in the contract.",
            })
            continue

        raw = call_llm(
            client,
            system=SYSTEM_PROMPT,
            user=CLAUSE_PROMPT.format(
                category=cat,
                clause="\n\n".join(top_chunks)[:2500],
            ),
        )

        # Strip markdown fences the model sometimes adds
        raw = re.sub(r'^```(?:json)?\s*', '', raw, flags=re.MULTILINE)
        raw = re.sub(r'\s*```$', '', raw, flags=re.MULTILINE)
        match = re.search(r'\{.*\}', raw, re.DOTALL)
        try:
            parsed = json.loads(match.group()) if match else {}
        except Exception:
            parsed = {}

        parsed.setdefault("risk_level", "MEDIUM")
        parsed.setdefault("risk_score", 5)
        parsed.setdefault("concern", "Review this clause carefully.")
        parsed.setdefault("problematic_text", None)
        parsed.setdefault("safer_alternative", "Consult a lawyer for a safer version.")
        parsed.setdefault("explanation", raw[:300])
        analyses.append({"category": cat, **parsed})

        time.sleep(3)   # stay within Groq free-tier rate limits

    # 5. Executive summary
    if progress: progress(0.92, "Writing executive summary...")
    summary = call_llm(
        client,
        system=SYSTEM_PROMPT,
        user=SUMMARY_PROMPT.format(
            doc_type=doc_type,
            analyses=json.dumps(analyses),
        ),
        max_tokens=350,
    )

    levels = [a.get("risk_level", "LOW").upper() for a in analyses]
    overall = "HIGH" if "HIGH" in levels else ("MEDIUM" if "MEDIUM" in levels else "LOW")

    if progress: progress(1.0, "Done!")

    return {
        "doc_type": doc_type,
        "overall_risk": overall,
        "summary": summary,
        "analyses": analyses,
        "stats": {"pages": page_count, "chunks": len(chunks), "categories": len(analyses)},
    }

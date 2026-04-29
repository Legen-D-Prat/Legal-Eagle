"""
engine.py — RAG pipeline for Legal Eagle.
Pure-Python vector store (numpy only) — no ChromaDB, no protobuf, no compatibility issues.
Uses PyMuPDF for PDF parsing and Groq (free) with LLaMA 3 as the LLM.
"""

import re, json, hashlib
import numpy as np
import fitz  # PyMuPDF
import requests
from prompts import SYSTEM_PROMPT, CLAUSE_PROMPT, SUMMARY_PROMPT

GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL   = "llama-3.1-8b-instant"

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


# ── Lightweight numpy vector store (replaces ChromaDB) ────────────────────────
class VectorStore:
    """Simple cosine-similarity vector store backed by numpy arrays."""

    def __init__(self):
        self.docs: list[str] = []
        self.matrix: np.ndarray | None = None  # shape (n, DIM)

    def add(self, documents: list[str], embeddings: list[list[float]]):
        self.docs = documents
        self.matrix = np.array(embeddings, dtype=np.float32)

    def query(self, query_embedding: list[float], n: int = 5) -> list[str]:
        if self.matrix is None or len(self.docs) == 0:
            return []
        qvec = np.array(query_embedding, dtype=np.float32)
        # Cosine similarity = dot product (both already unit-normalised)
        scores = self.matrix @ qvec
        top_k = min(n, len(self.docs))
        top_idx = np.argpartition(scores, -top_k)[-top_k:]
        top_idx = top_idx[np.argsort(scores[top_idx])[::-1]]
        return [self.docs[i] for i in top_idx]


# ── Hash-based embedding (no model downloads, no GPU) ─────────────────────────
def embed(texts: list[str]) -> list[list[float]]:
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


# ── Groq LLM call ──────────────────────────────────────────────────────────────
def call_llm(system: str, user: str, api_key: str, max_tokens: int = 600) -> str:
    import json as _json
    key = api_key.strip()
    payload = _json.dumps({
        "model": GROQ_MODEL,
        "max_tokens": max_tokens,
        "temperature": 0.2,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
        ],
    })
    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
    }
    resp = requests.post(GROQ_API_URL, headers=headers, data=payload,
                         allow_redirects=False, timeout=30)
    if resp.status_code in (301, 302, 307, 308):
        loc = resp.headers.get("Location", GROQ_API_URL)
        resp = requests.post(loc, headers=headers, data=payload, timeout=30)
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"].strip()


# ── Sentence Window Parser ─────────────────────────────────────────────────────
def parse_windows(text: str, window: int = 4, stride: int = 2) -> list[str]:
    sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text.strip())
    sentences = [s.strip() for s in sentences if len(s.strip()) > 30]
    chunks = []
    for i in range(0, len(sentences), stride):
        chunk = " ".join(sentences[i:i + window])
        if len(chunk) > 80:
            chunks.append(chunk)
    return chunks


# ── Main pipeline ──────────────────────────────────────────────────────────────
def analyse_contract(pdf_path: str, api_key: str, progress=None) -> dict:

    # 1. Extract PDF text
    if progress: progress(0.05, "Extracting text from PDF...")
    doc = fitz.open(pdf_path)
    full_text = "\n".join(page.get_text() for page in doc)
    page_count = len(doc)

    # 2. Detect document type
    if progress: progress(0.15, "Identifying document type...")
    doc_type = call_llm(
        system="You are a legal document classifier.",
        user=f"What type of legal document is this? Reply with ONLY one of: NDA, Lease, Service Agreement, Employment Contract, Other.\n\n{full_text[:800]}",
        api_key=api_key,
        max_tokens=10,
    )

    # 3. Sentence windows + embed + index
    if progress: progress(0.25, "Building vector index...")
    chunks = parse_windows(full_text)
    embeddings = embed(chunks)
    vs = VectorStore()
    vs.add(chunks, embeddings)

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
            system=SYSTEM_PROMPT,
            user=CLAUSE_PROMPT.format(category=cat, clause="\n\n".join(top_chunks)[:2500]),
            api_key=api_key,
        )

        # Strip markdown fences
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)
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

    # 5. Executive summary
    if progress: progress(0.92, "Writing executive summary...")
    summary = call_llm(
        system=SYSTEM_PROMPT,
        user=SUMMARY_PROMPT.format(doc_type=doc_type, analyses=json.dumps(analyses)),
        api_key=api_key,
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

import os, tempfile, base64
import streamlit as st
from engine import analyse_contract

st.set_page_config(page_title="Legal Eagle", page_icon="⚖️", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700&family=Inter:wght@400;500;600&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.main-title {
    font-family: 'Playfair Display', serif;
    font-size: 2.4rem;
    color: #1a1a2e;
    margin-bottom: 0;
}
.subtitle { color: #555; margin-top: 4px; font-size: 0.95rem; }

.risk-card {
    border-radius: 10px;
    padding: 16px 18px;
    margin-bottom: 12px;
    border-left: 5px solid #ccc;
    background: #fafafa;
}
.risk-HIGH   { border-left-color: #e53935; background: #fff5f5; }
.risk-MEDIUM { border-left-color: #fb8c00; background: #fff8f0; }
.risk-LOW    { border-left-color: #43a047; background: #f4fdf4; }

.badge {
    display: inline-block;
    padding: 2px 10px;
    border-radius: 20px;
    font-size: 0.75rem;
    font-weight: 600;
    color: white;
}
.badge-HIGH   { background: #e53935; }
.badge-MEDIUM { background: #fb8c00; }
.badge-LOW    { background: #43a047; }

.summary-box {
    background: #f0f4ff;
    border: 1px solid #c5d0f5;
    border-radius: 10px;
    padding: 18px 22px;
    line-height: 1.75;
    color: #222;
}

.stat-box {
    background: #fff;
    border: 1px solid #e5e7eb;
    border-radius: 8px;
    padding: 14px;
    text-align: center;
}
.stat-num { font-size: 1.8rem; font-weight: 700; color: #1a1a2e; }
.stat-lbl { font-size: 0.75rem; color: #888; text-transform: uppercase; letter-spacing: 0.05em; }

.overall-HIGH   { background: #fde8e8; border: 2px solid #e53935; border-radius: 10px; padding: 14px 20px; }
.overall-MEDIUM { background: #fff3e0; border: 2px solid #fb8c00; border-radius: 10px; padding: 14px 20px; }
.overall-LOW    { background: #e8f5e9; border: 2px solid #43a047; border-radius: 10px; padding: 14px 20px; }

.section-header {
    font-family: 'Playfair Display', serif;
    font-size: 1.2rem;
    color: #1a1a2e;
    border-bottom: 2px solid #e5e7eb;
    padding-bottom: 6px;
    margin: 20px 0 14px 0;
}

.api-info {
    background: #fffbea;
    border: 1px solid #f6d860;
    border-radius: 8px;
    padding: 12px 14px;
    font-size: 0.85rem;
    color: #555;
}
</style>
""", unsafe_allow_html=True)

EMOJI = {"HIGH": "🔴", "MEDIUM": "🟡", "LOW": "🟢"}
COLOR = {"HIGH": "#e53935", "MEDIUM": "#fb8c00", "LOW": "#43a047"}

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚖️ Legal Eagle")
    st.markdown("**Contract Auditor** — AI-powered risk analysis")
    st.divider()

    # Groq API key input
    st.markdown("### 🔑 Groq API Key")
    st.markdown(
        '<div class="api-info">Free at <a href="https://console.groq.com" target="_blank">console.groq.com</a> — no credit card needed.</div>',
        unsafe_allow_html=True
    )
    st.markdown("")

    # Check env var first (for Streamlit Cloud secrets), then let user type it
    default_key = os.environ.get("GROQ_API_KEY", "")
    api_key = st.text_input(
        "Enter your Groq API key",
        value=default_key,
        type="password",
        placeholder="gsk_...",
        label_visibility="collapsed",
    )

    st.divider()
    st.markdown("### 📂 Upload Contract")
    uploaded = st.file_uploader("Supported format: PDF", type=["pdf"], label_visibility="collapsed")

    st.divider()
    st.markdown("""
**How it works**
1. Paste your free Groq API key above
2. Upload any PDF contract
3. Text split into sentence windows
4. Chunks stored in ChromaDB (local)
5. LLaMA 3 evaluates 8 risk categories
6. Get safer clause alternatives
""")
    st.caption("Stack: Streamlit · LLaMA 3 (Groq) · ChromaDB · PyMuPDF")

# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown('<p class="main-title">⚖️ Legal Eagle: Contract Auditor</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Upload a PDF contract to identify high-risk clauses and get safer alternatives — powered by open-source LLaMA 3, completely free.</p>', unsafe_allow_html=True)
st.divider()

# ── Gate: need both API key and file ──────────────────────────────────────────
if not api_key:
    st.info("👈 Enter your free Groq API key in the sidebar to get started.")
    st.markdown("""
    **Getting a free Groq API key:**
    1. Go to [console.groq.com](https://console.groq.com)
    2. Sign up (free, no credit card)
    3. Go to **API Keys** → **Create API Key**
    4. Paste it in the sidebar
    """)
    st.stop()

if not uploaded:
    c1, c2, c3 = st.columns(3)
    c1.info("📄 Upload a PDF in the sidebar")
    c2.success("🔍 Analyses 8 legal risk categories")
    c3.warning("✅ Get safer clause alternatives")
    st.stop()

# ── Run Analysis ───────────────────────────────────────────────────────────────
pdf_bytes = uploaded.read()
cache_key = f"{uploaded.name}_{len(pdf_bytes)}"

if cache_key not in st.session_state:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as f:
        f.write(pdf_bytes)
        tmp = f.name
    bar = st.progress(0.0)
    status = st.empty()
    try:
        results = analyse_contract(
            tmp,
            api_key=api_key,
            progress=lambda p, m: (bar.progress(p), status.caption(m)),
        )
        st.session_state[cache_key] = results
    except Exception as e:
        st.error(f"Analysis failed: {e}")
        st.stop()
    finally:
        os.unlink(tmp)
        bar.empty()
        status.empty()

r = st.session_state[cache_key]

# ── Layout: PDF viewer left, analysis right ────────────────────────────────────
left, right = st.columns([1, 1], gap="large")

with left:
    st.markdown('<p class="section-header">📄 Contract Preview</p>', unsafe_allow_html=True)
    b64 = base64.b64encode(pdf_bytes).decode()
    st.markdown(
        f'<iframe src="data:application/pdf;base64,{b64}" width="100%" height="750px" '
        f'style="border:1px solid #e5e7eb; border-radius:10px;"></iframe>',
        unsafe_allow_html=True,
    )

with right:
    st.markdown('<p class="section-header">🔬 Risk Analysis</p>', unsafe_allow_html=True)

    # Stats row
    s = r["stats"]
    sc1, sc2, sc3, sc4 = st.columns(4)
    for col, num, lbl in [
        (sc1, r["doc_type"], "Type"),
        (sc2, s["pages"], "Pages"),
        (sc3, s["chunks"], "Chunks"),
        (sc4, s["categories"], "Categories"),
    ]:
        col.markdown(
            f'<div class="stat-box"><div class="stat-num">{num}</div>'
            f'<div class="stat-lbl">{lbl}</div></div>',
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # Overall risk
    ov = r["overall_risk"]
    st.markdown(
        f'<div class="overall-{ov}"><span style="font-size:1.1rem; font-weight:700; color:{COLOR[ov]};">'
        f'{EMOJI[ov]} Overall Risk: {ov}</span></div>',
        unsafe_allow_html=True,
    )

    # Tabs
    tab1, tab2, tab3 = st.tabs(["📋 Summary", "⚠️ Clauses", "📊 Heatmap"])

    with tab1:
        st.markdown(f'<div class="summary-box">{r["summary"]}</div>', unsafe_allow_html=True)

    with tab2:
        order = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}
        sorted_a = sorted(r["analyses"], key=lambda a: order.get(a.get("risk_level", "LOW"), 3))
        for a in sorted_a:
            cat  = a["category"]
            lvl  = a.get("risk_level", "LOW").upper()
            score = a.get("risk_score", 1)
            st.markdown(
                f'<div class="risk-card risk-{lvl}">'
                f'<strong>{EMOJI[lvl]} {cat}</strong> &nbsp;'
                f'<span class="badge badge-{lvl}">{lvl}</span> &nbsp;'
                f'<span style="color:#888; font-size:0.82rem;">Score: {score}/10</span>'
                f'</div>',
                unsafe_allow_html=True,
            )
            with st.expander(f"Details — {cat}", expanded=(lvl == "HIGH")):
                st.markdown(f"**Concern:** {a.get('concern', '—')}")
                if a.get("problematic_text"):
                    st.markdown("**Problematic language:**")
                    st.code(a["problematic_text"], language=None)
                st.markdown("**Safer alternative:**")
                st.success(a.get("safer_alternative", "—"))
                st.markdown(f"**Why:** {a.get('explanation', '—')}")

    with tab3:
        st.markdown("#### Risk Score by Category")
        for a in r["analyses"]:
            cat   = a["category"]
            score = a.get("risk_score", 1)
            lvl   = a.get("risk_level", "LOW").upper()
            c_lbl, c_bar = st.columns([2, 5])
            c_lbl.markdown(f"{EMOJI[lvl]} {cat}")
            c_bar.progress(score / 10, text=f"{score}/10")

        st.divider()
        high = sum(1 for a in r["analyses"] if a.get("risk_level", "").upper() == "HIGH")
        med  = sum(1 for a in r["analyses"] if a.get("risk_level", "").upper() == "MEDIUM")
        low  = sum(1 for a in r["analyses"] if a.get("risk_level", "").upper() == "LOW")
        x1, x2, x3 = st.columns(3)
        x1.metric("🔴 High Risk",   high)
        x2.metric("🟡 Medium Risk", med)
        x3.metric("🟢 Low Risk",    low)

st.divider()
st.caption("⚠️ This tool is for informational purposes only and does not constitute legal advice. Always consult a qualified lawyer before signing any contract.")

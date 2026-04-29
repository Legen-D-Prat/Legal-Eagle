# ⚖️ Legal Eagle: Contract Auditor

An AI-powered Streamlit web app that analyses PDF contracts for risky clauses and suggests safer alternatives. **100% free — uses open-source LLaMA 3 via Groq's free API.**

## Features
- 📄 PDF upload with in-browser viewer
- 🔍 Analyses 8 legal risk categories (Indemnity, Termination, Governing Law, etc.)
- 🤖 RAG pipeline: Sentence Window parsing → ChromaDB → LLaMA 3 (via Groq)
- ✅ Suggests safer clause alternatives based on industry standards
- 📊 Risk heatmap and executive summary

## Stack
| Component | Library | Cost |
|-----------|---------|------|
| UI | Streamlit | Free |
| LLM | LLaMA 3 8B via Groq API | **Free** |
| Vector Store | ChromaDB (in-memory) | Free |
| PDF Parsing | PyMuPDF | Free |
| Embeddings | Hash-based (no GPU needed) | Free |

## Getting a Free Groq API Key
1. Go to [console.groq.com](https://console.groq.com)
2. Sign up (no credit card needed)
3. Go to **API Keys** → **Create API Key**
4. Copy your key (starts with `gsk_...`)

## Run Locally

```bash
git clone https://github.com/YOUR_USERNAME/legal-eagle.git
cd legal-eagle
pip install -r requirements.txt
streamlit run app.py
```

Then paste your Groq API key in the sidebar.

## Deploy on Streamlit Cloud (Free)

1. Push this repo to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub repo → set main file to `app.py`
4. Go to **Advanced settings → Secrets** and add:
   ```toml
   GROQ_API_KEY = "gsk_your_key_here"
   ```
5. Click **Deploy** — done!

## Project Structure

```
legal-eagle/
├── app.py            # Streamlit frontend
├── engine.py         # RAG pipeline (PDF → ChromaDB → LLaMA 3)
├── prompts.py        # LLM system prompts
├── requirements.txt  # 5 dependencies, all free
└── README.md
```

## Disclaimer
For educational/informational purposes only. Not legal advice.

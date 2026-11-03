# Vidya-Daan: CSR Education Grant Proposal Auditor

A Streamlit RAG prototype that indexes an NGO education grant proposal and optional CSR reference PDFs, retrieves relevant passages using LangChain + LanceDB, and presents a preliminary keyword-based review checklist.

## Run locally (Windows / VS Code)

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run app.py
```

Open the local URL shown in the terminal (usually http://localhost:8501).

## Demo flow
1. Upload a selectable-text proposal PDF.
2. Optionally upload current official CSR guideline PDFs.
3. Click **Analyze documents**.
4. Review the preliminary checklist.
5. Ask a question and inspect retrieved evidence passages.
6. Download the preliminary review summary.

## Important
- First run downloads the sentence-transformers embedding model and may take several minutes, depending on internet speed.
- Scanned PDFs may need OCR before upload.
- This is a prototype, not a legal-compliance certification tool. Confirm all legal claims against current official MCA legislation/rules and qualified advice.
- Do not upload confidential personal or financial data to a public demo.

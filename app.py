
import streamlit as st
import re
import math
import hashlib
import tempfile

from pypdf import PdfReader

try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:
    from langchain.text_splitter import RecursiveCharacterTextSplitter

try:
    import lancedb
    from langchain_community.vectorstores import LanceDB
    LANCEDB_AVAILABLE = True
except ImportError:
    LANCEDB_AVAILABLE = False


# =========================================================
# 1. PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Vidya-Daan | CSR Grant Auditor",
    page_icon="📚",
    layout="wide"
)

st.markdown("""
<style>
    .stApp {
        background-color: #10141c;
        color: #f1f5f9;
    }

    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }

    .main-title {
        color: #80aaff;
        font-size: 36px;
        font-weight: 750;
    }

    .sub-title {
        color: #aab8ca;
        font-size: 16px;
    }

    .section-heading {
        color: #80aaff;
        font-size: 23px;
        font-weight: 700;
    }

    div[data-testid="stMetric"] {
        background-color: white;
        border: 1px solid #dce3ed;
        padding: 16px;
        border-radius: 10px;
    }

    div[data-testid="stMetric"] label,
    div[data-testid="stMetric"] label p {
        color: #35445a !important;
    }

    div[data-testid="stMetric"] [data-testid="stMetricValue"] {
        color: #17365d !important;
    }

    .workflow-card {
        background-color: white;
        border: 1px solid #dce3ed;
        border-radius: 10px;
        padding: 18px;
        min-height: 85px;
        color: #17365d !important;
        font-weight: 650;
        text-align: center;
    }

    .workflow-card p {
        color: #17365d !important;
    }
</style>
""", unsafe_allow_html=True)


# =========================================================
# 2. SIMPLE EMBEDDING MODEL
# =========================================================

class SimpleEmbeddings:

    def __init__(self, dimension=384):
        self.dimension = dimension

    def _embed(self, text):

        vector = [0.0] * self.dimension

        words = re.findall(
            r"[a-zA-Z0-9]+",
            text.lower()
        )

        features = list(words)

        for i in range(len(words) - 1):
            features.append(
                words[i] + "_" + words[i + 1]
            )

        for feature in features:

            hashed = hashlib.md5(
                feature.encode("utf-8")
            ).hexdigest()

            index = int(hashed[:8], 16) % self.dimension

            vector[index] += 1.0

        magnitude = math.sqrt(
            sum(value * value for value in vector)
        )

        if magnitude > 0:
            vector = [
                value / magnitude
                for value in vector
            ]

        return vector

    def embed_documents(self, texts):
        return [
            self._embed(text)
            for text in texts
        ]

    def embed_query(self, text):
        return self._embed(text)


# =========================================================
# 3. PDF TEXT EXTRACTION
# =========================================================

def extract_pdf_text(uploaded_file):

    reader = PdfReader(uploaded_file)

    pages = []

    for page_number, page in enumerate(
        reader.pages,
        start=1
    ):

        page_text = page.extract_text() or ""

        if page_text.strip():

            pages.append({
                "page": page_number,
                "text": page_text,
                "source": uploaded_file.name
            })

    return pages


# =========================================================
# 4. DOCUMENT CLASSIFICATION
# =========================================================

def classify_documents(uploaded_files):

    proposal_pages = []
    reference_pages = []

    proposal_names = []
    reference_names = []

    reference_keywords = [
        "reference",
        "guideline",
        "checklist",
        "policy",
        "manual"
    ]

    for uploaded_file in uploaded_files:

        filename = uploaded_file.name.lower()

        is_reference = any(
            keyword in filename
            for keyword in reference_keywords
        )

        pages = extract_pdf_text(uploaded_file)

        if is_reference:

            reference_pages.extend(pages)
            reference_names.append(uploaded_file.name)

        else:

            proposal_pages.extend(pages)
            proposal_names.append(uploaded_file.name)

    return (
        proposal_pages,
        reference_pages,
        proposal_names,
        reference_names
    )


# =========================================================
# 5. TEXT CHUNKING
# =========================================================

def create_chunks(pages):

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=120
    )

    chunks = []
    metadata = []

    for page in pages:

        page_chunks = splitter.split_text(
            page["text"]
        )

        for chunk in page_chunks:

            chunks.append(chunk)

            metadata.append({
                "page": page["page"],
                "source": page["source"]
            })

    return chunks, metadata


# =========================================================
# 6. REQUIREMENT CHECKLIST
# =========================================================

REQUIREMENTS = [

    {
        "name": "Organisation Registration",
        "groups": [
            [
                "registration certificate",
                "registration details",
                "registration number",
                "registered under",
                "trust registration",
                "society registration",
                "section 8 company"
            ]
        ]
    },

    {
        "name": "CSR-1 Registration",
        "groups": [
            ["csr-1", "csr 1", "csr1"]
        ]
    },

    {
        "name": "Tax and Exemption Documents",
        "groups": [
            [
                "12a",
                "80g",
                "tax exemption",
                "income tax exemption",
                "exemption certificate"
            ]
        ]
    },

    {
        "name": "School Permission",
        "groups": [
            [
                "school permission",
                "permission letter",
                "approval from school",
                "school authorization",
                "consent from school",
                "school approval"
            ]
        ]
    },

    {
        "name": "Audited Financial Statements",
        "groups": [
            [
                "audited financial statements",
                "audited financial",
                "audit report",
                "auditor report"
            ]
        ]
    },

    {
        "name": "Budget and Quotations",
        "groups": [
            [
                "budget",
                "project cost",
                "cost estimate",
                "financial allocation"
            ],
            [
                "quotation",
                "quotations",
                "vendor quote",
                "price quotation",
                "supplier quotation"
            ]
        ]
    },

    {
        "name": "Conflict of Interest Declaration",
        "groups": [
            [
                "conflict of interest",
                "declaration of interest",
                "interest declaration"
            ]
        ]
    },

    {
        "name": "Governing Body Approval",
        "groups": [
            [
                "governing body approval",
                "board resolution",
                "board approval",
                "trustee approval",
                "management approval"
            ]
        ]
    },

    {
        "name": "Monitoring and Evaluation",
        "groups": [
            [
                "baseline",
                "baseline data",
                "starting assessment"
            ],
            [
                "target",
                "targets",
                "measurable target",
                "expected outcome"
            ],
            [
                "reporting frequency",
                "monitoring frequency",
                "monthly report",
                "quarterly report",
                "reporting schedule"
            ],
            [
                "responsible person",
                "project coordinator",
                "monitoring officer",
                "responsibility"
            ]
        ]
    },

    {
        "name": "Post-Grant Sustainability",
        "groups": [
            [
                "maintenance plan",
                "maintenance funding",
                "post-grant",
                "sustainability plan",
                "ongoing funding",
                "continuation plan"
            ]
        ]
    },

    {
        "name": "Beneficiary Details",
        "groups": [
            [
                "beneficiary",
                "students",
                "children"
            ],
            [
                "school names",
                "names of schools",
                "participating schools",
                "government schools"
            ],
            [
                "class-wise",
                "disaggregated",
                "student breakdown",
                "beneficiary breakdown"
            ]
        ]
    }
]


# =========================================================
# 7. REQUIREMENT ANALYSIS
# =========================================================

NEGATIVE_PHRASES = [
    "not provided",
    "not supplied",
    "not attached",
    "not available",
    "not included",
    "not submitted",
    "missing",
    "pending",
    "yet to be provided",
    "to be submitted"
]


def get_paragraphs(text):

    paragraphs = re.split(r"\n+", text)

    cleaned = []

    for paragraph in paragraphs:

        paragraph = paragraph.strip()

        if len(paragraph) < 15:
            continue

        if "missing information is intentional" in paragraph.lower():
            continue

        cleaned.append(paragraph)

    return cleaned


def analyse_requirements(proposal_text):

    paragraphs = get_paragraphs(proposal_text)

    results = []

    for requirement in REQUIREMENTS:

        group_statuses = []
        evidence = []

        for aliases in requirement["groups"]:

            matching = []

            for paragraph in paragraphs:

                lower = paragraph.lower()

                if any(
                    alias.lower() in lower
                    for alias in aliases
                ):
                    matching.append(paragraph)

            has_negative = any(
                any(
                    phrase in paragraph.lower()
                    for phrase in NEGATIVE_PHRASES
                )
                for paragraph in matching
            )

            group_statuses.append(
                bool(matching) and not has_negative
            )

            for paragraph in matching:

                if paragraph not in evidence:
                    evidence.append(paragraph)

        completed = sum(group_statuses)
        total = len(group_statuses)

        if completed == total:
            status = "Present"

        elif completed == 0:
            status = "Missing"

        else:
            status = "Needs Review"

        results.append({
            "name": requirement["name"],
            "status": status,
            "evidence": evidence,
            "completed": completed,
            "total": total
        })

    return results


# =========================================================
# 8. READINESS INDICATOR
# =========================================================

def calculate_readiness(results):

    score = 0
    maximum_score = len(results) * 10

    for result in results:

        if result["status"] == "Present":
            score += 10

        elif result["status"] == "Needs Review":
            score += 5

    if maximum_score == 0:
        return 0

    return round(
        score / maximum_score * 100
    )


# =========================================================
# 9. IMPROVEMENT SUGGESTIONS
# =========================================================

SUGGESTIONS = {

    "Organisation Registration":
        "Add the organisation's registration certificate and registration details.",

    "CSR-1 Registration":
        "Provide CSR-1 registration details, where applicable.",

    "Tax and Exemption Documents":
        "Attach relevant tax and exemption documents, where applicable.",

    "School Permission":
        "Attach permission letters or approvals from participating schools.",

    "Audited Financial Statements":
        "Include the organisation's audited financial statements and audit records.",

    "Budget and Quotations":
        "Provide a detailed budget and supporting vendor quotations.",

    "Conflict of Interest Declaration":
        "Add a signed conflict-of-interest declaration.",

    "Governing Body Approval":
        "Attach governing-body approval or the relevant board resolution.",

    "Monitoring and Evaluation":
        "Specify baseline values, measurable targets, reporting frequency and responsible persons.",

    "Post-Grant Sustainability":
        "Explain how the project will be maintained and funded after the grant period.",

    "Beneficiary Details":
        "Provide participating school names and a clear beneficiary breakdown."
}


def find_improvements(results):

    return [
        {
            "name": item["name"],
            "suggestion": SUGGESTIONS[item["name"]]
        }
        for item in results
        if item["status"] != "Present"
    ]


# =========================================================
# 10. LANCEDB INDEXING
# =========================================================

def create_vector_database(chunks, metadata):

    if not LANCEDB_AVAILABLE:
        return None, "LanceDB packages are unavailable."

    if not chunks:
        return None, "No readable text sections found."

    try:

        directory = tempfile.mkdtemp(
            prefix="vidya_daan_"
        )

        connection = lancedb.connect(directory)

        vector_database = LanceDB.from_texts(
            texts=chunks,
            embedding=SimpleEmbeddings(),
            metadatas=metadata,
            connection=connection,
            table_name="vidya_daan_documents"
        )

        return vector_database, None

    except Exception as error:

        return None, str(error)


# =========================================================
# 11. QUESTION TYPE DETECTION
# =========================================================

REFERENCE_TERMS = [
    "csr guideline",
    "csr guidelines",
    "csr rule",
    "csr rules",
    "csr requirement",
    "csr requirements",
    "legal requirement",
    "eligibility",
    "compliance",
    "what is csr",
    "what does csr"
]


def identify_question_source(question):

    lower = question.lower()

    if any(
        term in lower
        for term in REFERENCE_TERMS
    ):
        return "reference"

    return "proposal"


# =========================================================
# 12. KEYWORD RETRIEVAL
# =========================================================

STOP_WORDS = {
    "what", "is", "the", "a", "an", "of", "in",
    "on", "for", "by", "from", "and", "to",
    "how", "does", "do", "can", "could", "please",
    "tell", "me", "about", "was", "were", "are",
    "this", "that", "it", "their", "they"
}


def get_query_terms(question):

    words = re.findall(
        r"[a-zA-Z0-9]+",
        question.lower()
    )

    return {
        word for word in words
        if word not in STOP_WORDS and len(word) > 1
    }


def retrieve_keyword_evidence(
    question,
    chunks,
    metadata,
    count=4
):

    terms = get_query_terms(question)

    lower_question = question.lower()

    budget_question = any(
        word in lower_question
        for word in [
            "budget",
            "amount",
            "cost",
            "funding",
            "requested",
            "grant"
        ]
    )

    scored = []

    for index, chunk in enumerate(chunks):

        lower_chunk = chunk.lower()

        score = sum(
            1 for term in terms
            if term in lower_chunk
        )

        if budget_question:

            if "₹" in chunk or "inr" in lower_chunk:
                score += 5

            if "requested" in lower_chunk:
                score += 3

            if "project cost" in lower_chunk:
                score += 3

            if "budget" in lower_chunk:
                score += 3

        if score > 0:

            scored.append({
                "score": score,
                "text": chunk,
                "metadata": metadata[index]
            })

    scored.sort(
        key=lambda item: item["score"],
        reverse=True
    )

    results = []

    for item in scored[:count]:

        results.append((
            item["text"],
            item["metadata"]
        ))

    return results


# =========================================================
# 13. SOURCE-AWARE QUESTION ANSWERING
# =========================================================

def get_answer(
    question,
    vector_database,
    proposal_chunks,
    proposal_metadata,
    reference_chunks,
    reference_metadata
):

    source_type = identify_question_source(question)

    if source_type == "reference":

        selected_chunks = reference_chunks
        selected_metadata = reference_metadata

        source_description = "CSR Reference Guidelines"

    else:

        selected_chunks = proposal_chunks
        selected_metadata = proposal_metadata

        source_description = "NGO Education Grant Proposal"

    if not selected_chunks:

        return (
            f"No {source_description} was uploaded or classified.",
            []
        )

    # Keyword retrieval is used first so the source is controlled.
    retrieved = retrieve_keyword_evidence(
        question,
        selected_chunks,
        selected_metadata,
        count=4
    )

    # Use LanceDB only as a fallback, and filter its results
    # to the selected document category.
    if not retrieved and vector_database is not None:

        try:

            documents = vector_database.similarity_search(
                question,
                k=10
            )

            for document in documents:

                metadata = document.metadata

                source_name = metadata.get(
                    "source",
                    ""
                )

                allowed_names = (
                    st.session_state["reference_names"]
                    if source_type == "reference"
                    else st.session_state["proposal_names"]
                )

                if source_name in allowed_names:

                    retrieved.append((
                        document.page_content,
                        metadata
                    ))

                if len(retrieved) >= 4:
                    break

        except Exception:
            retrieved = []

    if not retrieved:

        return (
            f"No relevant passage was found in the {source_description}.",
            []
        )

    response = (
        f"Retrieved relevant information from the "
        f"{source_description}. Review the evidence below "
        f"to find the answer."
    )

    return response, retrieved


# =========================================================
# 14. SIDEBAR
# =========================================================

with st.sidebar:

    st.markdown("## Vidya-Daan")

    st.caption("CSR Education Grant Proposal Auditor")

    st.divider()

    st.markdown("### Project Overview")

    st.write(
        "A document analysis prototype that helps identify "
        "missing information and review education grant proposals."
    )

    st.divider()

    st.markdown("### Main Modules")

    st.write("1. PDF Document Upload")
    st.write("2. Text Extraction")
    st.write("3. Document Indexing")
    st.write("4. Requirement Analysis")
    st.write("5. Readiness Indicator")
    st.write("6. Question Answering")

    st.divider()

    st.caption(
        "Academic demonstration prototype. "
        "Not an official CSR approval or legal verification system."
    )


# =========================================================
# 15. MAIN DASHBOARD
# =========================================================

st.markdown(
    '<div class="main-title">Vidya-Daan</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="sub-title">'
    'CSR Education Grant Proposal Auditor'
    '</div>',
    unsafe_allow_html=True
)

st.write("")

st.write(
    "Upload an NGO education grant proposal and optional CSR "
    "reference guidelines. The application analyses the proposal "
    "separately and uses reference documents for supporting information."
)

st.divider()


# =========================================================
# 16. DOCUMENT UPLOAD
# =========================================================

st.markdown(
    '<div class="section-heading">1. Upload Documents</div>',
    unsafe_allow_html=True
)

uploaded_files = st.file_uploader(
    "Upload your NGO proposal and CSR reference PDFs",
    type=["pdf"],
    accept_multiple_files=True
)

st.caption(
    "Reference files should contain words such as reference, "
    "guidelines, checklist, policy or manual in their filenames."
)

analyse_button = st.button(
    "Analyse Documents",
    type="primary",
    use_container_width=True
)


# =========================================================
# 17. DOCUMENT ANALYSIS
# =========================================================

if analyse_button:

    if not uploaded_files:

        st.warning("Please upload at least one PDF document.")

    else:

        try:

            with st.spinner("Reading and classifying documents..."):

                (
                    proposal_pages,
                    reference_pages,
                    proposal_names,
                    reference_names
                ) = classify_documents(uploaded_files)

            if not proposal_pages:

                st.error(
                    "No NGO proposal was identified. "
                    "Please check your proposal filename."
                )

            else:

                with st.spinner("Extracting and indexing text..."):

                    proposal_chunks, proposal_metadata = create_chunks(
                        proposal_pages
                    )

                    reference_chunks, reference_metadata = create_chunks(
                        reference_pages
                    )

                    all_chunks = (
                        proposal_chunks + reference_chunks
                    )

                    all_metadata = (
                        proposal_metadata + reference_metadata
                    )

                    vector_database, vector_error = (
                        create_vector_database(
                            all_chunks,
                            all_metadata
                        )
                    )

                proposal_text = "\n\n".join(
                    page["text"]
                    for page in proposal_pages
                )

                with st.spinner("Analysing proposal requirements..."):

                    requirement_results = analyse_requirements(
                        proposal_text
                    )

                    readiness_score = calculate_readiness(
                        requirement_results
                    )

                    improvements = find_improvements(
                        requirement_results
                    )

                st.session_state["analysed"] = True

                st.session_state["proposal_names"] = proposal_names
                st.session_state["reference_names"] = reference_names

                st.session_state["proposal_pages"] = proposal_pages
                st.session_state["reference_pages"] = reference_pages

                st.session_state["proposal_text"] = proposal_text

                st.session_state["proposal_chunks"] = proposal_chunks
                st.session_state["proposal_metadata"] = proposal_metadata

                st.session_state["reference_chunks"] = reference_chunks
                st.session_state["reference_metadata"] = reference_metadata

                st.session_state["all_chunks"] = all_chunks
                st.session_state["all_metadata"] = all_metadata

                st.session_state["vector_database"] = vector_database
                st.session_state["vector_error"] = vector_error

                st.session_state["requirements"] = requirement_results
                st.session_state["readiness"] = readiness_score
                st.session_state["improvements"] = improvements

                st.success(
                    "Document analysis completed successfully!"
                )

        except Exception as error:

            st.error(
                f"An error occurred during analysis: {error}"
            )


# =========================================================
# 18. DISPLAY RESULTS
# =========================================================

if st.session_state.get("analysed", False):

    results = st.session_state["requirements"]

    score = st.session_state["readiness"]

    st.divider()

    st.markdown(
        '<div class="section-heading">2. Analysis Dashboard</div>',
        unsafe_allow_html=True
    )

    present_count = sum(
        item["status"] == "Present"
        for item in results
    )

    missing_count = sum(
        item["status"] == "Missing"
        for item in results
    )

    review_count = sum(
        item["status"] == "Needs Review"
        for item in results
    )

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Readiness Indicator", f"{score}/100")
    col2.metric("Present", present_count)
    col3.metric("Missing", missing_count)
    col4.metric("Needs Review", review_count)

    st.progress(score / 100)

    st.caption(
        "Illustrative checklist score: Present = 10 points, "
        "Needs Review = 5 points, Missing = 0 points per item. "
        "This is not an official CSR score."
    )

    st.info(
        "A Present status means matching text was found. "
        "It does not confirm document authenticity or legal validity."
    )

    # -----------------------------------------------------
    # DOCUMENT LIST
    # -----------------------------------------------------

    st.divider()

    st.markdown(
        '<div class="section-heading">3. Uploaded Documents</div>',
        unsafe_allow_html=True
    )

    st.markdown("**NGO Proposal — requirement analysis**")

    for name in st.session_state["proposal_names"]:
        st.write(f"• {name}")

    st.markdown("**CSR Reference Documents — reference material**")

    if st.session_state["reference_names"]:

        for name in st.session_state["reference_names"]:
            st.write(f"• {name}")

    else:

        st.write("No reference document uploaded.")

    st.write(
        f"**Proposal pages:** "
        f"{len(st.session_state['proposal_pages'])}"
    )

    st.write(
        f"**Reference pages:** "
        f"{len(st.session_state['reference_pages'])}"
    )

    st.write(
        f"**Total indexed sections:** "
        f"{len(st.session_state['all_chunks'])}"
    )

    if st.session_state["vector_error"]:

        st.warning(
            "LanceDB indexing was unavailable. "
            "Keyword retrieval is still available. "
            f"Details: {st.session_state['vector_error']}"
        )

    else:

        st.success("LanceDB document indexing completed successfully.")

    # -----------------------------------------------------
    # REQUIREMENT ANALYSIS
    # -----------------------------------------------------

    st.divider()

    st.markdown(
        '<div class="section-heading">4. Requirement Analysis</div>',
        unsafe_allow_html=True
    )

    for result in results:

        if result["status"] == "Present":
            symbol = "🟢"

        elif result["status"] == "Needs Review":
            symbol = "🟡"

        else:
            symbol = "🔴"

        with st.expander(
            f"{symbol} {result['name']} — {result['status']}"
        ):

            st.write(
                f"**Evidence groups found:** "
                f"{result['completed']} of {result['total']}"
            )

            if result["evidence"]:

                st.markdown("**Related proposal text:**")

                for evidence in result["evidence"][:5]:
                    st.write(f"> {evidence}")

            else:

                st.write(
                    "No matching evidence was found in the NGO proposal."
                )

            if result["status"] != "Present":

                st.markdown("**Suggested improvement:**")

                st.write(
                    SUGGESTIONS[result["name"]]
                )

    # -----------------------------------------------------
    # MISSING INFORMATION
    # -----------------------------------------------------

    st.divider()

    st.markdown(
        '<div class="section-heading">5. Missing Information</div>',
        unsafe_allow_html=True
    )

    incomplete = [
        item for item in results
        if item["status"] != "Present"
    ]

    if incomplete:

        for item in incomplete:

            st.markdown(
                f"**{item['name']} — {item['status']}**"
            )

            st.write(
                SUGGESTIONS[item["name"]]
            )

    else:

        st.success(
            "All checklist items have matching text. "
            "This does not mean they have been legally verified."
        )

    # -----------------------------------------------------
    # QUESTION ANSWERING
    # -----------------------------------------------------

    st.divider()

    st.markdown(
        '<div class="section-heading">6. Ask Questions About Documents</div>',
        unsafe_allow_html=True
    )

    st.write(
        "Project questions search the NGO proposal. "
        "CSR rule and compliance questions search the reference guidelines."
    )

    question = st.text_input(
        "Enter your question",
        placeholder="Example: What is the total project budget requested by the NGO?",
        key="document_question"
    )

    if st.button("Find Answer", key="find_answer_button"):

        if not question.strip():

            st.warning("Please enter a question.")

        else:

            source_type = identify_question_source(question)

            if source_type == "reference":
                selected_source = "CSR Reference Guidelines"
            else:
                selected_source = "NGO Education Grant Proposal"

            answer, retrieved = get_answer(
                question,
                st.session_state["vector_database"],
                st.session_state["proposal_chunks"],
                st.session_state["proposal_metadata"],
                st.session_state["reference_chunks"],
                st.session_state["reference_metadata"]
            )

            st.markdown("**Response**")

            st.write(answer)

            st.caption(
                f"Selected source: {selected_source}"
            )

            if retrieved:

                st.markdown("**Supporting Evidence**")

                for index, (passage, meta) in enumerate(
                    retrieved,
                    start=1
                ):

                    source_name = meta.get(
                        "source",
                        "Uploaded PDF"
                    )

                    page_number = meta.get(
                        "page",
                        "Unknown"
                    )

                    with st.expander(
                        f"Evidence {index} — {source_name}, Page {page_number}"
                    ):

                        st.write(passage)

            else:

                st.warning(
                    "Try rephrasing your question using words "
                    "that appear in the document."
                )

    # -----------------------------------------------------
    # EXTRACTED TEXT PREVIEW
    # -----------------------------------------------------

    st.divider()

    st.markdown(
        '<div class="section-heading">7. Extracted Text Preview</div>',
        unsafe_allow_html=True
    )

    with st.expander("View NGO proposal text"):

        st.text(
            st.session_state["proposal_text"][:12000]
        )

    if st.session_state["reference_pages"]:

        reference_text = "\n\n".join(
            page["text"]
            for page in st.session_state["reference_pages"]
        )

        with st.expander("View CSR reference text"):

            st.text(reference_text[:12000])

    st.divider()

    st.caption(
        "Vidya-Daan | CSR Education Grant Proposal Auditor | "
        "Academic Project Demonstration"
    )


# =========================================================
# 19. INITIAL WORKFLOW
# =========================================================

else:

    st.info(
        "Upload your NGO proposal and optional CSR reference PDF, "
        "then select Analyse Documents to begin."
    )

    st.markdown("### Application Workflow")

    steps = [
        "PDF Upload",
        "Text Extraction",
        "Document Indexing",
        "Requirement Analysis",
        "Evidence Retrieval"
    ]

    columns = st.columns(5)

    for column, step in zip(columns, steps):

        with column:

            st.markdown(
                f"""
                <div class="workflow-card">
                    {step}
                </div>
                """,
                unsafe_allow_html=True
            )

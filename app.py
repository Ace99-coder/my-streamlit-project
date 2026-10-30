"""
app.py
--------
AI Based Plagiarism Detection System — Streamlit entry point.

This file wires together every module into the 7 UI pages described in
the project spec. It contains NO detection/classification/scoring logic
of its own — every number shown here comes from calling into modules/.
That separation is what makes the system testable (see tests/) and lets
the adaptive engine be reused headlessly (e.g. from a script or notebook)
without Streamlit at all.

Run with:  streamlit run app.py
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from database.database import Database
from modules.adaptive_weighting import AdaptiveWeightingEngine, WeightSelection
from modules.document_processor import Document, DocumentProcessor
from modules.exact_matcher import ExactMatcher
from modules.explainability_engine import Explanation, ExplainabilityEngine
from modules.hybrid_scorer import HybridScorer
from modules.ngram_detector import NgramDetector
from modules.plagiarism_classifier import ClassificationResult, PlagiarismClassifier
from modules.report_generator import ReportGenerator
from modules.semantic_detector import SemanticDetector
from modules.sentence_analyzer import SentenceAnalyzer, SentenceMatch
from modules.text_preprocessor import ProcessedText, TextPreprocessor
from modules.tfidf_detector import TfidfDetector
from utils.helpers import load_config, now_iso, PROJECT_ROOT
from utils.validators import ValidationError, validate_config_weights

logging.basicConfig(level=logging.INFO)

st.set_page_config(
    page_title="Adaptive AI-Based Plagiarism Detection System",
    page_icon="🔎",
    layout="wide",
)

DETECTOR_LABELS = {
    "exact": "Exact Match",
    "tfidf": "TF-IDF",
    "ngram": "N-Gram",
    "semantic": "Semantic (AI)",
}


# =============================================================================
# Cached resources — loaded once per session, not once per rerun.
# =============================================================================

@st.cache_resource(show_spinner=False)
def get_config() -> dict:
    cfg = load_config()
    validate_config_weights(cfg)  # fail fast if config.yaml was hand-edited into an invalid state
    return cfg


@st.cache_resource(show_spinner=False)
def get_pipeline():
    """Build every module once. The SemanticDetector loads (or attempts to
    download) the Sentence-BERT model, which is the single most expensive
    step in the whole app — caching this is what keeps page navigation
    fast after the first analysis."""
    cfg = get_config()
    return {
        "processor": DocumentProcessor(cfg),
        "preprocessor": TextPreprocessor(cfg),
        "exact": ExactMatcher(cfg),
        "tfidf": TfidfDetector(cfg),
        "ngram": NgramDetector(cfg),
        "semantic": SemanticDetector(cfg),
        "classifier": PlagiarismClassifier(cfg),
        "adaptive": AdaptiveWeightingEngine(cfg),
        "scorer": HybridScorer(cfg["scoring"]["plagiarism_suspected_threshold"]),
        "sentence_analyzer": SentenceAnalyzer(config=cfg),
        "explainer": ExplainabilityEngine(cfg),
        "report_generator": ReportGenerator(),
    }


@st.cache_resource(show_spinner=False)
def get_db() -> Optional[Database]:
    try:
        return Database()
    except Exception as exc:  # pragma: no cover - environment dependent
        logging.warning("Database unavailable, continuing without persistence: %s", exc)
        return None


# =============================================================================
# Core analysis pipeline — pure function, no Streamlit widgets inside.
# Connects Modules 1-2-3-4-5-6-7 for ONE (suspicious, source) document pair.
# =============================================================================

def analyze_pair(suspicious: Document, source: Document) -> dict:
    pipeline = get_pipeline()
    cfg = get_config()

    p_susp: ProcessedText = pipeline["preprocessor"].process(suspicious.raw_text)
    p_src: ProcessedText = pipeline["preprocessor"].process(source.raw_text)

    exact = pipeline["exact"].document_score(p_susp.original_sentences, p_src.original_sentences)
    tfidf = pipeline["tfidf"].document_score(" ".join(p_susp.clean_sentences), " ".join(p_src.clean_sentences))
    ngram = pipeline["ngram"].document_score(p_susp.tokens, p_src.tokens)
    semantic = pipeline["semantic"].document_score(p_susp.original_sentences, p_src.original_sentences)

    classification: ClassificationResult = pipeline["classifier"].classify(exact, tfidf, ngram, semantic)
    weight_selection: WeightSelection = pipeline["adaptive"].select(classification.plagiarism_type)
    fixed_selection = pipeline["adaptive"].fixed_weights()

    detector_scores = {"exact": exact, "tfidf": tfidf, "ngram": ngram, "semantic": semantic}
    comparison = pipeline["scorer"].compare(detector_scores, weight_selection.weights, fixed_selection.weights)
    hybrid_result = comparison["adaptive"]

    sentence_matches: List[SentenceMatch] = pipeline["sentence_analyzer"].analyze(
        p_susp.original_sentences, p_src.original_sentences, source.name
    )

    explanation: Explanation = pipeline["explainer"].explain(
        hybrid_result, classification, weight_selection, sentence_matches
    )

    return {
        "source_name": source.name,
        "detector_scores": detector_scores,
        "classification": classification,
        "weight_selection": weight_selection,
        "fixed_selection": fixed_selection,
        "comparison": comparison,
        "hybrid_result": hybrid_result,
        "sentence_matches": sentence_matches,
        "explanation": explanation,
        "processed_susp": p_susp,
        "processed_src": p_src,
    }


def run_full_analysis(suspicious: Document, sources: List[Document]) -> dict:
    """MODULE 8: Multiple Document Comparison.
    Runs analyze_pair against every reference document, then determines
    which source contributes the most suspicious content (highest adaptive
    score) as the 'primary' result the dashboard opens on."""
    per_source = {src.name: analyze_pair(suspicious, src) for src in sources}
    best_name = max(per_source, key=lambda n: per_source[n]["hybrid_result"].score)

    db = get_db()
    if db is not None:
        try:
            _persist_results(db, suspicious, sources, per_source)
        except Exception as exc:  # pragma: no cover - persistence is best-effort
            logging.warning("Could not persist analysis results: %s", exc)

    return {
        "suspicious": suspicious,
        "sources": sources,
        "per_source": per_source,
        "best_source_name": best_name,
    }


def _persist_results(db: Database, suspicious: Document, sources: List[Document], per_source: dict) -> None:
    susp_id = db.insert_document(suspicious.name, suspicious.raw_text)
    for src in sources:
        src_id = db.insert_document(src.name, src.raw_text)
        r = per_source[src.name]
        result_id = db.insert_result(
            suspicious_document_id=susp_id,
            source_document_id=src_id,
            exact_similarity=r["detector_scores"]["exact"],
            tfidf_similarity=r["detector_scores"]["tfidf"],
            ngram_similarity=r["detector_scores"]["ngram"],
            semantic_similarity=r["detector_scores"]["semantic"],
            detected_plagiarism_type=r["classification"].plagiarism_type,
            classification_confidence=r["classification"].confidence,
            fixed_score=r["comparison"]["fixed"].score,
            adaptive_score=r["hybrid_result"].score,
            analysis_date=now_iso(),
        )
        db.insert_sentence_matches(result_id, [
            {
                "suspicious_sentence": m.suspicious_sentence,
                "source_sentence": m.source_sentence,
                "source_document": m.source_document,
                "similarity_score": m.final_similarity,
                "plagiarism_type": m.classification,
            }
            for m in r["sentence_matches"]
        ])


# =============================================================================
# Small shared UI helpers
# =============================================================================

def _score_color(score: float) -> str:
    cfg = get_config()
    if score >= cfg["scoring"]["high_similarity"]:
        return "#d64545"
    if score >= cfg["scoring"]["medium_similarity"]:
        return "#e0a83a"
    return "#3a9e5f"


def _detector_bar_chart(detector_scores: Dict[str, float], title: str = "Detection Method Comparison") -> go.Figure:
    labels = [DETECTOR_LABELS[k] for k in ["exact", "tfidf", "ngram", "semantic"]]
    values = [detector_scores[k] for k in ["exact", "tfidf", "ngram", "semantic"]]
    fig = px.bar(
        x=labels, y=values, text=[f"{v:.1f}%" for v in values],
        labels={"x": "Detection Method", "y": "Similarity (%)"}, title=title,
        color=values, color_continuous_scale="RdYlGn_r", range_color=[0, 100],
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(yaxis_range=[0, 105], coloraxis_showscale=False, showlegend=False)
    return fig


def _weight_bar_chart(weights: List[float], title: str = "Adaptive Weight Selection") -> go.Figure:
    labels = [DETECTOR_LABELS[k] for k in ["exact", "tfidf", "ngram", "semantic"]]
    values = [w * 100 for w in weights]
    fig = px.bar(
        x=labels, y=values, text=[f"{v:.0f}%" for v in values],
        labels={"x": "Detection Method", "y": "Weight (%)"}, title=title,
        color_discrete_sequence=["#3a6ea5"],
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(yaxis_range=[0, 70])
    return fig


def _radar_chart(detector_scores: Dict[str, float], title: str = "Similarity Pattern") -> go.Figure:
    labels = [DETECTOR_LABELS[k] for k in ["exact", "tfidf", "ngram", "semantic"]]
    values = [detector_scores[k] for k in ["exact", "tfidf", "ngram", "semantic"]]
    fig = go.Figure(
        data=go.Scatterpolar(r=values + values[:1], theta=labels + labels[:1], fill="toself", name="Similarity")
    )
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
        showlegend=False, title=title,
    )
    return fig


def _adaptive_vs_fixed_chart(comparison: dict) -> go.Figure:
    fig = go.Figure(data=[
        go.Bar(name="Traditional Fixed Score", x=["Score"], y=[comparison["fixed"].score], marker_color="#9aa3af"),
        go.Bar(name="Adaptive Score", x=["Score"], y=[comparison["adaptive"].score], marker_color="#3a6ea5"),
    ])
    fig.update_layout(barmode="group", yaxis_range=[0, 105], title="Adaptive vs Fixed-Weight Scoring")
    return fig


def _sentence_distribution_chart(matches: List[SentenceMatch]) -> go.Figure:
    values = [m.final_similarity for m in matches]
    fig = px.histogram(
        x=values, nbins=15, labels={"x": "Sentence Similarity (%)", "y": "Number of Sentences"},
        title="Sentence-Level Similarity Distribution",
    )
    cfg = get_config()
    fig.add_vline(
        x=cfg["scoring"]["sentence_flag_threshold"], line_dash="dash", line_color="red",
        annotation_text="Flag threshold",
    )
    return fig


def _similarity_matrix_df(per_source: dict) -> pd.DataFrame:
    rows = []
    for name, r in per_source.items():
        row = {"Source Document": name}
        row.update({DETECTOR_LABELS[k]: r["detector_scores"][k] for k in ["exact", "tfidf", "ngram", "semantic"]})
        row["Adaptive Score"] = r["hybrid_result"].score
        row["Detected Type"] = r["classification"].plagiarism_type
        rows.append(row)
    return pd.DataFrame(rows)


# =============================================================================
# PAGE 1: HOME
# =============================================================================

PAGE_ICONS = {
    "Home": "🏠",
    "Upload & Analyze": "📤",
    "Results": "📊",
    "Adaptive Analysis": "🧠",
    "Sentence Analysis": "🔬",
    "Analytics": "📈",
    "Report": "📄",
}

_HOME_CSS = """
<style>
.block-container { padding-top: 2rem; max-width: 1200px; }
.pd-hero {
    background: linear-gradient(135deg, #1e3a8a 0%, #3a6ea5 55%, #06b6d4 100%);
    border-radius: 20px; padding: 3.2rem 2.8rem; color: #ffffff; margin-bottom: 1rem;
    box-shadow: 0 10px 30px rgba(30, 58, 138, 0.25);
}
.pd-hero .pd-badge {
    display: inline-block; background: rgba(255,255,255,0.18); border-radius: 999px;
    padding: 0.25rem 0.9rem; font-size: 0.8rem; font-weight: 600; letter-spacing: 0.04em;
    margin-bottom: 1rem;
}
.pd-hero .pd-title { font-size: 2.8rem; font-weight: 800; line-height: 1.15; margin: 0 0 0.8rem 0; }
.pd-hero .pd-tagline { font-size: 1.15rem; opacity: 0.92; max-width: 760px; margin: 0; }
.pd-section-title { font-size: 1.5rem; font-weight: 700; margin: 2.2rem 0 0.2rem 0; }
.pd-section-sub { opacity: 0.7; margin-bottom: 1rem; }
.pd-stat {
    border: 1px solid rgba(128,128,128,0.25); border-radius: 14px; padding: 1.1rem 1rem;
    text-align: center; background: rgba(128,128,128,0.06);
}
.pd-stat .pd-num { font-size: 2.1rem; font-weight: 800; color: #3a6ea5; line-height: 1.1; }
.pd-stat .pd-lbl { font-size: 0.85rem; opacity: 0.75; margin-top: 0.25rem; }
.pd-card {
    border: 1px solid rgba(128,128,128,0.25); border-radius: 14px; padding: 1.3rem 1.2rem;
    background: rgba(128,128,128,0.06); min-height: 190px; margin-bottom: 1rem;
    transition: transform 0.15s ease, box-shadow 0.15s ease;
}
.pd-card:hover { transform: translateY(-3px); box-shadow: 0 8px 22px rgba(0,0,0,0.12); }
.pd-card .pd-icon { font-size: 1.8rem; margin-bottom: 0.4rem; }
.pd-card .pd-card-title { font-size: 1.05rem; font-weight: 700; margin-bottom: 0.35rem; }
.pd-card .pd-card-body { font-size: 0.92rem; opacity: 0.8; line-height: 1.5; }
.pd-flow { display: flex; flex-wrap: wrap; align-items: stretch; gap: 0.5rem; }
.pd-step {
    flex: 1 1 140px; border: 1px solid rgba(128,128,128,0.25); border-radius: 12px;
    padding: 0.9rem 0.8rem; text-align: center; background: rgba(58,110,165,0.10);
}
.pd-step .pd-step-num {
    display: inline-block; width: 1.7rem; height: 1.7rem; line-height: 1.7rem; border-radius: 50%;
    background: #3a6ea5; color: #fff; font-weight: 700; font-size: 0.85rem; margin-bottom: 0.4rem;
}
.pd-step .pd-step-title { font-weight: 700; font-size: 0.95rem; }
.pd-step .pd-step-body { font-size: 0.8rem; opacity: 0.75; margin-top: 0.2rem; }
.pd-cta {
    border-radius: 18px; padding: 2rem 2rem; margin-top: 2rem; text-align: center;
    background: rgba(58,110,165,0.12); border: 1px solid rgba(58,110,165,0.35);
}
.pd-cta .pd-cta-title { font-size: 1.5rem; font-weight: 800; margin-bottom: 0.3rem; }
.pd-cta .pd-cta-body { opacity: 0.8; }
</style>
"""


def _go_to(page_name: str, use_samples: bool = False) -> None:
    """Button callback: switch the sidebar page (and optionally preselect the
    bundled sample documents on the Upload page)."""
    st.session_state["nav_page"] = page_name
    if use_samples:
        st.session_state["susp_source"] = "Use a bundled sample"
        st.session_state["src_source"] = "Use bundled sample reference"


def _stat_card(number: str, label: str) -> str:
    return f'<div class="pd-stat"><div class="pd-num">{number}</div><div class="pd-lbl">{label}</div></div>'


def _feature_card(icon: str, title: str, body: str) -> str:
    return (
        f'<div class="pd-card"><div class="pd-icon">{icon}</div>'
        f'<div class="pd-card-title">{title}</div><div class="pd-card-body">{body}</div></div>'
    )


def _flow_step(num: int, title: str, body: str) -> str:
    return (
        f'<div class="pd-step"><div class="pd-step-num">{num}</div>'
        f'<div class="pd-step-title">{title}</div><div class="pd-step-body">{body}</div></div>'
    )


def page_home() -> None:
    st.markdown(_HOME_CSS, unsafe_allow_html=True)

    # ---- Hero ---------------------------------------------------------------
    st.markdown(
        '<div class="pd-hero">'
        '<div class="pd-badge">🔎 ADAPTIVE · EXPLAINABLE · AI-POWERED</div>'
        '<div class="pd-title">AI Based Plagiarism<br>Detection System</div>'
        '<p class="pd-tagline">Detect, classify, and explain plagiarism using lexical and semantic '
        'similarity analysis. Know not just <em>how much</em> a document matches, '
        'but <em>what kind</em> of match it is.</p>'
        '</div>',
        unsafe_allow_html=True,
    )

    b1, b2, _ = st.columns([1.2, 1.2, 3])
    b1.button(
        "🚀 Start Analysis", type="primary", use_container_width=True,
        on_click=_go_to, args=("Upload & Analyze",),
    )
    b2.button(
        "🧪 Try Sample Documents", use_container_width=True,
        on_click=_go_to, args=("Upload & Analyze", True),
    )

    # ---- Stats --------------------------------------------------------------
    st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)
    s1, s2, s3, s4 = st.columns(4)
    s1.markdown(_stat_card("4", "Detection techniques"), unsafe_allow_html=True)
    s2.markdown(_stat_card("6", "Plagiarism types classified"), unsafe_allow_html=True)
    s3.markdown(_stat_card("PDF · DOCX · TXT", "Supported formats"), unsafe_allow_html=True)
    s4.markdown(_stat_card("Multi-source", "Compare against many references"), unsafe_allow_html=True)

    # ---- Features -----------------------------------------------------------
    st.markdown('<div class="pd-section-title">What it can do</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="pd-section-sub">Every score in the report traces back to evidence you can inspect.</div>',
        unsafe_allow_html=True,
    )
    features = [
        ("🧩", "Multi-technique detection",
         "Exact matching, TF-IDF, n-gram overlap and Sentence-BERT semantic similarity, each computed independently."),
        ("🏷️", "Plagiarism type classification",
         "Direct Copy, Partial Copy, Paraphrased, Structural, Mixed, or Low/No Plagiarism."),
        ("⚖️", "Adaptive weighting engine",
         "Re-weights the four techniques based on the detected pattern instead of using one formula for every document."),
        ("💡", "Explainable results",
         "Plain-language reasons tied to specific scores, weights and sentence matches."),
        ("📊", "Adaptive vs. fixed comparison",
         "See exactly what the adaptive engine changed versus a traditional fixed-weight baseline."),
        ("🔬", "Sentence-level evidence",
         "Pinpoints which sentences matched which source, and how they matched."),
    ]
    for row_start in (0, 3):
        cols = st.columns(3)
        for col, (icon, title, body) in zip(cols, features[row_start:row_start + 3]):
            col.markdown(_feature_card(icon, title, body), unsafe_allow_html=True)

    # ---- Workflow -----------------------------------------------------------
    st.markdown('<div class="pd-section-title">How it works</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="pd-section-sub">From upload to report in a single pipeline.</div>',
        unsafe_allow_html=True,
    )
    steps = [
        ("Upload", "Suspicious + reference documents"),
        ("Preprocess", "Extract and clean text"),
        ("Detect", "Exact · TF-IDF · N-Gram · Semantic"),
        ("Classify", "Identify the plagiarism type"),
        ("Weigh", "Pick adaptive weights"),
        ("Explain", "Score, evidence and PDF report"),
    ]
    st.markdown(
        '<div class="pd-flow">' + "".join(_flow_step(i, t, b) for i, (t, b) in enumerate(steps, start=1)) + "</div>",
        unsafe_allow_html=True,
    )

    st.info(
        "**The innovation, in one line:** instead of asking *'how similar is this document?'*, "
        "this system first asks *'what KIND of similarity is this?'* and lets that answer decide "
        "how the final score is computed."
    )

    # ---- Closing call to action --------------------------------------------
    st.markdown(
        '<div class="pd-cta"><div class="pd-cta-title">Ready to check a document?</div>'
        '<div class="pd-cta-body">Upload your own files or load the bundled samples to see the system in action.</div></div>',
        unsafe_allow_html=True,
    )
    _, mid, _ = st.columns([2, 1.5, 2])
    mid.button(
        "📤 Go to Upload & Analyze", type="primary", use_container_width=True,
        on_click=_go_to, args=("Upload & Analyze",), key="home_cta_bottom",
    )

# =============================================================================
# PAGE 2: UPLOAD AND ANALYZE
# =============================================================================

def _load_document_from_upload(uploaded_file) -> Document:
    processor = get_pipeline()["processor"]
    return processor.load_from_bytes(uploaded_file.read(), uploaded_file.name)


def _load_sample_documents() -> Dict[str, Document]:
    processor = get_pipeline()["processor"]
    samples_dir = PROJECT_ROOT / "data" / "samples"
    docs = {}
    for path in sorted(samples_dir.glob("*.txt")):
        docs[path.stem] = processor.load(path)
    return docs


def page_upload_analyze() -> None:
    st.title("📤 Upload & Analyze")
    cfg = get_config()

    st.subheader("1. Suspicious Document")
    susp_source = st.radio(
        "Provide the suspicious document via:",
        ["Upload a file", "Use a bundled sample"],
        horizontal=True, key="susp_source",
    )

    suspicious_doc: Optional[Document] = None
    if susp_source == "Upload a file":
        f = st.file_uploader(
            "Upload the document to check", type=["pdf", "txt", "docx"], key="susp_upload"
        )
        if f is not None:
            try:
                suspicious_doc = _load_document_from_upload(f)
            except Exception as exc:
                st.error(f"Could not read uploaded file: {exc}")
    else:
        samples = _load_sample_documents()
        sample_names = [n for n in samples if n != "source_reference"]
        choice = st.selectbox("Choose a sample suspicious document:", sample_names)
        if choice:
            suspicious_doc = samples[choice]
            with st.expander("Preview sample text"):
                st.write(suspicious_doc.raw_text)

    st.subheader("2. Reference Document(s)")
    src_source = st.radio(
        "Provide reference document(s) via:",
        ["Upload file(s)", "Use bundled sample reference"],
        horizontal=True, key="src_source",
    )

    source_docs: List[Document] = []
    if src_source == "Upload file(s)":
        files = st.file_uploader(
            "Upload one or more reference documents", type=["pdf", "txt", "docx"],
            accept_multiple_files=True, key="src_upload",
        )
        for f in files or []:
            try:
                source_docs.append(_load_document_from_upload(f))
            except Exception as exc:
                st.error(f"Could not read '{f.name}': {exc}")
    else:
        samples = _load_sample_documents()
        if "source_reference" in samples:
            source_docs.append(samples["source_reference"])
            with st.expander("Preview sample reference text"):
                st.write(samples["source_reference"].raw_text)

    st.divider()
    ready = suspicious_doc is not None and len(source_docs) > 0
    if not ready:
        st.warning("Provide both a suspicious document and at least one reference document to continue.")

    if st.button("🚀 Run Adaptive Plagiarism Analysis", type="primary", disabled=not ready):
        with st.spinner("Running detectors, classifying pattern, selecting adaptive weights..."):
            try:
                results = run_full_analysis(suspicious_doc, source_docs)
                st.session_state["analysis_results"] = results
                st.success(
                    f"Analysis complete — compared against {len(source_docs)} reference "
                    f"document(s). Open **Results** in the sidebar to view findings."
                )
            except ValidationError as exc:
                st.error(f"Validation error: {exc}")
            except Exception as exc:  # pragma: no cover
                st.error(f"Analysis failed: {exc}")
                logging.exception("Analysis pipeline failed")


# =============================================================================
# Shared: source-result selector (used by Results / Adaptive / Sentence pages)
# =============================================================================

def _require_results() -> Optional[dict]:
    results = st.session_state.get("analysis_results")
    if results is None:
        st.info("No analysis yet. Go to **Upload & Analyze** to run one.")
        return None
    return results


def _select_source_result(results: dict) -> tuple[str, dict]:
    names = list(results["per_source"].keys())
    default_idx = names.index(results["best_source_name"]) if results["best_source_name"] in names else 0
    if len(names) > 1:
        chosen = st.selectbox(
            "Reference document to view results for:", names, index=default_idx,
            help="The most similar source is pre-selected. Switch to inspect results against a different reference.",
        )
    else:
        chosen = names[0]
    return chosen, results["per_source"][chosen]


# =============================================================================
# PAGE 3: RESULTS
# =============================================================================

def page_results() -> None:
    st.title("📊 Results")
    results = _require_results()
    if results is None:
        return

    name, r = _select_source_result(results)
    hybrid = r["hybrid_result"]
    classification = r["classification"]

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Adaptive Plagiarism Score", f"{hybrid.score:.1f}%")
    c2.metric("Detected Type", classification.plagiarism_type)
    c3.metric("Classification Confidence", f"{classification.confidence:.1f}%")
    c4.metric("Most Similar Source", results["best_source_name"])
    c5.metric("Suspicious Sentences", f"{sum(m.is_flagged for m in r['sentence_matches'])} / {len(r['sentence_matches'])}")

    if hybrid.suspected:
        st.error(f"⚠️ Plagiarism suspected against **{name}** (score ≥ {get_config()['scoring']['plagiarism_suspected_threshold']}%).")
    else:
        st.success(f"✅ No significant plagiarism suspected against **{name}**.")

    st.divider()
    st.subheader("Detection Method Comparison")
    st.plotly_chart(_detector_bar_chart(r["detector_scores"]), use_container_width=True)

    cols = st.columns(4)
    for col, key in zip(cols, ["exact", "tfidf", "ngram", "semantic"]):
        col.metric(DETECTOR_LABELS[key], f"{r['detector_scores'][key]:.1f}%")

    if len(results["per_source"]) > 1:
        st.divider()
        st.subheader("Document Similarity Matrix (Module 8)")
        st.dataframe(_similarity_matrix_df(results["per_source"]), use_container_width=True)


# =============================================================================
# PAGE 4: ADAPTIVE ANALYSIS
# =============================================================================

def page_adaptive_analysis() -> None:
    st.title("🧠 Adaptive Analysis")
    results = _require_results()
    if results is None:
        return

    name, r = _select_source_result(results)
    classification = r["classification"]
    weight_selection = r["weight_selection"]
    comparison = r["comparison"]

    st.subheader("Detected Similarity Pattern")
    st.plotly_chart(_radar_chart(r["detector_scores"]), use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Detected Plagiarism Type", classification.plagiarism_type)
        st.metric("Classification Confidence", f"{classification.confidence:.1f}%")
    with col2:
        st.metric("Traditional Fixed Score", f"{comparison['fixed'].score:.1f}%")
        st.metric("Adaptive Score", f"{comparison['adaptive'].score:.1f}%", delta=f"{comparison['difference']:+.1f}")

    st.divider()
    st.subheader("Selected Dynamic Weights")
    st.plotly_chart(_weight_bar_chart(weight_selection.weights), use_container_width=True)
    st.caption(f"**Why these weights:** {weight_selection.reason}")

    st.divider()
    st.subheader("Module 10 — Adaptive vs. Fixed-Weight Scoring")
    st.plotly_chart(_adaptive_vs_fixed_chart(comparison), use_container_width=True)
    fixed_w = r["fixed_selection"].weights
    st.caption(
        f"Fixed baseline weights — Exact {fixed_w[0]*100:.0f}%, TF-IDF {fixed_w[1]*100:.0f}%, "
        f"N-Gram {fixed_w[2]*100:.0f}%, Semantic {fixed_w[3]*100:.0f}% (same for every document, "
        f"regardless of detected pattern)."
    )

    st.divider()
    st.subheader("Explanation")
    for i, reason in enumerate(r["explanation"].reasons, start=1):
        st.write(f"**{i}.** {reason}")
    st.info(r["explanation"].conclusion)


# =============================================================================
# PAGE 5: SENTENCE ANALYSIS
# =============================================================================

def page_sentence_analysis() -> None:
    st.title("🔬 Sentence Analysis")
    results = _require_results()
    if results is None:
        return

    name, r = _select_source_result(results)
    matches = r["sentence_matches"]

    if not matches:
        st.info("No sentences long enough to compare were found.")
        return

    col1, col2 = st.columns(2)
    with col1:
        filter_level = st.selectbox(
            "Filter by similarity", ["All", "High similarity (≥75%)", "Medium similarity (45-75%)", "Low similarity (<45%)"]
        )
    with col2:
        types_available = sorted({m.classification for m in matches})
        filter_type = st.selectbox("Filter by plagiarism type", ["All"] + types_available)

    def keep(m: SentenceMatch) -> bool:
        if filter_level == "High similarity (≥75%)" and m.final_similarity < 75:
            return False
        if filter_level == "Medium similarity (45-75%)" and not (45 <= m.final_similarity < 75):
            return False
        if filter_level == "Low similarity (<45%)" and m.final_similarity >= 45:
            return False
        if filter_type != "All" and m.classification != filter_type:
            return False
        return True

    filtered = [m for m in matches if keep(m)]
    st.caption(f"Showing {len(filtered)} of {len(matches)} compared sentences.")

    for m in filtered:
        badge = "🔴" if m.is_flagged else "⚪"
        with st.expander(f"{badge} {m.final_similarity:.1f}% — {m.suspicious_sentence[:90]}"):
            st.markdown(f"**Suspicious sentence:**  \n{m.suspicious_sentence}")
            st.markdown(f"**Matching source ({m.source_document}):**  \n{m.source_sentence}")
            cols = st.columns(4)
            cols[0].metric("Exact", f"{m.exact_similarity:.1f}%")
            cols[1].metric("TF-IDF", f"{m.tfidf_similarity:.1f}%")
            cols[2].metric("Semantic", f"{m.semantic_similarity:.1f}%")
            cols[3].metric("Final", f"{m.final_similarity:.1f}%")
            st.markdown(f"**Classification:** {m.classification}")


# =============================================================================
# PAGE 6: ANALYTICS DASHBOARD
# =============================================================================

def page_analytics() -> None:
    st.title("📈 Analytics Dashboard")
    results = _require_results()
    if results is None:
        return

    name, r = _select_source_result(results)

    row1 = st.columns(5)
    row1[0].metric("Adaptive Score", f"{r['hybrid_result'].score:.1f}%")
    row1[1].metric("Detected Type", r["classification"].plagiarism_type)
    row1[2].metric("Confidence", f"{r['classification'].confidence:.1f}%")
    row1[3].metric("Most Similar Source", results["best_source_name"])
    row1[4].metric("Suspicious Sentences", sum(m.is_flagged for m in r["sentence_matches"]))

    st.divider()
    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(_detector_bar_chart(r["detector_scores"]), use_container_width=True)
    with c2:
        st.plotly_chart(_weight_bar_chart(r["weight_selection"].weights), use_container_width=True)

    c3, c4 = st.columns(2)
    with c3:
        st.plotly_chart(_radar_chart(r["detector_scores"]), use_container_width=True)
    with c4:
        st.plotly_chart(_sentence_distribution_chart(r["sentence_matches"]), use_container_width=True)

    if len(results["per_source"]) > 1:
        st.divider()
        st.subheader("Document Similarity Matrix")
        df = _similarity_matrix_df(results["per_source"])
        st.dataframe(df, use_container_width=True)
        heat_cols = [DETECTOR_LABELS[k] for k in ["exact", "tfidf", "ngram", "semantic"]]
        fig = px.imshow(
            df[heat_cols].values, x=heat_cols, y=df["Source Document"],
            color_continuous_scale="RdYlGn_r", range_color=[0, 100], text_auto=".0f",
            title="Similarity Heatmap Across Source Documents",
        )
        st.plotly_chart(fig, use_container_width=True)

    st.divider()
    st.subheader("Plagiarism Type Summary")
    if len(results["per_source"]) > 1:
        type_counts = pd.Series([res["classification"].plagiarism_type for res in results["per_source"].values()]).value_counts()
        st.plotly_chart(px.pie(values=type_counts.values, names=type_counts.index, title="Detected Types Across Sources"), use_container_width=True)
    else:
        st.write(f"**{r['classification'].plagiarism_type}** — confidence {r['classification'].confidence:.1f}%")


# =============================================================================
# PAGE 7: REPORT
# =============================================================================

def page_report() -> None:
    st.title("📄 Report")
    results = _require_results()
    if results is None:
        return

    name, r = _select_source_result(results)
    st.write(
        f"Generate a downloadable PDF report for **{results['suspicious'].name}** vs. "
        f"**{name}**, including detection results, adaptive analysis, explainable findings, "
        f"and suspicious sentence evidence."
    )

    if st.button("📥 Generate PDF Report", type="primary"):
        with st.spinner("Building report..."):
            output_path = PROJECT_ROOT / "data" / "uploads" / f"plagiarism_report_{name}.pdf"
            report_gen = get_pipeline()["report_generator"]
            report_gen.generate(
                output_path=output_path,
                suspicious_doc_name=results["suspicious"].name,
                reference_doc_names=[name],
                hybrid_result=r["hybrid_result"],
                classification=r["classification"],
                weight_selection=r["weight_selection"],
                explanation=r["explanation"],
                sentence_matches=r["sentence_matches"],
                fixed_vs_adaptive=r["comparison"],
            )
            with open(output_path, "rb") as f:
                st.download_button(
                    "⬇️ Download Report", data=f.read(),
                    file_name=f"plagiarism_report_{name}.pdf", mime="application/pdf",
                )
            st.success("Report generated.")


# =============================================================================
# MAIN — sidebar navigation
# =============================================================================

def main() -> None:
    st.sidebar.title("🔎 PlagiarismAI")
    st.sidebar.caption("Adaptive plagiarism detection")
    st.sidebar.divider()
    if "nav_page" not in st.session_state:
        st.session_state["nav_page"] = "Home"
    page = st.sidebar.radio(
        "Go to",
        list(PAGE_ICONS.keys()),
        key="nav_page",
        format_func=lambda p: f"{PAGE_ICONS[p]}  {p}",
        label_visibility="collapsed",
    )

    pages = {
        "Home": page_home,
        "Upload & Analyze": page_upload_analyze,
        "Results": page_results,
        "Adaptive Analysis": page_adaptive_analysis,
        "Sentence Analysis": page_sentence_analysis,
        "Analytics": page_analytics,
        "Report": page_report,
    }
    pages[page]()

    st.sidebar.divider()
    st.sidebar.caption("AI Based Plagiarism Detection System v1.0")


if __name__ == "__main__":
    main()
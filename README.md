# AI Based Plagiarism Detection System

**An adaptive AI system that detects, classifies, and explains plagiarism using lexical and semantic similarity analysis.**

Traditional plagiarism checkers apply one fixed scoring formula to every document:

```
Final Score = 20% Exact + 25% TF-IDF + 20% N-Gram + 35% Semantic   (always, for every document)
```

The problem: **different types of plagiarism leave different fingerprints.** Direct copying scores high on every lexical method. Paraphrasing scores low on exact/lexical methods but high on semantic similarity. A fixed formula can't tell these apart — it just averages them together and hides the pattern.

This project instead:

1. Runs **four independent similarity techniques** on every document pair.
2. **Classifies the pattern** those four scores form into a plagiarism *type* (Direct Copy, Partial Copy, Paraphrased Plagiarism, Structural Similarity, Mixed, or Low/No Plagiarism).
3. **Dynamically re-weights** the four techniques based on that detected type — this is the adaptive engine, and it's the core innovation.
4. Generates an **explanation that traces back to actual computed values** — never a templated sentence.
5. Shows exactly what changed by comparing the **adaptive score against a traditional fixed-weight baseline**, side by side.

---

## 1. Architecture

```
Document Upload
      ↓
Text Extraction (PDF / DOCX / TXT)
      ↓
Text Preprocessing (NLTK: tokenize, stopwords, lemmatize)
      ↓
┌──────────────────────────────────────────────┐
│  Exact Match  │  TF-IDF  │  N-Gram  │ Semantic │   ← 4 independent detectors
└──────────────────────────────────────────────┘
      ↓
Similarity Pattern Analysis → Plagiarism Type Classification (rule-based, config-driven)
      ↓
Adaptive Weight Selection (looked up per detected type)
      ↓
Adaptive Plagiarism Score = Σ(weight × detector score)
      ↓
Explainable AI Analysis (traces every claim to a computed value)
      ↓
Sentence-Level Evidence + PDF Report + Streamlit Dashboard
```

## 2. Folder Structure

```
AI_Plagiarism_Detection/
├── app.py                        # Streamlit entry point — 7 UI pages
├── requirements.txt
├── README.md
├── config/
│   └── config.yaml               # every threshold & weight — nothing hardcoded
├── data/
│   ├── uploads/                  # runtime uploads + generated reports
│   ├── samples/                  # 4 demo documents + 1 reference doc
│   └── database.db               # created on first run
├── modules/
│   ├── document_processor.py     # Module 1 — upload & text extraction
│   ├── text_preprocessor.py      # Module 2 — NLP preprocessing
│   ├── exact_matcher.py          # Module 3.1 — SequenceMatcher exact/near-exact
│   ├── tfidf_detector.py         # Module 3.2 — TF-IDF + cosine
│   ├── ngram_detector.py         # Module 3.3 — uni/bi/trigram overlap
│   ├── semantic_detector.py      # Module 3.4 — Sentence-BERT semantic similarity
│   ├── plagiarism_classifier.py  # Module 4 — rule-based type classification
│   ├── adaptive_weighting.py     # Module 5 — ⭐ the core innovation
│   ├── hybrid_scorer.py          # Module 5 (cont.) — weighted score arithmetic
│   ├── sentence_analyzer.py      # Module 6 — sentence-level matching
│   ├── explainability_engine.py  # Module 7 — traceable explanations
│   └── report_generator.py       # Module 11 — PDF report (ReportLab)
├── database/
│   ├── database.py                # SQLite connection + CRUD
│   └── models.py                  # schema (Documents, AnalysisResults, SentenceMatches)
├── utils/
│   ├── helpers.py                 # config loading, scoring math, formatting
│   └── validators.py              # input & weight-vector validation
└── tests/
    ├── test_similarity.py
    ├── test_classifier.py
    └── test_adaptive_weighting.py
```

Modules 8 (multi-document comparison), 9 (analytics dashboard), and 10 (adaptive vs. fixed comparison) are implemented inside `app.py`, since they're primarily orchestration + visualization over the modules above rather than new detection logic.

## 3. Installation

```bash
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt

python3 -m nltk.downloader punkt stopwords wordnet punkt_tab
python3 -m spacy download en_core_web_sm   # optional, reserved for future extensions

streamlit run app.py
```

The first run of the semantic detector downloads `all-MiniLM-L6-v2` (~90MB) from HuggingFace — this requires internet access once. After that, it's cached locally by `sentence-transformers`.

## 4. Running Tests

```bash
pytest tests/ -v
# or, without pytest installed:
python3 tests/test_similarity.py
python3 tests/test_classifier.py
python3 tests/test_adaptive_weighting.py
```

25 tests cover: each detector in isolation, every classification rule, every adaptive weight vector summing to 1.0, and — most importantly — a test proving the adaptive engine produces a **materially different score than the fixed baseline** on a paraphrase-shaped profile, which is the entire point of the project.

## 5. Module-by-Module: How Each Piece Feeds the Adaptive Engine

| Module | What it does | Why the adaptive engine needs it |
|---|---|---|
| **1. Document Processor** | Extracts text from PDF/DOCX/TXT, keeps the *original* text alongside it | Every later module needs clean text; sentence highlighting needs the untouched original |
| **2. Text Preprocessor** | Lowercases, strips punctuation/stopwords, lemmatizes, splits sentences | Makes TF-IDF/n-gram comparable; deliberately keeps originals for exact-match & semantic |
| **3. Four Detectors** | Exact (difflib), TF-IDF+cosine, n-gram overlap, Sentence-BERT semantic | Four *independent* signals — no signal alone tells you the plagiarism type |
| **4. Classifier** | Turns 4 scores into ONE type label, config-driven rules | This label is what everything downstream keys off of |
| **5. Adaptive Weighting** | Looks up a weight vector *for that type* | ⭐ Instead of one formula, the formula itself changes per document |
| **6. Sentence Analyzer** | Per-sentence best match + sub-scores | Evidence for the explanation and the report |
| **7. Explainability Engine** | Turns scores + weights + matches into readable reasons | Every sentence is traceable — never invented |
| **8–10. (in app.py)** | Multi-doc comparison, dashboard, adaptive-vs-fixed | Proves the innovation visually, not just numerically |
| **11. Report Generator** | PDF export via ReportLab | Portfolio/viva-ready deliverable |

## 6. Adaptive Weight Profiles (from `config.yaml`)

| Detected Type | Exact | TF-IDF | N-Gram | Semantic | Why |
|---|---|---|---|---|---|
| Direct Copy | 40% | 30% | 20% | 10% | Verbatim copying — lexical signals dominate |
| Partial Copy | 30% | 25% | 25% | 20% | Mixed copied/original content |
| Paraphrased Plagiarism | 10% | 20% | 10% | 60% | Words differ, meaning doesn't |
| Structural Similarity | 10% | 20% | 20% | 50% | Structure/meaning preserved, wording differs |
| Mixed / Complex | 25% | 25% | 20% | 30% | No single pattern dominates |
| Low / No Significant Plagiarism | 25% | 25% | 25% | 25% | No strong pattern — equal weighting is the safe default |

Every vector is validated at startup (`utils/validators.py`) to sum to exactly 1.0 — a bad edit to `config.yaml` fails loudly rather than silently skewing every score.

## 7. Sample Documents (`data/samples/`)

| File | Demonstrates |
|---|---|
| `source_reference.txt` | The reference document everything else is checked against |
| `sample_direct_copy.txt` | Verbatim copying of most of the reference |
| `sample_partial_copy.txt` | Some sentences copied verbatim, interleaved with original commentary |
| `sample_paraphrased.txt` | Meaning preserved throughout, wording substantially reworded |
| `sample_low_similarity.txt` | Unrelated topic — should NOT be flagged |

## 8. A Note on Threshold Calibration

The classification thresholds in `config.yaml` were **empirically calibrated**, not guessed. For example: `difflib.SequenceMatcher` (used for exact matching) compares character sequences, not word boundaries — two genuinely *unrelated* English paragraphs still typically share **~28-35% ratio** just from common function words, punctuation, and sentence rhythm. Testing this baseline directly (see the calibration checks alongside `tests/test_similarity.py`) is why the "Low or No Significant Plagiarism" rule uses `exact_lt: 40` rather than a naive `exact_lt: 15` — the latter would misclassify almost every unrelated document pair.

## 9. Known Limitation: Offline Semantic Fallback

`sentence-transformers` needs to download `all-MiniLM-L6-v2` on first use. If that download isn't possible (no internet, restricted environment), `semantic_detector.py` automatically falls back to a **character n-gram TF-IDF cosine similarity** so the whole pipeline still runs rather than crashing. This fallback is logged clearly and is a real (if weaker) similarity signal — it catches shared word roots and substrings, but it does **not** capture deep meaning equivalence the way Sentence-BERT does. In practice this means:

- Direct Copy, Partial Copy, and Low/No Similarity classifications are unaffected (they rely mainly on exact/lexical signals).
- Paraphrased Plagiarism and Structural Similarity — which specifically depend on the semantic detector catching meaning preserved across very different wording — will under-score when running on the fallback, and score correctly once `sentence-transformers` has internet access to download its model.

This is a deliberate, honest design choice: **degrade gracefully and say so, rather than fail or fake a semantic score.**

## 10. Technology Stack

| Layer | Choice |
|---|---|
| Language | Python 3.10+ |
| Frontend | Streamlit |
| NLP | NLTK, spaCy (reserved), Sentence-Transformers |
| ML / Similarity | scikit-learn (TF-IDF, cosine), difflib, custom n-gram overlap |
| Database | SQLite (stdlib `sqlite3`) |
| Visualization | Plotly |
| PDF Reports | ReportLab |
| Testing | pytest (or the bundled `if __name__ == "__main__"` runners) |

## 11. Database Schema

```
Documents(document_id, document_name, document_text, upload_date)
AnalysisResults(result_id, suspicious_document_id, source_document_id,
                exact_similarity, tfidf_similarity, ngram_similarity, semantic_similarity,
                detected_plagiarism_type, classification_confidence,
                fixed_score, adaptive_score, analysis_date)
SentenceMatches(match_id, result_id, suspicious_sentence, source_sentence,
                source_document, similarity_score, plagiarism_type)
```

## 12. Extending the System

- **New plagiarism type?** Add a rule to `classification_rules` and a matching entry to `adaptive_weights` in `config.yaml` — no code changes needed. An unmatched type safely falls back to equal (25/25/25/25) weighting.
- **New detector?** Implement it as its own `modules/*.py` with a `document_score()` method returning 0-100, add it to `app.py`'s pipeline dict, and extend the weight vectors to 5 elements.
- **Swap the semantic model?** Change `detectors.semantic.model_name` in `config.yaml` to any `sentence-transformers` model name.

## 13. Suggested Talking Points (Viva / Interview)

- *"Why not just average the four scores?"* — Because averaging treats every plagiarism type identically. A paraphrased document and an unrelated document can both score ~30% on a naive average, for completely different reasons — averaging destroys the very information that would tell them apart.
- *"How do you know the adaptive weights are actually better, not just different?"* — `tests/test_adaptive_weighting.py::test_paraphrase_profile_scores_higher_adaptive_than_fixed` demonstrates this directly: on a paraphrase-shaped score profile, the adaptive score is measurably higher than the fixed baseline, and for a direct-copy profile (where all four detectors already agree) the two scores converge — showing the adaptive engine adds value exactly where detectors disagree, and stays out of the way where they don't.
- *"What happens if your config.yaml has a bug?"* — `utils/validators.py` checks every weight vector sums to 1.0 at startup and raises immediately, rather than silently producing a wrong score during analysis.

---

*Built as a Computer Science Engineering academic project — designed to be portfolio-, GitHub-, and viva-ready.*

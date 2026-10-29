"""
modules/report_generator.py
------------------------------
MODULE 11: PDF REPORT GENERATION

Connection to the adaptive system:
This is the "print view" of everything computed upstream — it doesn't
calculate anything new. It exists to package the detection results,
adaptive analysis, explanation, and sentence-level evidence into a
single downloadable artifact suitable for a viva/portfolio submission.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import List

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from modules.explainability_engine import Explanation
from modules.hybrid_scorer import HybridScoreResult
from modules.plagiarism_classifier import ClassificationResult
from modules.adaptive_weighting import WeightSelection
from modules.sentence_analyzer import SentenceMatch


class ReportGenerator:
    def __init__(self):
        self.styles = getSampleStyleSheet()
        self.styles.add(
            ParagraphStyle(name="H1Custom", fontSize=18, spaceAfter=12, textColor=colors.HexColor("#1f3a5f"))
        )
        self.styles.add(
            ParagraphStyle(name="H2Custom", fontSize=13, spaceBefore=14, spaceAfter=8, textColor=colors.HexColor("#1f3a5f"))
        )
        self.styles.add(ParagraphStyle(name="BodyCustom", fontSize=10, leading=14))

    def generate(
        self,
        output_path: str | Path,
        suspicious_doc_name: str,
        reference_doc_names: List[str],
        hybrid_result: HybridScoreResult,
        classification: ClassificationResult,
        weight_selection: WeightSelection,
        explanation: Explanation,
        sentence_matches: List[SentenceMatch],
        fixed_vs_adaptive: dict | None = None,
    ) -> Path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        doc = SimpleDocTemplate(
            str(output_path), pagesize=A4,
            leftMargin=2 * cm, rightMargin=2 * cm, topMargin=2 * cm, bottomMargin=2 * cm,
        )
        story = []

        story += self._title_section(suspicious_doc_name, reference_doc_names)
        story += self._detection_results_section(classification.scores)
        story += self._adaptive_analysis_section(classification, weight_selection, hybrid_result, fixed_vs_adaptive)
        story += self._explainable_section(explanation)
        story.append(PageBreak())
        story += self._sentence_section(sentence_matches)
        story += self._conclusion_section(explanation)

        doc.build(story)
        return output_path

    # -- sections -----------------------------------------------------------

    def _title_section(self, suspicious_name: str, reference_names: List[str]) -> list:
        s = self.styles
        elems = [
            Paragraph("AI Based Plagiarism Detection System", s["H1Custom"]),
            Paragraph("Adaptive Plagiarism Analysis Report", s["Heading3"]),
            Spacer(1, 10),
        ]
        meta_table = Table(
            [
                ["Document analyzed:", suspicious_name],
                ["Reference document(s):", ", ".join(reference_names)],
                ["Analysis date:", datetime.now().strftime("%Y-%m-%d %H:%M")],
            ],
            colWidths=[5 * cm, 10 * cm],
        )
        meta_table.setStyle(TableStyle([
            ("FONTSIZE", (0, 0), (-1, -1), 9),
            ("TEXTCOLOR", (0, 0), (0, -1), colors.HexColor("#555555")),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ]))
        elems.append(meta_table)
        elems.append(Spacer(1, 12))
        return elems

    def _detection_results_section(self, scores: dict) -> list:
        s = self.styles
        elems = [Paragraph("Detection Results", s["H2Custom"])]
        data = [["Method", "Similarity Score"]] + [
            [label, f"{scores[key]:.1f}%"]
            for key, label in [
                ("exact", "Exact Matching"),
                ("tfidf", "TF-IDF + Cosine Similarity"),
                ("ngram", "N-Gram Overlap"),
                ("semantic", "Semantic Similarity (Sentence-BERT)"),
            ]
        ]
        table = Table(data, colWidths=[9 * cm, 5 * cm])
        table.setStyle(self._standard_table_style())
        elems.append(table)
        return elems

    def _adaptive_analysis_section(self, classification, weight_selection, hybrid_result, fixed_vs_adaptive) -> list:
        s = self.styles
        elems = [Paragraph("Adaptive Analysis", s["H2Custom"])]
        rows = [
            ["Detected Plagiarism Type", classification.plagiarism_type],
            ["Classification Confidence", f"{classification.confidence:.1f}%"],
            ["Exact Match Weight", f"{weight_selection.weights[0]*100:.0f}%"],
            ["TF-IDF Weight", f"{weight_selection.weights[1]*100:.0f}%"],
            ["N-Gram Weight", f"{weight_selection.weights[2]*100:.0f}%"],
            ["Semantic Similarity Weight", f"{weight_selection.weights[3]*100:.0f}%"],
            ["Adaptive Plagiarism Score", f"{hybrid_result.score:.1f}%"],
        ]
        if fixed_vs_adaptive:
            rows.append(["Traditional Fixed Score", f"{fixed_vs_adaptive['fixed'].score:.1f}%"])
            rows.append(["Difference (Adaptive − Fixed)", f"{fixed_vs_adaptive['difference']:+.1f} points"])

        table = Table([["Field", "Value"]] + rows, colWidths=[9 * cm, 5 * cm])
        table.setStyle(self._standard_table_style())
        elems.append(table)
        elems.append(Spacer(1, 6))
        elems.append(Paragraph(f"<i>{weight_selection.reason}</i>", s["BodyCustom"]))
        return elems

    def _explainable_section(self, explanation: Explanation) -> list:
        s = self.styles
        elems = [Paragraph("Why Was This Flagged? (Explainable AI)", s["H2Custom"])]
        for i, reason in enumerate(explanation.reasons, start=1):
            elems.append(Paragraph(f"{i}. {reason}", s["BodyCustom"]))
            elems.append(Spacer(1, 4))
        return elems

    def _sentence_section(self, sentence_matches: List[SentenceMatch]) -> list:
        s = self.styles
        elems = [Paragraph("Suspicious Sentences", s["H2Custom"])]
        flagged = [m for m in sentence_matches if m.is_flagged]
        if not flagged:
            elems.append(Paragraph("No sentences exceeded the suspicion threshold.", s["BodyCustom"]))
            return elems

        for m in flagged:
            elems.append(Paragraph(f"<b>Suspicious:</b> {m.suspicious_sentence}", s["BodyCustom"]))
            elems.append(Paragraph(f"<b>Matched source ({m.source_document}):</b> {m.source_sentence}", s["BodyCustom"]))
            elems.append(Paragraph(
                f"<b>Similarity:</b> {m.final_similarity:.1f}%  |  "
                f"<b>Exact:</b> {m.exact_similarity:.1f}%  |  "
                f"<b>Semantic:</b> {m.semantic_similarity:.1f}%  |  "
                f"<b>Type:</b> {m.classification}",
                s["BodyCustom"],
            ))
            elems.append(Spacer(1, 8))
        return elems

    def _conclusion_section(self, explanation: Explanation) -> list:
        s = self.styles
        return [
            Paragraph("Final Conclusion", s["H2Custom"]),
            Paragraph(explanation.conclusion, s["BodyCustom"]),
        ]

    @staticmethod
    def _standard_table_style() -> TableStyle:
        return TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1f3a5f")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTSIZE", (0, 0), (-1, -1), 9),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cccccc")),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f4f6f8")]),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ("TOPPADDING", (0, 0), (-1, -1), 5),
        ])

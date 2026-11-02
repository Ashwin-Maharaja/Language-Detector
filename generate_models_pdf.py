"""
generate_models_pdf.py
======================
Generates a concise, publication-grade academic PDF summarizing all 8 Machine Learning
models used in the Multi-Script Language Detection Project (English, Hindi, Kannada).
"""

import os
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, KeepTogether, PageBreak, HRFlowable
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.pdfgen import canvas

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
PLOTS_DIR = os.path.join(PROJECT_ROOT, "plots")
PDF_OUTPUT_PATH = os.path.join(PROJECT_ROOT, "ML_Models_Summary.pdf")

class NumberedCanvas(canvas.Canvas):
    """Adds page numbers and header/footer to each page."""
    def __init__(self, *args, **kwargs):
        super(NumberedCanvas, self).__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_decorations(num_pages)
            super(NumberedCanvas, self).showPage()
        super(NumberedCanvas, self).save()

    def draw_page_decorations(self, page_count):
        self.saveState()
        self.setFont("Helvetica", 8)
        self.setFillColor(colors.HexColor("#64748b"))
        
        # Header (pages > 1)
        if self._pageNumber > 1:
            self.drawString(54, letter[1] - 36, "Multi-Script Language Detection — Machine Learning Models Summary")
            self.setStrokeColor(colors.HexColor("#e2e8f0"))
            self.setLineWidth(0.5)
            self.line(54, letter[1] - 42, letter[0] - 54, letter[1] - 42)
            
        # Footer
        text = f"Page {self._pageNumber} of {page_count}"
        self.drawRightString(letter[0] - 54, 32, text)
        self.drawString(54, 32, "English, Hindi (हिंदी), and Kannada (ಕನ್ನಡ) • Mini Project Evaluation")
        self.setStrokeColor(colors.HexColor("#e2e8f0"))
        self.setLineWidth(0.5)
        self.line(54, 44, letter[0] - 54, 44)
        self.restoreState()

def create_models_pdf():
    doc = SimpleDocTemplate(
        PDF_OUTPUT_PATH,
        pagesize=letter,
        leftMargin=54,
        rightMargin=54,
        topMargin=54,
        bottomMargin=54,
    )

    styles = getSampleStyleSheet()

    # Custom styles
    title_style = ParagraphStyle(
        "DocTitle",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=20,
        leading=24,
        textColor=colors.HexColor("#1e3a8a"),
        spaceAfter=4,
    )

    subtitle_style = ParagraphStyle(
        "DocSubtitle",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=10,
        leading=14,
        textColor=colors.HexColor("#475569"),
        spaceAfter=14,
    )

    h1_style = ParagraphStyle(
        "Heading1_Custom",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=12,
        leading=16,
        textColor=colors.HexColor("#0f172a"),
        spaceBefore=10,
        spaceAfter=6,
    )

    h2_style = ParagraphStyle(
        "Heading2_Custom",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=10,
        leading=13,
        textColor=colors.HexColor("#1e40af"),
        spaceBefore=6,
        spaceAfter=2,
    )

    body_style = ParagraphStyle(
        "Body_Custom",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=8.5,
        leading=11.5,
        textColor=colors.HexColor("#334155"),
        spaceAfter=4,
    )

    body_bold = ParagraphStyle(
        "Body_Bold",
        parent=body_style,
        fontName="Helvetica-Bold",
    )

    table_header = ParagraphStyle(
        "TableHeader",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=7.5,
        leading=9.5,
        textColor=colors.white,
        alignment=1,
    )

    table_cell = ParagraphStyle(
        "TableCell",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=7.5,
        leading=9.5,
        textColor=colors.HexColor("#0f172a"),
        alignment=0,
    )

    table_cell_center = ParagraphStyle(
        "TableCellCenter",
        parent=table_cell,
        alignment=1,
    )

    story = []

    # 1. Document Header
    story.append(Paragraph("Machine Learning Models Compendium", title_style))
    story.append(Paragraph(
        "<b>Multi-Script Language Detection (English, Hindi, Kannada)</b> | 8 ML Architectures Benchmarked with Stratified 5-Fold Cross-Validation",
        subtitle_style
    ))
    story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#1e3a8a"), spaceAfter=10))

    # 2. Executive Summary / Dataset Recap
    intro_text = (
        "This concise report provides a comprehensive reference of all <b>8 machine learning algorithms</b> "
        "(4 individual and 4 ensemble architectures) implemented, tuned, and evaluated for language detection. "
        "Feature engineering was performed using <b>character n-grams (range 2–5, 32,485 dimensions)</b> on a balanced corpus "
        "of 1,050 sentences (350 English, 350 Hindi, 350 Kannada). All models were evaluated on an 80/20 stratified split "
        "(Train: 840, Test: 210) across standard multi-class metrics."
    )
    story.append(Paragraph(intro_text, body_style))
    story.append(Spacer(1, 6))

    # 3. Master Benchmark Table
    story.append(Paragraph("1. Comprehensive Model Benchmark Comparison", h1_style))
    
    # [Model, Category, Test Acc, F1-Score, 5-Fold CV Mean, Training Time, Optimal Parameters]
    table_data = [
        [
            Paragraph("Model Name", table_header),
            Paragraph("Category", table_header),
            Paragraph("Test Acc", table_header),
            Paragraph("Weighted F1", table_header),
            Paragraph("5-Fold CV", table_header),
            Paragraph("Time (s)", table_header),
            Paragraph("Best Hyperparameters", table_header)
        ],
        [
            Paragraph("<b>Multinomial Naive Bayes</b>", table_cell),
            Paragraph("Individual (Baseline)", table_cell),
            Paragraph("<b>1.0000</b>", table_cell_center),
            Paragraph("1.0000", table_cell_center),
            Paragraph("1.0000 ± 0.00", table_cell_center),
            Paragraph("18.75s", table_cell_center),
            Paragraph("alpha=0.01, fit_prior=True", table_cell)
        ],
        [
            Paragraph("<b>Logistic Regression</b>", table_cell),
            Paragraph("Individual (Linear)", table_cell),
            Paragraph("<b>1.0000</b>", table_cell_center),
            Paragraph("1.0000", table_cell_center),
            Paragraph("1.0000 ± 0.00", table_cell_center),
            Paragraph("8.83s", table_cell_center),
            Paragraph("C=0.1, penalty='l2', solver='lbfgs'", table_cell)
        ],
        [
            Paragraph("<b>Linear SVM (Calibrated)</b>", table_cell),
            Paragraph("Individual (Max-Margin)", table_cell),
            Paragraph("<b>1.0000</b>", table_cell_center),
            Paragraph("1.0000", table_cell_center),
            Paragraph("1.0000 ± 0.00", table_cell_center),
            Paragraph("2.24s", table_cell_center),
            Paragraph("C=0.1, CalibratedClassifierCV(cv=3)", table_cell)
        ],
        [
            Paragraph("<b>Random Forest</b>", table_cell),
            Paragraph("Ensemble (Bagging)", table_cell),
            Paragraph("<b>1.0000</b>", table_cell_center),
            Paragraph("1.0000", table_cell_center),
            Paragraph("1.0000 ± 0.00", table_cell_center),
            Paragraph("7.41s", table_cell_center),
            Paragraph("n_estimators=50, max_depth=30", table_cell)
        ],
        [
            Paragraph("<b>Soft Voting Classifier</b>", table_cell),
            Paragraph("Meta-Ensemble", table_cell),
            Paragraph("<b>1.0000</b>", table_cell_center),
            Paragraph("1.0000", table_cell_center),
            Paragraph("1.0000 ± 0.00", table_cell_center),
            Paragraph("0.45s", table_cell_center),
            Paragraph("Top-3 models soft probability blend", table_cell)
        ],
        [
            Paragraph("<b>Gradient Boosting</b>", table_cell),
            Paragraph("Ensemble (Boosting)", table_cell),
            Paragraph("<b>1.0000</b>", table_cell_center),
            Paragraph("1.0000", table_cell_center),
            Paragraph("1.0000 ± 0.00", table_cell_center),
            Paragraph("57.58s", table_cell_center),
            Paragraph("n_estimators=50, lr=0.1, max_depth=3", table_cell)
        ],
        [
            Paragraph("<b>Stacking Classifier</b>", table_cell),
            Paragraph("Meta-Ensemble", table_cell),
            Paragraph("<b>1.0000</b>", table_cell_center),
            Paragraph("1.0000", table_cell_center),
            Paragraph("1.0000 ± 0.00", table_cell_center),
            Paragraph("1.85s", table_cell_center),
            Paragraph("Top-3 estimators, LR Meta-Learner", table_cell)
        ],
        [
            Paragraph("<b>Decision Tree</b>", table_cell),
            Paragraph("Individual (Tree)", table_cell),
            Paragraph("0.9905", table_cell_center),
            Paragraph("0.9905", table_cell_center),
            Paragraph("0.9964 ± 0.00", table_cell_center),
            Paragraph("3.25s", table_cell_center),
            Paragraph("criterion='gini', max_depth=20", table_cell)
        ],
    ]

    col_widths = [105, 95, 45, 45, 65, 40, 109]
    t = Table(table_data, colWidths=col_widths, repeatRows=1)
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1e3a8a")),
        ("ALIGN", (0, 0), (-1, -1), "LEFT"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8fafc")]),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
    ]))
    story.append(t)
    story.append(Spacer(1, 10))

    # 4. Detailed Profiles for Each of the 8 Models
    story.append(Paragraph("2. Individual & Ensemble Model Profiles", h1_style))

    models_info = [
        (
            "1. Multinomial Naive Bayes (MultinomialNB)",
            "Individual Classifier • Probabilistic Baseline",
            "Operates under Bayes' Theorem using discrete multinomial distributions of n-gram term frequencies. "
            "Assumes conditional feature independence given the language class.",
            "Extremely fast training and low memory footprint (~1.5 MB). Highly effective for sparse text matrices.",
            "Independence assumption causes probability polarization on mixed scripts. Tuned with Laplace smoothing alpha=0.01."
        ),
        (
            "2. Logistic Regression (One-vs-Rest / Multinomial)",
            "Individual Classifier • High-Dimensional Linear",
            "Models class probabilities using logistic sigmoid / softmax link functions over a linear combination of n-gram features. "
            "Fitted using the L-BFGS quasi-Newton solver with L2 regularization.",
            "Convex loss landscape guarantees global minimum. Produces well-calibrated, smooth probabilities.",
            "Linear decision boundary cannot capture non-linear feature interactions without explicit polynomial terms."
        ),
        (
            "3. Linear Support Vector Machine (LinearSVC + CalibratedCV)",
            "Individual Classifier • Margin Maximizer",
            "Maximizes the geometric margin separating language representations in 32,485-dimensional Hilbert space. "
            "Wrapped in CalibratedClassifierCV(cv=3) using Platt's sigmoid scaling to output calibrated probabilities.",
            "Strong theoretical foundation with high structural risk minimization; robust against high dimensionality.",
            "Native LinearSVC does not compute probabilities; calibration requires an extra cross-validation pass."
        ),
        (
            "4. Decision Tree Classifier (CART)",
            "Individual Classifier • Non-Parametric Tree",
            "Recursively partitions the feature space based on Gini Impurity. Configured with pre-pruning (max_depth=20, min_samples_split=2) to prevent overfitting.",
            "Fully interpretable white-box decision logic with visualizable tree splits.",
            "Only model to score 99.05% due to axis-aligned orthogonal cuts struggling with sparse n-gram spaces."
        ),
        (
            "5. Random Forest Classifier",
            "Ensemble Classifier • Bagging Architecture",
            "Constructs an ensemble of 50 de-correlated decision trees. Each tree is trained on a bootstrap sample of the training set with random feature subset selection (sqrt features).",
            "Reduces variance dramatically compared to a single decision tree. Resilient to noise and outliers.",
            "Larger serialisation footprint (~225 KB) and slower prediction than pure linear models."
        ),
        (
            "6. Gradient Boosting Classifier",
            "Ensemble Classifier • Boosting Architecture",
            "Builds a sequential additive ensemble of shallow regression trees (max_depth=3) minimizing multi-class deviance loss. Subsequent trees fit pseudo-residuals of earlier iterations.",
            "High representational capacity; excels at correcting hard classification boundary errors.",
            "Highest training latency (57.58s) due to the strictly sequential, non-parallel nature of gradient boosting."
        ),
        (
            "7. Soft Voting Classifier",
            "Meta-Ensemble • Probability Averaging",
            "Combines predicted posterior probability distributions across the top 3 individual classifiers (Multinomial NB, Logistic Regression, and Calibrated LinearSVC).",
            "Mitigates individual model idiosyncrasies and provides smooth, well-tempered confidence scores.",
            "Requires all base estimators to produce calibrated class probabilities."
        ),
        (
            "8. Stacking Classifier",
            "Meta-Ensemble • Two-Tier Heterogeneous Stacking",
            "Two-tier meta-learning setup: Level-0 base models (NB, LR, SVM) output cross-validated out-of-fold predictions. A Level-1 Logistic Regression meta-learner learns optimal blending weights.",
            "Learns which base model to trust for specific input regions rather than applying static averaging weights.",
            "Increased pipeline complexity and training overhead; requires rigorous out-of-fold cross-validation."
        ),
    ]

    for title, cat, mech, pro, con in models_info:
        model_flow = []
        model_flow.append(Paragraph(f"<b>{title}</b> — <font color='#64748b'>{cat}</font>", h2_style))
        model_flow.append(Paragraph(f"<b>Mechanism:</b> {mech}", body_style))
        model_flow.append(Paragraph(f"<b>Key Strengths:</b> {pro}", body_style))
        model_flow.append(Paragraph(f"<b>Observations / Trade-offs:</b> {con}", body_style))
        model_flow.append(Spacer(1, 4))
        story.append(KeepTogether(model_flow))

    # Page Break before visuals
    story.append(PageBreak())

    # 5. Visualizations & Evaluation Charts
    story.append(Paragraph("3. Model Performance Visualizations", h1_style))
    story.append(Paragraph(
        "The plots below illustrate comparative benchmark metrics across the candidate algorithms. "
        "All models exhibit near-perfect separation on monolingual text due to non-overlapping Unicode script ranges.",
        body_style
    ))
    story.append(Spacer(1, 6))

    img1_path = os.path.join(PLOTS_DIR, "model_comparison_accuracy.png")
    img2_path = os.path.join(PLOTS_DIR, "roc_curves.png")
    img3_path = os.path.join(PLOTS_DIR, "training_time_comparison.png")
    img4_path = os.path.join(PLOTS_DIR, "cross_validation_comparison.png")

    vis_data = [
        [
            Paragraph("<b>Test Accuracy Benchmark</b>", body_bold),
            Paragraph("<b>Macro-Average ROC Curves (AUC = 1.0000)</b>", body_bold)
        ],
        [
            Image(img1_path, width=3.3 * inch, height=1.75 * inch) if os.path.exists(img1_path) else Paragraph("Plot missing", body_style),
            Image(img2_path, width=3.3 * inch, height=1.75 * inch) if os.path.exists(img2_path) else Paragraph("Plot missing", body_style)
        ],
        [
            Paragraph("<b>Training Latency Benchmarking (seconds)</b>", body_bold),
            Paragraph("<b>5-Fold Cross-Validation Distributions</b>", body_bold)
        ],
        [
            Image(img3_path, width=3.3 * inch, height=1.75 * inch) if os.path.exists(img3_path) else Paragraph("Plot missing", body_style),
            Image(img4_path, width=3.3 * inch, height=1.75 * inch) if os.path.exists(img4_path) else Paragraph("Plot missing", body_style)
        ]
    ]

    vis_table = Table(vis_data, colWidths=[3.5 * inch, 3.5 * inch])
    vis_table.setStyle(TableStyle([
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
    ]))
    story.append(vis_table)
    story.append(Spacer(1, 10))

    # 6. Production Model Selection & Code-Mixing Fix
    story.append(Paragraph("4. Production Deployment Selection & Code-Mixing Handling", h1_style))
    selection_text = (
        "<b>Model Selection Decision:</b> While 7 models tied at 100% accuracy on pure monolingual text, "
        "the <b>Soft Voting Ensemble</b> (combining Naive Bayes, Logistic Regression, and Linear SVC) was selected for production deployment.<br/>"
        "<b>Why not Naive Bayes alone?</b> On mixed-script phrases (e.g., <i>'data science and तकनीक'</i>), Naive Bayes suffered from "
        "length-induced probability polarization (100% English, 0% Hindi), completely ignoring secondary languages.<br/>"
        "<b>The Hybrid Production Solution:</b> The deployed service in <code>src/deploy.py</code> combines the Soft Voting Ensemble with "
        "a physical character script analyzer (50% ML posterior + 50% physical character script ratio). "
        "This detects <b>Code-Mixed text</b> (e.g. <i>'Hello नमस्ते कैसे हो'</i> &rarr; Hindi dominant 74.4%, English 20.6%), "
        "providing nuanced, non-polarizing confidence outputs for real-world user queries."
    )
    story.append(Paragraph(selection_text, body_style))

    # Build the document
    doc.build(story, canvasmaker=NumberedCanvas)
    print(f"PDF generated successfully at: {PDF_OUTPUT_PATH}")

if __name__ == "__main__":
    create_models_pdf()

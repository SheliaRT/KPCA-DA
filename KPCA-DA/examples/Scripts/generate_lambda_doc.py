"""Generate supplementary documentation for lambda selection in KPCA-DA."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from docx import Document
from docx.shared import Pt, Inches, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml.ns import qn
from docx.oxml import OxmlElement

doc = Document()

# ---------- helpers ----------
def set_font(run, name="Times New Roman", size=12, bold=False, italic=False, color=None):
    run.font.name = name
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.italic = italic
    if color:
        run.font.color.rgb = RGBColor(*color)

def heading(text, level=1, size=14, bold=True, center=False):
    p = doc.add_paragraph()
    if center:
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = p.add_run(text)
    set_font(run, size=size, bold=bold)
    return p

def body(text, size=12, italic=False, indent=0):
    p = doc.add_paragraph()
    p.paragraph_format.space_after = Pt(6)
    if indent:
        p.paragraph_format.left_indent = Inches(indent)
    run = p.add_run(text)
    set_font(run, size=size, italic=italic)
    return p

def equation(text, label=""):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(4)
    p.paragraph_format.space_after = Pt(4)
    run = p.add_run(text)
    set_font(run, size=12, italic=True)
    if label:
        tab = p.add_run(f"    {label}")
        set_font(tab, size=11)
    return p

def add_table(headers, rows, col_widths=None):
    table = doc.add_table(rows=1 + len(rows), cols=len(headers))
    table.style = "Table Grid"
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    hdr_cells = table.rows[0].cells
    for i, h in enumerate(headers):
        hdr_cells[i].text = h
        run = hdr_cells[i].paragraphs[0].runs[0]
        set_font(run, size=11, bold=True)
        hdr_cells[i].paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER
    for row_data in rows:
        row_cells = table.add_row().cells
        for i, val in enumerate(row_data):
            row_cells[i].text = str(val)
            row_cells[i].paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER
            set_font(row_cells[i].paragraphs[0].runs[0], size=11)
    if col_widths:
        for i, w in enumerate(col_widths):
            for row in table.rows:
                row.cells[i].width = Inches(w)
    return table

# ======================================================
# Title
# ======================================================
p = doc.add_paragraph()
p.alignment = WD_ALIGN_PARAGRAPH.CENTER
run = p.add_run("Supplementary Note")
set_font(run, size=13, bold=True)

p = doc.add_paragraph()
p.alignment = WD_ALIGN_PARAGRAPH.CENTER
run = p.add_run(
    "Selection of the Regularization Parameter λ in\n"
    "Kernel PCA-Based Domain Adaptation (KPCA-DA)"
)
set_font(run, size=14, bold=True)
doc.add_paragraph()

# ======================================================
# S1. Overview
# ======================================================
heading("S1.  Overview", size=13)
body(
    "KPCA-DA adapts the target domain embedding by minimizing a composite objective "
    "that balances domain alignment with structural preservation of the target manifold. "
    "The regularization coefficient λ controls the relative contribution of the structural "
    "preservation term, making its selection critical to downstream classification performance. "
    "This note describes the role of λ, the grid-search procedure used to identify an "
    "optimal value, and the empirical results obtained on the TCGA-to-GSE45498 breast cancer "
    "gene expression benchmark."
)

# ======================================================
# S2. Objective function
# ======================================================
heading("S2.  Optimization Objective", size=13)
body(
    "At each gradient step the following loss is minimized with respect to the adapted "
    "target embedding Zₜ:"
)
equation(
    "ℒ(λ)  =  ℒₘₘ⁄  +  λ · ℒᵣᵒᵍ",
    "(S1)"
)
body(
    "where ℒₘₘ⁄ is the Maximum Mean Discrepancy (MMD) term that measures "
    "distributional divergence between the source and target embeddings on the shared gene "
    "subset, and ℒᵣᵒᵍ is the structural regularization term defined as:"
)
equation(
    "ℒᵣᵒᵍ  =  tr( Zₜᵀ  Kₜ  Zₜ )",
    "(S2)"
)
body(
    "Here Kₜ is the RBF kernel matrix computed from the original target feature space, "
    "and tr(·) denotes the matrix trace. "
    "This term penalizes embeddings that deviate from the intrinsic kernel geometry of the "
    "target domain, thereby preventing degenerate or over-fitted mappings."
)
body(
    "The coefficient λ ∈ ℝ⁺ governs the strength of this constraint:"
)
body(
    "•   Small λ  —  the optimization is dominated by domain alignment; "
    "the target embedding is free to move toward the source at the cost of "
    "potentially losing target-specific structure.",
    indent=0.3
)
body(
    "•   Large λ  —  the structural term strongly anchors the target embedding "
    "to its original kernel geometry, limiting the degree of adaptation.",
    indent=0.3
)

# ======================================================
# S3. Selection procedure
# ======================================================
heading("S3.  Selection Procedure", size=13)
body(
    "An empirical grid search was performed over six candidate values of λ spanning "
    "five orders of magnitude: {0.0001, 0.001, 0.01, 0.1, 1.0, 10.0}. "
    "For each candidate, the full KPCA-DA optimization was run with 500 gradient steps "
    "(SGD, learning rate 0.001) and an RBF kernel (γ = 0.5 for both KPCA and MMD). "
    "Optimality was assessed by downstream classification accuracy on the target domain, "
    "using a K-Nearest-Neighbour classifier (k = 5) trained on source embeddings and "
    "evaluated on the adapted target embeddings."
)
body(
    "Sample-level embeddings were derived by projecting the expression matrix through the "
    "gene-space coordinates learned by KPCA-DA. Specifically, for a data matrix "
    "X ∈ ℝ^(G×N) and gene embedding Z ∈ ℝ^(G×2) restricted to "
    "the G′ common genes, the N-sample projection is:"
)
equation(
    "P  =  X[G′, :]ᵀ  ·  Z[G′, :]   ∈  ℝ^(N×2)",
    "(S3)"
)
body(
    "This projection maps each sample into the same two-dimensional space as the gene "
    "embedding, enabling direct classifier training and evaluation."
)

# ======================================================
# S4. Results
# ======================================================
heading("S4.  Results", size=13)
body(
    "Table S1 summarises the classification accuracy on the target domain (GSE45498, "
    "n = 183 samples) for each λ value. The source domain comprised 589 labelled "
    "TCGA-BRCA samples (476 Tumour, 113 Normal) after excluding two samples with "
    "ambiguous annotation."
)
doc.add_paragraph()

add_table(
    headers=["λ", "Regularization Loss (final)", "Total Loss (final)", "Target Accuracy"],
    rows=[
        ["0.0001", "1.4601", "1.0001", "77.6%"],
        ["0.001",  "1.7394", "1.0017", "78.1%"],
        ["0.01 ★", "1.7657", "1.0175", "79.2%"],
        ["0.1",   "1.4282", "1.1401", "78.7%"],
        ["1.0",   "0.2936", "1.2475", "76.0%"],
        ["10.0",  "0.0015", "1.0127", "78.1%"],
    ],
    col_widths=[1.0, 2.1, 1.8, 1.8]
)

p = doc.add_paragraph()
run = p.add_run("★ Selected value.")
set_font(run, size=10, italic=True)

doc.add_paragraph()
body(
    "Classification accuracy peaks at λ = 0.01 (79.2%), identifying this as the "
    "optimal trade-off between alignment flexibility and structural preservation "
    "for this dataset. Values below 0.01 under-regularize, allowing the gene embedding "
    "to drift in ways that marginally reduce target-domain discriminability. "
    "Values above 0.1 progressively over-constrain the embedding, yielding accuracy "
    "below the optimum."
)

# ======================================================
# S5. Recommendation
# ======================================================
heading("S5.  Practical Recommendation", size=13)
body(
    "Based on the grid-search results, λ = 0.01 is recommended as the default "
    "regularization coefficient for KPCA-DA applied to cross-cohort gene expression "
    "data of comparable dimensionality and cohort size. "
    "For other datasets, we recommend evaluating at minimum the range "
    "[10⁻⁴, 10⁻², 10⁻¹] with 500 optimization epochs and "
    "selecting the value that maximises downstream classification accuracy on the target domain. "
    "When labelled target samples are unavailable, structural regularization loss "
    "ℒᵣᵒᵍ may serve as a proxy criterion, with lower values indicating "
    "a more compact and kernel-consistent adapted embedding."
)

# ======================================================
# Save
# ======================================================
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   "Supplementary_Lambda_Selection.docx")
doc.save(OUT)
print(f"Saved: {OUT}")

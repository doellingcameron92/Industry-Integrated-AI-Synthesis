"""Generate presentation/Mentor_Presentation.pptx (15-minute defense deck, speaker notes embedded)."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from PIL import Image as PILImage
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.util import Inches, Pt

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
OUT = HERE / "Mentor_Presentation.pptx"

NAVY = RGBColor(0x1F, 0x3A, 0x5F)
GREY = RGBColor(0x55, 0x55, 0x55)

card = json.loads((ROOT / "outputs/models/model_card.json").read_text())
unc = json.loads((ROOT / "outputs/models/uncertainty_reference.json").read_text())
results = json.loads((ROOT / "evaluation/results.json").read_text())
cv, ho = card["grouped_cv"], card["hold_out"]
nvb = pd.read_csv(ROOT / "outputs/models/neural_vs_boosting.csv", index_col=0)
reg_mae, base_mae = nvb.loc["test MAE", "Regularized MLP (BN+Dropout)"], nvb.loc["test MAE", "Baseline MLP"]
n_pass = sum(r["passed"] for r in results)

SLIDES: list[dict] = [
    dict(
        title="Health-Investment Policy Advisor",
        sub="An integrated statistical + deep-learning + generative + agentic system for public-sector decision support",
        bullets=["Cameron Doelling - AI Mastery Capstone", "Integrative Industry Synthesis: mentor presentation and defense"],
        notes="(1 min) Introduce the project: an AI advisor that helps ministries of health and development banks decide "
        "which health-investment levers are associated with longer lives, while staying honest about uncertainty. "
        "Four prior capstone projects are integrated; I will show what each contributes and how they interact.",
    ),
    dict(
        title="Industry problem: where should a ministry put the next dollar?",
        bullets=[
            "Sector: public policy / international development (ministries of health & finance, development-bank country teams)",
            "Decision: allocate a limited budget across water, sanitation, electricity, immunisation, schooling, health spend",
            "Data exists (World Bank WDI, 217 countries, 2000-2022) but arrives as spreadsheets, not answers",
            "Why AI fits: pattern recognition over public data + synthesis into a defensible brief",
            "Why it is risky: national averages hide inequity; a fluent but ungrounded brief can steer public money",
        ],
        notes="(1.5 min) Frame the decision and the stakes. Stress that the bottleneck is synthesis under time pressure, "
        "and that the failure mode I most fear is an over-confident or fabricated number reaching a cabinet paper.",
    ),
    dict(
        title="Integrated solution: one question in, one audited brief out",
        image=ROOT / "diagrams/workflow.png",
        notes="(1.5 min) Walk the seven steps: scope check, profile/peers/notes, scenario simulation, neural second opinion, "
        "draft + grounding audit, human approval, structured JSON answer. Every number in the answer traces to a tool call.",
    ),
    dict(
        title="Architecture: four prior projects, four layers",
        image=ROOT / "diagrams/architecture.png",
        notes="(1.5 min) Colours map layers to prior projects. Evidence layer (Data Science), second opinion (Deep Learning), "
        "brief generation (Generative AI), control layer (Agentic Workflows). The control layer decides when each may be used.",
    ),
    dict(
        title="Project 1 - Data Science Blog Post -> evidence layer",
        bullets=[
            "World Bank life-expectancy pipeline reused: leaky under-5 mortality dropped, sparse indicators dropped",
            "Country-grouped validation (no leakage between years of the same country)",
            f"Gradient boosting: grouped-CV R2 {cv['r2']:.3f}, MAE {cv['mae']:.2f} yrs; holdout R2 {ho['r2']:.3f}, MAE {ho['mae']:.2f} yrs",
            "Scenario simulator became the simulate_intervention tool; peer finder became peer_countries",
            "Known hard cases (NGA, ZAF, TCD, SWZ...) became an evidence-quality flag that lowers confidence",
        ],
        notes="(1.5 min) The blog post's methodological lessons are now hard constraints in code. The list of mis-predicted "
        "countries is the most valuable carry-over: it tells the agent when to trust itself less.",
    ),
    dict(
        title="Project 2 - Deep Learning Systems -> second opinion & uncertainty",
        bullets=[
            "Same controlled-experiment design as the Fashion-MNIST CNN: baseline vs BatchNorm + Dropout, all else equal",
            f"Regularised MLP test MAE {reg_mae:.2f} vs baseline {base_mae:.2f} yrs; smaller train/test gap",
            "Neural net does NOT beat boosting -> deployed as a second opinion, not the primary model",
            f"Boosting error when models agree: {unc['gbr_mae_when_models_agree']:.2f} yrs; when they disagree: {unc['gbr_mae_when_models_disagree']:.2f} yrs",
            "MC-dropout spread alone was ~uncorrelated with error; disagreement is what tracks error",
        ],
        notes="(1.5 min) Be candid: I kept the neural model because the experiment was informative, not because it won. "
        "The disagreement signal is the useful product; the dropout spread on its own would have been decorative.",
    ),
    dict(
        title="Project 3 - Generative AI -> brief writer + grounding audit",
        bullets=[
            "Pluggable writer: deterministic template offline; OpenAI writer when a key is present",
            "Memorisation/format audit from the Transformer project repurposed as a GroundingAudit",
            "Every number in the draft must match the EvidenceBundle (+/-0.05); disclaimer required; causal wording rejected",
            "A rejected LLM draft falls back to the template AND is reported as a caveat - safeguards must be visible",
            "Nigeria brief: 37/37 numbers grounded; fabricated test paragraph: rejected",
        ],
        notes="(1.5 min) Explain why the audit is strict and why I fixed a false rejection by enriching the bundle rather than "
        "loosening the check. Mention that a silent fallback is not governance.",
    ),
    dict(
        title="Project 4 - Agentic Workflows -> control layer",
        bullets=[
            "ReAct loop, Pydantic-typed tool contracts, tool allow-list, step/time/tool-call budgets",
            "Scope refusal (individual medical advice, political messaging, data concealment)",
            "Tool output scanned for prompt injection and treated as untrusted data",
            "save_brief is the only side effect: requires grounding pass AND human approval",
            "JSONL trace per run + episodic memory; structured output {answer, sources, confidence, caveats, refused}",
        ],
        notes="(1 min) Almost unchanged from the Research Triage Assistant; what is new is that the tools wrap the three other "
        "layers and the approval gate is coupled to the machine audit.",
    ),
    dict(
        title="Key tradeoffs",
        bullets=[
            "Two-model pipeline (slower, more complex) <-> usable 'how much should I trust this?' signal",
            "Deterministic planner by default (narrow phrasings) <-> reproducible, testable, auditable, free to run",
            "Strict grounding (occasional false rejections) <-> no unsupported statistic is ever saved",
            "Associational model only <-> honest about what the data can support; no causal claims",
            "Caching fitted models (~90 s once) <-> notebook, tests and evaluation re-run in seconds",
        ],
        notes="(1.5 min) For each tradeoff say what was given up and what was bought. The through-line: I traded fluency "
        "and flexibility for traceability, because the audience is a public institution.",
    ),
    dict(
        title="Ethics, governance and responsible use",
        bullets=[
            "Causal misreading: '+3.0 years' is an association; stated in every scenario, brief and answer",
            "Unequal reliability: HIV-burden / conflict / small states mis-predicted -> flagged, confidence reduced (model-card principle)",
            "Hallucination & indirect prompt injection: grounding audit + untrusted-data guardrail (OWASP LLM Top 10)",
            "Accountability: human approval before any side effect; full trace for audit (NIST AI RMF Govern/Manage)",
            "Scope: refuses individual medical questions and political persuasion",
        ],
        notes="(1.5 min) Tie each risk to a concrete mechanism in the code and to a framework citation. Mention the planted "
        "lobbyist injection in the Cambodia notes that is flagged and ignored.",
    ),
    dict(
        title="Evaluation: 11 scenarios, 13 tests",
        bullets=[
            f"{n_pass}/{len(results)} scenarios pass: golden paths, scope refusal, injection, approval denied, tool outage, hallucinating writer, low evidence quality, memory recall",
            "Found & fixed: brief could be drafted with zero successful simulations (tool-outage scenario)",
            "Found & fixed: hallucinating writer 'passed' via silent fallback -> now recorded as rejected_draft caveat",
            "Found: MC-dropout spread uncorrelated with error -> added cross-model disagreement",
            "Golden path (Nigeria): confidence 0.30 - low on purpose (low evidence quality + model disagreement)",
        ],
        notes="(1.5 min) Emphasise that the evaluation changed the system. Low confidence on Nigeria is the system working, "
        "not failing: Nigeria is one of the countries the blog post identified as systematically over-predicted.",
    ),
    dict(
        title="Limitations",
        bullets=[
            "National, annual data - cannot target districts or years; hides within-country inequity",
            "Levers simulated independently; no cost model -> ranks associations, not value for money",
            "Rule-based planner handles a narrow set of phrasings; regex guardrails are evadable",
            "Confidence is a heuristic adjustment, not a calibrated probability",
            "LLM writer path untested against a live model in this submission (no API key)",
        ],
        notes="(1 min) Own the limits plainly. These are also printed in the notebook's boundaries section.",
    ),
    dict(
        title="Professional relevance and next steps",
        bullets=[
            "Demonstrates: validated model -> tooling; controlled DL experiments; verifiable generative output; governed agents",
            "Demonstrates judgment about what NOT to build: no causal claims, no autonomy, no individual advice",
            "Next: difference-in-differences / synthetic-control layer using the panel structure",
            "Next: cost model (years gained per dollar); conformal intervals from grouped folds",
            "Next: reviewed approval queue with second-signer rule for low-evidence-quality briefs",
        ],
        notes="(1 min) Close by connecting to employer expectations and invite questions. Total ~15 minutes.",
    ),
]


def add_slide(prs: Presentation, spec: dict) -> None:
    layout = prs.slide_layouts[6]  # blank
    s = prs.slides.add_slide(layout)
    tb = s.shapes.add_textbox(Inches(0.5), Inches(0.3), Inches(12.3), Inches(0.9))
    p = tb.text_frame.paragraphs[0]
    p.text = spec["title"]
    p.font.size = Pt(30)
    p.font.bold = True
    p.font.color.rgb = NAVY
    top = 1.25
    if spec.get("sub"):
        st = s.shapes.add_textbox(Inches(0.5), Inches(top), Inches(12.3), Inches(0.8))
        q = st.text_frame.paragraphs[0]
        q.text = spec["sub"]
        q.font.size = Pt(18)
        q.font.color.rgb = GREY
        st.text_frame.word_wrap = True
        top += 1.0
    if spec.get("image"):
        w, h = PILImage.open(spec["image"]).size
        max_w, max_h = 12.1, 7.3 - top
        scale = min(max_w / w, max_h / h)
        pic_w = w * scale
        s.shapes.add_picture(str(spec["image"]), Inches((13.333 - pic_w) / 2), Inches(top), width=Inches(pic_w))
    if spec.get("bullets"):
        body = s.shapes.add_textbox(Inches(0.6), Inches(top), Inches(12.1), Inches(5.5))
        tf = body.text_frame
        tf.word_wrap = True
        for i, b in enumerate(spec["bullets"]):
            para = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
            para.text = "\u2022  " + b
            para.font.size = Pt(18)
            para.space_after = Pt(10)
    s.notes_slide.notes_text_frame.text = spec["notes"]


def build() -> None:
    prs = Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)
    for spec in SLIDES:
        add_slide(prs, spec)
    prs.save(str(OUT))
    md = ["# Speaker notes - Mentor Presentation (15 minutes)\n"]
    for i, spec in enumerate(SLIDES, 1):
        md.append(f"## Slide {i}: {spec['title']}\n\n{spec['notes']}\n")
    (HERE / "Speaker_Notes.md").write_text("\n".join(md), encoding="utf-8")
    print(f"wrote {OUT.relative_to(ROOT)} ({len(SLIDES)} slides) and presentation/Speaker_Notes.md")


if __name__ == "__main__":
    build()

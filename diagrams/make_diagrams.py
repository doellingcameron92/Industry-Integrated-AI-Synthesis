"""Render the architecture and data-flow diagrams to diagrams/*.png (matplotlib only)."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

OUT = Path(__file__).resolve().parent

COLORS = {"data": "#4C78A8", "dl": "#F58518", "gen": "#54A24B", "agent": "#B279A2", "human": "#E45756", "grey": "#8C8C8C"}


def box(ax, x, y, w, h, title, lines, color, title_size=10):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.15",
                                linewidth=2, edgecolor=color, facecolor=color + "22"))
    ax.text(x + w / 2, y + h - 0.28, title, ha="center", va="top", fontsize=title_size, fontweight="bold", color=color)
    offset = 0.62 + 0.3 * title.count("\n")
    ax.text(x + 0.15, y + h - offset, "\n".join(lines), ha="left", va="top", fontsize=7.6, linespacing=1.35)


def arrow(ax, p, q, label=None, color="#333333", style="-|>", ls="-"):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle=style, mutation_scale=14, linewidth=1.6, color=color, linestyle=ls))
    if label:
        ax.text((p[0] + q[0]) / 2, (p[1] + q[1]) / 2 + 0.12, label, ha="center", fontsize=7.8, color=color,
                bbox=dict(facecolor="white", edgecolor="none", pad=1))


def architecture() -> None:
    fig, ax = plt.subplots(figsize=(16, 8.6))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 8.6)
    ax.axis("off")
    ax.text(7, 8.35, "Health-Investment Policy Advisor: integrated architecture", ha="center", fontsize=15, fontweight="bold")
    ax.text(7, 8.0, "Colour = originating capstone project", ha="center", fontsize=9.5, color="#555555")

    # Human + agent
    box(ax, 0.3, 5.3, 2.6, 2.3, "Policy analyst\n(human)", [
        "Asks a population-level question",
        "Approves / denies save_brief",
        "Reads brief, caveats, trace"], COLORS["human"])
    box(ax, 3.6, 5.3, 6.8, 2.3, "Control layer: PolicyAdvisorAgent  [Agentic Workflows]", [
        "ReAct loop (plan -> typed tool call -> observe -> answer), step/tool/runtime budgets",
        "Guardrails: scope refusal, injection scan on tool output, approval gate, allow-list",
        "Typed tool contracts (Pydantic)   |   Short-term + episodic memory   |   JSONL trace log",
        "Structured FinalAnswer {answer, sources, confidence, caveats, refused}",
        "Planner: deterministic MockPlanner (default) or OpenAI tool-calling client"], COLORS["agent"])
    box(ax, 10.9, 5.3, 2.8, 2.3, "Outputs\n", [
        "outputs/briefs/*.md (approved only)",
        "logs/agent_trace.jsonl",
        "memory/episodic_memory.json",
        "RunResult JSON"], COLORS["grey"])

    # Tool layers
    box(ax, 0.3, 0.5, 4.2, 3.3, "Evidence layer\n[Data Science Blog Post]", [
        "World Bank WDI 2000-2022, 217 countries",
        "Cleaning: drop leaky under-5 mortality, sparse cols,",
        "  log-transform money/population, per-country interpolation",
        "GradientBoostingRegressor, country-grouped CV & hold-out",
        "  (CV R2 0.86, MAE 2.4y)",
        "Tools: country_profile, peer_countries,",
        "  simulate_intervention (+ extrapolation & quality flags)"], COLORS["data"])
    box(ax, 4.9, 0.5, 4.2, 3.3, "Second opinion\n[Deep Learning Systems]", [
        "Controlled experiment: baseline MLP vs",
        "  BatchNorm + Dropout MLP on same grouped split",
        "Regularised net: lower test MAE, smaller train/test gap",
        "MC-Dropout spread + cross-model disagreement",
        "  (GBR error 2.65y when models agree vs 3.60y when not)",
        "Tool: uncertainty_estimate -> flags, confidence level"], COLORS["dl"])
    box(ax, 9.5, 0.5, 4.2, 3.3, "Brief generation\n[Generative AI Applications]", [
        "EvidenceBundle (typed, tool-produced numbers only)",
        "Writer: TemplateBriefWriter (offline) or LLM writer",
        "GroundingAudit (from memorisation/format audit):",
        "  every number must trace to evidence (tolerance 0.05)",
        "  disclaimer required, causal wording rejected,",
        "  4-gram overlap reported",
        "Tools: draft_brief, save_brief (audit-gated)"], COLORS["gen"])

    arrow(ax, (2.9, 6.9), (3.6, 6.9), "task")
    arrow(ax, (3.6, 5.6), (2.9, 5.6), "approval request", COLORS["human"], ls="--")
    arrow(ax, (10.4, 6.5), (10.9, 6.5), "result")
    arrow(ax, (5.3, 5.3), (2.9, 3.8), "profile / peers / simulate", COLORS["data"])
    arrow(ax, (7.0, 5.3), (7.0, 3.8), "uncertainty", COLORS["dl"])
    arrow(ax, (8.8, 5.3), (11.2, 3.8), "draft / save", COLORS["gen"])
    arrow(ax, (4.5, 0.9), (4.9, 0.9), None, COLORS["grey"])
    ax.text(4.7, 0.55, "shared split", ha="center", fontsize=6.8, color=COLORS["grey"])
    arrow(ax, (9.1, 0.9), (9.5, 0.9), None, COLORS["grey"])
    ax.text(9.3, 0.55, "evidence JSON", ha="center", fontsize=6.8, color=COLORS["grey"])
    ax.text(7, 0.15, "Tool outputs are treated as untrusted data; the only side-effecting tool (save_brief) requires a passed "
                     "grounding audit AND human approval.", ha="center", fontsize=8.5, style="italic", color="#444444")
    fig.savefig(OUT / "architecture.png", dpi=170, bbox_inches="tight")
    plt.close(fig)


def workflow() -> None:
    fig, ax = plt.subplots(figsize=(14, 3.6))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 3.6)
    ax.axis("off")
    ax.text(7, 3.35, "Golden-path run: 'Should Nigeria prioritise water & sanitation or double health spending? Save the brief.'",
            ha="center", fontsize=11.5, fontweight="bold")
    steps = [
        ("1 Scope check", "regex scope guard\nno medical / political", COLORS["agent"]),
        ("2 Profile + peers\n+ notes", "country_profile\npeer_countries\ncountry_notes (scanned)", COLORS["data"]),
        ("3 Simulate", "A: water 95, san. 90\nB: 2x health spend\n+ caveats", COLORS["data"]),
        ("4 Second opinion", "uncertainty_estimate\nMC-dropout, disagreement", COLORS["dl"]),
        ("5 Draft + audit", "draft_brief\nGroundingAudit 23/23", COLORS["gen"]),
        ("6 Approval", "human approves\nsave_brief", COLORS["human"]),
        ("7 Answer", "JSON: answer, sources\nconfidence 0.30, caveats", COLORS["agent"]),
    ]
    w, gap = 1.7, 0.27
    x = 0.3
    for title, body, color in steps:
        ax.add_patch(FancyBboxPatch((x, 0.5), w, 2.3, boxstyle="round,pad=0.02,rounding_size=0.12", linewidth=2,
                                    edgecolor=color, facecolor=color + "22"))
        ax.text(x + w / 2, 2.55, title, ha="center", va="top", fontsize=9, fontweight="bold", color=color)
        ax.text(x + w / 2, 1.75, body, ha="center", va="top", fontsize=7.8, linespacing=1.4)
        if x + w + gap < 14:
            arrow(ax, (x + w, 1.6), (x + w + gap, 1.6))
        x += w + gap
    ax.text(7, 0.15, "Every step is appended to logs/agent_trace.jsonl; a failed audit or a denied approval stops the save and is "
                     "surfaced in caveats.", ha="center", fontsize=8.3, style="italic", color="#444444")
    fig.savefig(OUT / "workflow.png", dpi=170, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    architecture()
    workflow()
    print("wrote", OUT / "architecture.png", OUT / "workflow.png")

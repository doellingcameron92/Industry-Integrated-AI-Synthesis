"""Generate policy_advisor_integrated.ipynb (then execute with nbconvert)."""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parent
nb = nbf.v4.new_notebook()
cells = []


def md(text: str) -> None:
    cells.append(nbf.v4.new_markdown_cell(text.strip()))


def code(text: str) -> None:
    cells.append(nbf.v4.new_code_cell(text.strip()))


md("""
# Health-Investment Policy Advisor: integrated walkthrough

**Industry:** public policy / international development (ministries of health and finance, development-bank country teams).
**Problem:** with a fixed budget, which population-level levers (water, sanitation, electricity, immunisation, schooling,
health spending) are associated with the largest life-expectancy gains for a given country, and how much should the
answer be trusted?

This notebook runs the four integrated layers end-to-end. Each layer carries forward a prior capstone project:

| Layer | Prior project | What is reused |
|---|---|---|
| Evidence | Data Science Blog Post | World Bank data pipeline, leakage-aware cleaning, country-grouped gradient boosting, intervention simulation, peer comparison |
| Second opinion | Deep Learning Systems | Controlled baseline-vs-regularised (BatchNorm + Dropout) experiment, generalisation-gap and confidence/error analysis |
| Brief generation | Generative AI Applications | Generative writer + output-quality audit (memorisation / format validity) reused as a grounding audit |
| Control | Agentic Workflows | ReAct orchestration, typed tools, guardrails, approval gates, memory, trace logging, structured output |

![architecture](diagrams/architecture.png)
""")

code("""
import json, pandas as pd, matplotlib.pyplot as plt
from pathlib import Path
from policy_advisor.evidence import EvidenceBase, LEVERS
from policy_advisor.neural import NeuralComparison
from policy_advisor.generation import TemplateBriefWriter, GroundingAudit, make_bundle
from policy_advisor.agent import PolicyAdvisorAgent, Guardrails, AgentConfig
pd.set_option("display.width", 140)
""")

md("""
## 1. Evidence layer (Data Science Blog Post)

The blog-post project established that (a) under-5 mortality must be dropped because it is target leakage, (b) sparse
indicators (smoking, literacy, Gini) hurt more than they help, and (c) validation must be **grouped by country**, otherwise
a model that has seen Nigeria-2019 is asked to "predict" Nigeria-2020. All three lessons are baked into `EvidenceBase`.
Building takes ~40 s the first time; afterwards the fitted pipeline is cached in `outputs/models/`.
""")
code("""
evidence = EvidenceBase.load_or_build()
print(json.dumps(evidence.model_card(), indent=2)[:1500])
""")
code("""
imp = evidence.importance.head(10)
ax = imp.iloc[::-1].plot.barh(figsize=(7, 4), color="#4C78A8")
ax.set_title("Permutation importance (hold-out, country-grouped)"); ax.set_xlabel("Increase in MAE (years) when shuffled")
plt.tight_layout(); plt.show()
""")
md("""
### Evidence tools the agent will call
`country_profile` returns the latest indicators plus an **evidence-quality flag** derived from grouped-CV residuals: the
blog post found that countries such as South Africa and Nigeria were systematically mis-predicted (HIV burden, conflict),
so the system now says so up-front instead of hiding it.
""")
code("""
profile = evidence.country_profile("Nigeria")
{k: profile[k] for k in ["country", "year", "life_expectancy", "model_prediction", "evidence_quality"]}
""")
code("""
peers = evidence.peer_countries("Nigeria", n=6)
print("peer median life expectancy:", peers["peer_median_life_expectancy"], "| n peers:", peers["n_peers"])
pd.DataFrame(peers["over_performers"])
""")
md("""
`simulate_intervention` is the blog post's scenario tool, hardened for policy use: unknown levers are rejected, values
outside the observed range are flagged as extrapolation, and any gain smaller than the model error is called out.
""")
code("""
scenarios = {
    "A: Water 95% + sanitation 90%": {"basic_water_pct": 95, "basic_sanitation_pct": 90},
    "B: Electricity 95%": {"electricity_access_pct": 95},
    "C: Double health spending": {"health_exp_per_capita": round(2 * profile["health_exp_per_capita"], 1)},
    "D: Secondary enrolment 85%": {"secondary_enrollment_pct": 85},
}
rows = []
for label, iv in scenarios.items():
    r = evidence.simulate_intervention("Nigeria", iv)
    rows.append({"scenario": label, "baseline": r["baseline_prediction"], "scenario_pred": r["scenario_prediction"],
                 "gain_years": r["predicted_gain_years"], "caveats": " | ".join(r["caveats"]) or "-"})
pd.DataFrame(rows)
""")
md("""
The tree model attributes almost no *marginal* gain to doubling health spending once GDP, water, sanitation and
electricity are held fixed, even though spending is strongly correlated with life expectancy in the raw data. This is a
useful, honest finding for a ministry (money alone, without the infrastructure it usually buys, does not move the
indicator in this model), and a reminder that the model is associational: it cannot say what a real spending programme
would cause.
""")

md("""
## 2. Second opinion (Deep Learning Systems)

The deep-learning project compared a baseline CNN with a BatchNorm + Dropout CNN under identical training conditions and
showed that regularisation closed the generalisation gap and reduced over-confident errors. The same **controlled
experiment** is repeated here on tabular data: two MLPs, same split, same optimiser, one differing only by BN + Dropout.
""")
code("""
neural = NeuralComparison.load_or_build(evidence)
neural.summary.round(3)
""")
code("""
fig, axes = plt.subplots(1, 2, figsize=(11, 3.6), sharey=True)
for ax, (name, hist) in zip(axes, [("Baseline MLP", neural.baseline_history), ("Regularized MLP (BN+Dropout)", neural.regularized_history)]):
    ax.plot(hist["epoch"], hist["train_mse"] ** 0.5, label="train RMSE"); ax.plot(hist["epoch"], hist["val_mse"] ** 0.5, label="hold-out RMSE")
    ax.set_title(name); ax.set_xlabel("epoch"); ax.legend()
axes[0].set_ylabel("RMSE (years)"); axes[0].set_ylim(0, 10); plt.tight_layout(); plt.show()
""")
md("""
As in the CNN experiment, the regularised network generalises better (lower hold-out MAE, smaller train/hold-out gap).
It does **not** beat gradient boosting, so it is used as a *second opinion* rather than a replacement: the
`uncertainty_estimate` tool reports Monte-Carlo-dropout spread and, more usefully, the **disagreement** between the two
model families. On hold-out countries, gradient-boosting error is substantially higher when the models disagree.
""")
code("""
ref = neural.mc_std_reference
print(f"GBR MAE when models agree: {ref['gbr_mae_when_models_agree']:.2f}y | when they disagree (> CV MAE): "
      f"{ref['gbr_mae_when_models_disagree']:.2f}y | share disagreeing: {ref['share_disagree']:.0%}")
neural.uncertainty_estimate("Nigeria")
""")

md("""
## 3. Brief generation + grounding audit (Generative AI Applications)

The generative project evaluated a character-level Transformer with a **memorisation audit** (n-gram overlap, longest shared
string) and **format-validity checks**. Here the same idea protects a policy brief: the writer only sees a typed
`EvidenceBundle`, and `GroundingAudit` verifies that every number in the draft traces back to a tool output, that a
non-causal disclaimer is present, and that no causal wording slipped in. Without an `OPENAI_API_KEY` the deterministic
`TemplateBriefWriter` is used; with one, an LLM drafts and the audit decides whether the draft is acceptable.
""")
code("""
sims = []
for label, iv in scenarios.items():
    r = evidence.simulate_intervention("Nigeria", iv); r["label"] = label; sims.append(r)
bundle = make_bundle(profile, sims, evidence.peer_countries("Nigeria"), neural.uncertainty_estimate("Nigeria"), evidence.cv_metrics["mae"])
brief = TemplateBriefWriter().draft(bundle, audience="Federal Ministry of Health")
report = GroundingAudit().run(brief, bundle)
print(report.model_dump_json(indent=2))
""")
code("""
from IPython.display import Markdown
Markdown(brief)
""")
code("""
bad = "A WHO study shows that sanitation reform will add 7.9 years to life expectancy and cut mortality by 41.7%."
GroundingAudit().run(bad, bundle).model_dump()
""")

md("""
## 4. Control layer (Agentic Workflows)

`PolicyAdvisorAgent` adapts the Research Triage Assistant: a ReAct loop with typed tools, scope refusal, prompt-injection
scanning of tool output, an approval gate on the only side-effecting tool (`save_brief`), episodic memory and a JSONL
trace. The planner is a deterministic `MockPlanner` by default so the whole run is reproducible offline.
""")
code("""
agent = PolicyAdvisorAgent(evidence, neural, approval_callback=Guardrails.auto_approve,
                           config=AgentConfig(log_path="logs/notebook_trace.jsonl", memory_path="memory/notebook_memory.json"))
result = agent.run("Should Nigeria prioritise water and sanitation or double health spending? Save the brief.")
print(json.dumps(result.model_dump(exclude={"grounding"}), indent=2))
""")
code("""
trace = [json.loads(line) for line in Path("logs/notebook_trace.jsonl").read_text().splitlines() if json.loads(line).get("run_id") == result.run_id]
pd.DataFrame(trace).reindex(columns=["step", "kind", "name", "approved", "reason", "error", "latency_ms"]).fillna("")
""")
md("### Guardrails in action")
code("""
for task in ["What dose of amoxicillin should my child take?",
             "Compare options for Cambodia and save a brief.",   # analyst notes for KHM contain a prompt injection
             "What should Atlantis invest in?"]:
    r = agent.run(task)
    print(f"{task}\\n  -> status={r.status} refused={r.refused} confidence={r.confidence} tool_calls={r.tool_calls}\\n  {r.answer[:160]}\\n")
""")
code("""
denied = PolicyAdvisorAgent(evidence, neural, approval_callback=Guardrails.auto_deny,
                            config=AgentConfig(log_path="logs/notebook_trace.jsonl", memory_path="memory/notebook_memory.json"))
r = denied.run("Assess sanitation investment for Ghana and save the brief.")
r.status, r.brief_path, r.caveats[-1]
""")

md("""
## 5. Scenario evaluation

`evaluation/scenarios.json` holds 11 scenarios (golden paths, scope refusals, prompt injection, denied approval, tool
outage, hallucinating writer, low evidence quality, memory recall). Run `python -m evaluation.run_evaluation` to
regenerate `evaluation/results.md`; the summary table is shown below.
""")
code("""
res = json.loads(Path("evaluation/results.json").read_text())
pd.DataFrame([{"id": r["id"], "category": r["category"], **{k: r["observed"][k] for k in ["status", "tool_calls", "confidence", "brief_saved", "grounding_passed", "injection_flagged"]}, "passed": r["passed"]} for r in res])
""")

md("""
## 6. Boundaries of the system

* **Capability:** ranks *associational* scenarios from World Bank indicators for one country at a time; reports model error,
  evidence quality and cross-model disagreement; drafts a grounded brief.
* **Not a capability:** causal effect estimation, cost-effectiveness, sub-national targeting, individual medical advice,
  political messaging. The scope guard refuses the last two; the brief text states the first three.
* **Responsibility:** the human analyst owns the decision. The agent never writes a file without approval, never uses a
  number that did not come from a tool, and leaves a complete trace for audit.
""")

nb["cells"] = cells
nb["metadata"] = {"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
                  "language_info": {"name": "python"}}
nbf.write(nb, ROOT / "policy_advisor_integrated.ipynb")
print("wrote policy_advisor_integrated.ipynb")

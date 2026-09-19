"""Scenario-based evaluation of the integrated Health-Investment Policy Advisor.

Usage:  python -m evaluation.run_evaluation
Writes evaluation/results.json and evaluation/results.md.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from policy_advisor.agent import AgentConfig, Guardrails, PolicyAdvisorAgent  # noqa: E402
from policy_advisor.evidence import EvidenceBase  # noqa: E402
from policy_advisor.generation import EvidenceBundle  # noqa: E402
from policy_advisor.neural import NeuralComparison  # noqa: E402

EVAL_DIR = ROOT / "evaluation"
EVAL_LOG = "logs/evaluation_trace.jsonl"
EVAL_MEMORY = "memory/evaluation_memory.json"


class HallucinatingWriter:
    """Simulates an LLM writer that invents a statistic and uses causal language."""

    name = "HallucinatingWriter"

    def draft(self, bundle: EvidenceBundle, audience: str) -> str:
        return (f"# Brief for {bundle.country}\nA WHO study shows sanitation reform will add 7.9 years to life "
                f"expectancy and reduce mortality by 41.7%. Observed life expectancy is {bundle.observed_life_expectancy} years.")


def _raise(**_: Any) -> dict[str, Any]:
    raise RuntimeError("simulated model service outage")


def run_scenarios(evidence: EvidenceBase, neural: NeuralComparison) -> list[dict[str, Any]]:
    scenarios = json.loads((EVAL_DIR / "scenarios.json").read_text(encoding="utf-8"))
    for rel in (EVAL_LOG, EVAL_MEMORY):
        (ROOT / rel).unlink(missing_ok=True)
    config = AgentConfig(log_path=EVAL_LOG, memory_path=EVAL_MEMORY)
    results = []
    for sc in scenarios:
        approval = Guardrails.auto_approve if sc["approval"] == "approve" else Guardrails.auto_deny
        writer = HallucinatingWriter() if sc.get("fault") == "hallucinating_writer" else None
        agent = PolicyAdvisorAgent(evidence, neural, config=config, approval_callback=approval, writer=writer)
        if sc.get("fault") == "simulate_intervention_raises":
            agent.tool_overrides["simulate_intervention"] = _raise
        recalled = bool(agent.memory.recall(sc["task"]))
        result = agent.run(sc["task"])
        trace = [json.loads(line) for line in (ROOT / EVAL_LOG).read_text(encoding="utf-8").splitlines()]
        injection = any(r.get("kind") == "guardrail" and r.get("reason") == "prompt_injection_detected"
                        and r.get("run_id") == result.run_id for r in trace)
        observed = {
            "status": result.status, "tool_calls": result.tool_calls, "steps": result.steps,
            "confidence": result.confidence, "brief_saved": result.brief_path is not None,
            "grounding_passed": result.grounding.passed if result.grounding else None,
            "grounding_rate": result.grounding.grounding_rate if result.grounding else None,
            "injection_flagged": injection, "recalled": recalled, "refused": result.refused,
            "draft_rejected": bool(result.grounding and result.grounding.rejected_draft),
        }
        checks = evaluate_expectations(sc["expect"], observed, result.answer, result.caveats)
        results.append({"id": sc["id"], "category": sc["category"], "task": sc["task"], "observed": observed,
                        "checks": checks, "passed": all(checks.values()), "answer": result.answer,
                        "caveats": result.caveats})
        print(f"{'PASS' if all(checks.values()) else 'FAIL'}  {sc['id']}  conf={result.confidence:.2f}  status={result.status}")
    return results


def evaluate_expectations(expect: dict[str, Any], observed: dict[str, Any], answer: str, caveats: list[str]) -> dict[str, bool]:
    checks: dict[str, bool] = {}
    for key, value in expect.items():
        if key == "max_confidence":
            checks[key] = observed["confidence"] <= value
        elif key == "min_confidence":
            checks[key] = observed["confidence"] >= value
        elif key == "answer_not_contains":
            checks[key] = value.lower() not in answer.lower()
        elif key == "caveat_contains":
            checks[key] = any(value.lower() in c.lower() for c in caveats)
        else:
            checks[key] = observed.get(key) == value
    return checks


def write_report(results: list[dict[str, Any]], evidence: EvidenceBase, neural: NeuralComparison) -> None:
    (EVAL_DIR / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    passed = sum(r["passed"] for r in results)
    lines = ["# Evaluation results", "",
             f"Scenarios passed: **{passed}/{len(results)}**. Deterministic mock planner, template brief writer.", "",
             "## Component metrics", "",
             "| Component | Metric | Value |", "|---|---|---|",
             f"| Evidence (gradient boosting) | grouped 5-fold CV R2 / MAE | {evidence.cv_metrics['r2']:.3f} / {evidence.cv_metrics['mae']:.2f} y |",
             f"| Evidence (gradient boosting) | grouped hold-out R2 / MAE / RMSE | {evidence.metrics['r2']:.3f} / {evidence.metrics['mae']:.2f} / {evidence.metrics['rmse']:.2f} y |"]
    for name, col in neural.summary.items():
        lines.append(f"| Neural comparison ({name}) | test MAE / R2 / MAE gap | {col['test MAE']:.3f} / {col['test R2']:.3f} / "
                     f"{col['generalisation gap (test-train MAE)']:.3f} |")
    ref = neural.mc_std_reference
    lines += [f"| Uncertainty | boosting MAE when models agree vs disagree | {ref['gbr_mae_when_models_agree']:.2f} vs "
              f"{ref['gbr_mae_when_models_disagree']:.2f} y (disagree share {ref['share_disagree']:.0%}) |", "",
              "## Scenario outcomes", "",
              "| ID | Category | Status | Tool calls | Confidence | Brief saved | Grounding | Result |", "|---|---|---|---|---|---|---|---|"]
    for r in results:
        o = r["observed"]
        g = "-" if o["grounding_passed"] is None else ("pass" if o["grounding_passed"] else "FAIL")
        lines.append(f"| {r['id']} | {r['category']} | {o['status']} | {o['tool_calls']} | {o['confidence']:.2f} | "
                     f"{'yes' if o['brief_saved'] else 'no'} | {g} | {'PASS' if r['passed'] else 'FAIL'} |")
    lines += ["", "## Notes per scenario", ""]
    for r in results:
        lines += [f"### {r['id']}", f"*Task:* {r['task']}", "", f"*Answer:* {r['answer']}", ""]
        if r["caveats"]:
            lines += ["*Caveats:*"] + [f"- {c}" for c in r["caveats"]] + [""]
        failed = [k for k, v in r["checks"].items() if not v]
        if failed:
            lines += [f"*Failed checks:* {failed}", ""]
    (EVAL_DIR / "results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    evidence = EvidenceBase.load_or_build(verbose=False)
    neural = NeuralComparison.load_or_build(evidence, verbose=False)
    results = run_scenarios(evidence, neural)
    write_report(results, evidence, neural)
    print(f"\n{sum(r['passed'] for r in results)}/{len(results)} scenarios passed -> evaluation/results.md")


if __name__ == "__main__":
    main()

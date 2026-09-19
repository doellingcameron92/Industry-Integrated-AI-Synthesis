"""Generative layer: policy-brief drafting and grounding audit.

Carries forward the generative-AI project. That project trained a character-level
Transformer and, crucially, *audited* what it generated: n-gram overlap against the
training corpus (memorisation), word/format validity and a temperature sweep. The
same discipline is applied here to a language model that drafts policy briefs.
The brief is generated from a structured evidence bundle, and every number in the
brief must be traceable to that bundle; anything that is not is a hallucination
and blocks the brief from being saved. With no API key the deterministic
TemplateBriefWriter is used, so the whole system runs offline and reproducibly.
"""

from __future__ import annotations

import json
import os
import re
from typing import Any, Protocol

from pydantic import BaseModel, Field

NUMBER_RE = re.compile(r"(?<![\w.])[-+]?\d+(?:\.\d+)?(?![\w.]*[a-z])")


class EvidenceBundle(BaseModel):
    """Everything the writer is allowed to say, collected from tool outputs."""

    country: str
    iso3: str
    year: int
    region: str
    income_group: str
    observed_life_expectancy: float
    baseline_prediction: float
    scenarios: list[dict[str, Any]] = Field(default_factory=list)
    peers: dict[str, Any] = Field(default_factory=dict)
    uncertainty: dict[str, Any] = Field(default_factory=dict)
    evidence_quality: dict[str, Any] = Field(default_factory=dict)
    model_mae_years: float

    @property
    def observed_minus_predicted(self) -> float:
        return round(self.observed_life_expectancy - self.baseline_prediction, 1)

    def numbers(self) -> set[float]:
        """All numeric values present anywhere in the bundle, plus their rounded forms."""
        found: set[float] = set()

        def walk(obj: Any) -> None:
            if isinstance(obj, bool):
                return
            if isinstance(obj, (int, float)):
                found.add(round(float(obj), 2))
                found.add(round(float(obj), 1))
                found.add(float(round(obj)))
            elif isinstance(obj, dict):
                for v in obj.values():
                    walk(v)
            elif isinstance(obj, (list, tuple, set)):
                for v in obj:
                    walk(v)

        walk(self.model_dump())
        walk(self.observed_minus_predicted)
        return found


class BriefWriter(Protocol):
    name: str

    def draft(self, bundle: EvidenceBundle, audience: str) -> str: ...


class TemplateBriefWriter:
    """Deterministic writer used when no OPENAI_API_KEY is available."""

    name = "TemplateBriefWriter"

    def draft(self, bundle: EvidenceBundle, audience: str = "ministry of health") -> str:
        b = bundle
        lines = [f"# Health-investment options for {b.country} ({b.iso3}), {b.year} data",
                 f"*Prepared for: {audience}. Evidence: World Bank development indicators; gradient-boosting "
                 f"life-expectancy model (grouped-CV MAE {b.model_mae_years} years).*", "",
                 "## Where the country stands",
                 f"{b.country} ({b.region}, {b.income_group}) recorded a life expectancy of "
                 f"{b.observed_life_expectancy} years. Given its indicator profile the model expects "
                 f"{b.baseline_prediction} years; the gap of {b.observed_minus_predicted} "
                 f"years reflects factors the indicators do not capture (conflict, epidemics, data quality)."]
        if b.peers.get("peer_median_life_expectancy") is not None:
            lines.append(f"Peer countries in the same region and income group ({b.peers.get('n_peers')} countries) "
                         f"have a median life expectancy of {b.peers['peer_median_life_expectancy']} years.")
            over = b.peers.get("over_performers") or []
            if over:
                top = over[0]
                lines.append(f"The strongest wealth-adjusted over-performer is {top['country']} "
                             f"({top['life_expectancy']} years, {top['residual_vs_model']:+} versus the model).")
        lines += ["", "## Scenario comparison"]
        if not b.scenarios:
            lines.append("No intervention scenarios were simulated.")
        for s in b.scenarios:
            levers = ", ".join(f"{k} {v['before']} -> {v['after']}" for k, v in s["applied"].items())
            lines.append(f"- **{s['label']}** ({levers}): predicted {s['scenario_prediction']} years, "
                         f"a change of {s['predicted_gain_years']:+} years versus the modelled baseline.")
            for c in s.get("caveats", []):
                lines.append(f"  - Caveat: {c}")
        if b.scenarios:
            best = max(b.scenarios, key=lambda s: s["predicted_gain_years"])
            lines += ["", "## Reading the evidence",
                      f"The largest modelled gain comes from **{best['label']}** ({best['predicted_gain_years']:+} years). "
                      f"Differences smaller than the model error of {b.model_mae_years} years should not be treated as "
                      "meaningful. The model is associational: it describes countries that already have these indicator "
                      "levels and cannot prove that a programme would cause the change."]
        lines += ["", "## Confidence"]
        u = b.uncertainty
        if u:
            lines.append(f"A regularised neural network gives an independent estimate of {u.get('neural_prediction_mean')} "
                         f"years (Monte-Carlo dropout spread {u.get('neural_mc_dropout_std')} years) against "
                         f"{u.get('gradient_boosting_prediction')} years from the boosting model; overall confidence is "
                         f"**{u.get('confidence')}**.")
        q = b.evidence_quality
        if q.get("flags"):
            lines.append("Evidence-quality flags: " + "; ".join(q["flags"]) + ".")
        lines += ["", "## Recommended next steps",
                  "Validate the indicator values with the national statistics office, cost each option, and pilot before "
                  "scaling. This brief was drafted by an AI assistant and must be reviewed by a qualified analyst before use."]
        return "\n".join(lines)


class OpenAIBriefWriter:
    """LLM writer; only constructed when OPENAI_API_KEY is present."""

    name = "OpenAIBriefWriter"

    def __init__(self, model: str = "gpt-4o-mini", temperature: float = 0.3) -> None:
        from openai import OpenAI

        self.client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        self.model = model
        self.temperature = temperature

    def draft(self, bundle: EvidenceBundle, audience: str = "ministry of health") -> str:
        prompt = (
            "You write short, neutral policy briefs for a "
            f"{audience}. Use ONLY the numbers in the JSON evidence below; do not invent statistics, "
            "and state that the model is associational, not causal. Markdown, <= 350 words.\n\n"
            + json.dumps(bundle.model_dump(), indent=1)
        )
        response = self.client.chat.completions.create(
            model=self.model, temperature=self.temperature,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.choices[0].message.content or ""


def select_writer() -> BriefWriter:
    if os.environ.get("OPENAI_API_KEY"):
        return OpenAIBriefWriter()
    return TemplateBriefWriter()


class GroundingReport(BaseModel):
    total_numbers: int
    grounded_numbers: int
    ungrounded: list[str]
    grounding_rate: float
    contains_disclaimer: bool
    contains_causal_language: bool
    ngram_overlap_with_evidence: float
    passed: bool
    rejected_draft: GroundingReport | None = None


class GroundingAudit:
    """Checks a draft brief against its evidence bundle.

    Numbers: every number in the brief must appear in the bundle (rounded match).
    Language: the brief must include a review disclaimer and avoid causal claims.
    Overlap: share of the brief's 4-grams that also appear in the serialised
    evidence -- the memorisation-audit idea reused to show how much of the brief is
    verbatim data versus synthesised prose (informational, not pass/fail).
    """

    CAUSAL_RE = re.compile(r"\b(will (?:add|increase|raise|save|extend)|guarantees?|is proven to|causes)\b", re.I)
    DISCLAIMER_RE = re.compile(r"(associational|not causal|reviewed by|must be reviewed)", re.I)
    IGNORE_RE = re.compile(r"(^#+\s|\b(?:19|20)\d{2}\b|\d+-gram|<=\s*\d+|\b\d+\s*words)")

    def __init__(self, tolerance: float = 0.051, min_rate: float = 1.0) -> None:
        self.tolerance = tolerance
        self.min_rate = min_rate

    @staticmethod
    def _ngrams(text: str, n: int = 4) -> set[tuple[str, ...]]:
        tokens = re.findall(r"[a-z0-9]+", text.lower())
        return {tuple(tokens[i:i + n]) for i in range(max(0, len(tokens) - n + 1))}

    def run(self, brief: str, bundle: EvidenceBundle) -> GroundingReport:
        allowed = bundle.numbers()
        years = {float(bundle.year)}
        candidates = []
        for line in brief.splitlines():
            if line.startswith("#"):
                continue
            for match in NUMBER_RE.finditer(line):
                token = match.group()
                value = float(token)
                if value in years or (value.is_integer() and 1990 <= value <= 2035):
                    continue
                candidates.append(value)
        ungrounded = [f"{v:g}" for v in candidates
                      if not any(abs(v - a) <= self.tolerance for a in allowed)]
        total, grounded = len(candidates), len(candidates) - len(ungrounded)
        rate = grounded / total if total else 1.0
        evidence_text = json.dumps(bundle.model_dump())
        brief_grams, ev_grams = self._ngrams(brief), self._ngrams(evidence_text)
        overlap = len(brief_grams & ev_grams) / len(brief_grams) if brief_grams else 0.0
        disclaimer = bool(self.DISCLAIMER_RE.search(brief))
        causal = bool(self.CAUSAL_RE.search(brief))
        return GroundingReport(
            total_numbers=total, grounded_numbers=grounded, ungrounded=ungrounded, grounding_rate=round(rate, 3),
            contains_disclaimer=disclaimer, contains_causal_language=causal,
            ngram_overlap_with_evidence=round(overlap, 3),
            passed=rate >= self.min_rate and disclaimer and not causal,
        )


def make_bundle(profile: dict[str, Any], scenarios: list[dict[str, Any]], peers: dict[str, Any] | None,
                uncertainty: dict[str, Any] | None, model_mae_years: float) -> EvidenceBundle:
    return EvidenceBundle(
        country=profile["country"], iso3=profile["iso3"], year=profile["year"], region=profile["region"],
        income_group=profile["income_group"], observed_life_expectancy=profile["life_expectancy"],
        baseline_prediction=profile["model_prediction"], scenarios=scenarios, peers=peers or {},
        uncertainty=uncertainty or {}, evidence_quality=profile.get("evidence_quality", {}),
        model_mae_years=round(model_mae_years, 1),
    )

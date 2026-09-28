"""Run-level orchestration: scope guardrails -> parse -> generalize -> audit -> render -> approval gate.

Mirrors ``policy_advisor.agent``: typed request/result contracts, regex scope
guardrails, untrusted-input scanning, a JSONL trace per run and an explicit
approval callback. Nothing is written outside ``logs/`` until a human/candidate
approves the draft. Traces record counts, offsets and categories only, never
resume text, so the log itself does not become a store of candidate PII.
"""

from __future__ import annotations

import json
import re
import time
import uuid
from collections.abc import Callable
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

from .audit import AuditReport, audit_profile, finalize_report
from .generalize import BulletRewriter, GeneralizationConfig, Generalizer, Granularity, select_rewriter
from .lexicon import INJECTION_RE
from .parse import ResumeExtractor, parse_resume, select_extractor
from .render import render_docx, render_markdown, render_pdf
from .schema import BlindedProfile, ResumeProfile

ROOT = Path(__file__).resolve().parents[1]
OutputFormat = Literal["md", "pdf", "docx"]
RunStatus = Literal["pending_approval", "approved", "rejected", "refused", "blocked"]

OUT_OF_SCOPE_RULES: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("demographic_inference", re.compile(
        r"\b(?:infer|guess|predict|determine|estimate|identify|detect|tell\s+me|what\s+is|what's|work\s+out|figure\s+out|deduce)\b"
        r".{0,60}\b(?:gender|sex|age|race|ethnicity|ethnic|religion|religious|nationality|national\s+origin|citizenship|disability|"
        r"pregnan\w*|sexual\s+orientation|marital|veteran|how\s+old|where\s+.*\s+from)\b", re.I)),
    ("candidate_ranking", re.compile(
        r"\b(?:rank|ranking|score|scoring|rate|rating|shortlist|compare|sort|order|grade|stack[- ]rank)\b.{0,40}"
        r"\b(?:candidates?|applicants?|resumes?|cvs?|profiles?|people|them)\b|"
        r"\b(?:best|strongest|top|weakest)\s+(?:candidate|applicant|resume|fit)\b", re.I)),
    ("hiring_recommendation", re.compile(
        r"\b(?:should\s+(?:we|i)\s+(?:hire|interview|reject|advance)|recommend\w*\s+.{0,30}\b(?:hire|hiring|reject|interview|advance)|"
        r"hire\s+or\s+(?:not|reject)|(?:good|bad|strong|weak)\s+fit|worth\s+(?:hiring|interviewing)|hiring\s+decision|"
        r"(?:approve|reject)\s+(?:this|the)\s+(?:candidate|applicant))\b", re.I)),
)
REFUSAL_MESSAGES = {
    "demographic_inference": "This service does not infer or estimate demographic or protected attributes.",
    "candidate_ranking": "This service does not rank, score or compare candidates.",
    "hiring_recommendation": "This service does not make hiring or interview recommendations.",
    "prompt_injection": "The request contains instructions that attempt to override the service's rules.",
}
SCOPE_MESSAGE = ("resume_blinder only rewrites a resume into a standardized blinded draft for candidate review. "
                 "Screening decisions remain with trained human reviewers under your organization's AEDT/EEOC process.")


class ServiceConfig(BaseModel):
    log_dir: str = "logs/resume_blinder"
    output_dir: str = "outputs/resume_blinder"
    min_confidence: float = Field(default=0.5, ge=0, le=1)


class BlindingRequest(BaseModel):
    source_path: str | None = None
    text: str | None = None
    instruction: str = "Produce a standardized blinded resume."
    granularity: Granularity = "standard"
    output_formats: list[OutputFormat] = Field(default_factory=lambda: ["md", "pdf", "docx"])
    as_of: date | None = None


class ApprovalDecision(BaseModel):
    approved: bool
    reviewer: str = "candidate"
    note: str = ""


class Draft(BaseModel):
    run_id: str
    markdown: str
    audit: AuditReport
    removed: dict[str, int]


class BlindingResult(BaseModel):
    run_id: str
    status: RunStatus
    message: str = ""
    refusal_reason: str | None = None
    draft: Draft | None = None
    blinded: BlindedProfile | None = None
    profile: ResumeProfile | None = None
    outputs: dict[str, str] = Field(default_factory=dict)
    decision: ApprovalDecision | None = None
    trace_path: str = ""

    @property
    def confidence(self) -> float:
        return self.draft.audit.confidence if self.draft else 0.0


ApprovalCallback = Callable[[Draft], ApprovalDecision]


def auto_approve(draft: Draft) -> ApprovalDecision:
    return ApprovalDecision(approved=True, reviewer="auto_approve (tests/evaluation only)")


def auto_deny(draft: Draft) -> ApprovalDecision:
    return ApprovalDecision(approved=False, reviewer="auto_deny")


def interactive(draft: Draft) -> ApprovalDecision:
    print(draft.markdown)
    print(f"confidence={draft.audit.confidence} flagged={len(draft.audit.ungrounded) + len(draft.audit.leaked)}")
    answer = input("Candidate: approve this blinded resume for saving/sharing? [y/N] ").strip().lower()
    return ApprovalDecision(approved=answer in {"y", "yes"}, reviewer="interactive")


def check_scope(instruction: str) -> str | None:
    if INJECTION_RE.search(instruction):
        for reason, rx in OUT_OF_SCOPE_RULES:
            if rx.search(instruction):
                return reason
        return "prompt_injection"
    for reason, rx in OUT_OF_SCOPE_RULES:
        if rx.search(instruction):
            return reason
    return None


class RunTrace:
    """One JSONL file per run under ``logs/``."""

    def __init__(self, log_dir: Path, run_id: str) -> None:
        log_dir.mkdir(parents=True, exist_ok=True)
        self.path = log_dir / f"{run_id}.jsonl"
        self.run_id = run_id
        self.step = 0

    def log(self, kind: str, payload: dict[str, Any], latency_ms: float | None = None) -> None:
        self.step += 1
        record = {"ts": datetime.now(timezone.utc).isoformat(), "run_id": self.run_id, "step": self.step, "kind": kind, **payload}
        if latency_ms is not None:
            record["latency_ms"] = round(latency_ms, 1)
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, default=str) + "\n")


class ResumeBlinderService:
    def __init__(self, config: ServiceConfig | None = None, extractor: ResumeExtractor | None = None,
                 rewriter: BulletRewriter | None = None, root: Path = ROOT) -> None:
        self.config = config or ServiceConfig()
        self.extractor = extractor or select_extractor()
        self.rewriter = rewriter or select_rewriter()
        self.root = root
        self._pending: dict[str, tuple[Draft, BlindedProfile, list[OutputFormat], RunTrace]] = {}

    def run(self, request: BlindingRequest, approval: ApprovalCallback | None = None) -> BlindingResult:
        run_id = uuid.uuid4().hex[:12]
        trace = RunTrace(self.root / self.config.log_dir, run_id)
        trace.log("run_start", {"source_format": Path(request.source_path).suffix.lstrip(".") if request.source_path else "txt",
                                "granularity": request.granularity, "extractor": self.extractor.name, "rewriter": self.rewriter.name,
                                "output_formats": request.output_formats})

        refusal = check_scope(request.instruction)
        if refusal:
            trace.log("guardrail", {"reason": "out_of_scope", "category": refusal})
            trace.log("run_end", {"status": "refused"})
            return BlindingResult(run_id=run_id, status="refused", refusal_reason=refusal,
                                  message=f"{REFUSAL_MESSAGES[refusal]} {SCOPE_MESSAGE}", trace_path=str(trace.path))

        t0 = time.perf_counter()
        profile = parse_resume(request.source_path, text=request.text, extractor=self.extractor)
        trace.log("parse", {"extractor": profile.extractor, "chars": len(profile.raw_text), "roles": len(profile.roles),
                            "education": len(profile.education), "skills": len(profile.skills),
                            "certifications": len(profile.certifications), "achievements": len(profile.achievements),
                            "parse_coverage": profile.parse_coverage, "photos": profile.identity.photos},
                  latency_ms=(time.perf_counter() - t0) * 1000)
        for span in profile.flagged_injections:
            trace.log("guardrail", {"reason": "prompt_injection_detected", "source": "resume_text",
                                    "span": [span.start, span.end], "action": "excluded_from_profile"})

        t0 = time.perf_counter()
        config = GeneralizationConfig(granularity=request.granularity, as_of=request.as_of)
        blinded = Generalizer(config, self.rewriter).generalize(profile)
        trace.log("generalize", {"removed": blinded.removed, "lines": len(blinded.items())},
                  latency_ms=(time.perf_counter() - t0) * 1000)

        clean, report, tokens = audit_profile(profile, blinded)
        markdown = render_markdown(clean, draft=True)
        report = finalize_report(report, markdown, tokens, clean)
        trace.log("audit", {"total_lines": report.total_lines, "grounded_lines": report.grounded_lines,
                            "grounding_rate": report.grounding_rate,
                            "ungrounded": [{"section": f.section, "reason": f.reason} for f in report.ungrounded],
                            "leakage_dropped": [{"section": f.section, "categories": f.categories} for f in report.leaked],
                            "leakage_tokens_checked": report.leakage_tokens_checked, "document_leaks": report.document_leaks,
                            "placeholder_artifacts": report.placeholder_artifacts, "confidence": report.confidence,
                            "passed": report.passed})
        draft = Draft(run_id=run_id, markdown=markdown, audit=report, removed=clean.removed)
        result = BlindingResult(run_id=run_id, status="pending_approval", draft=draft, blinded=clean, profile=profile,
                                trace_path=str(trace.path),
                                message="Draft ready. It must be approved by the candidate before it is saved or shared.")
        if not report.passed or report.confidence < self.config.min_confidence:
            trace.log("guardrail", {"reason": "audit_failed", "document_leaks": report.document_leaks,
                                    "confidence": report.confidence, "action": "blocked_before_approval"})
            trace.log("run_end", {"status": "blocked"})
            return result.model_copy(update={"status": "blocked", "message": "Audit failed; the draft cannot be approved or saved."})

        trace.log("approval_requested", {"confidence": report.confidence})
        self._pending[run_id] = (draft, clean, list(request.output_formats), trace)
        if approval is None:
            return result
        return self.finalize(result, approval(draft))

    def finalize(self, result: BlindingResult, decision: ApprovalDecision) -> BlindingResult:
        """Apply the candidate's decision; only an explicit approval writes documents."""
        if result.run_id not in self._pending:
            raise ValueError(f"run {result.run_id} has no pending draft")
        draft, clean, formats, trace = self._pending.pop(result.run_id)
        trace.log("approval_decision", {"approved": decision.approved, "reviewer": decision.reviewer})
        if not decision.approved:
            trace.log("run_end", {"status": "rejected"})
            return result.model_copy(update={"status": "rejected", "decision": decision,
                                             "message": "Candidate did not approve; nothing was saved."})
        final_md = render_markdown(clean, draft=False)
        out_dir = self.root / self.config.output_dir / result.run_id
        outputs: dict[str, str] = {}
        if "md" in formats:
            out_dir.mkdir(parents=True, exist_ok=True)
            path = out_dir / "blinded_resume.md"
            path.write_text(final_md, encoding="utf-8")
            outputs["md"] = str(path)
        if "pdf" in formats:
            outputs["pdf"] = str(render_pdf(final_md, out_dir / "blinded_resume.pdf"))
        if "docx" in formats:
            outputs["docx"] = str(render_docx(final_md, out_dir / "blinded_resume.docx"))
        trace.log("saved", {"formats": sorted(outputs)})
        trace.log("run_end", {"status": "approved"})
        return result.model_copy(update={"status": "approved", "decision": decision, "outputs": outputs,
                                         "message": "Approved by candidate; documents saved."})


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Rewrite a resume into a standardized blinded draft for candidate approval.")
    parser.add_argument("resume", help="PDF, DOCX or text resume")
    parser.add_argument("--granularity", choices=["coarse", "standard", "fine"], default="standard")
    parser.add_argument("--formats", default="md,pdf,docx")
    args = parser.parse_args()
    service = ResumeBlinderService()
    result = service.run(BlindingRequest(source_path=args.resume, granularity=args.granularity,
                                         output_formats=[f for f in args.formats.split(",") if f]),  # type: ignore[misc]
                         approval=interactive)
    print(json.dumps({"run_id": result.run_id, "status": result.status, "confidence": result.confidence,
                      "outputs": result.outputs, "trace": result.trace_path, "message": result.message}, indent=2))


if __name__ == "__main__":
    main()

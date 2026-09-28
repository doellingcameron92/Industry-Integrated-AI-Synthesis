"""Resume blinding service: parse -> generalize -> audit -> render, gated on candidate approval."""

from .audit import AuditReport, audit_profile, build_leakage_list, finalize_report
from .generalize import GeneralizationConfig, Generalizer, generalize
from .parse import HeuristicExtractor, OpenAIExtractor, load_document, parse_resume
from .render import render_docx, render_markdown, render_pdf
from .schema import BlindedProfile, GroundedLine, ResumeProfile, SourceSpan, SpannedText
from .service import (
    ApprovalDecision,
    BlindingRequest,
    BlindingResult,
    Draft,
    ResumeBlinderService,
    ServiceConfig,
    auto_approve,
    auto_deny,
    check_scope,
    interactive,
)

__all__ = [
    "ApprovalDecision", "AuditReport", "BlindedProfile", "BlindingRequest", "BlindingResult", "Draft", "GeneralizationConfig",
    "Generalizer", "GroundedLine", "HeuristicExtractor", "OpenAIExtractor", "ResumeBlinderService", "ResumeProfile",
    "ServiceConfig", "SourceSpan", "SpannedText", "audit_profile", "auto_approve", "auto_deny", "build_leakage_list",
    "check_scope", "finalize_report", "generalize", "interactive", "load_document", "parse_resume", "render_docx",
    "render_markdown", "render_pdf",
]

"""Grounding and leakage audits for blinded output.

Grounding (mirrors ``policy_advisor.generation.GroundingAudit``): every output
line must cite at least one valid source span from the original resume, the
span must not come from an identity field or a flagged injected instruction, and
the line's words and numbers must be explainable by its spans plus the fixed
generalization vocabulary. Ungrounded lines are dropped, flagged and lower
confidence.

Leakage: a token list is built from the original resume's identifiers,
organization/school/location names and every protected-attribute term it
contains. Any output line matching the list is dropped; the final rendered
document must have zero matches.
"""

from __future__ import annotations

import re
from typing import Literal

from pydantic import BaseModel, Field

from .lexicon import (
    AFFINITY_RE,
    EMAIL_RE,
    GENDERED_PRONOUN_RE,
    GENERIC_ORG_WORDS,
    INJECTION_RE,
    NAMED_AFFINITY_RE,
    PHONE_RE,
    PROTECTED_RE,
    REFERENCE,
    URL_RE,
    ZIP_RE,
)
from .schema import BlindedProfile, GroundedLine, ResumeProfile, SourceSpan, SpannedText

WORD_RE = re.compile(r"[a-z0-9]+(?:['’][a-z]+)?", re.I)
NUMBER_RE = re.compile(r"\d+(?:[.,]\d+)*")
DURATION_RE = re.compile(r"^(?:\d+ years?)?(?: ?\d+ months?)?$")
PLACEHOLDER_RE = re.compile(r"\[(?:redacted|removed|name|email|phone|address|employer|school|x+)\]|█|\*{3,}|x{4,}", re.I)
REWRITE_VOCAB = {"the", "a", "an", "employer", "employer's", "university", "university's", "state", "and", "of"}
REWRITE_VOCAB |= {w.lower() for title in REFERENCE["neutral_titles"].values() for w in WORD_RE.findall(title)}
LEAK_MIN_LEN = 3
COMMON_WORDS = {"the", "and", "for", "with", "from", "work", "permit", "status", "two", "three", "one", "none", "yes", "not"}


def _words(text: str) -> set[str]:
    return {w.lower().replace("’", "'") for w in WORD_RE.findall(text)}


def _template_vocab() -> set[str]:
    labels: list[str] = [*REFERENCE["industries"].values(), *REFERENCE["size_bands"].values(), *REFERENCE["degree_levels"].values(),
                         *REFERENCE["institutions"].values(), REFERENCE["unknown_institution"], "United States", "Remote", "Role",
                         "Degree", "Career break", "years", "months", "year", "month"]
    labels += [tier for _, tier in REFERENCE["institution_keywords"]]
    labels += [f"{region} US" for _, region in REFERENCE["us_states"].values()]
    labels += [r for pair in REFERENCE["countries"].values() for r in pair]
    vocab: set[str] = set()
    for label in labels:
        vocab |= _words(label)
    return vocab


TEMPLATE_VOCAB = _template_vocab()
TEMPLATE_NUMBERS = {n for label in REFERENCE["size_bands"].values() for n in NUMBER_RE.findall(label)}


class FlaggedLine(BaseModel):
    section: str
    text: str
    reason: str
    categories: list[str] = Field(default_factory=list)


class LeakageToken(BaseModel):
    token: str
    category: str
    kind: Literal["text", "digits"] = "text"


class AuditReport(BaseModel):
    total_lines: int
    grounded_lines: int
    grounding_rate: float
    ungrounded: list[FlaggedLine] = Field(default_factory=list)
    leaked: list[FlaggedLine] = Field(default_factory=list)
    leakage_tokens_checked: int = 0
    document_leaks: list[str] = Field(default_factory=list)
    placeholder_artifacts: int = 0
    injections_flagged: int = 0
    parse_coverage: float = 1.0
    confidence: float = 1.0
    passed: bool = False


# ---------------------------------------------------------------------------
# Grounding
# ---------------------------------------------------------------------------

def protected_spans(profile: ResumeProfile) -> list[SourceSpan]:
    ident = profile.identity
    items: list[SpannedText] = [ident.name] if ident.name else []
    items += [*ident.emails, *ident.phones, *ident.links, *ident.addresses, *ident.pronouns, *ident.demographics, *ident.personal]
    return [*(s for item in items for s in item.spans), *profile.flagged_injections]


def _overlaps(a: SourceSpan, b: SourceSpan) -> bool:
    return a.start < b.end and b.start < a.end


def grounding_issue(line: GroundedLine, profile: ResumeProfile, forbidden: list[SourceSpan]) -> str | None:
    """Return why ``line`` is not traceable to the original resume, or ``None`` when it is grounded."""
    if not line.text.strip():
        return "empty_line"
    if not line.spans:
        return "no_source_span"
    raw = profile.raw_text
    for span in line.spans:
        if span.end > len(raw) or span.start >= span.end or raw[span.start:span.end] != span.text:
            return "invalid_source_span"
        if any(_overlaps(span, f) for f in forbidden):
            return "span_from_identity_or_injection"
    source = " ".join(s.text for s in line.spans)
    source_words = _words(source)
    source_numbers = set(NUMBER_RE.findall(source))
    words = _words(line.text)
    numbers = set(NUMBER_RE.findall(line.text))
    if line.derivation == "verbatim":
        return None if line.text in source else "verbatim_text_not_in_source"
    if line.derivation == "rewritten":
        extra = words - source_words - REWRITE_VOCAB
        if numbers - source_numbers:
            return "number_not_in_source"
        return f"words_not_in_source:{','.join(sorted(extra))}" if extra else None
    date_spans = [s for s in line.spans if s.field.endswith((".start", ".end"))]
    if line.derivation == "computed":
        if len(date_spans) < 2:
            return "computed_without_date_spans"
        if line.text != "Career break" and not DURATION_RE.match(line.text):
            return "unexpected_computed_text"
        return None
    duration_numbers = {n for m in re.finditer(r"(\d+) (?:years?|months?)", line.text) for n in m.groups()}
    if duration_numbers and len(date_spans) < 2:
        return "duration_without_date_spans"
    extra_numbers = numbers - source_numbers - TEMPLATE_NUMBERS - duration_numbers
    if extra_numbers:
        return "number_not_in_source"
    extra = words - source_words - TEMPLATE_VOCAB - {n for n in words if n.isdigit()}
    return f"words_not_in_source:{','.join(sorted(extra))}" if extra else None


# ---------------------------------------------------------------------------
# Leakage
# ---------------------------------------------------------------------------

def _distinctive_tokens(value: str) -> list[str]:
    tokens = [t.strip(".,'’()") for t in re.split(r"[\s/]+", value)]
    return [t for t in tokens if len(t) >= LEAK_MIN_LEN and t.lower() not in GENERIC_ORG_WORDS and t.lower() not in COMMON_WORDS
            and not t.isdigit()]


def build_leakage_list(profile: ResumeProfile) -> list[LeakageToken]:
    """Tokens from the original resume that must never appear in blinded output."""
    out: dict[tuple[str, str], LeakageToken] = {}

    def add(token: str, category: str, kind: Literal["text", "digits"] = "text") -> None:
        token = token.strip(" ,;:|()")
        if len(token) >= (7 if kind == "digits" else LEAK_MIN_LEN):
            out.setdefault((token.lower(), kind), LeakageToken(token=token, category=category, kind=kind))

    ident = profile.identity
    if ident.name:
        add(ident.name.value, "name")
        for part in re.split(r"[\s-]+", ident.name.value):
            add(part, "name")
    for item in ident.emails:
        add(item.value, "email")
        add(item.value.split("@")[0], "email")
    for item in ident.phones:
        add(item.value, "phone")
        add(re.sub(r"\D", "", item.value), "phone", "digits")
    for item in ident.links:
        add(item.value, "link")
        handle = item.value.rstrip("/").rsplit("/", 1)[-1]
        if handle != item.value:
            add(handle, "link")
    for item in ident.addresses:
        add(item.value, "address")
        for zip_code in ZIP_RE.findall(item.value):
            add(zip_code, "address")
        for token in _distinctive_tokens(item.value):
            if token.lower() not in {"road", "street", "avenue", "lane", "drive"}:
                add(token, "address")
    for item in ident.pronouns:
        add(item.value, "pronouns")
    for item in [*ident.demographics, *ident.personal]:
        value = item.value.split(":", 1)[-1].strip()
        add(value, "demographic")
        for token in re.findall(r"\b[A-Z][\w'’-]{2,}", value):
            add(token, "demographic")
        for year in re.findall(r"\b(?:19|20)\d{2}\b", value):
            out.setdefault((year, "text"), LeakageToken(token=year, category="demographic"))
    for role in profile.roles:
        if role.employer:
            add(role.employer.value, "employer")
            for token in _distinctive_tokens(role.employer.value):
                add(token, "employer")
        if role.location and role.location.value.strip().lower() != "remote":
            add(role.location.value, "location")
            add(role.location.value.split(",")[0], "location")
        for date_item in (role.start, role.end):
            if date_item and re.search(r"\d{4}", date_item.value):
                add(date_item.value, "date")
    for edu in profile.education:
        if edu.institution:
            add(edu.institution.value, "institution")
            for token in _distinctive_tokens(edu.institution.value):
                add(token, "institution")
        if edu.location:
            add(edu.location.value, "location")
        for item in edu.dates:
            add(item.value, "date") if not item.value.isdigit() else out.setdefault(
                (item.value, "text"), LeakageToken(token=item.value, category="date"))
    raw = profile.raw_text
    for rx, category in ((PROTECTED_RE, "protected_attribute"), (GENDERED_PRONOUN_RE, "pronoun"),
                         (NAMED_AFFINITY_RE, "affinity_group"), (AFFINITY_RE, "affinity_group")):
        for m in rx.finditer(raw):
            add(m.group(0), category)
    for span in profile.flagged_injections:
        add(span.text.strip(), "injected_instruction")
        for m in INJECTION_RE.finditer(span.text):
            add(m.group(0), "injected_instruction")
    return sorted(out.values(), key=lambda t: (t.category, t.token.lower()))


def compile_leakage(tokens: list[LeakageToken]) -> tuple[list[tuple[re.Pattern[str], LeakageToken]], list[LeakageToken]]:
    text_patterns = [(re.compile(r"(?<![\w])" + re.escape(t.token) + r"(?![\w])", re.I), t) for t in tokens if t.kind == "text"]
    return text_patterns, [t for t in tokens if t.kind == "digits"]


def find_leaks(text: str, compiled: tuple[list[tuple[re.Pattern[str], LeakageToken]], list[LeakageToken]]) -> list[str]:
    text_patterns, digit_tokens = compiled
    hits = [t.category for rx, t in text_patterns if rx.search(text)]
    digits = re.sub(r"\D", "", text)
    hits += [t.category for t in digit_tokens if t.token in digits]
    for rx, category in ((EMAIL_RE, "email_pattern"), (URL_RE, "link_pattern"), (PROTECTED_RE, "protected_attribute_pattern"),
                         (GENDERED_PRONOUN_RE, "pronoun_pattern"), (INJECTION_RE, "injection_pattern")):
        if rx.search(text):
            hits.append(category)
    if any(sum(ch.isdigit() for ch in m.group()) >= 10 for m in PHONE_RE.finditer(text)):
        hits.append("phone_pattern")
    return sorted(set(hits))


# ---------------------------------------------------------------------------
# Audit entry points
# ---------------------------------------------------------------------------

def _remove(blinded: BlindedProfile, target: GroundedLine) -> None:
    for name in ("skills", "education", "certifications", "achievements"):
        items: list[GroundedLine] = getattr(blinded, name)
        if any(x is target for x in items):
            items[:] = [x for x in items if x is not target]
            return
    for entry in blinded.experience:
        entry.bullets = [b for b in entry.bullets if b is not target]
        if entry.meta is target:
            entry.meta = None
        if entry.title is target:
            evidence = entry.meta or next(iter(entry.bullets), None)
            entry.title = GroundedLine(text="Role", spans=evidence.spans, derivation="generalized") if evidence else None
    blinded.experience = [e for e in blinded.experience if e.title is not None]


def audit_profile(profile: ResumeProfile, blinded: BlindedProfile) -> tuple[BlindedProfile, AuditReport, list[LeakageToken]]:
    clean = blinded.model_copy(deep=True)
    forbidden = protected_spans(profile)
    tokens = build_leakage_list(profile)
    compiled = compile_leakage(tokens)
    ungrounded: list[FlaggedLine] = []
    leaked: list[FlaggedLine] = []
    items = clean.items()
    for section, line in items:
        issue = grounding_issue(line, profile, forbidden)
        if issue:
            ungrounded.append(FlaggedLine(section=section, text=line.text, reason=issue))
            _remove(clean, line)
            continue
        hits = find_leaks(line.text, compiled)
        if hits or PLACEHOLDER_RE.search(line.text):
            leaked.append(FlaggedLine(section=section, text=line.text, reason="leakage", categories=hits or ["placeholder"]))
            _remove(clean, line)
    total = len(items)
    grounded = total - len(ungrounded)
    report = AuditReport(
        total_lines=total, grounded_lines=grounded, grounding_rate=round(grounded / total, 3) if total else 0.0,
        ungrounded=ungrounded, leaked=leaked, leakage_tokens_checked=len(tokens),
        injections_flagged=len(profile.flagged_injections), parse_coverage=profile.parse_coverage)
    return clean, report, tokens


def finalize_report(report: AuditReport, document: str, tokens: list[LeakageToken], clean: BlindedProfile) -> AuditReport:
    """Scan the rendered document itself and compute confidence."""
    compiled = compile_leakage(tokens)
    doc_leaks: list[str] = []
    for line in document.splitlines():
        doc_leaks += find_leaks(line, compiled)
    placeholders = len(PLACEHOLDER_RE.findall(document))
    confidence = 1.0
    confidence -= min(0.5, 0.1 * len(report.ungrounded))
    confidence -= min(0.3, 0.05 * len(report.leaked))
    confidence -= 0.1 if report.injections_flagged else 0.0
    confidence -= 0.5 * (1 - report.parse_coverage)
    if not clean.skills or not any(e.kind == "role" for e in clean.experience):
        confidence -= 0.2
    return report.model_copy(update={
        "document_leaks": sorted(set(doc_leaks)), "placeholder_artifacts": placeholders,
        "confidence": round(max(0.0, min(1.0, confidence)), 2),
        "passed": not doc_leaks and placeholders == 0,
    })

"""Typed contracts for the resume blinder.

Every value extracted from a resume carries the character spans of the original
text it came from, and every line of the blinded output carries the spans it was
derived from. That is what makes the grounding audit possible: an output line
with no valid span back into the original resume cannot be shown.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

SourceFormat = Literal["pdf", "docx", "txt"]
Derivation = Literal["verbatim", "rewritten", "generalized", "computed"]


class SourceSpan(BaseModel):
    """A half-open character range ``[start, end)`` into ``ResumeProfile.raw_text``."""

    model_config = ConfigDict(frozen=True)

    start: int = Field(ge=0)
    end: int = Field(ge=0)
    text: str
    field: str = ""


class SpannedText(BaseModel):
    value: str
    spans: list[SourceSpan] = Field(default_factory=list)


class Role(BaseModel):
    title: SpannedText | None = None
    employer: SpannedText | None = None
    location: SpannedText | None = None
    start: SpannedText | None = None
    end: SpannedText | None = None
    bullets: list[SpannedText] = Field(default_factory=list)


class Education(BaseModel):
    degree: SpannedText | None = None
    field_of_study: SpannedText | None = None
    institution: SpannedText | None = None
    location: SpannedText | None = None
    dates: list[SpannedText] = Field(default_factory=list)
    details: list[SpannedText] = Field(default_factory=list)


class Certification(BaseModel):
    name: SpannedText
    dates: list[SpannedText] = Field(default_factory=list)


class Identity(BaseModel):
    """Direct identifiers and protected attributes. Never rendered; used to build the leakage list."""

    name: SpannedText | None = None
    emails: list[SpannedText] = Field(default_factory=list)
    phones: list[SpannedText] = Field(default_factory=list)
    links: list[SpannedText] = Field(default_factory=list)
    addresses: list[SpannedText] = Field(default_factory=list)
    pronouns: list[SpannedText] = Field(default_factory=list)
    demographics: list[SpannedText] = Field(default_factory=list)
    personal: list[SpannedText] = Field(default_factory=list)
    photos: int = 0


class ResumeProfile(BaseModel):
    raw_text: str
    source_format: SourceFormat
    extractor: str
    identity: Identity = Field(default_factory=Identity)
    summary: list[SpannedText] = Field(default_factory=list)
    roles: list[Role] = Field(default_factory=list)
    education: list[Education] = Field(default_factory=list)
    skills: list[SpannedText] = Field(default_factory=list)
    certifications: list[Certification] = Field(default_factory=list)
    achievements: list[SpannedText] = Field(default_factory=list)
    flagged_injections: list[SourceSpan] = Field(default_factory=list)
    unassigned: list[SourceSpan] = Field(default_factory=list)
    parse_coverage: float = 1.0


class GroundedLine(BaseModel):
    """One unit of blinded output and the evidence it was derived from."""

    text: str
    spans: list[SourceSpan] = Field(default_factory=list)
    derivation: Derivation = "verbatim"


class BlindedEntry(BaseModel):
    kind: Literal["role", "career_break"] = "role"
    title: GroundedLine | None = None
    meta: GroundedLine | None = None
    bullets: list[GroundedLine] = Field(default_factory=list)


class BlindedProfile(BaseModel):
    skills: list[GroundedLine] = Field(default_factory=list)
    experience: list[BlindedEntry] = Field(default_factory=list)
    education: list[GroundedLine] = Field(default_factory=list)
    certifications: list[GroundedLine] = Field(default_factory=list)
    achievements: list[GroundedLine] = Field(default_factory=list)
    removed: dict[str, int] = Field(default_factory=dict)
    notes: list[str] = Field(default_factory=list)

    def items(self) -> list[tuple[str, GroundedLine]]:
        out: list[tuple[str, GroundedLine]] = [("skills", s) for s in self.skills]
        for entry in self.experience:
            for line in (entry.title, entry.meta):
                if line is not None:
                    out.append(("experience", line))
            out += [("experience", b) for b in entry.bullets]
        out += [("education", e) for e in self.education]
        out += [("certifications", c) for c in self.certifications]
        out += [("achievements", a) for a in self.achievements]
        return out

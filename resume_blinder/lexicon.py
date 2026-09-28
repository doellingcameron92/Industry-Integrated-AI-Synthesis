"""Shared patterns: identity/contact detection, protected-attribute terms and injection phrases."""

from __future__ import annotations

import json
import re
from pathlib import Path

REFERENCE = json.loads((Path(__file__).with_name("reference.json")).read_text(encoding="utf-8"))

MONTHS = r"(?:jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*\.?"
DATE_TOKEN = rf"(?:{MONTHS}\s+(?:19|20)\d{{2}}|(?:0?[1-9]|1[0-2])/(?:19|20)\d{{2}}|(?:19|20)\d{{2}})"
END_TOKEN = rf"(?:{DATE_TOKEN}|present|current|now|today)"
DATE_RANGE_RE = re.compile(rf"\b(?P<start>{DATE_TOKEN})\s*(?:-|–|—|to|until)\s*(?P<end>{END_TOKEN})\b", re.I)
DATE_TOKEN_RE = re.compile(rf"\b{DATE_TOKEN}\b", re.I)
YEAR_RE = re.compile(r"\b(?:19|20)\d{2}\b")

EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")
URL_RE = re.compile(r"(?:https?://\S+|www\.\S+|\b(?:[\w-]+\.)+(?:com|org|net|io|dev|me|co|ai)/\S*)", re.I)
PHONE_RE = re.compile(r"(?<!\w)(?:\+?\d{1,3}[\s.-]?)?(?:\(\d{2,4}\)[\s.-]?)?\d{2,4}[\s.-]\d{2,4}(?:[\s.-]\d{2,4})?(?!\w)")
ZIP_RE = re.compile(r"\b\d{5}(?:-\d{4})?\b|\b\d{2}-\d{3}\b")
STREET_RE = re.compile(
    r"\b\d{1,6}\s+(?:[A-Z][\w'.-]*\s+){0,4}(?:Street|St|Avenue|Ave|Road|Rd|Boulevard|Blvd|Lane|Ln|Drive|Dr|Way|Court|Ct|Place|Pl|"
    r"Terrace|Parkway|Pkwy|ul\.|Strasse|Straße)\b\.?", re.I)

LABELED_FIELD_RE = re.compile(
    r"^\s*(?P<label>pronouns|date of birth|dob|birth ?date|born|age|gender|sex|nationality|citizenship|marital status|family status|"
    r"religion|ethnicity|race|veteran status|disability(?: status)?|place of birth|visa status|work permit|address|home address|"
    r"photo|children|dependents)\s*[:\-]\s*(?P<value>.+)$", re.I)
PRONOUN_DECL_RE = re.compile(r"\(?\b(?:she/her(?:/hers)?|he/him(?:/his)?|they/them(?:/theirs)?|ze/zir|xe/xem)\b\)?", re.I)

GENDERED_PRONOUN_RE = re.compile(r"\b(?:he|she|him|her|his|hers|himself|herself)\b", re.I)

PROTECTED_TERMS = [
    # gender and family status
    r"wom[ae]n(?:'s)?", r"female", r"male", r"\bmen\b", r"mothers?", r"moms?", r"fathers?", r"dads?", r"maternity", r"paternity",
    r"pregnan\w*", r"wife", r"husband", r"married", r"divorced", r"widow\w*", r"spouse", r"single (?:mother|father|parent)",
    r"children", r"kids", r"sorority", r"fraternity", r"ladies", r"girls?", r"boys?",
    # age
    r"age[d]?\s*\d{2}", r"\d{2} years old", r"date of birth", r"retiree", r"millennial", r"boomer", r"gen[- ]z",
    # race, ethnicity, national origin, citizenship
    r"african[- ]american", r"black(?!\s+(?:friday|box|belt|hat))", r"hispanic", r"latin[aox]s?", r"latine", r"asian(?:[- ]american)?",
    r"caucasian", r"native american", r"indigenous", r"pacific islander", r"first nations", r"people of colou?r", r"person of colou?r",
    r"bipoc", r"minority", r"citizen\w*", r"green card", r"visa", r"immigra\w*", r"native speaker", r"mother tongue",
    r"heritage speaker", r"first[- ]generation", r"nationality", r"national origin",
    # religion
    r"christian", r"catholic", r"protestant", r"baptist", r"methodist", r"lutheran", r"mormon", r"muslim", r"islamic", r"jewish",
    r"hindu", r"buddhist", r"sikh", r"church", r"mosque", r"synagogue", r"temple", r"parish", r"bible", r"faith[- ]based", r"religious",
    # disability and health status
    r"disab\w*", r"wheelchair", r"deaf", r"hard of hearing", r"autis\w*", r"adhd", r"dyslexi\w*", r"chronic illness",
    # veteran status
    r"veterans?(?:'s?)?", r"military spouse",
    # sexual orientation and gender identity
    r"gay", r"lesbians?", r"bisexual", r"lgbt\w*\+?", r"queer", r"transgender", r"non-?binary", r"pride (?:network|erg|alliance|group)",
    r"pronouns",
]
PROTECTED_RE = re.compile(r"(?<![\w-])(?:" + "|".join(PROTECTED_TERMS) + r")(?![\w-])", re.I)

AFFINITY_GROUPS = (
    r"wom[ae]n(?:'s)?|black|latinx|latin[ao]s?|hispanic|asian(?: american)?|lgbtq?\+?|queer|pride|veterans?(?:')?|christian|muslim|"
    r"jewish|hindu|disability|working parents?|parents?|mothers?|moms?"
)
AFFINITY_RE = re.compile(
    rf"\b(?:the\s+)?(?:{AFFINITY_GROUPS})\s+(?:in\s+(?:tech|data|stem|ai|engineering|computing|business|leadership|healthcare)\b"
    r"|(?:[\w&-]+\s+){0,2}(?:employee resource group|erg|network|society|association|alliance|chapter|club|caucus|affinity group|"
    r"collective|council|coalition|forum)\b)", re.I)
NAMED_AFFINITY_ORGS = [
    "Society of Women Engineers", "National Society of Black Engineers", "Society of Hispanic Professional Engineers",
    "National Association of Hispanic Nurses", "National Black Nurses Association", "Girls Who Code", "Black Girls Code",
    "Women Who Code", "Lesbians Who Tech", "Out in Tech", "AnitaB.org", "Grace Hopper Celebration",
]
NAMED_AFFINITY_RE = re.compile(r"\b(?:the\s+)?(?:" + "|".join(re.escape(n) for n in NAMED_AFFINITY_ORGS) + r")\b", re.I)

INJECTION_RE = re.compile(
    r"(ignore\s+(?:all\s+)?(?:previous|prior|above)\s+instructions|disregard\s+(?:your|the|all|any)\s+(?:rules|instructions|guidelines)|"
    r"you\s+are\s+now|system\s+prompt|reveal\s+your|override\s+(?:the|your)|"
    r"(?:note|message|instruction)s?\s+(?:to|for)\s+(?:the\s+)?(?:ai|llm|model|assistant|screening|ats|recruiter\s+bot)|"
    r"\b(?:rank|rate|score|mark|classify)\s+(?:this|the|me|my)\s+(?:candidate|applicant|resume|cv|profile)|"
    r"\b(?:top|best|strongest|ideal)\s+(?:candidate|applicant)\b|recommend\s+(?:an?\s+)?(?:immediate\s+)?(?:hire|hiring|interview)|"
    r"\bhire\s+(?:me|this\s+(?:candidate|applicant))|move\s+(?:me|this\s+(?:candidate|applicant))\s+to\s+the\s+(?:top|next\s+round))", re.I)

GENERIC_ORG_WORDS = {
    "the", "of", "and", "for", "at", "in", "inc", "llc", "ltd", "llp", "gmbh", "corp", "corporation", "company", "co", "group",
    "partners", "systems", "services", "solutions", "health", "healthcare", "bank", "financial", "university", "college", "institute",
    "school", "state", "community", "technology", "technologies", "tech", "labs", "analytics", "consulting", "global",
    "international", "national", "center", "centre", "hospital", "medical", "department", "city", "county", "public", "regional",
    "foundation", "data", "software", "general", "unified", "district", "clinic", "house", "works", "technical", "research",
    "sciences", "science", "studies", "network", "agency", "office", "logistik", "logistics", "importers", "toys", "world", "wide",
}

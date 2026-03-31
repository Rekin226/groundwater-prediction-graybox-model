# Journal Style Profiles

Load the matching profile when the user selects a journal. Override the HJ defaults in
SKILL.md with any item listed here. Fields not listed inherit the HJ default.

---

## 1. Hydrogeology Journal (HJ) — Springer / IAH

| Field | Value |
|---|---|
| Publisher | Springer |
| Abstract | ≤ 250 words, no citations, no headings |
| Body word limit | ~ 8,000 words (excl. abstract, refs, captions) |
| Headings | Decimal numbered, max 3 levels: `1.`, `1.1`, `1.1.1` |
| In-text citation | `(Author Year)` or `Author (Year)` — Harvard |
| Multiple citations | Alphabetical, semicolon-separated: `(Abbott 1991; Barakat et al. 1995)` |
| Reference format | `Lastname IN (Year) Title. Journal Abbrev Vol(Issue):pages. https://doi.org/xx` |
| Journal name abbrev | ISSN LTWA (e.g., "Hydrogeology J", "Water Resour Res", "J Hydrol") |
| Fig caption | Below figure |
| Table caption | Above table |
| Key Points | Not required |
| Highlights | Not required |
| Supplementary | Allowed |

---

## 2. Water Resources Research (WRR) — AGU / Wiley

| Field | Value |
|---|---|
| Publisher | AGU / Wiley |
| Abstract | ≤ 250 words |
| Key Points | **Required**: exactly 3 bullets, ≤ 140 characters each, placed before abstract |
| Body word limit | ≤ 12,000 words |
| Headings | NOT numbered — plain bold headings (Introduction, Methods, Results…) |
| In-text citation | `(Author, Year)` or `Author (Year)` — AGU author-date |
| Multiple citations | Alphabetical, semicolon-separated: `(Abbott, 1991; Barakat et al., 1995)` |
| Reference format | `Lastname, I. N. (Year). Title. *Journal Name*, *Vol*(Issue), pages. https://doi.org/xx` |
| Journal name | Full name, italic |
| Fig caption | Below figure |
| Table caption | Above table |
| Plain language summary | Recommended (100–150 words for non-specialist audience) |

**Semantic Scholar query to verify format:** `venue:"Water Resources Research" year:2023`

---

## 3. Journal of Hydrology (JH) — Elsevier

| Field | Value |
|---|---|
| Publisher | Elsevier |
| Abstract | ≤ 250 words |
| Highlights | **Required**: 3–5 bullet points, ≤ 85 characters each |
| Body word limit | ~ 10,000 words |
| Headings | Numbered: `1.`, `1.1.`, `1.1.1.` (note trailing full stop) |
| In-text citation | `(Author, Year)` — Elsevier Harvard |
| Multiple citations | Alphabetical: `(Abbott, 1991; Barakat et al., 1995)` |
| Reference format | `Lastname, I.N., Lastname2, I.N., Year. Title. J. Hydrol. Vol, pages. https://doi.org/xx` |
| Journal name abbrev | Abbreviated with full stop: "J. Hydrol.", "Water Resour. Res." |
| Fig caption | Below figure |
| Table caption | Above table |
| CRediT statement | Required (author contributions) |

**Semantic Scholar query to verify format:** `venue:"Journal of Hydrology" year:2023`

---

## 4. Hydrology and Earth System Sciences (HESS) — EGU / Copernicus

| Field | Value |
|---|---|
| Publisher | EGU / Copernicus (open access) |
| Abstract | ≤ 300 words |
| Body word limit | No strict limit; typically 8,000–12,000 words |
| Headings | Numbered: `1`, `1.1`, `1.1.1` (no trailing punctuation) |
| In-text citation | `(Author, Year)` or `Author (Year)` |
| Multiple citations | Alphabetical: `(Abbott, 1991; Barakat et al., 1995)` |
| Reference format | `Lastname, I. N. and Lastname2, I. N.: Title, HESS, Vol, pages, https://doi.org/xx, Year.` |
| Note | Use "and" (not "&") between authors in reference list; colons after author block |
| Fig caption | Below figure, prefixed `Figure N.` |
| Table caption | Above table, prefixed `Table N.` |
| Code availability | Required section if code is used |
| Data availability | Required section |

**Semantic Scholar query to verify format:** `venue:"Hydrology and Earth System Sciences" year:2023`

---

## 5. Groundwater — NGWA / Wiley

| Field | Value |
|---|---|
| Publisher | NGWA / Wiley |
| Abstract | ≤ 250 words |
| Body word limit | ~ 6,000 words (concise format) |
| Headings | NOT numbered — bold plain headings |
| In-text citation | `(Author Year)` or `Author (Year)` — similar to HJ Harvard |
| Multiple citations | Alphabetical: `(Abbott 1991; Barakat et al. 1995)` |
| Reference format | `Lastname IN, Lastname2 IN. Year. Title. Groundwater Vol(Issue):pages. https://doi.org/xx` |
| Fig caption | Below figure |
| Table caption | Above table |
| Note | Audience is practitioners as well as researchers — keep language accessible |

**Semantic Scholar query to verify format:** `venue:"Groundwater" year:2023`

---

## Using Semantic Scholar to verify citation conventions

For any journal, run this query after loading the profile to sample recent in-text and
reference-list style from actual published papers:

```
tool: mcp__semantic-scholar__search_papers
query: "<journal full name> groundwater hydrology"
fields: paperId,title,authors,year,venue,externalIds
limit: 3
filter: year >= 2022
```

Inspect `venue` to confirm match, then use `get_paper` on a paperId to fetch full metadata
including `externalIds.DOI`. This validates that your citation format matches current
journal practice.

# Document Structure Constraints  
### (For PDF Structure Understanding System)

This document defines the structural and formatting constraints required for PDF documents used in training and evaluating the PDF Structure Understanding System.

These constraints apply to document structure and layout behavior — not to content generation length rules used for LLM prompting.

---

## 1. General Document Requirements

- Documents must be **text-based PDFs** (not scanned images).
- Content must follow a **clear hierarchical structure**.
- Layout should preferably be **single-column academic style**.
- Avoid heavy use of:
  - Large tables
  - Complex mathematical equations
  - Diagram-dominated pages
- Use a limited number of fonts (maximum 3–4 per document).
- Avoid decorative or stylized typography.

Document length is **not restricted** at the system level.  
Documents may be short or long.  

---

## 2. Mandatory Structural Hierarchy

Each document must follow a clear structural order:
Example

1. Title (Main Heading)
2. Abstract
3. Main Sections
4. Conclusion
5. Optional:
   - Future Work
   - Limitations
   - References

### Strict Hierarchy Rule


Rules:

- Headings must appear before their paragraphs.
- Subheadings must appear inside sections.
- Do not introduce headings after explanatory paragraphs.
- Do not break structural order.
- No paragraph should appear without belonging to a logical section (except Abstract).

---

## 3. Heading Constraints

Headings must:

- Be visually distinguishable from paragraphs.
- Appear more prominent (bold, bold-italic, or larger font size).
- Be structurally clear and meaningful.
- Be standalone section indicators.

Heading appearance does not strictly require literal bold, but must be visually stronger than paragraph text.

Headings must NOT:

- Be split across two consecutive lines forming a single logical heading.
- Be single characters (e.g., “n”, “i”).
- Be only numbers (e.g., “1”, “2”).
- Be 1–2 letter fragments.
- Be random short tokens.
- Appear after paragraphs that introduce them.

---

## 4. Subheading Constraints

Subheadings:

- Usually short (1–5 words).
- Typically bold or visually emphasized.
- Must appear inside a section.
- Must not exist without content beneath them.
- Must not duplicate the main heading directly.

Allowed:
- A subheading may relate closely to the section topic but must remain hierarchically nested.

Not allowed:
- Standalone subheadings without section context.
- Fragmented or meaningless subheadings.
- Subheadings visually indistinguishable from paragraphs.

---

## 5. Paragraph Constraints

Paragraphs must:

- Appear under a heading or subheading.
- Be multi-line and naturally written.
- Not be excessively short (single-word or fragment lines).
- Not be excessively long as a single uninterrupted block.
- Maintain reasonable readability length.

Guideline:
Paragraphs under headings should not be too short and not excessively long whenever possible.

Paragraphs must NOT:

- Be randomly bold.
- Visually resemble headings.
- Be isolated structural fragments.

---

## 6. Font & Styling Rules

- Use no more than 3–4 fonts per document.
- Avoid decorative fonts.
- Avoid symbol-heavy content.
- Avoid excessive capitalization.
- Avoid inconsistent styling that destroys visual hierarchy.

Headings should be visually stronger than paragraphs through:
- Bold
- Bold-italic
- Slightly larger font size
- Spacing adjustments

Minor inconsistencies are allowed but hierarchy must remain clear.

---

## 7. Allowed Imperfections (Intentional)

To simulate real-world documents, the following are acceptable:

- Minor inconsistent bold usage.
- Slight variation in heading size.
- One or two less-clear subheadings.
- Slight formatting irregularities.

However:

- Structural hierarchy must remain intact.
- Headings must remain distinguishable.
- Content must remain logically grouped.

---

## 8. Content Domain Scope

Documents should generally belong to domains such as:

- Software systems
- AI-assisted tools
- Web platforms
- Data analysis systems
- Automation tools
- Academic mini-project style systems

Avoid:

- Heavy equation-dominated research papers
- Table-dominated reports
- Slide-style PDFs
- Multi-column journal layouts (unless explicitly testing layout robustness)

---

## 9. Structural Integrity Rules

- No empty heading sections.
- No duplicate stacked headings without content.
- Subheadings must contain content beneath them.
- No random isolated tokens as headings.
- No layout patterns that break reading order.

---

## 10. Purpose

These constraints ensure:

- Reliable heading classification.
- Clean structural parsing.
- Stable structured JSON generation.
- Effective vector chunking.
- High-quality semantic search and retrieval.
- Reduced layout-based extraction errors.

---

This document defines the controlled structural characteristics expected for PDFs used in training and evaluating the PDF Structure Understanding System.

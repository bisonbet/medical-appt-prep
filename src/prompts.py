"""
Prompt templates for the three output types:
  1. Symptom timeline
  2. Questions to ask the doctor
  3. Relevant background information
  4. Combined prep report
"""

from __future__ import annotations


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _format_section(label: str, content: str) -> str:
    content = content.strip()
    if not content:
        return ""
    return f"**{label}:**\n{content}"


def _build_context_block(symptoms: str, notes: str, medications: str) -> str:
    parts = []
    if symptoms.strip():
        parts.append(_format_section("Symptoms", symptoms))
    if notes.strip():
        parts.append(_format_section("Additional Notes", notes))
    if medications.strip():
        parts.append(_format_section("Current Medications", medications))
    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Timeline prompt
# ---------------------------------------------------------------------------

def build_timeline_prompt(symptoms: str, notes: str, medications: str) -> str:
    context = _build_context_block(symptoms, notes, medications)
    return f"""You are helping a patient organize their symptom history before a doctor's appointment.

Write only the patient-facing timeline. Do not write thoughts, planning,
analysis, or comments about this prompt. Do not diagnose.

Create 3-5 concise bullets that:
- Organize symptoms by timing when possible.
- Note patterns, triggers, aggravating factors, and relieving factors.
- Highlight what seems new, ongoing, or changing.
- Do not invent timing. If timing is not provided, say "timing not specified."

--- Patient Information ---
{context}
---

Start directly with the bullets. Do not include a heading, introduction, notes
section, or subsection."""


# ---------------------------------------------------------------------------
# Questions prompt
# ---------------------------------------------------------------------------

def build_questions_prompt(symptoms: str, notes: str, medications: str) -> str:
    context = _build_context_block(symptoms, notes, medications)
    return f"""You are helping a patient prepare thoughtful questions for their upcoming doctor's appointment.

Write only patient-facing questions. Do not write thoughts, planning, analysis,
answers, or comments about this prompt. Do not diagnose.

Create 5 concise questions the patient can ask their doctor. Include questions
about possible causes, tests, medication concerns, urgency, and follow-up. Keep
each question under 20 words.

--- Patient Information ---
{context}
---

Start directly with question 1. Do not include a heading or introduction."""


# ---------------------------------------------------------------------------
# Relevant information prompt
# ---------------------------------------------------------------------------

def build_relevant_info_prompt(symptoms: str, notes: str, medications: str) -> str:
    context = _build_context_block(symptoms, notes, medications)
    return f"""You are helping a patient prepare for a doctor's appointment.

Write only patient-facing background information. Do not write thoughts,
planning, analysis, or comments about this prompt. Do not diagnose or claim
certainty.

Create exactly 4 concise bullets that may help the patient talk with their
doctor. Keep each bullet under 22 words. Mention general red flags and
medication-related considerations only when relevant. Make the last bullet a
short reminder that this is informational only and not a substitute for
professional medical advice.

--- Patient Information ---
{context}
---

Start directly with the bullets. Do not include a heading, introduction,
subsection, or "important considerations" section."""


# ---------------------------------------------------------------------------
# Combined prep report prompt
# ---------------------------------------------------------------------------

def build_prep_report_prompt(symptoms: str, notes: str, medications: str) -> str:
    context = _build_context_block(symptoms, notes, medications)
    return f"""You are helping a patient prepare for a doctor's appointment.

Write only the patient-facing report. Do not write thoughts, planning, analysis,
reasoning notes, a checklist of instructions, or comments about this prompt.
Do not diagnose or claim certainty. Keep the report useful for discussion with
a qualified healthcare professional. Aim for 220-360 words total.

--- Patient Information ---
{context}
---

Use exactly this format and start with TIMELINE:

TIMELINE:
- 3-6 concise bullets organizing symptoms by timing, patterns, triggers, and what is new or ongoing.

QUESTIONS:
1. 5-7 concise questions for the doctor about possible causes, tests, medications, urgency, and follow-up.

RELEVANT_INFO:
- 3-5 concise bullets with helpful background, general red flags when relevant, and medication-related considerations.
- End with one short reminder that this is informational only and not a substitute for professional medical advice."""

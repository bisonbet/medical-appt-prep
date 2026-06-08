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

Based on the information below, create a clear **symptom timeline** that:
- Lists symptoms in chronological order (when they first appeared, how they changed)
- Notes any patterns, triggers, or aggravating/relieving factors
- Highlights when symptoms are new vs. ongoing
- Is concise but complete (use bullet points or a numbered list)
- Does NOT diagnose — only organizes the reported information

--- Patient Information ---
{context}
---

Respond with only the formatted symptom timeline. Start directly with the content."""


# ---------------------------------------------------------------------------
# Questions prompt
# ---------------------------------------------------------------------------

def build_questions_prompt(symptoms: str, notes: str, medications: str) -> str:
    context = _build_context_block(symptoms, notes, medications)
    return f"""You are helping a patient prepare thoughtful questions for their upcoming doctor's appointment.

Based on the information below, generate a list of **specific, high-value questions** the patient should ask their doctor. Include questions about:
- Possible causes or diagnoses for their symptoms
- Any diagnostic tests that might be relevant
- Medication interactions or adjustments
- Lifestyle or management advice
- When to seek urgent care
- Follow-up expectations

Format as a numbered list. Keep each question concise and direct. Do NOT answer the questions — only generate them.

--- Patient Information ---
{context}
---

Respond with only the numbered list of questions. Start directly with question 1."""


# ---------------------------------------------------------------------------
# Relevant information prompt
# ---------------------------------------------------------------------------

def build_relevant_info_prompt(symptoms: str, notes: str, medications: str) -> str:
    context = _build_context_block(symptoms, notes, medications)
    return f"""You are a knowledgeable medical assistant helping a patient understand their situation before seeing a doctor.

Based on the information below, provide **relevant background information** that would help the patient have an informed conversation with their doctor. Include:
- Brief overview of common conditions associated with these symptoms (2-3 most likely)
- Any known interactions between the listed medications and reported symptoms
- General red flags or warning signs worth mentioning
- Relevant lifestyle factors that could be contributing

Use plain language. Keep each section brief (2-4 sentences). Do NOT diagnose. End with a clear reminder that this is informational only and not a substitute for professional medical advice.

--- Patient Information ---
{context}
---

Respond with the formatted background information. Use clear section headers."""


# ---------------------------------------------------------------------------
# Combined prep report prompt
# ---------------------------------------------------------------------------

def build_prep_report_prompt(symptoms: str, notes: str, medications: str) -> str:
    context = _build_context_block(symptoms, notes, medications)
    return f"""You are helping a patient prepare for a doctor's appointment.

Use the patient-provided information below to produce three separate sections.
Do not diagnose. Do not claim certainty. Keep the content useful for discussion
with a qualified healthcare professional.

--- Patient Information ---
{context}
---

Return exactly these section markers in this order:

TIMELINE:
- Organize symptoms chronologically when possible.
- Note patterns, triggers, aggravating factors, relieving factors, and new vs ongoing symptoms.
- Use concise bullets.

QUESTIONS:
- Generate specific, high-value questions the patient can ask their doctor.
- Include questions about possible causes, tests, medication concerns, urgency, and follow-up.
- Use a numbered list.

RELEVANT_INFO:
- Provide brief background information that may help the patient have an informed conversation.
- Mention general red flags and medication-related considerations when relevant.
- End with a reminder that this is informational only and not a substitute for professional medical advice.

Start directly with TIMELINE:"""

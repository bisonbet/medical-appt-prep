import { Client } from "https://cdn.jsdelivr.net/npm/@gradio/client/dist/index.min.js";

const storageKey = "medical-appt-prep-theme";
const validThemes = new Set(["system", "light", "dark"]);
const fields = {
  symptoms: document.querySelector("#symptoms"),
  notes: document.querySelector("#notes"),
  medicationName: document.querySelector("#medication-name"),
  medicationInstructions: document.querySelector("#medication-instructions"),
  medications: document.querySelector("#medications"),
  suggestions: document.querySelector("#medication-suggestions"),
  form: document.querySelector("#prep-form"),
  addMedication: document.querySelector("#add-medication"),
  demo: document.querySelector("#demo-button"),
  clear: document.querySelector("#clear-button"),
  generate: document.querySelector("#generate-button"),
  status: document.querySelector("#status-message"),
  exportStatus: document.querySelector("#export-status"),
  timeline: document.querySelector("#timeline-output"),
  questions: document.querySelector("#questions-output"),
  relevant: document.querySelector("#relevant-output"),
  exportButtons: document.querySelectorAll(".export-action"),
  emailReport: document.querySelector("#email-report"),
  pdfReport: document.querySelector("#pdf-report"),
  printReport: document.querySelector("#print-report"),
  copyReport: document.querySelector("#copy-report"),
  portalCopy: document.querySelector("#portal-copy"),
  downloadReport: document.querySelector("#download-report"),
};

let gradioClientPromise;
let medicationTimer;
let medicationAbortController;
let reportReady = false;
let lastReport = {
  timeline: "",
  questions: "",
  relevant: "",
};
let lastDemoIndex = -1;

const demoCases = [
  {
    label: "headache visit",
    symptoms: "Dull headache behind both eyes for about 10 days. It is usually worse late afternoon after computer work and sometimes comes with light sensitivity. Ibuprofen helps a little but it keeps coming back.",
    notes: "Vision feels a little blurry by the end of the workday. Sleeping 5-6 hours recently. No recent head injury. Wants to ask whether this could relate to eyestrain, stress, blood pressure, or medication changes.",
    medications: "Lisinopril - 10 mg once daily in the morning\nSertraline - 50 mg once daily\nIbuprofen - 400 mg as needed, used 3 times this week",
  },
  {
    label: "knee pain visit",
    symptoms: "Right knee pain for 3 weeks after increasing weekend pickleball. Pain is on the inside of the knee, worse going downstairs, and there is mild swelling after activity. Rest and ice help.",
    notes: "No fall or major twist. Can walk but avoids longer walks. Wants to stay active and understand what exam findings, imaging, or physical therapy options might matter.",
    medications: "Atorvastatin - 20 mg once nightly\nMetformin ER - 500 mg with dinner\nAcetaminophen - 500 mg as needed for knee pain",
  },
  {
    label: "stomach symptoms visit",
    symptoms: "Burning upper stomach discomfort most evenings for 2 months. Worse after tomato sauce, coffee, and late meals. Sometimes wakes up with sour taste in mouth.",
    notes: "Started a new job schedule and eats dinner later. No known food allergies. Wants help describing patterns and asking what lifestyle changes, tests, or medication review might be appropriate.",
    medications: "Levothyroxine - 75 mcg every morning before food\nOmeprazole - 20 mg as needed, used a few times this month\nCalcium carbonate antacid - as needed after meals",
  },
  {
    label: "breathing follow-up",
    symptoms: "Intermittent cough and chest tightness for 4 weeks, especially after climbing stairs or being around cold air. Albuterol helps within a few minutes. Symptoms are more noticeable at night.",
    notes: "Seasonal allergies have been worse this month. No fever now. Wants to ask whether inhaler use, allergy control, or pulmonary testing should be reviewed.",
    medications: "Albuterol inhaler - 2 puffs as needed\nFluticasone nasal spray - 1 spray each nostril daily\nCetirizine - 10 mg once daily",
  },
  {
    label: "fatigue visit",
    symptoms: "Low energy for about 6 weeks. Feels cold more often, has dry skin, and is having trouble concentrating in the afternoon. Symptoms are gradual rather than sudden.",
    notes: "Work stress is higher, but sleep schedule is fairly consistent. Had thyroid medication adjusted last year. Wants a concise way to discuss fatigue, mood, sleep, and possible lab questions.",
    medications: "Levothyroxine - 88 mcg every morning\nBupropion XL - 150 mg once daily\nVitamin D3 - 2000 IU daily",
  },
];

function currentTheme() {
  const stored = localStorage.getItem(storageKey);
  return validThemes.has(stored) ? stored : "system";
}

function applyTheme(theme) {
  const nextTheme = validThemes.has(theme) ? theme : "system";
  document.documentElement.dataset.theme = nextTheme;
  if (nextTheme === "system") {
    localStorage.removeItem(storageKey);
  } else {
    localStorage.setItem(storageKey, nextTheme);
  }

  document.querySelectorAll("[data-theme-option]").forEach((button) => {
    const selected = button.dataset.themeOption === nextTheme;
    button.classList.toggle("selected", selected);
    button.setAttribute("aria-pressed", String(selected));
  });
}

function connectClient() {
  if (!gradioClientPromise) {
    gradioClientPromise = Client.connect(window.location.origin);
  }
  return gradioClientPromise;
}

function escapeHtml(value) {
  return String(value || "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function inlineMarkdown(value) {
  return escapeHtml(value).replace(/\*\*(.*?)\*\*/g, "<strong>$1</strong>");
}

function renderMarkdown(value) {
  const text = String(value || "").trim();
  if (!text) {
    return '<p class="empty">Nothing to show yet.</p>';
  }

  const lines = text.split(/\r?\n/).map((line) => line.trim()).filter(Boolean);
  const html = [];
  let listOpen = false;
  let ordered = false;

  for (const line of lines) {
    const bullet = line.match(/^[-*]\s+(.*)$/);
    const numbered = line.match(/^\d+[.)]\s+(.*)$/);
    if (bullet || numbered) {
      const nextOrdered = Boolean(numbered);
      if (!listOpen || ordered !== nextOrdered) {
        if (listOpen) {
          html.push(ordered ? "</ol>" : "</ul>");
        }
        ordered = nextOrdered;
        listOpen = true;
        html.push(ordered ? "<ol>" : "<ul>");
      }
      html.push(`<li>${inlineMarkdown((bullet || numbered)[1])}</li>`);
      continue;
    }

    if (listOpen) {
      html.push(ordered ? "</ol>" : "</ul>");
      listOpen = false;
    }
    html.push(`<p>${inlineMarkdown(line)}</p>`);
  }

  if (listOpen) {
    html.push(ordered ? "</ol>" : "</ul>");
  }
  return html.join("");
}

function plainMarkdown(value) {
  return String(value || "")
    .replace(/\*\*(.*?)\*\*/g, "$1")
    .replace(/<[^>]*>/g, "")
    .trim();
}

function setStatus(message) {
  if (!fields.status) {
    return;
  }
  fields.status.textContent = message;
}

function setExportStatus(message) {
  if (!fields.exportStatus) {
    return;
  }
  fields.exportStatus.textContent = message;
}

function setExportReady(isReady) {
  reportReady = isReady;
  fields.exportButtons?.forEach((button) => {
    button.disabled = !isReady;
  });
  setExportStatus(isReady ? "Ready to email, print, copy, or save the full report." : "Generate a report to enable full-report exports.");
}

function setLoading(isLoading) {
  if (!fields.generate) {
    return;
  }
  fields.generate.disabled = isLoading;
  fields.generate.classList.toggle("loading", isLoading);
  fields.generate.querySelector(".button-label").textContent = isLoading
    ? "Organizing..."
    : "Generate Prep Report";
}

function selectTab(tabName) {
  document.querySelectorAll("[data-tab]").forEach((button) => {
    const selected = button.dataset.tab === tabName;
    button.classList.toggle("selected", selected);
    button.setAttribute("aria-selected", String(selected));
  });
  document.querySelectorAll("[data-panel]").forEach((panel) => {
    panel.classList.toggle("selected", panel.dataset.panel === tabName);
  });
}

function reportSections() {
  return [
    ["Timeline", lastReport.timeline],
    ["Questions", lastReport.questions],
    ["Relevant Info", lastReport.relevant],
  ];
}

function hasGeneratedReport() {
  return reportReady && reportSections().some(([, value]) => plainMarkdown(value));
}

function resetReportState(statusMessage) {
  fields.timeline.innerHTML = renderMarkdown("Your timeline will appear here.");
  fields.questions.innerHTML = renderMarkdown("Questions for your visit will appear here.");
  fields.relevant.innerHTML = renderMarkdown("Relevant background information will appear here.");
  lastReport = { timeline: "", questions: "", relevant: "" };
  setExportReady(false);
  setStatus(statusMessage || "Your report will appear after you generate it.");
  selectTab("timeline");
}

function randomDemoIndex() {
  if (demoCases.length < 2) {
    return 0;
  }
  let nextIndex = Math.floor(Math.random() * demoCases.length);
  if (nextIndex === lastDemoIndex) {
    nextIndex = (nextIndex + 1) % demoCases.length;
  }
  lastDemoIndex = nextIndex;
  return nextIndex;
}

function fillDemoCase() {
  const demo = demoCases[randomDemoIndex()];
  fields.symptoms.value = demo.symptoms;
  fields.notes.value = demo.notes;
  fields.medications.value = demo.medications;
  fields.medicationName.value = "";
  fields.medicationInstructions.value = "";
  hideSuggestions();
  resetReportState(`Sample test data loaded: ${demo.label}. Review or edit it, then generate the prep report.`);
  fields.symptoms.focus();
}

function currentPrepText({ portal = false } = {}) {
  const preparedAt = new Date().toLocaleString([], {
    year: "numeric",
    month: "short",
    day: "numeric",
    hour: "numeric",
    minute: "2-digit",
  });
  const lines = [
    "Medical Appointment Prep",
    `Prepared: ${preparedAt}`,
    "",
  ];

  if (!portal) {
    lines.push(
      "This is informational only and not a substitute for professional medical advice.",
      "",
    );
  }

  if (fields.symptoms.value.trim()) {
    lines.push("Symptoms / Concerns", fields.symptoms.value.trim(), "");
  }
  if (fields.notes.value.trim()) {
    lines.push("Additional Notes", fields.notes.value.trim(), "");
  }
  if (fields.medications.value.trim()) {
    lines.push("Medications", fields.medications.value.trim(), "");
  }

  for (const [heading, value] of reportSections()) {
    const body = plainMarkdown(value);
    if (body) {
      lines.push(heading, body, "");
    }
  }

  if (portal) {
    lines.push("Note: This was generated as appointment-prep text for clinician discussion.");
  }

  return lines.join("\n").replace(/\n{3,}/g, "\n\n").trim();
}

function reportHtml({ autoPrint = false } = {}) {
  const sections = reportSections()
    .map(([heading, value]) => `
      <section class="print-section">
        <h2>${escapeHtml(heading)}</h2>
        <div>${renderMarkdown(value)}</div>
      </section>
    `)
    .join("");
  const preparedAt = escapeHtml(new Date().toLocaleString([], {
    year: "numeric",
    month: "long",
    day: "numeric",
    hour: "numeric",
    minute: "2-digit",
  }));
  const medications = fields.medications.value.trim()
    ? `<section class="print-section print-context"><h2>Medications</h2><pre>${escapeHtml(fields.medications.value.trim())}</pre></section>`
    : "";
  const notes = fields.notes.value.trim()
    ? `<section class="print-section print-context"><h2>Additional Notes</h2><pre>${escapeHtml(fields.notes.value.trim())}</pre></section>`
    : "";
  const symptoms = fields.symptoms.value.trim()
    ? `<section class="print-section print-context"><h2>Symptoms / Concerns</h2><pre>${escapeHtml(fields.symptoms.value.trim())}</pre></section>`
    : "";

  const printScript = autoPrint
    ? `<script>
      (() => {
        function printWhenReady() {
          window.setTimeout(() => {
            window.focus();
            window.print();
          }, 150);
        }
        if (document.readyState === "complete") {
          printWhenReady();
        } else {
          window.addEventListener("load", printWhenReady, { once: true });
        }
      })();
    </script>`
    : "";

  return `<!doctype html>
<html>
  <head>
    <meta charset="utf-8" />
    <title>Medical Appointment Prep</title>
    <style>
      :root { color-scheme: light; }
      body {
        color: #24211d;
        font-family: ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
        line-height: 1.48;
        margin: 0;
        padding: 32px;
      }
      .print-page {
        border: 1px solid #ded5c8;
        border-radius: 8px;
        margin: 0 auto;
        max-width: 840px;
        overflow: hidden;
      }
      header {
        background: #f7f3ec;
        border-bottom: 1px solid #ded5c8;
        padding: 28px 30px;
      }
      .kicker {
        color: #00897b;
        font-size: 12px;
        font-weight: 800;
        margin: 0 0 8px;
        text-transform: uppercase;
      }
      h1 {
        font-size: 34px;
        line-height: 1.1;
        margin: 0 0 8px;
      }
      .prepared {
        color: #6d6258;
        margin: 0;
      }
      main {
        padding: 8px 30px 30px;
      }
      .print-section {
        border-bottom: 1px solid #eee6dc;
        padding: 20px 0;
      }
      .print-section:last-child {
        border-bottom: 0;
      }
      h2 {
        font-size: 18px;
        margin: 0 0 10px;
      }
      p {
        margin: 0 0 10px;
      }
      ul, ol {
        margin: 0;
        padding-left: 22px;
      }
      li {
        margin-bottom: 8px;
      }
      pre {
        font: inherit;
        white-space: pre-wrap;
        margin: 0;
      }
      .disclaimer {
        background: #fff8e8;
        border: 1px solid #efd8a8;
        border-radius: 8px;
        color: #5f4b19;
        margin-top: 20px;
        padding: 12px 14px;
      }
      @page {
        margin: 0.55in;
      }
      @media print {
        body { padding: 0; }
        .print-page { border: 0; max-width: none; }
      }
    </style>
  </head>
  <body>
    <article class="print-page">
      <header>
        <p class="kicker">Appointment prep</p>
        <h1>Medical Appointment Prep</h1>
        <p class="prepared">Prepared ${preparedAt}</p>
      </header>
      <main>
        ${symptoms}
        ${notes}
        ${medications}
        ${sections}
        <p class="disclaimer">This is informational only and not a substitute for professional medical advice.</p>
      </main>
    </article>
    ${printScript}
  </body>
</html>`;
}

function ensureReportReady(actionName) {
  if (hasGeneratedReport()) {
    return true;
  }
  setExportStatus(`Generate a report before using ${actionName}.`);
  return false;
}

async function copyText(text, successMessage) {
  try {
    await navigator.clipboard.writeText(text);
  } catch (_error) {
    const textarea = document.createElement("textarea");
    textarea.value = text;
    textarea.setAttribute("readonly", "");
    textarea.style.position = "fixed";
    textarea.style.left = "-9999px";
    document.body.appendChild(textarea);
    textarea.select();
    document.execCommand("copy");
    textarea.remove();
  }
  setExportStatus(successMessage);
}

function openPrintReport({ pdf = false } = {}) {
  if (!ensureReportReady(pdf ? "Export PDF" : "Print")) {
    return;
  }
  const frame = document.createElement("iframe");
  frame.setAttribute("aria-hidden", "true");
  frame.style.border = "0";
  frame.style.height = "0";
  frame.style.position = "fixed";
  frame.style.right = "0";
  frame.style.bottom = "0";
  frame.style.width = "0";

  const cleanup = () => window.setTimeout(() => frame.remove(), 1000);
  const triggerPrint = () => {
    const printFrame = frame.contentWindow;
    if (!printFrame) {
      cleanup();
      return;
    }
    printFrame.addEventListener("afterprint", cleanup, { once: true });
    window.setTimeout(cleanup, 60000);
    printFrame.focus();
    printFrame.print();
  };

  frame.addEventListener("load", () => window.setTimeout(triggerPrint, 150), { once: true });
  frame.srcdoc = reportHtml();
  document.body.appendChild(frame);
  setExportStatus(pdf ? "Print dialog opening. Choose Save as PDF." : "Print dialog opening.");
}

function emailReport() {
  if (!ensureReportReady("Email")) {
    return;
  }
  const subject = "Medical appointment prep";
  const body = `${currentPrepText()}\n\nReview before sending. Email may include health details.`;
  window.location.href = `mailto:?subject=${encodeURIComponent(subject)}&body=${encodeURIComponent(body)}`;
  setExportStatus("Email draft opened. Review it before sending.");
}

function downloadReport() {
  if (!ensureReportReady("Download Text")) {
    return;
  }
  const blob = new Blob([currentPrepText()], { type: "text/plain;charset=utf-8" });
  const link = document.createElement("a");
  const date = new Date().toISOString().slice(0, 10);
  link.href = URL.createObjectURL(blob);
  link.download = `medical-appointment-prep-${date}.txt`;
  document.body.appendChild(link);
  link.click();
  link.remove();
  URL.revokeObjectURL(link.href);
  setExportStatus("Text file downloaded.");
}

function addMedication() {
  const name = fields.medicationName.value.trim();
  const instructions = fields.medicationInstructions.value.trim();
  if (!name && !instructions) {
    fields.medicationName.focus();
    return;
  }

  const line = name && instructions ? `${name} - ${instructions}` : name || instructions;
  fields.medications.value = fields.medications.value.trim()
    ? `${fields.medications.value.trim()}\n${line}`
    : line;
  fields.medicationName.value = "";
  fields.medicationInstructions.value = "";
  hideSuggestions();
  fields.medicationName.focus();
}

function hideSuggestions() {
  fields.suggestions.hidden = true;
  fields.suggestions.innerHTML = "";
}

async function fetchMedicationSuggestions(query) {
  if (medicationAbortController) {
    medicationAbortController.abort();
  }
  medicationAbortController = new AbortController();
  const response = await fetch(`/api/medications?q=${encodeURIComponent(query)}`, {
    signal: medicationAbortController.signal,
  });
  if (!response.ok) {
    return [];
  }
  const data = await response.json();
  return Array.isArray(data.choices) ? data.choices : [];
}

async function updateMedicationSuggestions() {
  const query = fields.medicationName.value.trim();
  if (!query) {
    hideSuggestions();
    return;
  }

  try {
    const choices = await fetchMedicationSuggestions(query);
    if (!choices.length) {
      hideSuggestions();
      return;
    }
    fields.suggestions.innerHTML = choices
      .map((choice) => `<button type="button" role="option">${escapeHtml(choice)}</button>`)
      .join("");
    fields.suggestions.hidden = false;
  } catch (error) {
    if (error.name !== "AbortError") {
      hideSuggestions();
    }
  }
}

async function generateReport(event) {
  event.preventDefault();
  if (!fields.symptoms.value.trim()) {
    setStatus("Add symptoms or concerns before generating your report.");
    fields.symptoms.focus();
    return;
  }

  setLoading(true);
  setExportReady(false);
  setStatus("Organizing your notes into a visit-ready report...");
  fields.timeline.innerHTML = renderMarkdown("Working on the timeline...");
  fields.questions.innerHTML = renderMarkdown("Drafting questions...");
  fields.relevant.innerHTML = renderMarkdown("Collecting relevant background notes...");
  selectTab("timeline");

  try {
    const client = await connectClient();
    const result = await client.predict("/generate", {
      symptoms: fields.symptoms.value,
      notes: fields.notes.value,
      medications: fields.medications.value,
    });
    const data = Array.isArray(result.data?.[0]) ? result.data[0] : result.data;
    const payload = Array.isArray(data) && data.length === 1 && typeof data[0] === "object"
      ? data[0]
      : data;
    const [timeline, questions, relevant] = Array.isArray(payload)
      ? payload
      : [payload?.timeline, payload?.questions, payload?.relevant];
    lastReport = {
      timeline: timeline || "",
      questions: questions || "",
      relevant: relevant || "",
    };
    fields.timeline.innerHTML = renderMarkdown(timeline);
    fields.questions.innerHTML = renderMarkdown(questions);
    fields.relevant.innerHTML = renderMarkdown(relevant);
    setExportReady(true);
    setStatus("Your report is ready. Review each tab before your visit.");
  } catch (error) {
    setStatus("The report could not be generated. Check the backend and try again.");
    fields.timeline.innerHTML = renderMarkdown(`Generation failed: ${error.message || error}`);
    fields.questions.innerHTML = renderMarkdown("");
    fields.relevant.innerHTML = renderMarkdown("");
    setExportReady(false);
  } finally {
    setLoading(false);
  }
}

document.querySelectorAll("[data-theme-option]").forEach((button) => {
  button.addEventListener("click", () => applyTheme(button.dataset.themeOption));
});

document.querySelectorAll("[data-tab]").forEach((button) => {
  button.addEventListener("click", () => selectTab(button.dataset.tab));
});

applyTheme(currentTheme());

if (fields.form) {
  fields.form.addEventListener("submit", generateReport);
  fields.addMedication.addEventListener("click", addMedication);
  fields.medicationInstructions.addEventListener("keydown", (event) => {
    if (event.key === "Enter") {
      event.preventDefault();
      addMedication();
    }
  });
  fields.clear.addEventListener("click", () => {
    fields.form.reset();
    hideSuggestions();
    resetReportState("Your report will appear after you generate it.");
  });
  fields.demo.addEventListener("click", fillDemoCase);
  fields.emailReport.addEventListener("click", emailReport);
  fields.pdfReport.addEventListener("click", () => openPrintReport({ pdf: true }));
  fields.printReport.addEventListener("click", () => openPrintReport());
  fields.copyReport.addEventListener("click", () => {
    if (ensureReportReady("Copy All")) {
      copyText(currentPrepText(), "Full prep copied.");
    }
  });
  fields.portalCopy.addEventListener("click", () => {
    if (ensureReportReady("Portal Copy")) {
      copyText(currentPrepText({ portal: true }), "Portal-friendly copy copied.");
    }
  });
  fields.downloadReport.addEventListener("click", downloadReport);
  fields.medicationName.addEventListener("input", () => {
    clearTimeout(medicationTimer);
    medicationTimer = setTimeout(updateMedicationSuggestions, 160);
  });
  fields.medicationName.addEventListener("blur", () => {
    setTimeout(hideSuggestions, 140);
  });
  fields.suggestions.addEventListener("mousedown", (event) => {
    const option = event.target.closest("button");
    if (!option) {
      return;
    }
    fields.medicationName.value = option.textContent;
    hideSuggestions();
  });
  setExportReady(false);
}

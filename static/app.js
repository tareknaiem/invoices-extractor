/* app.js — frontend logic for the Invoice Extractor dashboard.
 *
 * Talks to the FastAPI backend exposed by server.py:
 *   GET    /api/config          load saved config
 *   POST   /api/config          save config
 *   DELETE /api/config          clear config
 *   POST   /api/extract         upload files, get results + download_id
 *   GET    /api/download/{id}   fetch the generated workbook
 */

"use strict";

const $ = (sel) => document.querySelector(sel);

// ---------------------------------------------------------------------------
// Predefined API providers. Selecting one auto-fills the Base URL and fills
// the Model dropdown. "__custom__" switches Base URL + Model to free-text.
// ---------------------------------------------------------------------------
const PROVIDERS = {
  "Google Gemini": {
    url: "https://generativelanguage.googleapis.com/v1beta/openai/",
    models: [
      "gemini-1.5-flash",
      "gemini-1.5-flash-8b",
      "gemini-1.5-pro",
      "gemini-2.0-flash",
      "gemini-2.0-flash-lite-preview-02-05",
      "gemini-2.0-pro-exp-02-05",
    ],
  },
  OpenRouter: {
    url: "https://openrouter.ai/api/v1",
    models: [
      "google/gemini-2.0-flash-lite-preview-02-05:free",
      "meta-llama/llama-3.2-11b-vision-instruct:free",
    ],
  },
  Groq: {
    url: "https://api.groq.com/openai/v1",
    models: ["llama-3.2-90b-vision-preview", "llama-3.2-11b-vision-preview"],
  },
  DeepSeek: {
    url: "https://api.deepseek.com",
    models: ["deepseek-chat"],
  },
  Qwen: {
    url: "https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
    models: ["qwen-vl-plus", "qwen-vl-max"],
  },
  "Together AI": {
    url: "https://api.together.xyz/v1",
    models: [
      "meta-llama/Llama-3.2-11B-Vision-Instruct-Turbo",
      "meta-llama/Llama-3.2-90B-Vision-Instruct-Turbo",
    ],
  },
  "Mistral AI": {
    url: "https://api.mistral.ai/v1",
    models: ["pixtral-12b-2409", "pixtral-large-latest"],
  },
};
const CUSTOM = "__custom__";

// ---------------------------------------------------------------------------
// Element references (IDs must match index.html exactly)
// ---------------------------------------------------------------------------
const configForm = $("#config-form");
const providerInput = $("#provider");
const baseUrlInput = $("#base_url");
const modelInput = $("#model");
const modelOptions = $("#model-options");
const fetchModelsBtn = $("#fetchModelsBtn");
const apiKeyInput = $("#api_key");
const toggleKeyBtn = $("#toggle-key");
const keyStatus = $("#key-status");
const clearBtn = $("#clear-btn");
const configMsg = $("#config-msg");

const dropzone = $("#dropzone");
const fileInput = $("#file-input");
const fileList = $("#file-list");
const processBtn = $("#process-btn");
const clearFilesBtn = $("#clear-files-btn");

const progressWrap = $("#progress-wrap");
const progressBar = $("#progress-bar");
const progressLabel = $("#progress-label");

const alerts = $("#alerts");
const results = $("#results");
const resultsTable = $("#results-table");
const downloadBtn = $("#download-btn");

// Files chosen for the current batch (array of File).
let selectedFiles = [];

// ---------------------------------------------------------------------------
// Provider dropdown logic
// ---------------------------------------------------------------------------
function isPredefined(name) {
  return Object.prototype.hasOwnProperty.call(PROVIDERS, name);
}

// The model is always read from the free-text combo-box input.
function currentModel() {
  return modelInput.value.trim();
}

// The effective provider label sent to the server.
function currentProvider() {
  const v = providerInput.value;
  return v === CUSTOM ? "Custom" : v;
}

// Update Base URL + the model suggestion list for the selected provider.
// The Model field stays a free-text combo-box; a datalist supplies suggestions.
function applyProvider(name) {
  // Reset the suggestion list for the (possibly new) provider.
  modelOptions.innerHTML = "";
  if (isPredefined(name)) {
    const p = PROVIDERS[name];
    // Base URL: auto-filled + locked for predefined providers.
    baseUrlInput.readOnly = true;
    baseUrlInput.value = p.url;
    baseUrlInput.classList.add("bg-slate-100", "text-slate-600", "cursor-not-allowed");
    baseUrlInput.classList.remove("bg-white");
    // Populate the model suggestion datalist.
    p.models.forEach((m) => {
      const opt = document.createElement("option");
      opt.value = m;
      modelOptions.appendChild(opt);
    });
  } else {
    // Custom (or nothing selected): editable Base URL + free-text Model.
    baseUrlInput.readOnly = false;
    baseUrlInput.classList.remove("bg-slate-100", "text-slate-600", "cursor-not-allowed");
    baseUrlInput.classList.add("bg-white");
  }
}

providerInput.addEventListener("change", () => {
  applyProvider(providerInput.value);
  // When entering Custom, start from blank fields for fresh entry.
  if (providerInput.value === CUSTOM) {
    baseUrlInput.value = "";
    modelInput.value = "";
  }
});
// ---------------------------------------------------------------------------
// Config: load, save, clear
// ---------------------------------------------------------------------------
async function loadConfig() {
  try {
    const res = await fetch("/api/config");
    if (!res.ok) throw new Error("Failed to load config");
    const cfg = await res.json();
    const provider = cfg.provider || "";
    if (provider && isPredefined(provider)) {
      providerInput.value = provider;
      applyProvider(provider);
      // Restore the saved model into the free-text field.
      modelInput.value = cfg.model || "";
    } else {
      // Custom provider (or an unrecognised saved name) -> free-text fields.
      providerInput.value = provider ? CUSTOM : "";
      applyProvider(CUSTOM);
      baseUrlInput.value = cfg.base_url || "";
      modelInput.value = cfg.model || "";
    }
    if (cfg.has_key) {
      keyStatus.textContent = "✓ A key is saved on the server.";
      keyStatus.classList.remove("hidden");
      apiKeyInput.placeholder = "Leave blank to keep the saved key";
    }
  } catch (err) {
    showConfigMsg("Could not load saved configuration.", "error");
  }
}

function showConfigMsg(text, type) {
  configMsg.textContent = text;
  configMsg.className =
    "mt-3 text-sm rounded-lg px-3 py-2 " +
    (type === "error"
      ? "bg-red-50 text-red-700 border border-red-200"
      : "bg-emerald-50 text-emerald-700 border border-emerald-200");
  configMsg.classList.remove("hidden");
}

configForm.addEventListener("submit", async (e) => {
  e.preventDefault();
  const data = new FormData();
  data.append("provider", currentProvider());
  data.append("base_url", baseUrlInput.value.trim());
  data.append("model", currentModel());
  data.append("api_key", apiKeyInput.value.trim());

  try {
    const res = await fetch("/api/config", { method: "POST", body: data });
    const payload = await res.json().catch(() => ({}));
    if (!res.ok) {
      showConfigMsg(payload.detail || "Could not save configuration.", "error");
      return;
    }
    showConfigMsg(
      `Saved to .env as ${payload.key_name || "AI_API_KEY"}.`,
      "success"
    );
    keyStatus.textContent = "✓ A key is saved on the server.";
    keyStatus.classList.remove("hidden");
    apiKeyInput.value = "";
    apiKeyInput.placeholder = "Leave blank to keep the saved key";
  } catch (err) {
    showConfigMsg("Network error while saving.", "error");
  }
});

clearBtn.addEventListener("click", async () => {
  try {
    const res = await fetch("/api/config", { method: "DELETE" });
    if (!res.ok) throw new Error("clear failed");
    providerInput.value = "";
    applyProvider(CUSTOM);
    baseUrlInput.value = "";
    modelOptions.innerHTML = "";
    modelInput.value = "";
    apiKeyInput.value = "";
    keyStatus.classList.add("hidden");
    showConfigMsg("Configuration cleared from .env.", "success");
  } catch (err) {
    showConfigMsg("Could not clear configuration.", "error");
  }
});

toggleKeyBtn.addEventListener("click", () => {
  const isPwd = apiKeyInput.type === "password";
  apiKeyInput.type = isPwd ? "text" : "password";
  toggleKeyBtn.textContent = isPwd ? "🙈" : "👁";
});

// ---------------------------------------------------------------------------
// File selection (drag & drop + browse)
// ---------------------------------------------------------------------------
const ALLOWED_EXT = ["pdf", "jpg", "jpeg", "png"];

function extOf(name) {
  const i = name.lastIndexOf(".");
  return i === -1 ? "" : name.slice(i + 1).toLowerCase();
}

function addFiles(fileArr) {
  for (const file of fileArr) {
    if (!ALLOWED_EXT.includes(extOf(file.name))) continue;
    const dup = selectedFiles.some(
      (f) => f.name === file.name && f.size === file.size
    );
    if (!dup) selectedFiles.push(file);
  }
  renderFileList();
}

function renderFileList() {
  fileList.innerHTML = "";
  selectedFiles.forEach((file, idx) => {
    const li = document.createElement("li");
    li.className =
      "flex items-center justify-between rounded-lg border border-slate-200 bg-slate-50 px-3 py-2 text-sm";
    li.innerHTML = `
      <span class="flex items-center gap-2 truncate">
        <span class="text-slate-400">📎</span>
        <span class="truncate">${escapeHtml(file.name)}</span>
        <span class="text-xs text-slate-400">(${formatBytes(file.size)})</span>
      </span>
      <button data-idx="${idx}" class="remove-file text-slate-400 hover:text-red-600 text-xs">✕</button>`;
    fileList.appendChild(li);
  });

  processBtn.disabled = selectedFiles.length === 0;
  clearFilesBtn.classList.toggle("hidden", selectedFiles.length === 0);
}

fileList.addEventListener("click", (e) => {
  const btn = e.target.closest(".remove-file");
  if (!btn) return;
  selectedFiles.splice(Number(btn.dataset.idx), 1);
  renderFileList();
});

// "browse": clicking the dropzone opens the hidden file input.
dropzone.addEventListener("click", () => fileInput.click());

// Selecting files via the file dialog.
fileInput.addEventListener("change", () => {
  addFiles([...fileInput.files]);
  fileInput.value = "";
});

// Drag & drop visual feedback.
["dragenter", "dragover"].forEach((evt) =>
  dropzone.addEventListener(evt, (e) => {
    e.preventDefault();
    dropzone.classList.add("border-brand-500", "bg-brand-50");
  })
);
["dragleave", "drop"].forEach((evt) =>
  dropzone.addEventListener(evt, (e) => {
    e.preventDefault();
    dropzone.classList.remove("border-brand-500", "bg-brand-50");
  })
);

// Dropping files onto the dropzone.
dropzone.addEventListener("drop", (e) => {
  if (e.dataTransfer && e.dataTransfer.files && e.dataTransfer.files.length) {
    addFiles([...e.dataTransfer.files]);
  }
});

// Clear the current selection.
clearFilesBtn.addEventListener("click", () => {
  selectedFiles = [];
  renderFileList();
  resetResults();
});

// ---------------------------------------------------------------------------
// Extraction
// ---------------------------------------------------------------------------
processBtn.addEventListener("click", async () => {
  if (!selectedFiles.length) return;

  resetResults();
  setProcessing(true);

  const data = new FormData();
  // Send the current form values so a user can run without re-saving the key;
  // blank fields fall back to the server's saved config.
  data.append("provider", currentProvider());
  data.append("base_url", baseUrlInput.value.trim());
  data.append("model", currentModel());
  data.append("api_key", apiKeyInput.value.trim());
  selectedFiles.forEach((f) => data.append("files", f));

  // Indeterminate progress animation while the request is in flight.
  let pct = 5;
  progressLabel.textContent = `Processing ${selectedFiles.length} invoice(s)…`;
  const ticker = setInterval(() => {
    pct = Math.min(pct + 3, 90);
    progressBar.style.width = pct + "%";
  }, 400);

  try {
    const res = await fetch("/api/extract", { method: "POST", body: data });
    const payload = await res.json().catch(() => ({}));
    if (!res.ok) {
      showAlert(payload.detail || "Extraction failed.", "error");
      return;
    }
    progressBar.style.width = "100%";
    renderResults(payload);
  } catch (err) {
    showAlert("Network error during extraction.", "error");
  } finally {
    clearInterval(ticker);
    setProcessing(false);
  }
});

function setProcessing(on) {
  processBtn.disabled = on;
  processBtn.innerHTML = on
    ? '<span class="spinner"></span> Processing…'
    : "🚀 Process Invoices";
  progressWrap.classList.toggle("hidden", !on);
  if (on) progressBar.style.width = "5%";
}

function resetResults() {
  alerts.innerHTML = "";
  results.classList.add("hidden");
  resultsTable.querySelector("thead").innerHTML = "";
  resultsTable.querySelector("tbody").innerHTML = "";
  downloadBtn.classList.add("hidden");
  downloadBtn.removeAttribute("href");
}

// ---------------------------------------------------------------------------
// Rendering
// ---------------------------------------------------------------------------
function showAlert(text, type) {
  const div = document.createElement("div");
  const styles = {
    success: "bg-emerald-50 text-emerald-800 border-emerald-200",
    error: "bg-red-50 text-red-800 border-red-200",
    warning: "bg-amber-50 text-amber-900 border-amber-200",
    info: "bg-slate-50 text-slate-700 border-slate-200",
  };
  div.className = `rounded-lg border px-4 py-3 text-sm ${styles[type] || styles.info}`;
  div.innerHTML = text;
  alerts.appendChild(div);
}

function renderResults(payload) {
  // Per-file failures.
  (payload.results || [])
    .filter((r) => !r.ok)
    .forEach((r) =>
      showAlert(`❌ <strong>${escapeHtml(r.name)}</strong> — ${escapeHtml(r.error || "")}`, "error")
    );

  if (payload.succeeded > 0) {
    showAlert(
      `✅ Processed ${payload.succeeded} of ${payload.total} invoice(s) successfully.`,
      "success"
    );
  }

  // QA warnings for invoices that need a second look.
  if (payload.flagged && payload.flagged.length) {
    const listing = payload.flagged
      .map((f) => `• <strong>${escapeHtml(f.name)}</strong>: ${escapeHtml(f.review)}`)
      .join("<br>");
    showAlert(
      "⚠️ Please double-check these invoice(s) before using the numbers:<br>" + listing,
      "warning"
    );
  }

  if (payload.failed > 0 && payload.succeeded === 0) {
    showAlert("No invoices could be processed. Review the errors above and try again.", "info");
  }

  // Data table.
  if (payload.rows && payload.rows.length && payload.columns) {
    renderTable(payload.columns, payload.rows);
    results.classList.remove("hidden");
  }

  // Download button.
  if (payload.download_id) {
    downloadBtn.href = `/api/download/${payload.download_id}`;
    downloadBtn.classList.remove("hidden");
  }
}

function renderTable(columns, rows) {
  const thead = resultsTable.querySelector("thead");
  const tbody = resultsTable.querySelector("tbody");
  thead.innerHTML = "";
  tbody.innerHTML = "";

  const headRow = document.createElement("tr");
  columns.forEach((col) => {
    const th = document.createElement("th");
    th.className = "px-4 py-2 text-left font-semibold border-b border-slate-200";
    th.textContent = col;
    headRow.appendChild(th);
  });
  thead.appendChild(headRow);

  rows.forEach((row) => {
    const tr = document.createElement("tr");
    tr.className = "hover:bg-slate-50";
    columns.forEach((col) => {
      const td = document.createElement("td");
      td.className = "px-4 py-2 border-b border-slate-100 text-slate-700";
      const value = row[col];
      td.textContent = value === null || value === undefined ? "" : value;
      if (/[\u0600-\u06FF]/.test(String(value ?? ""))) td.setAttribute("data-ar", "1");
      tr.appendChild(td);
    });
    tbody.appendChild(tr);
  });
}

// ---------------------------------------------------------------------------
// Utilities
// ---------------------------------------------------------------------------
function escapeHtml(str) {
  return String(str ?? "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

function formatBytes(bytes) {
  if (bytes < 1024) return bytes + " B";
  if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + " KB";
  return (bytes / (1024 * 1024)).toFixed(1) + " MB";
}

// ---------------------------------------------------------------------------
// Fetch Models button: ask the backend to list the provider's models
// (proxied server-side to avoid CORS), then populate the model datalist.
// ---------------------------------------------------------------------------
fetchModelsBtn.addEventListener("click", async () => {
  const originalText = fetchModelsBtn.textContent;
  fetchModelsBtn.disabled = true;
  fetchModelsBtn.textContent = "⏳...";

  const data = new FormData();
  data.append("base_url", baseUrlInput.value.trim());
  data.append("api_key", apiKeyInput.value.trim());

  try {
    const res = await fetch("/api/models", { method: "POST", body: data });
    const payload = await res.json().catch(() => ({}));
    if (!res.ok) {
      window.alert(payload.error || "Could not fetch models.");
      fetchModelsBtn.textContent = originalText;
      fetchModelsBtn.disabled = false;
      return;
    }

    // Replace the datalist suggestions with the fetched models.
    modelOptions.innerHTML = "";
    (payload.models || []).forEach((m) => {
      const opt = document.createElement("option");
      opt.value = m;
      modelOptions.appendChild(opt);
    });

    // Briefly confirm success, then revert.
    fetchModelsBtn.textContent = "✅ Done";
    setTimeout(() => {
      fetchModelsBtn.textContent = originalText;
      fetchModelsBtn.disabled = false;
    }, 1500);
  } catch (err) {
    window.alert("Network error while fetching models.");
    fetchModelsBtn.textContent = originalText;
    fetchModelsBtn.disabled = false;
  }
});


// ---------------------------------------------------------------------------
// Init
// ---------------------------------------------------------------------------
loadConfig();

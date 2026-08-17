/* ═══════════════════════════════════════════════════════
   RAG Eval UI — Application Logic
   Handles: session CRUD, upload, QA generation, evaluation,
   session persistence, and DOM rendering.
   ═══════════════════════════════════════════════════════ */

const API = '';  // Same origin — no prefix needed

// ──── State ────
let sessions = [];
let activeSessionId = null;
let activeSessionData = null;  // Full session detail


// ═══════════════════════════════════════
// INIT
// ═══════════════════════════════════════

document.addEventListener('DOMContentLoaded', () => {
    bindEvents();
    bootSplash();
});


/** Hold the splash for a minimum dwell, then dismiss once the first session
 *  fetch has settled. Timed off real work rather than a fixed delay, so it
 *  never hides a slow first load behind an animation that has already ended. */
async function bootSplash() {
    const splash = document.getElementById('splash');
    const MIN_DWELL = 1500;              // long enough to read the wordmark
    const started = Date.now();

    try {
        await loadSessions();
    } finally {
        // Behind the splash, stage the guided flow so the popup is already
        // composed when the wordmark fades — nothing else is on screen.
        openDraftWizard();

        const remaining = Math.max(0, MIN_DWELL - (Date.now() - started));
        setTimeout(() => {
            if (!splash) return;
            splash.classList.add('gone');
            // Remove from the tree after the fade so it can never trap focus.
            setTimeout(() => splash.remove(), 700);
        }, remaining);
    }
}


/** Open the popup on a blank flow, with no session created yet.
 *
 *  The session row is minted on the first upload instead of here, so simply
 *  loading the page never leaves an empty session behind in the history. */
function openDraftWizard() {
    activeSessionId = null;
    activeSessionData = null;
    renderSessionList();

    document.getElementById('welcomeScreen').style.display = 'none';
    document.getElementById('sessionWorkspace').style.display = 'flex';
    document.getElementById('sessionTitle').textContent = 'New session';
    const wizTitle = document.getElementById('wizardSessionTitle');
    if (wizTitle) wizTitle.textContent = 'New session — nothing uploaded yet';

    // Blank slate: no documents, no test set, no runs.
    renderDocumentList([]);
    document.getElementById('uploadArea').style.display = '';
    document.getElementById('btnGenerate').disabled = true;
    document.getElementById('qaPreview').style.display = 'none';
    renderPreviousRuns([]);
    document.getElementById('newRunResults').innerHTML = '';

    goToStage(1, { force: true });
}


/** Create the session the flow has been filling in. Called lazily on the first
 *  upload so a draft that is abandoned costs nothing. */
async function ensureSession() {
    if (activeSessionId) return activeSessionId;
    const now = new Date();
    const name = `Session ${now.toLocaleDateString('en-US', { month: 'short', day: 'numeric' })} ${now.toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit' })}`;
    const res = await fetch(`${API}/api/sessions`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ name }),
    });
    if (!res.ok) throw new Error(`Could not start a session (${res.status})`);
    const session = await res.json();
    sessions.unshift(session);
    activeSessionId = session.id;
    activeSessionData = session;
    renderSessionList();
    document.getElementById('sessionTitle').textContent = session.name;
    const wizTitle = document.getElementById('wizardSessionTitle');
    if (wizTitle) wizTitle.textContent = session.name;
    return session.id;
}


function bindEvents() {
    // New session
    document.getElementById('btnNewSession').addEventListener('click', createSession);

    // Upload area
    const uploadArea = document.getElementById('uploadArea');
    const fileInput = document.getElementById('fileInput');
    uploadArea.addEventListener('click', () => fileInput.click());
    uploadArea.addEventListener('dragover', e => { e.preventDefault(); uploadArea.classList.add('dragover'); });
    uploadArea.addEventListener('dragleave', () => uploadArea.classList.remove('dragover'));
    uploadArea.addEventListener('drop', e => {
        e.preventDefault();
        uploadArea.classList.remove('dragover');
        if (e.dataTransfer.files.length > 0) handleFileUpload(e.dataTransfer.files);
    });
    fileInput.addEventListener('change', e => {
        if (e.target.files.length > 0) handleFileUpload(e.target.files);
        e.target.value = '';   // let the same file be re-picked after a removal
    });

    // Generate QA
    document.getElementById('btnGenerate').addEventListener('click', generateQA);

    // Download QA
    document.getElementById('btnDownloadQA').addEventListener('click', downloadQA);

    // Next step
    document.getElementById('btnNextStep').addEventListener('click', () => goToStage(3));

    // Stage rail
    document.querySelectorAll('.rail-step').forEach(el => {
        el.addEventListener('click', () => goToStage(Number(el.dataset.stage)));
    });

    // Per-stage Back / Continue
    document.querySelectorAll('.stage-next, .stage-back').forEach(btn => {
        btn.addEventListener('click', () => goToStage(Number(btn.dataset.goto)));
    });

    // Finish -> dismiss the popup for good and reveal history + everything
    document.getElementById('btnFinish')?.addEventListener('click', () => {
        setReviewMode(true);
        showToast('Session complete — here is everything, end to end', 'success');
    });

    // Leave the flow early. With no session yet there is nothing to review, so
    // this lands on the welcome screen beside the history instead.
    document.getElementById('btnHistoryJump')?.addEventListener('click', () => {
        if (activeSessionId) setReviewMode(true);
        else showWelcome();
    });
    document.getElementById('btnWizardJump')?.addEventListener('click', () => goToStage(currentStage, { force: true }));

    // Config form
    document.getElementById('configForm').addEventListener('submit', e => {
        e.preventDefault();
        runEvaluation();
    });

    // Temperature slider
    document.getElementById('temperature').addEventListener('input', e => {
        document.getElementById('tempValue').textContent = parseFloat(e.target.value).toFixed(1);
    });

    // Search type → show/hide index type
    document.querySelectorAll('input[name="searchType"]').forEach(radio => {
        radio.addEventListener('change', updateIndexTypeVisibility);
    });
    updateIndexTypeVisibility();
}


// ═══════════════════════════════════════
// SESSIONS
// ═══════════════════════════════════════

async function loadSessions() {
    try {
        const res = await fetch(`${API}/api/sessions`);
        sessions = await res.json();
        renderSessionList();
    } catch (err) {
        console.error('Failed to load sessions:', err);
    }
}


/** "New session" reopens the guided flow. The row itself is created by
 *  ensureSession() on the first upload, so clicking this repeatedly — or
 *  reloading the page — never litters the history with empty sessions. */
function createSession() {
    openDraftWizard();
}


async function deleteSession(sessionId, event) {
    event.stopPropagation();
    if (!confirm('Delete this session and all its data?')) return;
    try {
        await fetch(`${API}/api/sessions/${sessionId}`, { method: 'DELETE' });
        sessions = sessions.filter(s => s.id !== sessionId);
        if (activeSessionId === sessionId) {
            activeSessionId = null;
            activeSessionData = null;
            showWelcome();
        }
        renderSessionList();
        showToast('Session deleted', 'info');
    } catch (err) {
        showToast('Failed to delete session', 'error');
    }
}


async function selectSession(sessionId) {
    activeSessionId = sessionId;
    renderSessionList();

    // Load full session detail
    try {
        const res = await fetch(`${API}/api/sessions/${sessionId}`);
        activeSessionData = await res.json();
        renderSessionWorkspace();
    } catch (err) {
        showToast('Failed to load session', 'error');
    }
}


function renderSessionList() {
    const list = document.getElementById('sessionList');

    if (sessions.length === 0) {
        list.innerHTML = `
            <div class="empty-state" id="emptyState">
                <p>No sessions yet</p>
                <p class="subtle">Click + to create one</p>
            </div>`;
        return;
    }

    list.innerHTML = sessions.map(s => {
        // Amber marker rides the session list too, so unrated runs cannot be
        // escaped simply by navigating away from the run that produced them.
        const pending = s.unverified_runs || 0;
        return `
        <div class="session-item ${s.id === activeSessionId ? 'active' : ''}" onclick="selectSession('${s.id}')">
            <div class="session-icon ${pending ? 'needs-feedback' : ''}" title="${pending ? pending + ' run(s) awaiting your feedback' : ''}">${pending ? '⚠' : (s.has_qa ? '✅' : '📄')}</div>
            <div class="session-info">
                <div class="session-name">${escapeHtml(s.name)}</div>
                <div class="session-meta">${formatDate(s.created_at)}${s.document_filename ? ' · ' + escapeHtml(s.document_filename) : ''}</div>
            </div>
            ${pending ? `<span class="session-unverified-count" title="${pending} run(s) awaiting feedback">${pending}</span>` : ''}
            <button class="btn-delete-session" onclick="deleteSession('${s.id}', event)" title="Delete session">
                <svg width="14" height="14" viewBox="0 0 14 14" fill="none"><path d="M2 4h10M5 4V3a1 1 0 011-1h2a1 1 0 011 1v1M9 4v7a1 1 0 01-1 1H6a1 1 0 01-1-1V4" stroke="currentColor" stroke-width="1.5" stroke-linecap="round"/></svg>
            </button>
        </div>
    `; }).join('');
}


function renderSessionWorkspace() {
    document.getElementById('welcomeScreen').style.display = 'none';
    document.getElementById('sessionWorkspace').style.display = 'flex';

    const data = activeSessionData;
    document.getElementById('sessionTitle').textContent = data.name;
    // Mirrored into the popup's top bar, which covers the session header.
    const wizTitle = document.getElementById('wizardSessionTitle');
    if (wizTitle) wizTitle.textContent = data.name;

    // Reset UI
    const uploadArea = document.getElementById('uploadArea');
    const qaPreview = document.getElementById('qaPreview');
    const btnGenerate = document.getElementById('btnGenerate');

    // Fall back to the legacy single field for sessions predating multi-upload.
    const docs = data.documents && data.documents.length
        ? data.documents
        : (data.document_filename ? [{ filename: data.document_filename, size: null }] : []);

    if (docs.length) {
        renderDocumentList(docs);
        btnGenerate.disabled = false;
    } else {
        renderDocumentList([]);
        uploadArea.style.display = '';
        btnGenerate.disabled = true;
    }

    if (data.qa_json && data.qa_json.length > 0) {
        // QA already generated — show preview
        renderQAPreview(data.qa_json);
        qaPreview.style.display = '';
        // Mark step 1 completed
        setStepCompleted(1);

        // Always render previous runs (clears old ones if empty)
        renderPreviousRuns(data.run_configs || []);

        // A session with completed runs is something you review, not something
        // you are still filling in — so it opens in the full view.
        if (data.run_configs && data.run_configs.length > 0) {
            setReviewMode(true);
            currentStage = 5;
            refreshStageProgress();
        } else {
            goToStage(3, { force: true });
        }
    } else {
        qaPreview.style.display = 'none';
        renderPreviousRuns([]); // clear old runs
        goToStage(docs.length ? 2 : 1, { force: true });
    }
}


function showWelcome() {
    clearStageModes();
    document.getElementById('welcomeScreen').style.display = '';
    document.getElementById('sessionWorkspace').style.display = 'none';
}


// ═══════════════════════════════════════
// STEP 1: UPLOAD & QA
// ═══════════════════════════════════════

async function handleFileUpload(files) {
    const list = Array.from(files || []);
    if (!list.length) return;

    // A draft flow has no session yet — the first upload is what makes it real.
    try {
        await ensureSession();
    } catch (err) {
        showToast(err.message, 'error');
        return;
    }

    const formData = new FormData();
    // Field name is `files` (plural) — the endpoint appends rather than replaces,
    // so documents can be added to a session incrementally.
    list.forEach(f => formData.append('files', f));

    try {
        showToast(`Uploading ${list.length} file(s)…`, 'info');
        const res = await fetch(`${API}/api/sessions/${activeSessionId}/upload`, {
            method: 'POST',
            body: formData,
        });
        if (!res.ok) {
            const err = await res.json().catch(() => ({}));
            throw new Error(err.detail || `Upload failed (${res.status})`);
        }
        const result = await res.json();

        applyDocuments(result.documents || []);

        if (result.skipped && result.skipped.length) {
            showToast(`Skipped: ${result.skipped.join('; ')}`, 'info');
        }
        if (result.saved && result.saved.length) {
            showToast(`Uploaded ${result.saved.length} file(s)`, 'success');
        }
    } catch (err) {
        showToast('Upload failed: ' + err.message, 'error');
    }
}


async function removeDocument(filename) {
    if (!activeSessionId) return;
    if (!confirm(`Remove ${filename}? Any generated test set will be cleared, since it no longer matches the corpus.`)) return;
    try {
        const res = await fetch(
            `${API}/api/sessions/${activeSessionId}/documents/${encodeURIComponent(filename)}`,
            { method: 'DELETE' });
        if (!res.ok) {
            const err = await res.json().catch(() => ({}));
            throw new Error(err.detail || `Failed (${res.status})`);
        }
        const result = await res.json();
        applyDocuments(result.documents || []);
        showToast(`Removed ${filename}`, 'info');
    } catch (err) {
        showToast(err.message, 'error');
    }
}


/** Single place that reconciles UI state with the server's document list.
 *  Changing the corpus invalidates the test set, so the QA preview is cleared
 *  here too — leaving it visible would imply it still matches the documents. */
function applyDocuments(documents) {
    if (activeSessionData) {
        activeSessionData.documents = documents;
        activeSessionData.qa_json = null;
        activeSessionData.document_filename = documents.length ? documents[0].filename : null;
    }
    renderDocumentList(documents);

    document.getElementById('btnGenerate').disabled = documents.length === 0;
    document.getElementById('qaPreview').style.display = 'none';
    const prov = document.getElementById('qaProvenance');
    if (prov) { prov.innerHTML = ''; prov.style.display = 'none'; }

    refreshStageProgress();

    const idx = sessions.findIndex(s => s.id === activeSessionId);
    if (idx !== -1) {
        sessions[idx].document_filename = documents.length ? documents[0].filename : null;
        sessions[idx].document_count = documents.length;
        sessions[idx].has_qa = false;
    }
    renderSessionList();
}


function formatBytes(n) {
    if (!n && n !== 0) return '';
    if (n < 1024) return `${n} B`;
    if (n < 1024 * 1024) return `${(n / 1024).toFixed(0)} KB`;
    return `${(n / 1024 / 1024).toFixed(1)} MB`;
}


function renderDocumentList(documents) {
    const area = document.getElementById('uploadArea');
    const listEl = document.getElementById('documentList');
    const docs = documents || [];

    if (!docs.length) {
        listEl.style.display = 'none';
        listEl.innerHTML = '';
        area.style.display = '';
        return;
    }

    // Keep the drop area visible so more files can be added without hunting
    // for a button — the list sits beneath it.
    area.style.display = '';
    listEl.style.display = '';
    const total = docs.reduce((a, d) => a + (d.size || 0), 0);

    listEl.innerHTML = `
        <div class="doc-list-head">
            <span>${docs.length} document${docs.length === 1 ? '' : 's'}</span>
            ${total ? `<span class="doc-total">${formatBytes(total)} total</span>` : ''}
        </div>
        ${docs.map(d => `
            <div class="doc-row">
                <svg class="doc-icon" width="16" height="16" viewBox="0 0 20 20" fill="none"><rect x="3" y="2" width="14" height="16" rx="2" stroke="currentColor" stroke-width="1.5"/><path d="M7 7h6M7 10h4M7 13h5" stroke="currentColor" stroke-width="1.5" stroke-linecap="round"/></svg>
                <span class="doc-name">${escapeHtml(d.filename)}</span>
                <span class="doc-size">${formatBytes(d.size)}</span>
                <button class="doc-remove" data-filename="${escapeHtml(d.filename)}" title="Remove this document">
                    <svg width="14" height="14" viewBox="0 0 14 14" fill="none"><path d="M3.5 3.5l7 7M10.5 3.5l-7 7" stroke="currentColor" stroke-width="1.6" stroke-linecap="round"/></svg>
                </button>
            </div>`).join('')}`;
}


// Delegated so rows rendered later are covered too.
document.addEventListener('click', (ev) => {
    const btn = ev.target.closest('.doc-remove');
    if (btn) removeDocument(btn.dataset.filename);
});


async function generateQA() {
    if (!activeSessionId) return;

    const btn = document.getElementById('btnGenerate');
    const numQuestions = document.getElementById('numQuestions').value || 6;
    setButtonLoading(btn, true);

    try {
        const res = await fetch(`${API}/api/sessions/${activeSessionId}/generate-qa?num_questions=${numQuestions}`, {
            method: 'POST'
        });
        if (!res.ok) {
            let errMsg = 'Generation failed';
            try {
                const err = await res.json();
                errMsg = err.detail || errMsg;
            } catch (e) {
                errMsg = `Server error (${res.status})`;
            }
            throw new Error(errMsg);
        }

        const data = await res.json();

        // Update local state
        activeSessionData.qa_json = data.preview;  // We have preview, full is on server

        // Render preview
        renderQAPreview(data.preview, data.total, data.meta);
        document.getElementById('qaPreview').style.display = '';
        if (activeSessionData) activeSessionData.qa_json = data.preview;
        refreshStageProgress();
        if (!document.body.classList.contains('mode-review')) goToStage(3);

        // Update session in sidebar
        const idx = sessions.findIndex(s => s.id === activeSessionId);
        if (idx !== -1) sessions[idx].has_qa = true;
        renderSessionList();

        showToast(`${data.total} QA test cases generated`, 'success');
    } catch (err) {
        showToast(err.message, 'error');
    } finally {
        setButtonLoading(btn, false);
    }
}


function renderQAPreview(qaPairs, total, meta) {
    const tbody = document.getElementById('qaPreviewBody');
    tbody.innerHTML = qaPairs.map((qa, i) => `
        <tr>
            <td>${i + 1}</td>
            <td>${escapeHtml(qa.question)}</td>
            <td>${escapeHtml(qa.answer)}</td>
        </tr>
    `).join('');

    const countEl = document.getElementById('qaCount');
    countEl.textContent = `${qaPairs.length} test cases generated`;

    // Provenance: a ragas test set and a fallback one are not equivalent, so
    // which generator ran must never be left to guesswork.
    const banner = document.getElementById('qaProvenance');
    if (!banner) return;
    if (!meta) { banner.innerHTML = ''; banner.style.display = 'none'; return; }

    const LABELS = {
        deepeval: 'Generated by <strong>DeepEval</strong> — Synthesizer with evolutions',
    };
    const isFallback = meta.generator === 'fallback-llm';
    const notes = (meta.notes || []).length
        ? `<ul class="qa-prov-notes">${meta.notes.map(n => `<li>${escapeHtml(n)}</li>`).join('')}</ul>`
        : '';
    banner.className = `qa-provenance ${isFallback ? 'warn' : 'ok'}`;
    banner.style.display = '';
    banner.innerHTML = `
        <div class="qa-prov-head">
            ${LABELS[meta.generator]
                || `⚠ Generated by <strong>fallback single-prompt LLM</strong> — not a framework test set`}
        </div>
        <div class="qa-prov-meta">
            model <code>${escapeHtml(meta.model || '?')}</code>
            ${meta.embedding_model ? `· embeddings <code>${escapeHtml(meta.embedding_model)}</code>` : ''}
            · requested ${meta.requested} · produced ${meta.produced}
        </div>
        ${notes}`;
}


async function downloadQA() {
    if (!activeSessionId) return;
    try {
        const res = await fetch(`${API}/api/sessions/${activeSessionId}/download-qa`);
        if (!res.ok) throw new Error('Download failed');
        const blob = await res.blob();
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `qa_testcases_${activeSessionId}.json`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
        showToast('Download started', 'success');
    } catch (err) {
        showToast('Download failed', 'error');
    }
}


// ═══════════════════════════════════════
// STEP 2: CONFIG & EVALUATE
// ═══════════════════════════════════════

function updateIndexTypeVisibility() {
    const searchType = document.querySelector('input[name="searchType"]:checked')?.value;
    const group = document.getElementById('indexTypeGroup');
    if (searchType === 'keyword') {
        group.style.display = 'none';
    } else {
        group.style.display = '';
    }
}


async function runEvaluation() {
    if (!activeSessionId) return;

    const btn = document.getElementById('btnRunEval');
    setButtonLoading(btn, true);

    const searchType = document.querySelector('input[name="searchType"]:checked')?.value || 'semantic';

    const payload = {
        chat_model: document.getElementById('chatModel').value,
        embedding_model: document.getElementById('embeddingModel').value,
        vectordb: document.getElementById('vectordb').value,
        chunk_size: parseInt(document.getElementById('chunkSize').value),
        chunk_overlap: parseInt(document.getElementById('chunkOverlap').value),
        reranker_model: document.getElementById('rerankerModel').value || null,
        search_type: searchType,
        index_type: (searchType !== 'keyword') ? (document.getElementById('indexType').value || null) : null,
        top_k: parseInt(document.getElementById('topK').value),
        temperature: parseFloat(document.getElementById('temperature').value),
    };

    try {
        const res = await fetch(`${API}/api/sessions/${activeSessionId}/evaluate`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload),
        });

        if (!res.ok) {
            const err = await res.json();
            throw new Error(err.detail || 'Evaluation failed');
        }

        const runConfig = await res.json();
        appendRunCard(runConfig, document.getElementById('newRunResults'));
        if (activeSessionData) {
            activeSessionData.run_configs = [...(activeSessionData.run_configs || []), runConfig];
        }
        refreshStageProgress();
        // Land on the analytics the run just produced rather than leaving the
        // user on the form they already submitted.
        if (!document.body.classList.contains('mode-review')) goToStage(4);
        showToast('Evaluation complete', 'success');
        // Blocking prompt: the human check is the point, so it is not optional.
        openFeedbackModal(runConfig);
    } catch (err) {
        showToast(err.message, 'error');
    } finally {
        setButtonLoading(btn, false);
    }
}


function renderPreviousRuns(runConfigs) {
    const container = document.getElementById('previousRuns');
    container.innerHTML = '';
    runConfigs.forEach(rc => appendRunCard(rc, container));
}


// Below this a metric is treated as a problem worth acting on.
const WEAK = 0.7;
// At or above this a metric is considered healthy.
const STRONG = 0.8;

// 384-dimension models — the cheapest tier. Worth suggesting an upgrade from
// when retrieval quality is the bottleneck.
const SMALL_EMBEDDINGS = [
    'BAAI/bge-small-en-v1.5', 'BAAI/bge-small-en', 'BAAI/bge-small-zh-v1.5',
    'snowflake/snowflake-arctic-embed-xs', 'snowflake/snowflake-arctic-embed-s',
    'sentence-transformers/all-MiniLM-L6-v2',
    'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2',
    'bge-small',
];

// Chat models with a context window too small to hold many retrieved chunks.
const SMALL_CONTEXT_MODELS = { 'allam-2-7b': 4096 };

/**
 * Turn a run's scores into concrete next actions.
 *
 * Every suggestion names the knob AND its current value, because "raise top_k"
 * is useless without knowing it is currently 3. Only knobs that genuinely
 * affect the pipeline are ever suggested — never vectordb or index_type.
 *
 * Returns [{ tone, title, actions: [string] }].
 */
function buildDiagnostics(batch, cfg) {
    const f = batch.faithfulness;
    const ar = batch.answer_relevancy;
    const cp = batch.context_precision;
    const cr = batch.context_recall;

    const has = v => v !== undefined && v !== null;
    const out = [];

    // ── Retrieval is missing information the answer needs ──────────────────
    if (has(cr) && cr < WEAK) {
        const actions = [];
        if (cfg.top_k < 10) actions.push(`Raise <strong>Top K</strong> (now ${cfg.top_k}) so more chunks are considered.`);
        if (cfg.chunk_size < 1024) actions.push(`Increase <strong>Chunk Size</strong> (now ${cfg.chunk_size}) so each chunk carries more surrounding context.`);
        if (cfg.search_type === 'semantic') actions.push(`Switch <strong>Search Type</strong> to <strong>hybrid</strong> — it also catches exact keyword matches that embeddings miss.`);
        if (cfg.search_type === 'keyword') actions.push(`Switch <strong>Search Type</strong> to <strong>hybrid</strong> or <strong>semantic</strong> — keyword search misses paraphrased wording.`);
        if (cfg.chunk_overlap < Math.round(cfg.chunk_size * 0.2)) actions.push(`Raise <strong>Chunk Overlap</strong> (now ${cfg.chunk_overlap}) so facts spanning a boundary aren't split.`);
        if (cfg.search_type !== 'keyword' && SMALL_EMBEDDINGS.includes(cfg.embedding_model)) {
            actions.push(`Move to a larger <strong>Embedding Model</strong> (now <code>${escapeHtml(cfg.embedding_model)}</code>, 384d) — a 768d or 1024d model separates similar passages better.`);
        }
        out.push({
            tone: 'bad',
            title: `Context Recall is low — the retrieved chunks don't contain everything the answer needs.`,
            actions,
        });
    }

    // ── Retrieval pulls the right thing but buries it in noise ─────────────
    if (has(cp) && cp < WEAK) {
        const actions = [];
        if (cfg.top_k > 3) actions.push(`Lower <strong>Top K</strong> (now ${cfg.top_k}) — you are pulling in chunks that dilute the relevant one.`);
        if (!cfg.reranker_model) actions.push(`Add a <strong>Reranker</strong> — it reorders retrieved chunks so the relevant one lands first, which is exactly what this metric rewards.`);
        if (cfg.search_type === 'keyword') actions.push(`Switch <strong>Search Type</strong> to <strong>semantic</strong> or <strong>hybrid</strong> for better ranking.`);
        if (cfg.chunk_size > 1024) actions.push(`Reduce <strong>Chunk Size</strong> (now ${cfg.chunk_size}) — large chunks mix relevant and irrelevant text together.`);
        out.push({
            tone: 'bad',
            title: `Context Precision is low — relevant chunks are being retrieved, but not ranked near the top.`,
            actions,
        });
    }

    // ── The generator is inventing beyond its context ──────────────────────
    if (has(f) && f < WEAK) {
        const actions = [];
        if (cfg.temperature > 0.3) actions.push(`Lower <strong>Temperature</strong> (now ${cfg.temperature}) — higher values make the model embellish beyond the context.`);
        if (/8b-instant|20b|7b/.test(cfg.chat_model || '')) actions.push(`Try a larger <strong>Chat Model</strong> (now <code>${escapeHtml(cfg.chat_model)}</code>) — smaller models stray from the provided context more often.`);
        if (has(cr) && cr >= STRONG) actions.push(`Retrieval is healthy here, so this is a <strong>generation</strong> problem, not a retrieval one — focus on the model and temperature.`);
        else actions.push(`Fix Context Recall first — a model given incomplete context is forced to fill gaps itself.`);
        out.push({
            tone: 'bad',
            title: `Faithfulness is low — the answers make claims the retrieved context does not support.`,
            actions,
        });
    }

    // ── Answers don't address the question ─────────────────────────────────
    if (has(ar) && ar < WEAK) {
        const actions = [];
        if (has(cr) && cr < WEAK) {
            actions.push(`This usually follows weak retrieval — when context is missing, the model replies "the context does not contain the answer", which scores 0 here. Fix Context Recall first.`);
        } else {
            actions.push(`Retrieval looks fine, so the generator is the weak point — try a different <strong>Chat Model</strong> (now <code>${escapeHtml(cfg.chat_model)}</code>).`);
            if (cfg.temperature > 0.5) actions.push(`Lower <strong>Temperature</strong> (now ${cfg.temperature}) to keep answers on-topic.`);
        }
        actions.push(`Note: this metric rewards answers that <em>address</em> the question — a confidently wrong answer can still score high. Read it next to Faithfulness.`);
        out.push({
            tone: 'warn',
            title: `Answer Relevancy is low — answers aren't directly addressing the questions.`,
            actions,
        });
    }

    // ── Config sanity: context window too small for the retrieved chunks ───
    const ctxLimit = SMALL_CONTEXT_MODELS[cfg.chat_model];
    if (ctxLimit) {
        // ~4 characters per token is the usual rough conversion.
        const approxTokens = Math.round((cfg.top_k * cfg.chunk_size) / 4);
        if (approxTokens > ctxLimit * 0.6) {
            out.push({
                tone: 'warn',
                title: `<code>${escapeHtml(cfg.chat_model)}</code> only has a ${ctxLimit.toLocaleString()}-token context window.`,
                actions: [`Top K ${cfg.top_k} × Chunk Size ${cfg.chunk_size} is roughly ${approxTokens.toLocaleString()} tokens of context — chunks may be silently truncated before the model sees them. Lower <strong>Top K</strong> or <strong>Chunk Size</strong>, or pick a 131k-context model.`],
            });
        }
    }

    // ── Everything healthy: suggest tuning for cost instead ────────────────
    const scored = [f, ar, cp, cr].filter(has);
    if (scored.length && scored.every(v => v >= STRONG)) {
        const actions = [`Try lowering <strong>Top K</strong> (now ${cfg.top_k}) or using a smaller <strong>Chat Model</strong> — if the scores hold, you get the same quality for less cost and latency.`];
        if (cfg.chunk_size > 512) actions.push(`Try a smaller <strong>Chunk Size</strong> (now ${cfg.chunk_size}) to cut embedding and prompt cost.`);
        out.push({
            tone: 'good',
            title: `All metrics are healthy — this config is working.`,
            actions,
        });
    }

    return out.filter(d => d.actions.length);
}


// Aspect tags map to pipeline stages, so a thumbs-down points somewhere fixable.
const FEEDBACK_ASPECTS = [
    { key: 'scores_disagree',  label: "Scores don't match my read" },
    { key: 'bad_answers',      label: 'Generated answers are wrong' },
    { key: 'bad_retrieval',    label: 'Retrieved context was irrelevant' },
    { key: 'bad_questions',    label: 'Test questions are poor' },
    { key: 'bad_ground_truth', label: 'Expected answers are wrong' },
    { key: 'too_slow',         label: 'Too slow / too expensive' },
];


/** The test cases a rating is actually about, as a scrollable list.
 *
 *  A batch average alone cannot be judged: "faithfulness 65.6%" is only wrong or
 *  right relative to the answers it scored. So each case shows its question, the
 *  answer the pipeline produced, the reference answer it was compared against,
 *  and — where DeepEval recorded them — that case's own four scores. A user who
 *  spots a case scored 1.0 on an answer that invented a figure now has something
 *  concrete to write in the comment box.
 *
 *  Per-case scores are the same numbers the batch average is computed from, not a
 *  second opinion, so the list and the tiles can never disagree.
 */
const CASE_METRIC_LABELS = [
    ['faithfulness', 'Faith'],
    ['answer_relevancy', 'Relev'],
    ['context_precision', 'Prec'],
    ['context_recall', 'Recall'],
];

function caseReviewHtml(runConfig, opts = {}) {
    const cases = (runConfig && runConfig.results) || [];
    if (!cases.length) {
        return `<div class="fb-cases">
            <div class="fb-cases-head"><span class="fb-cases-title">Test cases</span></div>
            <div class="fb-cases-empty">No per-case rows were stored for this run.</div>
        </div>`;
    }

    const rows = cases.map((r, i) => {
        const m = r.metrics || {};
        const chips = CASE_METRIC_LABELS
            .filter(([k]) => typeof m[k] === 'number')
            .map(([k, label]) => {
                const v = m[k];
                const level = v >= 0.8 ? 'high' : v >= 0.6 ? 'medium' : 'low';
                return `<span class="fb-case-chip ${level}">
                            <b>${escapeHtml(label)}</b> ${(v * 100).toFixed(0)}%
                        </span>`;
            }).join('');

        return `<article class="fb-case">
            <div class="fb-case-top">
                <span class="fb-case-n">${i + 1}</span>
                <p class="fb-case-q">${escapeHtml(r.question || '—')}</p>
            </div>
            ${chips ? `<div class="fb-case-chips">${chips}</div>` : ''}
            <div class="fb-case-pair">
                <div class="fb-case-col">
                    <span class="fb-case-tag">Generated</span>
                    <p>${escapeHtml(r.generated_answer || '—')}</p>
                </div>
                <div class="fb-case-col expected">
                    <span class="fb-case-tag">Expected</span>
                    <p>${escapeHtml(r.expected_answer || '—')}</p>
                </div>
            </div>
        </article>`;
    }).join('');

    const n = cases.length;
    return `<div class="fb-cases">
        <div class="fb-cases-head">
            <span class="fb-cases-title">Read the ${n} case${n === 1 ? '' : 's'} these scores came from</span>
            <span class="fb-cases-hint">${n > 2 ? 'Scroll for the rest' : ''}</span>
        </div>
        <div class="fb-cases-scroll" ${opts.tall ? 'data-tall="1"' : ''}>${rows}</div>
    </div>`;
}


/** @param opts.withCases  include the scrollable case list above the thumbs.
 *  The run card already prints the full table right below its panel, so only
 *  the standalone feedback stage needs it — otherwise the same rows would
 *  appear twice on one screen. */
function feedbackPanelHtml(runConfig, opts = {}) {
    const existing = runConfig.feedback || [];
    if (existing.length) {
        const f = existing[0];
        const icon = f.rating === 'up' ? '👍' : '👎';
        const tags = (f.aspects || []).map(a =>
            `<span class="fb-tag-static">${escapeHtml(
                (FEEDBACK_ASPECTS.find(x => x.key === a) || {}).label || a)}</span>`).join('');
        return `<div class="fb-panel submitted" data-run="${runConfig.id}">
            <div class="fb-done">
                <span class="fb-done-icon">${icon}</span>
                <div>
                    <strong>Feedback recorded</strong>
                    ${f.comment ? `<div class="fb-done-comment">"${escapeHtml(f.comment)}"</div>` : ''}
                    ${tags ? `<div class="fb-done-tags">${tags}</div>` : ''}
                    <div class="fb-done-meta">${f.synced_to_langsmith
                        ? 'Saved to Postgres and mirrored to LangSmith'
                        : 'Saved to Postgres'}</div>
                </div>
                <button class="fb-again" data-run="${runConfig.id}">Add more</button>
            </div>
        </div>`;
    }

    // Progressive disclosure: thumbs first, then the comment box and tags once a
    // rating is picked. Keeps the panel compact until the user opts in.
    return `<div class="fb-panel" data-run="${runConfig.id}">
        <div class="fb-head">
            <span class="fb-title">Do these results look right to you?</span>
            <span class="fb-sub">Every score above came from one model judging another. Your answer is the only human check in the loop.</span>
        </div>
        ${opts.withCases ? caseReviewHtml(runConfig, { tall: true }) : ''}
        <div class="fb-actions">
            <button class="fb-thumb up"   data-run="${runConfig.id}" data-rating="up"   title="The scores match my own read">👍 Looks right</button>
            <button class="fb-thumb down" data-run="${runConfig.id}" data-rating="down" title="Something here is off">👎 Something's off</button>
            <span class="fb-chosen">Pick one, then add detail below</span>
        </div>
        <div class="fb-body" id="fb-body-${runConfig.id}" style="display:none;">
            <label class="fb-label" for="fb-comment-${runConfig.id}">Your comments</label>
            <textarea class="fb-comment" id="fb-comment-${runConfig.id}" rows="3"
                placeholder="What specifically was right or wrong? e.g. 'faithfulness says 91% but two answers invented figures that aren't in the context'. Concrete detail is what makes this actionable."></textarea>
            <div class="fb-label fb-label-tags">Which part? <span class="fb-optional">optional</span></div>
            <div class="fb-aspects">
                ${FEEDBACK_ASPECTS.map(a => `
                    <label class="fb-tag">
                        <input type="checkbox" value="${a.key}"> ${escapeHtml(a.label)}
                    </label>`).join('')}
            </div>
            <div class="fb-submit-row">
                <span class="fb-hint">A rating is required; the comment is what makes it useful.</span>
                <button class="fb-submit" data-run="${runConfig.id}" disabled>Submit feedback</button>
            </div>
        </div>
    </div>`;
}


/** Everything that must happen once a rating is recorded, in one place.
 *
 *  Both the inline panel and the modal call this. They used to each do their
 *  own DOM updates, and the inline one silently forgot to clear the UNVERIFIED
 *  badge — so the card still read as unrated until a page refresh. One shared
 *  function means that cannot drift apart again.
 */
function markRunVerified(panel, rating, saved, runId) {
    // Patch the in-memory model, not just the DOM. Anything that re-renders
    // later — Finish, a stage switch, entering review mode — rebuilds its HTML
    // from activeSessionData, so a rating that lives only in the DOM silently
    // reverts to "please rate this" the moment one of those happens.
    const id = runId || (panel && panel.dataset.run);
    if (id && activeSessionData) {
        const rc = (activeSessionData.run_configs || [])
            .find(r => String(r.id) === String(id));
        // The modal path prepends its own copy first; don't double-count it.
        if (rc && !(rc.feedback || []).length) {
            rc.feedback = [saved || { rating }];
        }
    }

    if (panel) {
        panel.classList.add('submitted');
        panel.innerHTML = `<div class="fb-done">
            <span class="fb-done-icon">${rating === 'up' ? '👍' : '👎'}</span>
            <div><strong>Thank you — feedback recorded</strong>
                <div class="fb-done-meta">${saved && saved.synced_to_langsmith
                    ? 'Saved to Postgres and mirrored to LangSmith'
                    : 'Saved to Postgres'}</div></div>
        </div>`;
    }

    // Clear the amber marking on the whole card, not just the panel inside it.
    const runCard = panel ? panel.closest('.run-card') : null;
    if (runCard) {
        runCard.classList.remove('unverified');
        const badge = runCard.querySelector('.run-unverified');
        if (badge) {
            badge.className = 'run-verified';
            badge.textContent = '✓ Human-verified';
            badge.removeAttribute('title');
        }
    }

    // The rail's stage-5 tick is derived from the same model field.
    refreshStageProgress();

    // And the sidebar count, so the debt clears everywhere at once.
    loadSessions();
}


async function submitFeedback(runConfigId, panel) {
    const rating = panel.dataset.rating;
    if (!rating) return;
    const aspects = [...panel.querySelectorAll('.fb-aspects input:checked')].map(i => i.value);
    const comment = panel.querySelector('.fb-comment')?.value || '';
    const btn = panel.querySelector('.fb-submit');
    if (btn) { btn.disabled = true; btn.textContent = 'Saving…'; }

    try {
        const res = await fetch(
            `${API}/api/sessions/${activeSessionId}/runs/${runConfigId}/feedback`,
            { method: 'POST', headers: { 'Content-Type': 'application/json' },
              body: JSON.stringify({ rating, comment, aspects }) });
        if (!res.ok) {
            const err = await res.json().catch(() => ({}));
            throw new Error(err.detail || `Failed (${res.status})`);
        }
        const saved = await res.json();
        markRunVerified(panel, rating, saved);
        showToast('Feedback saved', 'success');
    } catch (e) {
        if (btn) { btn.disabled = false; btn.textContent = 'Submit feedback'; }
        showToast(e.message, 'error');
    }
}


// One delegated listener for all run cards, present and future.
document.addEventListener('click', (ev) => {
    const thumb = ev.target.closest('.fb-thumb');
    if (thumb) {
        const panel = thumb.closest('.fb-panel');
        panel.dataset.rating = thumb.dataset.rating;
        panel.querySelectorAll('.fb-thumb').forEach(b => b.classList.remove('chosen'));
        thumb.classList.add('chosen');
        // Picking a rating reveals the comment box and tags, and unlocks submit.
        const body = panel.querySelector('.fb-body');
        if (body) body.style.display = '';
        const submit = panel.querySelector('.fb-submit');
        if (submit) submit.disabled = false;
        const chosen = panel.querySelector('.fb-chosen');
        if (chosen) chosen.textContent = thumb.dataset.rating === 'up'
            ? 'Marked as looking right'
            : 'Marked as off — which part?';
        return;
    }
    const submit = ev.target.closest('.fb-submit');
    if (submit) {
        submitFeedback(submit.dataset.run, submit.closest('.fb-panel'));
        return;
    }
    const again = ev.target.closest('.fb-again');
    if (again) {
        const panel = again.closest('.fb-panel');
        panel.classList.remove('submitted');
        panel.removeAttribute('data-rating');
        panel.innerHTML = feedbackPanelHtml({ id: again.dataset.run, feedback: [] })
            .replace(/^<div class="fb-panel"[^>]*>/, '').replace(/<\/div>$/, '');
    }
});


function appendRunCard(runConfig, container) {
    const card = document.createElement('div');
    card.className = 'run-card';

    // `informational: true` marks knobs that are recorded but do NOT change the
    // pipeline, so two runs differing only in those are not a real comparison.
    const badges = [
        { label: 'Model', value: runConfig.chat_model },
        { label: 'Embed', value: runConfig.embedding_model },
        { label: 'VectorDB', value: runConfig.vectordb, informational: true },
        { label: 'Chunk', value: `${runConfig.chunk_size}/${runConfig.chunk_overlap}` },
        { label: 'Search', value: runConfig.search_type },
        { label: 'Top K', value: runConfig.top_k },
        { label: 'Temp', value: runConfig.temperature },
    ];
    if (runConfig.index_type) {
        badges.push({ label: 'Index', value: runConfig.index_type, informational: true });
    }
    if (runConfig.reranker_model) {
        badges.push({ label: 'Reranker', value: runConfig.reranker_model });
    }

    const badgesHtml = badges.map(b => {
        const info = b.informational
            ? ` <span class="badge-info" title="Recorded for reference only — this setting does not change the pipeline, so it cannot change the scores.">ⓘ</span>`
            : '';
        return `<span class="config-badge${b.informational ? ' informational' : ''}">` +
               `<span class="badge-label">${b.label}:</span> ${escapeHtml(String(b.value))}${info}</span>`;
    }).join('');

    // LangSmith button
    const lsBtn = runConfig.langsmith_experiment_url
        ? `<a class="btn-langsmith" href="${escapeHtml(runConfig.langsmith_experiment_url)}" target="_blank" rel="noopener">
               <span class="ls-dot"></span> View in LangSmith
           </a>`
        : '';

    // The four metrics, scored over the whole test set as one batch.
    const BATCH_METRICS = [
        { key: 'faithfulness',      label: 'Faithfulness',      desc: 'Share of claims in the answer that are supported by the retrieved context. Measures hallucination.' },
        { key: 'answer_relevancy',  label: 'Answer Relevancy',  desc: 'How directly the answer addresses the question. Note: an answer can be relevant but wrong.' },
        { key: 'context_precision', label: 'Context Precision', desc: 'Rank-aware — are the chunks relevant to the ground truth ranked near the top?' },
        { key: 'context_recall',    label: 'Context Recall',    desc: 'Share of the ground-truth answer that the retrieved context actually covers.' },
    ];

    const batch = runConfig.metrics || {};
    const meta = batch._meta || {};
    const numCases = meta.num_test_cases ?? (runConfig.results || []).length;

    const scoreTilesHtml = BATCH_METRICS.map(({ key, label, desc }) => {
        const val = batch[key];
        if (val === undefined || val === null) {
            return `<div class="ragas-tile empty" title="${escapeHtml(desc)}">
                        <div class="ragas-tile-label">${label}</div>
                        <div class="ragas-tile-value">—</div>
                        <div class="ragas-tile-sub">not scored</div>
                    </div>`;
        }
        const pct = (val * 100).toFixed(1);
        const level = val >= 0.8 ? 'high' : val >= 0.6 ? 'medium' : 'low';
        const scored = (meta.scored_counts || {})[key];
        // Show coverage whenever it is short of the full batch, so a mean over
        // a subset can never be mistaken for a mean over everything.
        const sub = (scored !== undefined && scored < numCases)
            ? `<span class="ragas-partial">${scored} of ${numCases} cases</span>`
            : `${numCases} test cases`;
        return `<div class="ragas-tile ${level}" title="${escapeHtml(desc)}">
                    <div class="ragas-tile-label">${label}</div>
                    <div class="ragas-tile-value">${pct}%</div>
                    <div class="ragas-tile-sub">${sub}</div>
                </div>`;
    }).join('');

    const framework = 'DeepEval';

    const cc = meta.cost_controls;
    const costHtml = cc ? ` · caps: truths ≤ ${cc.truths_extraction_limit}`
        + `${cc.include_reason ? '' : ', reasons off'}` : '';

    const provenanceHtml = meta.judge_model ? `
        <div class="ragas-provenance">
            Scored by <strong>${framework}</strong> · generator <code>${escapeHtml(meta.generator_model || '?')}</code>
            · judge <code>${escapeHtml(meta.judge_model)}</code>
            · embeddings <code>${escapeHtml(meta.embedding_model || '?')}</code>${costHtml}
        </div>` : '';

    const notes = meta.notes || [];
    const notesHtml = notes.length
        ? `<div class="ragas-notes"><strong>⚠ Run notes</strong><ul>${
              notes.map(n => `<li>${escapeHtml(n)}</li>`).join('')
          }</ul></div>`
        : '';

    // Action hints. `title` and `actions` are built from trusted local strings
    // and already-escaped config values, so they are intentionally not escaped
    // again here — that would render the inline <strong>/<code> markup as text.
    const diagnostics = Object.keys(batch).some(k => !k.startsWith('_'))
        ? buildDiagnostics(batch, runConfig)
        : [];
    const diagnosticsHtml = diagnostics.length
        ? `<div class="ragas-diagnostics">
               <div class="diag-heading">What to change next</div>
               ${diagnostics.map(d => `
                   <div class="diag-item ${d.tone}">
                       <div class="diag-title">${d.title}</div>
                       <ul>${d.actions.map(a => `<li>${a}</li>`).join('')}</ul>
                   </div>`).join('')}
           </div>`
        : '';

    const metricsPanelHtml = `
        <div class="ragas-panel">
            <div class="ragas-panel-header">
                <span class="ragas-title">DeepEval — overall across ${numCases} test case${numCases === 1 ? '' : 's'}</span>
            </div>
            <div class="ragas-grid">${scoreTilesHtml}</div>
            ${diagnosticsHtml}
            ${provenanceHtml}
            ${notesHtml}
        </div>`;

    const resultsHtml = (runConfig.results || []).map((r, i) => `
            <tr>
                <td>${i + 1}</td>
                <td>${escapeHtml(r.question)}</td>
                <td>${escapeHtml(r.generated_answer || '—')}</td>
                <td>${escapeHtml(r.expected_answer || '—')}</td>
            </tr>
        `).join('');

    // A run nobody has rated stays visibly marked. The modal interrupts once;
    // this badge outlives it, so skipping never becomes quietly free.
    const isRated = (runConfig.feedback || []).length > 0;
    if (!isRated) card.className = "run-card unverified";
    const verifyBadge = isRated
        ? `<span class="run-verified">✓ Human-verified</span>`
        : `<span class="run-unverified">⚠ Unverified</span>`;

    card.innerHTML = `
        <div class="run-card-header">
            <div class="run-config-summary">${badgesHtml}</div>
            <div style="display:flex;align-items:center;gap:10px;flex-shrink:0;">
                ${verifyBadge}
                <span class="run-timestamp">${formatDate(runConfig.created_at)}</span>
                ${lsBtn}
            </div>
        </div>
        <div class="run-card-body">
            ${metricsPanelHtml}
            ${feedbackPanelHtml(runConfig)}
            <div class="run-card-table-wrapper">
                <table class="data-table">
                    <thead>
                        <tr>
                            <th>#</th>
                            <th>Question</th>
                            <th>Generated Answer</th>
                            <th>Expected Answer</th>
                        </tr>
                    </thead>
                    <tbody>${resultsHtml}</tbody>
                </table>
            </div>
        </div>
    `;

    container.appendChild(card);
}


// ═══════════════════════════════════════
// UI HELPERS
// ═══════════════════════════════════════

/** Reflect real progress on the stepper: stage 1 is complete once a test set
 *  exists, stage 2 once at least one run has been evaluated. Driven by data,
 *  not by which stage you happen to be looking at. */
function refreshStageProgress() {
    const hasQA = !!(activeSessionData && activeSessionData.qa_json);
    const hasRuns = !!(activeSessionData && (activeSessionData.run_configs || []).length);
    const s1 = document.querySelector('.step[data-step="1"]');
    const s2 = document.querySelector('.step[data-step="2"]');
    if (s1) s1.classList.toggle('completed', hasQA);
    if (s2) s2.classList.toggle('completed', hasRuns);
}


/* ═══════════════════════════════════════
   STAGE NAVIGATION
   ═══════════════════════════════════════
   Two modes over the same DOM:
     · wizard  — one stage visible at a time, guided, with a rail and footers
     · review  — every stage stacked, which is the dense single-page view

   Nothing is duplicated between them: the same cards, the same listeners, the
   same rendered run cards. Only visibility changes, so no functionality can
   drift between the two views.                                              */

const TOTAL_STAGES = 5;
let currentStage = 1;

/** What is actually done, derived from data rather than from navigation. */
function stageState() {
    const d = activeSessionData || {};
    const docs = (d.documents && d.documents.length) ? d.documents.length
               : (d.document_filename ? 1 : 0);
    const runs = (d.run_configs || []).length;
    const rated = (d.run_configs || []).some(r => (r.feedback || []).length);
    return {
        1: docs > 0,
        2: !!(d.qa_json && d.qa_json.length),
        3: runs > 0,
        4: runs > 0,
        5: rated,
    };
}

/** A stage is reachable once the one before it is satisfied — you cannot
 *  configure a run before a test set exists, and so on. */
function stageReachable(n) {
    const done = stageState();
    if (n <= 1) return true;
    for (let i = 1; i < n; i++) if (!done[i]) return false;
    return true;
}

function goToStage(n, opts = {}) {
    n = Math.min(Math.max(n, 1), TOTAL_STAGES);
    if (!opts.force && !stageReachable(n)) {
        const done = stageState();
        const blocker = [1, 2, 3, 4].find(i => !done[i]) || 1;
        showToast(STAGE_BLOCKED[blocker] || 'Finish the previous stage first', 'info');
        n = blocker;
    }
    currentStage = n;
    setReviewMode(false);

    document.querySelectorAll('.stage').forEach(el => {
        el.classList.toggle('active', Number(el.dataset.stage) === n);
    });
    refreshStageProgress();

    // The card body is its own scroll region, so reset that rather than the
    // page — the page does not scroll while the modal is up.
    const host = document.getElementById('stageHost');
    if (host) host.scrollTo({ top: 0, behavior: 'smooth' });

    if (n === 5) renderFeedbackStage();
    if (n === 4) refreshAnalyticsEmpty();
}

const STAGE_BLOCKED = {
    1: 'Upload at least one document first',
    2: 'Generate a test set first',
    3: 'Run an evaluation first',
    4: 'Run an evaluation first',
};

/** Paint the rail: completed, current, and locked states. */
function refreshStageProgress() {
    const done = stageState();
    document.querySelectorAll('.rail-step').forEach(el => {
        const n = Number(el.dataset.stage);
        el.classList.toggle('done', !!done[n] && n !== currentStage);
        el.classList.toggle('current', n === currentStage);
        el.classList.toggle('locked', !stageReachable(n));
    });
    // Footer buttons gate on the same state, so the rail and the buttons can
    // never disagree about whether you may advance.
    document.querySelectorAll('.stage-next').forEach(btn => {
        const target = Number(btn.dataset.goto);
        btn.disabled = !stageReachable(target);
    });
    document.querySelectorAll('.stage-hint').forEach(h => {
        const n = Number(h.dataset.hint);
        h.style.display = done[n] ? 'none' : '';
    });
    const finish = document.getElementById('btnFinish');
    if (finish) finish.disabled = !done[3];

    // Card top bar: counter and the hairline progress fill.
    const count = document.getElementById('wizardStageCount');
    if (count) count.textContent = `${currentStage} / ${TOTAL_STAGES}`;
    const bar = document.getElementById('wizardProgressBar');
    if (bar) bar.style.width = `${(currentStage / TOTAL_STAGES) * 100}%`;
}

function refreshAnalyticsEmpty() {
    const empty = document.getElementById('analyticsEmpty');
    if (!empty) return;
    const hasCards = document.querySelectorAll('#newRunResults .run-card, #previousRuns .run-card').length > 0;
    empty.style.display = hasCards ? 'none' : '';
}

/** Stage 5 shows the feedback panel for the most recent run. Reuses the same
 *  renderer the run card uses, so both paths stay identical. */
function renderFeedbackStage() {
    const host = document.getElementById('feedbackStageHost');
    if (!host) return;
    const runs = (activeSessionData && activeSessionData.run_configs) || [];
    const latest = runs[runs.length - 1];
    if (!latest) {
        host.innerHTML = `<div class="stage-empty"><p>Nothing to rate yet.</p>
            <p class="subtle">Run an evaluation first.</p></div>`;
        return;
    }
    // This stage has no results table of its own, so the panel carries the case
    // list — a rating with nothing to read would be a guess.
    host.innerHTML = metricsRecapHtml(latest) + feedbackPanelHtml(latest, { withCases: true });
}


/** A compact restatement of the batch scores for the feedback stage, so the
 *  numbers being rated are on screen next to the cases and the thumbs. */
function metricsRecapHtml(runConfig) {
    const batch = runConfig.metrics || {};
    const tiles = [
        ['Faithfulness', batch.faithfulness],
        ['Answer Relevancy', batch.answer_relevancy],
        ['Context Precision', batch.context_precision],
        ['Context Recall', batch.context_recall],
    ].filter(([, v]) => typeof v === 'number');
    if (!tiles.length) return '';

    const n = (runConfig.results || []).length;
    return `<div class="fb-recap">
        <div class="fb-recap-head">DeepEval — overall across ${n} test case${n === 1 ? '' : 's'}</div>
        <div class="fb-recap-grid">
            ${tiles.map(([label, v]) => {
                const level = v >= 0.8 ? 'high' : v >= 0.6 ? 'medium' : 'low';
                return `<div class="fb-recap-tile ${level}">
                    <span class="fb-recap-label">${escapeHtml(label)}</span>
                    <span class="fb-recap-value">${(v * 100).toFixed(1)}%</span>
                </div>`;
            }).join('')}
        </div>
    </div>`;
}

/** The one switch between the two presentations.
 *
 *  on  = review  — the popup dissolves, history comes back, every stage is
 *                  stacked flat: uploads, test set, config, analytics, feedback.
 *  off = wizard  — the popup card is up and it is the only thing on screen.
 *
 *  The two classes are mutually exclusive by construction here, so there is no
 *  way to end up half-modal. */
function setReviewMode(on) {
    document.body.classList.toggle('mode-review', on);
    document.body.classList.toggle('mode-wizard', !on);
    const jumpW = document.getElementById('btnWizardJump');
    if (jumpW) jumpW.style.display = on ? '' : 'none';
    if (on) {
        document.querySelectorAll('.stage').forEach(el => el.classList.add('active'));
        refreshAnalyticsEmpty();
        renderFeedbackStage();
        renderReviewSummary();
        // The page scrolls again once the modal is gone.
        const main = document.getElementById('mainContent');
        if (main) main.scrollTo({ top: 0, behavior: 'smooth' });
    }
}


/** The review view drops the input stages — an upload box and a config form are
 *  nothing to review once the run is done. What is worth keeping is the
 *  provenance those stages established, so it is restated here as plain text:
 *  which corpus, how many golden cases, how many evaluations. */
function renderReviewSummary() {
    const host = document.getElementById('reviewSummary');
    if (!host) return;
    const d = activeSessionData;
    if (!d) { host.innerHTML = ''; return; }

    const docs = (d.documents && d.documents.length)
        ? d.documents.map(x => x.filename)
        : (d.document_filename ? [d.document_filename] : []);
    const cases = (d.qa_json || []).length;
    const runs = (d.run_configs || []).length;

    const item = (k, v, title) =>
        `<div class="rs-item"><span class="rs-k">${k}</span>` +
        `<span class="rs-v"${title ? ` title="${escapeHtml(title)}"` : ''}>${v}</span></div>`;

    host.innerHTML =
        item('Corpus', docs.length
                ? `${docs.length} document${docs.length === 1 ? '' : 's'} · ${escapeHtml(docs.join(', '))}`
                : '<span class="rs-none">none</span>',
             docs.join('\n')) +
        item('Test set', cases ? `${cases} golden case${cases === 1 ? '' : 's'}`
                              : '<span class="rs-none">none</span>') +
        item('Evaluations', runs ? `${runs} run${runs === 1 ? '' : 's'}`
                                : '<span class="rs-none">none</span>');
}

/** No session selected: neither presentation applies, so drop both classes or
 *  the welcome screen would stay hidden behind a stale backdrop. */
function clearStageModes() {
    document.body.classList.remove('mode-wizard', 'mode-review');
}

/** Backwards-compatible shim: existing call sites still say "step 1 / step 2".
 *  Step 2 meant "configure and evaluate", which is stage 3 now. */
function switchToStep(stepNum) {
    goToStage(stepNum === 2 ? 3 : 1, { force: true });
}

function setStepCompleted() { refreshStageProgress(); }


function setButtonLoading(btn, loading) {
    const text = btn.querySelector('.btn-text');
    const loader = btn.querySelector('.btn-loader');
    if (loading) {
        text.style.display = 'none';
        loader.style.display = '';
        btn.disabled = true;
    } else {
        text.style.display = '';
        loader.style.display = 'none';
        btn.disabled = false;
    }
}


function showToast(message, type = 'info') {
    const container = document.getElementById('toastContainer');
    const toast = document.createElement('div');
    toast.className = `toast ${type}`;
    toast.textContent = message;
    container.appendChild(toast);
    setTimeout(() => {
        toast.style.animation = 'toastOut 0.3s ease forwards';
        setTimeout(() => toast.remove(), 300);
    }, 3000);
}


function formatDate(dateStr) {
    try {
        // Postgres returns microseconds (6 digits) — JS Date only handles milliseconds (3 digits)
        // Also strip trailing timezone and normalize
        const normalized = dateStr
            .replace(/(\.(\d{3}))\d+/, '$1')  // truncate microseconds → milliseconds
            .replace(/([+-]\d{2}:\d{2})$/, 'Z') // replace +05:30 style tz with Z (we'll use local display)
            .replace(/Z$/, '+00:00');            // normalise for Date constructor
        const d = new Date(normalized);
        if (isNaN(d.getTime())) return dateStr;
        return d.toLocaleDateString('en-US', { month: 'short', day: 'numeric' })
            + ' · ' + d.toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit' });
    } catch {
        return dateStr;
    }
}


function escapeHtml(str) {
    const div = document.createElement('div');
    div.textContent = str;
    return div.innerHTML;
}


// ═══════════════════════════════════════
// MANDATORY FEEDBACK MODAL
// ═══════════════════════════════════════
//
// Fires automatically when an evaluation completes, and cannot be dismissed
// without submitting: no close button, no backdrop click, no Esc.
//
// Both rating buttons are given identical visual weight on purpose. A forced
// prompt with one prominent button collects reflex clicks on that button, and
// this data exists specifically to check whether the metrics match human
// judgement — a reflex 👍 is worse than no rating at all.
//
// The one escape hatch appears only after a failed save, so a backend outage
// cannot leave the user stuck in a modal they have no way to satisfy.

let fbModalRun = null;      // run config awaiting feedback
let fbModalRating = null;
let fbModalFailures = 0;


function openFeedbackModal(runConfig) {
    fbModalRun = runConfig;
    fbModalRating = null;
    fbModalFailures = 0;

    const overlay = document.getElementById('fbModalOverlay');
    if (!overlay || !runConfig) return;

    // Show the numbers being judged — asking "do these look right?" without
    // restating them makes the question unanswerable.
    const batch = runConfig.metrics || {};
    const meta = batch._meta || {};
    const tiles = [
        ['Faithfulness', batch.faithfulness],
        ['Answer Relevancy', batch.answer_relevancy],
        ['Context Precision', batch.context_precision],
        ['Context Recall', batch.context_recall],
    ].filter(([, v]) => v !== undefined && v !== null);

    document.getElementById('fbModalScores').innerHTML = tiles.length
        ? tiles.map(([label, v]) => {
            const level = v >= 0.8 ? 'high' : v >= 0.6 ? 'medium' : 'low';
            return `<div class="fb-score ${level}">
                        <span class="fb-score-label">${escapeHtml(label)}</span>
                        <span class="fb-score-value">${(v * 100).toFixed(1)}%</span>
                    </div>`;
          }).join('')
        : '<div class="fb-score empty">No scores were recorded for this run.</div>';

    // The evidence, not just the totals — see caseReviewHtml().
    document.getElementById('fbModalCases').innerHTML = caseReviewHtml(runConfig);

    document.getElementById('fbModalAspects').innerHTML = FEEDBACK_ASPECTS.map(a => `
        <label class="fb-tag">
            <input type="checkbox" value="${a.key}"> ${escapeHtml(a.label)}
        </label>`).join('');

    document.getElementById('fbModalComment').value = '';
    document.getElementById('fbModalSubmit').disabled = true;
    document.getElementById('fbModalSubmit').textContent = 'Submit feedback';
    document.getElementById('fbModalHint').textContent = 'Pick 👍 or 👎 above to continue.';
    document.getElementById('fbModalEscape').style.display = 'none';
    document.querySelectorAll('#fbModalOverlay .fb-thumb').forEach(b => b.classList.remove('chosen'));

    overlay.style.display = 'flex';
    document.body.classList.add('fb-modal-open');
    setTimeout(() => document.getElementById('fbModalComment')?.focus(), 250);
}


function closeFeedbackModal() {
    const overlay = document.getElementById('fbModalOverlay');
    if (overlay) overlay.style.display = 'none';
    document.body.classList.remove('fb-modal-open');
    fbModalRun = null;
    fbModalRating = null;
}


function chooseModalRating(rating) {
    fbModalRating = rating;
    document.querySelectorAll('#fbModalOverlay .fb-thumb').forEach(b =>
        b.classList.toggle('chosen', b.dataset.rating === rating));
    document.getElementById('fbModalSubmit').disabled = false;
    document.getElementById('fbModalHint').textContent = rating === 'up'
        ? 'Marked as looking right — a comment makes it far more useful.'
        : 'Marked as off — tell us which part, so it points somewhere fixable.';
}


async function submitModalFeedback() {
    if (!fbModalRun || !fbModalRating) return;

    const btn = document.getElementById('fbModalSubmit');
    const aspects = [...document.querySelectorAll('#fbModalAspects input:checked')].map(i => i.value);
    const comment = document.getElementById('fbModalComment').value || '';

    btn.disabled = true;
    btn.textContent = 'Saving…';

    try {
        const res = await fetch(
            `${API}/api/sessions/${activeSessionId}/runs/${fbModalRun.id}/feedback`,
            { method: 'POST', headers: { 'Content-Type': 'application/json' },
              body: JSON.stringify({ rating: fbModalRating, comment, aspects }) });
        if (!res.ok) {
            const err = await res.json().catch(() => ({}));
            throw new Error(err.detail || `Failed (${res.status})`);
        }
        const saved = await res.json();

        // Reflect it on the run card immediately, so the inline panel and the
        // modal never disagree about whether feedback exists.
        fbModalRun.feedback = [saved, ...(fbModalRun.feedback || [])];
        // Same shared path the inline panel uses — badge, card and sidebar all
        // clear together, so the two entry points can never disagree.
        markRunVerified(
            document.querySelector(`.fb-panel[data-run="${fbModalRun.id}"]`),
            fbModalRating, saved, fbModalRun.id,
        );

        closeFeedbackModal();
        showToast('Thank you — feedback saved', 'success');
    } catch (e) {
        fbModalFailures += 1;
        btn.disabled = false;
        btn.textContent = 'Retry submit';
        document.getElementById('fbModalHint').textContent = e.message;
        // Two failures means something is genuinely wrong on the server side.
        // Offer a way out rather than holding the app hostage to it.
        if (fbModalFailures >= 2) {
            document.getElementById('fbModalEscape').style.display = '';
        }
        showToast('Could not save feedback: ' + e.message, 'error');
    }
}


document.getElementById('fbModalUp')?.addEventListener('click', () => chooseModalRating('up'));
document.getElementById('fbModalDown')?.addEventListener('click', () => chooseModalRating('down'));
document.getElementById('fbModalSubmit')?.addEventListener('click', submitModalFeedback);
document.getElementById('fbModalEscapeBtn')?.addEventListener('click', () => {
    closeFeedbackModal();
    showToast('Feedback skipped — the run is still saved', 'info');
});

// Swallow backdrop clicks: the overlay must not be a dismiss target.
document.getElementById('fbModalOverlay')?.addEventListener('click', (ev) => {
    if (ev.target.id === 'fbModalOverlay') ev.stopPropagation();
});

// Block Esc while the modal is open, and keep Tab inside it so the user cannot
// tab out into the page behind.
document.addEventListener('keydown', (ev) => {
    const overlay = document.getElementById('fbModalOverlay');
    if (!overlay || overlay.style.display === 'none') return;
    if (ev.key === 'Escape') { ev.preventDefault(); ev.stopPropagation(); return; }
    if (ev.key === 'Tab') {
        const focusable = overlay.querySelectorAll('button:not([disabled]), textarea, input[type="checkbox"]');
        if (!focusable.length) return;
        const first = focusable[0], last = focusable[focusable.length - 1];
        if (ev.shiftKey && document.activeElement === first) { ev.preventDefault(); last.focus(); }
        else if (!ev.shiftKey && document.activeElement === last) { ev.preventDefault(); first.focus(); }
    }
}, true);

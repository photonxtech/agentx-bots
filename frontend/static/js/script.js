console.log("SCRIPT LOADED");

let sessionId = null;

// ==================== Elements ====================

const pdfInput = document.getElementById("pdfFile");
const uploadStatus = document.getElementById("uploadStatus");

const chatWindow = document.getElementById("chatWindow");
const chatInput = document.getElementById("chatInput");
const sendButton = document.getElementById("sendBtn");

const historyContainer = document.getElementById("historyList");
const docListContainer = document.getElementById("docList");
const newChatBtn = document.getElementById("newChatBtn");


// ==================== Rendering helpers ====================

function isImageFile(name) {
    return /\.(png|jpe?g|webp)$/i.test(name);
}

// Detects fallback / "I don't know" style answers so we can suppress
// the Sources panel (and the metrics button) for them — retrieval ran,
// but nothing useful was found, so showing "Sources"/"Calculate Metrics"
// next to "I don't know" is misleading.
function isFallbackAnswer(text) {
    if (!text) return false;
    const t = text.trim();

    const patterns = [
        /^i don'?t know\.?$/i,
        /^i do not know\.?$/i,
        /^i'?m not sure\.?$/i,
        /^oops i don'?t know\.?$/i,
        /i don'?t have (enough|sufficient) information/i,
        /i (could not|couldn'?t) find/i,
        /no relevant information/i,
        /not (mentioned|found|available) in the (provided|given|uploaded)? ?(context|document|pdf)/i,
        /the (document|context|pdf) does(n'?t| not) (contain|mention|include)/i,
    ];

    return patterns.some(p => p.test(t));
}

function renderDocList(pdfNames) {
    if (!pdfNames || pdfNames.length === 0) {
        docListContainer.innerHTML = `<span class="empty-hint">No documents in this chat yet.</span>`;
        return;
    }

    docListContainer.innerHTML = "";

    pdfNames.forEach(name => {
        const item = document.createElement("div");
        item.className = "doc-item";

        const chip = document.createElement("div");
        chip.className = "doc-chip";
        chip.innerText = `${isImageFile(name) ? "🖼️" : "📎"} ${name}`;

        const removeBtn = document.createElement("button");
        removeBtn.className = "doc-remove-btn";
        removeBtn.innerText = "✕";
        removeBtn.title = "Remove file";
        removeBtn.onclick = async (e) => {
            e.stopPropagation();
            if (!sessionId) return;
            try {
                await fetch(`/sessions/${sessionId}/files/${encodeURIComponent(name)}`, { method: "DELETE" });
                const detail = await (await fetch(`/sessions/${sessionId}`)).json();
                renderDocList(detail.pdf_names);
                await loadSessionList();
            } catch (err) {
                console.error("Error removing file:", err);
            }
        };

        item.appendChild(chip);
        item.appendChild(removeBtn);
        docListContainer.appendChild(item);
    });
}

// renderMessage now returns the message body element instead of building
// the whole bubble (sources + metrics) as one static innerHTML blob. That's
// needed so the metrics section can be a real DOM node with its own click
// handler (the "Calculate Metrics" button), not just a string of markup.
function renderMessage(role, content) {
    const wrapper = document.createElement("div");
    wrapper.className = role === "user" ? "msg-row msg-row-user" : "msg-row msg-row-assistant";

    let body;
    if (role === "user") {
        body = document.createElement("div");
        body.className = "user-message";
        body.innerHTML = content;
    } else {
        body = document.createElement("div");
        body.className = "assistant-message";
        body.innerHTML = marked.parse(content);
    }

    wrapper.appendChild(body);
    chatWindow.appendChild(wrapper);
    return body;
}

function sourcesHtml(sources, tokenUsage, answerText = "") {

    let parsedSources = sources || [];
    console.log("SOURCES FROM BACKEND:", parsedSources);

    if (typeof parsedSources === "string") {
        try {
            parsedSources = JSON.parse(parsedSources);
        } catch {
            parsedSources = [];
        }
    }

// Don't show sources next to a fallback / "I don't know" answer —
    // retrieval ran, but nothing it found was actually used.
    const suppressSources = isFallbackAnswer(answerText);

    const sourcesBlock = (parsedSources.length && !suppressSources)
        ? `<details class="sources-toggle">
                <summary>Sources (${parsedSources.length})</summary>

                ${parsedSources.map(s => {

                    // s.page comes in 3 different shapes depending on which
                    // backend branch answered this turn:
                    //   - a 0-indexed int from normal chunk retrieval -> show page+1
                    //   - a numeric STRING like "247" from the direct page-lookup
                    //     branch (get_answer's page_lookup path) -> this is already
                    //     the human-facing page number the user typed, so it's
                    //     shown as-is, NOT +1'd (that branch never had a 0-indexed
                    //     int to begin with, and treating it as one previously
                    //     always fell through to "Entire document" instead, since
                    //     `typeof "247" === "number"` is false).
                    //   - the literal string "full document" from the broad-summary
                    //     branch -> "Entire document".
                    let pageText = "Entire document";

                    if (typeof s.page === "number") {
                        pageText = "Page: " + (s.page + 1);
                    } else if (typeof s.page === "string" && /^\d+$/.test(s.page.trim())) {
                        pageText = "Page: " + s.page.trim();
                    }

                    return `
                        <div class="source-line">
                            <strong>📄 ${s.source}</strong><br>
                            ${pageText}
                        </div>
                    `;
                }).join("")}

            </details>`
        : "";

    const tokenBlock = (tokenUsage && !suppressSources)
        ? `<div class="token-caption">
                ${tokenUsage.prompt_tokens ?? 0} prompt ·
                ${tokenUsage.completion_tokens ?? 0} completion ·
                ${tokenUsage.total_tokens ?? 0} total
           </div>`
        : "";

    return sourcesBlock + tokenBlock;
}

function metricsHtml(metrics, answerText = "") {
    if (isFallbackAnswer(answerText)) return "";
    if (!metrics) metrics = {};

    const labels = {
        faithfulness: "Faithfulness",
        answer_relevancy: "Answer Relevance",
        context_precision: "Context Precision",
        context_relevancy: "Context Relevance",
    };

    const items = Object.entries(labels)
        .map(([key, label]) => {
            const val = metrics[key];
            const display = (val === null || val === undefined) ? "N/A" : val;
            return `<span class="metric-badge">${label}: ${display}</span>`;
        })
        .join("");

    return `<div class="metrics-row">${items}</div>`;
}

// True only if the metrics object actually has at least one computed value —
// distinguishes "never calculated yet" ({} from /chat) from "calculated but
// the judge failed for everything" ({faithfulness: null, ...}).
function hasAnyMetricValue(metrics) {
    if (!metrics) return false;
    return Object.values(metrics).some(v => v !== null && v !== undefined);
}

// Renders the "Calculate Metrics" button. Clicking it POSTs this exact
// turn's question/answer/sources to /metrics/calculate — metrics are only
// ever computed for a turn the user actually asks about, never automatically.
function renderCalculateButton(container, { question, answer, sources, turnId }) {
    const btn = document.createElement("button");
    btn.className = "calc-metrics-btn";
    btn.type = "button";
    btn.innerText = "Calculate Metrics";

    btn.onclick = async () => {
        btn.disabled = true;
        btn.innerText = "Calculating...";

        try {
            const response = await fetch("/metrics/calculate", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ question, answer, sources: sources || [], turn_id: turnId ?? null })
            });

            if (!response.ok) throw new Error(`HTTP ${response.status}`);

            const data = await response.json();
            container.innerHTML = metricsHtml(data.metrics, answer);
        } catch (err) {
            console.error("Error calculating metrics:", err);
            btn.disabled = false;
            btn.innerText = "Calculate Metrics (retry)";
        }
    };

    container.innerHTML = "";
    container.appendChild(btn);
}

// Appends the sources block + the metrics section (badges if already
// computed, otherwise the button) under a rendered assistant message body.
function appendSourcesAndMetrics(body, { question, answer, sources, tokenUsage, metrics, turnId }) {
    const sourcesDiv = document.createElement("div");
    sourcesDiv.innerHTML = sourcesHtml(sources, tokenUsage, answer);
    body.appendChild(sourcesDiv);
    if (isFallbackAnswer(answer)) return;
    const metricsContainer = document.createElement("div");
    metricsContainer.className = "metrics-container";
    body.appendChild(metricsContainer);
    if (hasAnyMetricValue(metrics)) {
        metricsContainer.innerHTML = metricsHtml(metrics, answer);
    } else {
        renderCalculateButton(metricsContainer, { question, answer, sources, turnId });
    }
}

function renderChatHistory(chatHistory) {
    chatWindow.innerHTML = "";
    (chatHistory || []).forEach(turn => {
        renderMessage("user", turn.question);
        const body = renderMessage("assistant", turn.answer);
        appendSourcesAndMetrics(body, {
            question: turn.question,
            answer: turn.answer,
            sources: turn.sources,
            tokenUsage: null,
            metrics: turn.metrics
        });
    });
    chatWindow.scrollTop = chatWindow.scrollHeight;
}

function showTypingIndicator() {
    const wrapper = document.createElement("div");
    wrapper.className = "msg-row msg-row-assistant";
    wrapper.id = "typingIndicator";

    const body = document.createElement("div");
    body.className = "assistant-message";
    body.innerHTML = `<div class="typing-dots"><span></span><span></span><span></span></div>`;
    wrapper.appendChild(body);

    chatWindow.appendChild(wrapper);
    chatWindow.scrollTop = chatWindow.scrollHeight;
}

function removeTypingIndicator() {
    const el = document.getElementById("typingIndicator");
    if (el) el.remove();
}


// ==================== Sessions ====================

async function loadSessionList() {
    try {
        const response = await fetch("/sessions");
        const sessions = await response.json();

        historyContainer.innerHTML = "";

        sessions.forEach(session => {
            const created = new Date(session.created_at);
            const names = session.pdf_names || [];
            const label = names.length === 0
                ? "New chat (no PDF yet)"
                : names.length === 1
                    ? names[0]
                    : `${names[0]} +${names.length - 1} more`;

            const row = document.createElement("div");
            row.className = "history-row";

            const card = document.createElement("div");
            card.className = "history-item" + (session.session_id === sessionId ? " active" : "");
            card.innerHTML = `<strong>${label}</strong><br><small>${created.toLocaleString()}</small>`;
            card.onclick = () => switchSession(session.session_id);

            const delBtn = document.createElement("button");
            delBtn.className = "delete-btn";
            delBtn.innerText = "🗑️";
            delBtn.onclick = async (e) => {
                e.stopPropagation();

                if (!confirm("Delete this chat and all uploaded documents?")) {
                    return;
                }

                try {
                    const response = await fetch(`/sessions/${session.session_id}`, {
                        method: "DELETE"
                    });

                    if (!response.ok) {
                        const err = await response.text();
                        throw new Error(err);
                    }

                    if (session.session_id === sessionId) {
                        sessionId = null;
                        chatWindow.innerHTML = "";
                        renderDocList([]);
                        uploadStatus.innerText = "";
                    }

                    await loadSessionList();

                } catch (err) {
                console.error(err);
                    alert("Failed to delete the chat.");
                }
            };

            row.appendChild(card);
            row.appendChild(delBtn);
            historyContainer.appendChild(row);
        });

    } catch (err) {
        console.error("Error loading sessions:", err);
    }
}

async function switchSession(id) {
    try {
        const response = await fetch(`/sessions/${id}`);
        if (!response.ok) return;
        const data = await response.json();

        sessionId = id;
        renderDocList(data.pdf_names);
        renderChatHistory(data.chat_history);
        uploadStatus.innerText = "";

        await loadSessionList(); // re-render so the active highlight moves
    } catch (err) {
        console.error("Error switching session:", err);
    }
}

newChatBtn.addEventListener("click", async () => {
    try {
        const response = await fetch("/sessions/new", { method: "POST" });
        const data = await response.json();
        sessionId = data.session_id;
        chatWindow.innerHTML = "";
        renderDocList([]);
        uploadStatus.innerText = "";
        await loadSessionList();
    } catch (err) {
        console.error("Error creating new chat:", err);
    }
});


// ==================== Upload file(s) ====================

pdfInput.addEventListener("change", async () => {
    const files = pdfInput.files;
    if (!files || files.length === 0) return;

    const formData = new FormData();
    for (const file of files) {
        formData.append("files", file);
    }
    if (sessionId) {
        formData.append("session_id", sessionId);
    }

    uploadStatus.innerText = "Uploading...";

    try {
        const response = await fetch("/upload", {
            method: "POST",
            body: formData
        });

        const data = await response.json();
        sessionId = data.session_id;

        const names = data.files.map(f => f.filename).join(", ");
        uploadStatus.innerText = `:) Added : ${names}`;

        const detail = await (await fetch(`/sessions/${sessionId}`)).json();
        renderDocList(detail.pdf_names);

        await loadSessionList();
        pdfInput.value = "";
    } catch (err) {
        console.error(err);
        uploadStatus.innerText = ":( Upload failed";
    }
});


// ==================== Chat ====================

sendButton.addEventListener("click", sendMessage);

chatInput.addEventListener("keypress", function (e) {
    if (e.key === "Enter") {
        sendMessage();
    }
});

async function sendMessage() {
    const question = chatInput.value.trim();
    if (!question) return;

    if (!sessionId) {
        alert("Please upload a PDF or photo first.");
        return;
    }

    renderMessage("user", question);
    chatInput.value = "";
    chatWindow.scrollTop = chatWindow.scrollHeight;

    showTypingIndicator();

    try {
        const response = await fetch("/chat", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ session_id: sessionId, question: question })
        });

        const data = await response.json();

        removeTypingIndicator();

        if (data.error) {
            renderMessage("assistant", `❌ ${data.error}`);
        } else {
            const body = renderMessage("assistant", data.answer);
            appendSourcesAndMetrics(body, {
                question: question,
                answer: data.answer,
                sources: data.sources,
                tokenUsage: data.token_usage,
                metrics: data.metrics,
                turnId: data.turn_id
            });
        }

        chatWindow.scrollTop = chatWindow.scrollHeight;
    } catch (err) {
        console.error(err);
        removeTypingIndicator();
        renderMessage("assistant", "❌ Error getting response.");
    }
}


// ==================== Initial Load ====================
(async function init() {
    await loadSessionList();
    try {
        const response = await fetch("/sessions");
        const sessions = await response.json();
        if (sessions.length > 0) {
            await switchSession(sessions[0].session_id);
        }
    } catch (err) {
        console.error("Error on initial load:", err);
    }
})();


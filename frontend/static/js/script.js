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
// the Sources panel for them (retrieval still ran, but nothing useful
// was found, so showing "Sources" next to "I don't know" is misleading).
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

function renderMessage(role, content, extraHtml = "") {
    const wrapper = document.createElement("div");
    wrapper.className = role === "user" ? "msg-row msg-row-user" : "msg-row msg-row-assistant";

    if (role === "user") {
        const bubble = document.createElement("div");
        bubble.className = "user-message";
        bubble.innerHTML = content;
        wrapper.appendChild(bubble);
    } else {
        const body = document.createElement("div");
        body.className = "assistant-message";
        body.innerHTML = marked.parse(content) + extraHtml;
        wrapper.appendChild(body);
    }

    chatWindow.appendChild(wrapper);
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

                    let pageText = "Entire document";

                    if (typeof s.page === "number") {
                        pageText = "Page: " + (s.page + 1);
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

function renderChatHistory(chatHistory) {
    chatWindow.innerHTML = "";
    (chatHistory || []).forEach(turn => {
        renderMessage("user", turn.question);
        renderMessage("assistant", turn.answer, sourcesHtml(turn.sources, null, turn.answer) + metricsHtml(turn.metrics, turn.answer));
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
            renderMessage("assistant", data.answer, sourcesHtml(data.sources, data.token_usage, data.answer) + metricsHtml(data.metrics, data.answer));
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
// Client state management
let sessions = {};
let activeSessionId = null;

const quotes = [
    ["Focus on being productive instead of busy.", "Tim Ferriss"],
    ["The secret of getting ahead is getting started.", "Mark Twain"],
    ["Action is the foundational key to all success.", "Pablo Picasso"],
    ["It always seems impossible until it's done.", "Nelson Mandela"],
    ["Efficiency is doing things right; effectiveness is doing the right things.", "Peter Drucker"],
    ["Productivity is never an accident. It is always the result of a commitment to excellence.", "Paul J. Meyer"]
];

// Find and wrap query keywords with styled <mark> tags in text snippets
function highlightKeywords(text, query) {
    if (!text || !query) return text;

    // Clean and tokenise the query to find significant keywords
    const cleanQuery = query.toLowerCase()
        .replace(/[.,\/#!$%\^&\*;:{}=\-_`~()?]/g, " ")
        .replace(/\s+/g, " ")
        .trim();

    const words = cleanQuery.split(" ")
        .filter(w => w.length > 2)
        .filter(w => !["the", "and", "for", "are", "you", "with", "this", "that", "from", "these", "those", "what", "which", "who", "whom", "here", "there", "their", "they", "them", "then", "than", "your", "were", "been", "have", "has", "had", "does", "done", "will", "would", "shall", "should", "can", "could", "may", "might", "must", "about", "above", "below", "under", "over", "only", "other", "some", "such"].includes(w));

    if (words.length === 0) return text;

    // De-duplicate words
    const uniqueWords = Array.from(new Set(words));

    // Sort words by length descending to match longer words first in the regex alternation
    uniqueWords.sort((a, b) => b.length - a.length);

    // Escape regex special characters
    const escapedWords = uniqueWords.map(w => w.replace(/[-\/\\^$*+?.()|[\]{}]/g, '\\$&'));

    // Create a single regex with alternation of all keywords
    const regex = new RegExp(`\\b((${escapedWords.join("|")})[a-zA-Z]*)\\b`, "gi");

    return text.replace(regex, '<mark class="highlighted-term" style="background-color: rgba(251, 191, 36, 0.4); color: inherit; border-radius: 2px; padding: 0 2px; font-weight: 500;">$1</mark>');
}

// Convert basic markdown to HTML for chat rendering
function markdownToHtml(text, sources, msgIdx) {
    if (!text) return "";
    // Escape HTML entities first
    let html = text
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;");

    // Bold: **text** or __text__
    html = html.replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>");
    html = html.replace(/__(.+?)__/g, "<strong>$1</strong>");
    // Italic: *text* or _text_ (not inside bold)
    html = html.replace(/(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)/g, "<em>$1</em>");

    // Citations: [1], [2], etc.
    if (sources && sources.length > 0) {
        html = html.replace(/\[(\d+)\]/g, (match, p1) => {
            const idx = parseInt(p1, 10);
            const src = sources.find(s => s.index === idx);
            if (src) {
                const docName = src.source_name || "Source";
                const pageLabel = src.page !== undefined && src.page !== null && src.page !== 'N/A' ? `, Page ${src.page}` : "";
                const displayName = `${docName}${pageLabel}`;
                return `<a href="#" class="citation-link" onclick="highlightSource(${idx}, event, ${msgIdx})">[${displayName}]</a>`;
            }
            return `<a href="#" class="citation-link" onclick="highlightSource(${idx}, event, ${msgIdx})">[${idx}]</a>`;
        });
    } else {
        html = html.replace(/\[(\d+)\]/g, '<a href="#" class="citation-link" onclick="highlightSource($1, event)">[$1]</a>');
    }

    // --- List / heading handling ---
    html = html.replace(/^[\*\-]\s*$/gm, "<li></li>");
    html = html.replace(/^\s{4,}[\*\-] (.+)$/gm, "<li class='nested'>$1</li>");
    html = html.replace(/^\d+\.\s+(.+)$/gm, "<li class='numbered'>$1</li>");
    html = html.replace(/^[\*\-] (.+)$/gm, "<li>$1</li>");
    html = html.replace(/<li(?: class='(?:nested|numbered)')?>\s*<\/li>\s*\n?/g, "");
    html = html.replace(/^(?!<li)(.+:)\s*$/gm, "<div class='list-heading'><strong>$1</strong></div>");
    html = html.replace(/(<div class='list-heading'>.*?<\/div>)\s*\n/g, "$1");

    const collapseListWhitespace = (match) => match.replace(/>\s+</g, "><").replace(/\s+$/, "");

    html = html.replace(/(?:<li class='numbered'>.*?<\/li>\s*)+/gs, (m) => `<ol class="rag-ol">${collapseListWhitespace(m)}</ol>`);
    html = html.replace(/(?:<li(?: class='nested')?>.*?<\/li>\s*)+/gs, (m) => `<ul class="rag-ul">${collapseListWhitespace(m)}</ul>`);

    // Line breaks
    html = html.replace(/\n{2,}/g, "</p><p>");
    html = html.replace(/\n/g, "<br>");
    return `<p>${html}</p>`;
}

// Helper to determine greeting based on hour
function getGreeting() {
    const hour = new Date().getHours();
    if (hour < 12) return "Good morning";
    if (hour < 17) return "Good afternoon";
    return "Good evening";
}

// Initialize application
document.addEventListener("DOMContentLoaded", () => {
    // Setup UI listeners
    setupUIIEventListeners();

    // Fetch sessions from server-side persistent file
    fetch("/sessions")
        .then(res => res.json())
        .then(data => {
            sessions = data || {};
            // Render initial states once data is loaded
            startNewChatState();
            renderRecentSessions();
        })
        .catch(err => {
            console.error("Error loading chat history:", err);
            sessions = {};
            startNewChatState();
            renderRecentSessions();
        });

    fetchDbCount();
    fetchTokenUsage();
});

// Setup DOM event listeners
function setupUIIEventListeners() {
    // New Chat Button
    document.getElementById("btn-new-chat").addEventListener("click", () => {
        startNewChatState();
    });

    // Landing Page Submit Click & Enter
    document.getElementById("btn-submit-landing").addEventListener("click", () => {
        handleQuerySubmit(document.getElementById("landing-search-input").value, true);
    });
    document.getElementById("landing-search-input").addEventListener("keydown", (e) => {
        if (e.key === "Enter") {
            handleQuerySubmit(e.target.value, true);
        }
    });

    // Chat Page Submit Click & Enter
    document.getElementById("btn-submit-chat").addEventListener("click", () => {
        handleQuerySubmit(document.getElementById("chat-input").value, false);
    });
    document.getElementById("chat-input").addEventListener("keydown", (e) => {
        if (e.key === "Enter") {
            handleQuerySubmit(e.target.value, false);
        }
    });

    // Drag and drop PDF setup
    const dropZone = document.getElementById("drop-zone");
    const fileInput = document.getElementById("file-input");
    const fileInfoContainer = document.getElementById("file-info-container");
    const fileNameDiv = document.getElementById("file-name");
    const uploadStatusDiv = document.getElementById("upload-status");

    dropZone.addEventListener("click", () => fileInput.click());

    dropZone.addEventListener("dragover", (e) => {
        e.preventDefault();
        dropZone.style.borderColor = "#ffffff";
    });

    dropZone.addEventListener("dragleave", () => {
        dropZone.style.borderColor = "var(--border-color)";
    });

    dropZone.addEventListener("drop", (e) => {
        e.preventDefault();
        dropZone.style.borderColor = "var(--border-color)";
        if (e.dataTransfer.files.length > 0) {
            fileInput.files = e.dataTransfer.files;
            handleFileSelection();
        }
    });

    fileInput.addEventListener("change", () => {
        if (fileInput.files.length > 0) {
            handleFileSelection();
        }
    });

    // Action button to run Ingestion API
    document.getElementById("btn-index").addEventListener("click", () => {
        if (fileInput.files.length === 0) return;

        const file = fileInput.files[0];
        const formData = new FormData();
        formData.append("file", file);

        uploadStatusDiv.innerHTML = '<span class="spinner"></span> Indexing file...';
        document.getElementById("btn-index").disabled = true;

        fetch("/upload", {
            method: "POST",
            body: formData
        })
            .then(res => res.json())
            .then(data => {
                if (data.success) {
                    uploadStatusDiv.innerHTML = `<span style="color: #22c55e;">🎉 Indexed ${data.chunks} chunks!</span>`;
                    fetchDbCount();
                    // Reset file picker
                    fileInput.value = "";
                    fileInfoContainer.style.display = "none";
                } else {
                    const errMsg = data.detail || data.error || "Unknown server error";
                    uploadStatusDiv.innerHTML = `<span style="color: #ef4444;">⚠️ Error: ${errMsg}</span>`;
                }
            })
            .catch(err => {
                uploadStatusDiv.innerHTML = `<span style="color: #ef4444;">⚠️ Ingestion failed: ${err.message}</span>`;
            })
            .finally(() => {
                document.getElementById("btn-index").disabled = false;
            });
    });

    // Clear Document & Cache button
    document.getElementById("btn-clear-db").addEventListener("click", () => {
        const clearStatus = document.getElementById("clear-status");
        const btn = document.getElementById("btn-clear-db");

        if (!confirm("⚠️ This will permanently delete all indexed chunks and vectors from the database. Continue?")) return;

        clearStatus.innerHTML = '<span class="spinner"></span> Clearing...';
        btn.disabled = true;

        fetch("/clear", { method: "POST" })
            .then(res => res.json())
            .then(data => {
                if (data.success) {
                    clearStatus.innerHTML = `<span style="color: #22c55e;">✅ Document & cache cleared!</span>`;
                    // Reset chunk count badge
                    document.getElementById("badge-chunk-count").textContent = "Chunks: 0";
                    // Clear any upload status messages
                    document.getElementById("upload-status").innerHTML = "";
                    document.getElementById("file-info-container").style.display = "none";
                    document.getElementById("file-input").value = "";

                    // Hide active document info box
                    const activeDocInfo = document.getElementById("active-doc-info");
                    if (activeDocInfo) {
                        activeDocInfo.style.display = "none";
                    }

                    // Go back to landing page but keep sessions intact
                    startNewChatState();
                    renderRecentSessions();

                    // Auto-hide success message after 3 seconds
                    setTimeout(() => { clearStatus.innerHTML = ""; }, 3000);
                } else {
                    clearStatus.innerHTML = `<span style="color: #ef4444;">⚠️ Failed to clear.</span>`;
                }
            })
            .catch(err => {
                clearStatus.innerHTML = `<span style="color: #ef4444;">⚠️ Error: ${err.message}</span>`;
            })
            .finally(() => {
                btn.disabled = false;
            });
    });
}

// Fetch total chunks in Database
function fetchDbCount() {
    fetch("/db_count")
        .then(res => res.json())
        .then(data => {
            document.getElementById("badge-chunk-count").textContent = `Chunks: ${data.count}`;

            // Show/hide currently active document info in the sidebar
            const activeDocInfo = document.getElementById("active-doc-info");
            const activeDocName = document.getElementById("active-doc-name");
            if (activeDocInfo && activeDocName) {
                if (data.count > 0 && data.source) {
                    activeDocName.textContent = data.source;
                    activeDocInfo.style.display = "block";
                } else {
                    activeDocInfo.style.display = "none";
                }
            }
        })
        .catch(err => console.error("Error fetching db count:", err));
}

// Fetch token usage stats from backend
function fetchTokenUsage() {
    fetch("/token_usage")
        .then(res => res.json())
        .then(data => updateTokenUI(data))
        .catch(err => console.error("Error fetching token usage:", err));
}

// Update the token dropup panel with live data
function updateTokenUI(data) {
    const fmt = n => n.toLocaleString();
    const pct = data.pct_used || 0;

    // Trigger button label
    document.getElementById("token-trigger-label").textContent =
        `Tokens: ${fmt(data.total_used)} used`;

    // Panel header
    const providerIcon = data.provider === "groq" ? "⚡" : "✨";
    const providerName = data.provider.charAt(0).toUpperCase() + data.provider.slice(1);
    document.getElementById("token-provider-label").textContent =
        `${providerIcon} ${providerName} · ${data.model}`;
    document.getElementById("token-requests").textContent = `${data.requests} request${data.requests !== 1 ? "s" : ""}`;

    // Progress bar
    const bar = document.getElementById("token-progress-bar");
    bar.style.width = `${Math.min(pct, 100)}%`;
    bar.style.background = pct < 50 ? "#22c55e" : pct < 80 ? "#f59e0b" : "#ef4444";

    // Stats
    document.getElementById("token-used").textContent = fmt(data.total_used);
    document.getElementById("token-remaining").textContent = fmt(data.remaining);
    document.getElementById("token-limit").textContent = fmt(data.daily_limit);
    document.getElementById("token-prompt").textContent = fmt(data.prompt_tokens);
    document.getElementById("token-completion").textContent = fmt(data.completion_tokens);

    // Header badge
    const badge = document.getElementById("badge-model");
    if (badge) badge.textContent = `${data.provider} · ${data.model.split("-").slice(0, 3).join("-")}`;
}

// Toggle the token dropup panel open/close
window.toggleTokenPanel = function () {
    const panel = document.getElementById("token-panel");
    const dropup = document.getElementById("token-dropup");
    const isOpen = dropup.classList.toggle("open");
    if (isOpen) fetchTokenUsage(); // Refresh when opening
};

// Update file selection container UI
function handleFileSelection() {
    const file = document.getElementById("file-input").files[0];
    document.getElementById("file-name").textContent = file.name;
    document.getElementById("file-info-container").style.display = "block";
    document.getElementById("upload-status").innerHTML = "";
}

// Reset workspace to Centered Landing state
function startNewChatState() {
    activeSessionId = null;
    document.getElementById("landing-search-input").value = "";
    document.getElementById("chat-input").value = "";

    // Pick greeting & quotes
    document.getElementById("landing-greeting").textContent = `${getGreeting()}, Nikhilesh`;
    const randomQuote = quotes[Math.floor(Math.random() * quotes.length)];
    document.getElementById("landing-quote").textContent = `“${randomQuote[0]}” — ${randomQuote[1]}`;

    // Toggle Viewports
    document.getElementById("landing-view").style.display = "flex";
    document.getElementById("chat-view").style.display = "none";

    renderRecentSessions();
}

// Save conversations to server-side persistent file
function saveSessionsToServer() {
    fetch("/sessions", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ sessions: sessions })
    }).catch(err => console.error("Error saving chat history to server:", err));
}

// Submit user query to backend pipeline
function handleQuerySubmit(queryText, isInitial) {
    if (!queryText.trim()) return;

    // Create new session if initial, or append to active session
    if (isInitial || !activeSessionId) {
        activeSessionId = "session_" + Date.now();
        sessions[activeSessionId] = {
            title: queryText.trim(),
            messages: []
        };
        // Switch viewport to Chat view
        document.getElementById("landing-view").style.display = "none";
        document.getElementById("chat-view").style.display = "flex";
    }

    // Clear inputs
    document.getElementById("landing-search-input").value = "";
    document.getElementById("chat-input").value = "";

    const session = sessions[activeSessionId];
    session.messages.push({
        role: "user",
        content: queryText
    });
    saveSessionsToServer();

    // Render current message log
    renderActiveSessionMessages();

    // Append loading assistant bubble
    const messagesContainer = document.getElementById("chat-messages");
    const loadingBubble = document.createElement("div");
    loadingBubble.className = "chat-bubble-row";
    loadingBubble.id = "loading-bubble";
    loadingBubble.innerHTML = `
        <div class="chat-bubble-label">Assistant</div>
        <div class="chat-bubble-body"><span class="spinner"></span> Thinking...</div>
    `;
    messagesContainer.appendChild(loadingBubble);
    messagesContainer.scrollTop = messagesContainer.scrollHeight;

    // The index where the assistant message will be pushed
    const assistantIndex = session.messages.length;

    // Hit backend RAG pipeline API
    fetch("/query", {
        method: "POST",
        headers: {
            "Content-Type": "application/json"
        },
        body: JSON.stringify({
            query: queryText,
            session_id: activeSessionId,
            message_index: assistantIndex
        })
    })
        .then(res => res.json().then(data => ({ ok: res.ok, status: res.status, data })))
        .then(({ ok, status, data }) => {
            // Remove loading indicator bubble
            const loader = document.getElementById("loading-bubble");
            if (loader) loader.remove();

            let answerContent;
            let sources = [];

            if (!ok || !data.answer) {
                // API returned an error (e.g. 500/503 from API overload)
                const detail = data.detail || "Unknown error";
                if (detail.includes("503") || detail.includes("high demand") || detail.includes("UNAVAILABLE")) {
                    answerContent = "⚠️ The AI model is currently busy. Please try again in a few seconds.";
                } else {
                    answerContent = `⚠️ Error: ${detail}`;
                }
            } else {
                answerContent = data.answer;
                sources = data.sources || [];
            }

            session.messages.push({
                role: "assistant",
                content: answerContent,
                sources: sources,
                eval_contexts: data.eval_contexts || [],
                observability: data.observability || null
            });

            // Save to server
            saveSessionsToServer();

            // Re-render + refresh token counter
            renderActiveSessionMessages();
            renderRecentSessions();
            fetchTokenUsage();
        })
        .catch(err => {
            const loader = document.getElementById("loading-bubble");
            if (loader) loader.remove();

            session.messages.push({
                role: "assistant",
                content: `⚠️ Network error: ${err.message}. Please check your connection and try again.`,
                sources: []
            });
            renderActiveSessionMessages();
        });
}

// Render active conversation logs
function renderActiveSessionMessages() {
    const messagesContainer = document.getElementById("chat-messages");
    messagesContainer.innerHTML = "";

    if (!activeSessionId || !sessions[activeSessionId]) return;

    const messages = sessions[activeSessionId].messages;
    messages.forEach((msg, idx) => {
        const bubble = document.createElement("div");
        bubble.className = "chat-bubble-row";

        const labelText = msg.role === "user" ? "You" : "Assistant";
        let contextHtml = "";

        // If assistant has source citations, render collapsible expander and observability details
        if (msg.role === "assistant") {
            let obsHtml = "";
            if (msg.observability) {
                const obs = msg.observability;
                const cachedLabel = obs.cached ? "<span class='badge-cache' style='background:#3b82f6; color:#fff; padding:2px 6px; border-radius:4px; font-size:0.7rem; font-weight:bold; margin-right:8px;'>CACHED</span>" : "";
                obsHtml = `
                    <div class="observability-bar" style="margin-top:6px; font-size:0.75rem; color:var(--text-muted); display:flex; flex-wrap:wrap; gap:12px; align-items:center; border-top: 1px solid var(--border-color); padding-top:6px;">
                        ${cachedLabel}
                        <button class="btn-eval" id="btn-eval-${idx}" onclick="evaluateAnswer(${idx})" style="background:none; border:none; color:#3b82f6; cursor:pointer; font-weight:bold; font-size:0.75rem; padding:0; margin-left:auto;">📊 Evaluate RAG</button>
                    </div>
                    <div class="eval-result-box" id="eval-result-${idx}" style="display:none; background:rgba(0,0,0,0.05); padding:8px; border-radius:6px; margin-top:6px;"></div>
                `;
            }

            let sourcesHtml = "";
            if (msg.sources && msg.sources.length > 0) {
                let sourcesContent = "";
                const userQuery = idx > 0 ? messages[idx - 1].content : "";

                msg.sources.forEach(src => {
                    const confidenceColor = src.confidence === "HIGH" ? "#22c55e" : src.confidence === "MEDIUM" ? "#f59e0b" : "#ef4444";
                    const citation = src.citation || `[Source: Page ${src.page}]`;
                    const snippet = src.text ? src.text.trim() : "";
                    const highlightedSnippet = highlightKeywords(snippet, userQuery);

                    sourcesContent += `
                        <div class="source-item" id="source-${idx}-${src.index}" style="margin-bottom:12px; padding:6px; border-bottom:1px dashed var(--border-color);">
                            <div class="source-item-header" style="display:flex; align-items:center; font-size:0.8rem; font-weight:bold;">
                                <span class="source-index-num" style="color:#3b82f6; margin-right:6px;">[${src.index}]</span>
                                <span class="source-citation">${citation}</span>
                                <span class="confidence-badge" style="background:${confidenceColor}; color:#fff; padding:1px 5px; border-radius:3px; font-size:0.65rem; font-weight:bold; margin-left:auto;">${src.confidence}</span>
                            </div>
                            <div class="monospace-box" style="margin-top:4px; font-size:0.75rem; color:var(--text-main); font-family:monospace; background:rgba(0,0,0,0.08); padding:6px; border-radius:4px; line-height:1.3;">"${highlightedSnippet}"</div>
                        </div>
                    `;
                });

                sourcesHtml = `
                    <div class="source-expander" id="expander-${idx}" style="margin-top:8px; border:1px solid var(--border-color); border-radius:6px; padding:6px 10px; background:rgba(255,255,255,0.02);">
                        <span class="source-expander-header" onclick="toggleSourceExpander(${idx})" style="cursor:pointer; font-size:0.8rem; font-weight:bold; color:#3b82f6; display:flex; align-items:center; justify-content:space-between; user-select:none;">
                            <span>🔍 Show Retrieved Context Sources (${msg.sources.length})</span>
                            <span class="expander-arrow" style="font-size:0.6rem;">▼</span>
                        </span>
                        <div class="source-expander-content" style="display:none; margin-top:8px; border-top:1px solid var(--border-color); padding-top:8px;">
                            ${sourcesContent}
                        </div>
                    </div>
                `;
            }

            contextHtml = obsHtml + sourcesHtml;
        }

        const renderedContent = msg.role === "assistant"
            ? markdownToHtml(msg.content, msg.sources, idx)
            : `<p>${msg.content.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/\n/g, "<br>")}</p>`;

        bubble.innerHTML = `
            <div class="chat-bubble-label">${labelText}</div>
            <div class="chat-bubble-body">${renderedContent}</div>
            ${contextHtml}
        `;
        messagesContainer.appendChild(bubble);
    });

    messagesContainer.scrollTop = messagesContainer.scrollHeight;
}

// Helper to open/close collapsible details
window.toggleSourceExpander = function (index) {
    const expander = document.getElementById(`expander-${index}`);
    if (expander) {
        const content = expander.querySelector(".source-expander-content");
        const arrow = expander.querySelector(".expander-arrow");
        const isOpen = content.style.display === "block";
        content.style.display = isOpen ? "none" : "block";
        arrow.textContent = isOpen ? "▼" : "▲";
    }
}

// Helper to evaluate chatbot response via LLM-as-a-judge
window.evaluateAnswer = function (messageIndex) {
    const session = sessions[activeSessionId];
    if (!session) return;
    const msg = session.messages[messageIndex];
    if (!msg) return;

    const evalBtn = document.getElementById(`btn-eval-${messageIndex}`);
    if (evalBtn) {
        evalBtn.disabled = true;
        evalBtn.style.opacity = "0.5";
        evalBtn.style.cursor = "default";
        evalBtn.innerHTML = "⏳ Evaluating...";
    }

    const resultBox = document.getElementById(`eval-result-${messageIndex}`);
    resultBox.style.display = "block";
    resultBox.innerHTML = `<span class="spinner" style="border:2px solid #3b82f6; border-top:2px solid transparent; width:12px; height:12px; border-radius:50%; display:inline-block; animation:spin 1s linear infinite; vertical-align:middle; margin-right:6px;"></span> Evaluating response quality...`;

    // Retrieve full context strings if stored in state, otherwise map from sources
    const contexts = msg.eval_contexts && msg.eval_contexts.length > 0
        ? msg.eval_contexts
        : (msg.sources || []).map(s => s.text);  // fallback for old cached sessions

    // REQUIRED AUDIT LOGS: print state details before executing evaluate
    console.log(`[EVALUATE] Number of contexts: ${contexts.length}`);
    contexts.forEach((c, idx) => {
        console.log(`  [Context ${idx + 1}] Length: ${c ? c.length : 0}, First 100 chars: "${c ? c.substring(0, 100).replace(/\n/g, ' ') : ""}"`);
    });
    console.log(`[EVALUATE] Whether contexts are empty: ${contexts.length === 0}`);

    fetch("/evaluate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
            query: session.messages[messageIndex - 1]?.content || "",
            answer: msg.content,
            contexts: contexts,
            session_id: activeSessionId,
            message_index: messageIndex
        })
    })
        .then(res => {
            if (!res.ok) {
                return res.json().then(errData => {
                    throw new Error(errData.detail || "Server returned an error");
                });
            }
            return res.json();
        })
        .then(data => {
            if (evalBtn) {
                evalBtn.style.display = "none"; // Hide button once successfully evaluated
            }

            let metricsHtml = "";
            const metricNames = {
                faithfulness: "Faithfulness",
                relevance: "Answer Relevance",
                context_precision: "Context Precision",
                context_relevance: "Context Relevance",
                context_recall: "Context Recall",
                answer_correctness: "Answer Correctness"
            };

            for (const key in metricNames) {
                const score = data[key];
                let scoreText = "";
                let color = "var(--text-muted)";

                if (score !== undefined && score !== null) {
                    // Scores outside this range are from a stale/legacy API
                    // response (the current evaluator contract is 0.0 to 1.0).
                    if (typeof score !== "number" || score < 0 || score > 1) {
                        scoreText = "Invalid score — restart API";
                        metricsHtml += `<span style="margin-right: 16px; display: inline-block; white-space: nowrap;">${metricNames[key]}: <span style="color:#ef4444; font-weight:bold;">${scoreText}</span></span>`;
                        continue;
                    }
                    // Evaluator returns normalized fractional scores (0.0 to 1.0).
                    scoreText = score;
                    if (key === "noise_sensitivity") {
                        color = score <= 0.2 ? "#22c55e" : score <= 0.6 ? "#f59e0b" : "#ef4444";
                    } else {
                        color = score >= 0.8 ? "#22c55e" : score >= 0.4 ? "#f59e0b" : "#ef4444";
                    }
                } else {
                    scoreText = "N/A";
                }

                metricsHtml += `<span style="margin-right: 16px; display: inline-block; white-space: nowrap;">${metricNames[key]}: <span style="color:${color}; font-weight:bold;">${scoreText}</span></span>`;
            }

            let notesHtml = "";
            if (data.notes && data.notes.length > 0) {
                notesHtml = `<div style="font-size:0.7rem; color:var(--text-muted); margin-top:6px; border-top: 1px dashed var(--border-color); padding-top:4px; line-height: 1.3;">${data.notes.map(n => `ℹ️ ${n}`).join("<br>")}</div>`;
            }

            resultBox.innerHTML = `
            <div style="font-size:0.75rem;">
                <div style="display:flex; flex-wrap:wrap; gap:12px; margin-bottom:6px; font-weight:bold; border-bottom: 1px dashed var(--border-color); padding-bottom:4px; line-height: 1.5;">
                    ${metricsHtml}
                </div>
                ${notesHtml}
            </div>
        `;
        })
        .catch(err => {
            if (evalBtn) {
                evalBtn.disabled = false;
                evalBtn.style.opacity = "1";
                evalBtn.style.cursor = "pointer";
                evalBtn.innerHTML = "📊 Evaluate RAG";
            }
            resultBox.innerHTML = `<span style="color:#ef4444; font-size:0.75rem;">⚠️ Evaluation failed: ${err.message}</span>`;
        });
}

// Clickable citation link target highlighter
window.highlightSource = function (index, event, msgIdx) {
    if (event) event.preventDefault();

    // Find target using the msgIdx-scoped ID if available
    const targetId = msgIdx !== undefined ? `source-${msgIdx}-${index}` : `source-${index}`;
    let targetElement = document.getElementById(targetId);

    // Fallback if not found
    if (!targetElement) {
        const sourceItems = document.querySelectorAll(`[id^="source-"]`);
        sourceItems.forEach(el => {
            if (el.id.endsWith(`-${index}`)) {
                targetElement = el;
            }
        });
    }

    if (targetElement) {
        const expander = targetElement.closest(".source-expander");
        if (expander) {
            const content = expander.querySelector(".source-expander-content");
            const arrow = expander.querySelector(".expander-arrow");
            content.style.display = "block";
            arrow.textContent = "▲";
        }
        targetElement.scrollIntoView({ behavior: "smooth", block: "center" });
        targetElement.style.transition = "background-color 0.5s ease";
        targetElement.style.backgroundColor = "rgba(59, 130, 246, 0.25)";
        setTimeout(() => {
            targetElement.style.backgroundColor = "";
        }, 2000);
    }
}

// Render list of recent chat links in sidebar
function renderRecentSessions() {
    const listContainer = document.getElementById("recent-conversations-list");
    listContainer.innerHTML = "";

    const sessionIds = Object.keys(sessions).reverse();

    if (sessionIds.length === 0) {
        listContainer.innerHTML = `<div style="font-size:0.8rem; color:var(--text-muted); padding-left:10px;">No previous chats yet</div>`;
        return;
    }

    sessionIds.forEach(id => {
        const wrapper = document.createElement("div");
        wrapper.className = "recent-chat-wrapper";

        const item = document.createElement("button");
        item.className = "recent-chat-item";
        if (id === activeSessionId) item.classList.add("active");

        const title = sessions[id].title;
        item.textContent = title.length > 20 ? `💬 ${title.substring(0, 20)}...` : `💬 ${title}`;

        item.addEventListener("click", () => {
            activeSessionId = id;
            document.getElementById("landing-view").style.display = "none";
            document.getElementById("chat-view").style.display = "flex";
            renderActiveSessionMessages();
            renderRecentSessions();
        });

        const actionsDiv = document.createElement("div");
        actionsDiv.className = "recent-chat-actions";

        const editBtn = document.createElement("button");
        editBtn.className = "btn-action";
        editBtn.title = "Rename Chat";
        editBtn.innerHTML = "✏️";
        editBtn.addEventListener("click", (e) => {
            e.stopPropagation();
            const newTitle = prompt("Rename conversation:", sessions[id].title);
            if (newTitle !== null) {
                const trimmedTitle = newTitle.trim();
                if (trimmedTitle) {
                    sessions[id].title = trimmedTitle;
                    saveSessionsToServer();
                    renderRecentSessions();
                }
            }
        });

        const deleteBtn = document.createElement("button");
        deleteBtn.className = "btn-action";
        deleteBtn.title = "Delete Chat";
        deleteBtn.innerHTML = "🗑️";
        deleteBtn.addEventListener("click", (e) => {
            e.stopPropagation();
            if (confirm(`Are you sure you want to delete the chat "${sessions[id].title}"?`)) {
                delete sessions[id];
                saveSessionsToServer();
                if (activeSessionId === id) {
                    startNewChatState();
                } else {
                    renderRecentSessions();
                }
            }
        });

        actionsDiv.appendChild(editBtn);
        actionsDiv.appendChild(deleteBtn);

        wrapper.appendChild(item);
        wrapper.appendChild(actionsDiv);

        listContainer.appendChild(wrapper);
    });
}

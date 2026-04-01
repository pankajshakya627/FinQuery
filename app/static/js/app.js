/**
 * HDFC RAG Assistant — Client-side Chat Controller
 * Handles: query submission, message rendering, pipeline animation, health polling
 */

const API_BASE = '';
let isLoading = false;
let conversationId = null;

// ── DOM Elements ──
const chatArea = document.getElementById('chatArea');
const messagesContainer = document.getElementById('messagesContainer');
const welcomeScreen = document.getElementById('welcomeScreen');
const queryInput = document.getElementById('queryInput');
const sendBtn = document.getElementById('sendBtn');

// ── Initialize ──
document.addEventListener('DOMContentLoaded', () => {
    queryInput.addEventListener('keydown', handleKeyDown);
    queryInput.addEventListener('input', handleInput);
    checkHealth();
    setInterval(checkHealth, 30000);
});

// ── Input Handling ──
function handleKeyDown(e) {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        sendQuery();
    }
}

function handleInput() {
    // Auto-resize textarea
    queryInput.style.height = 'auto';
    queryInput.style.height = Math.min(queryInput.scrollHeight, 160) + 'px';

    // Character count
    const count = queryInput.value.length;
    document.getElementById('charCount').textContent = `${count} / 2048`;
}

// ── Send Query ──
async function sendQuery() {
    const question = queryInput.value.trim();
    if (!question || isLoading) return;

    isLoading = true;
    sendBtn.disabled = true;

    // Hide welcome, show messages
    if (welcomeScreen) welcomeScreen.style.display = 'none';

    // Add user message
    appendMessage('user', question);
    queryInput.value = '';
    queryInput.style.height = 'auto';
    document.getElementById('charCount').textContent = '0 / 2048';

    // Show loading
    const loadingEl = appendLoading();

    // Animate pipeline
    animatePipeline();

    try {
        const response = await fetch(`${API_BASE}/api/query`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                question: question,
                top_k: 5,
                similarity_threshold: 0.3,
                include_sources: true,
                conversation_id: conversationId,
            }),
        });

        const data = await response.json();

        // Remove loading
        loadingEl.remove();
        resetPipeline();

        // Add assistant message
        appendMessage('assistant', data.answer, data.sources, data.retrieval_stats, data.processing_time_ms);

        // Track conversation
        if (data.conversation_id) {
            conversationId = data.conversation_id;
        }

    } catch (err) {
        loadingEl.remove();
        resetPipeline();
        appendMessage('assistant', `⚠️ Error: ${err.message}. Make sure the FastAPI server is running.`);
    }

    isLoading = false;
    sendBtn.disabled = false;
    queryInput.focus();
}

// ── Ask from suggested question ──
function askQuestion(question) {
    queryInput.value = question;
    sendQuery();
}

// ── Message Rendering ──
function appendMessage(role, content, sources = null, stats = null, timeMs = null) {
    const msgDiv = document.createElement('div');
    msgDiv.className = `message ${role}`;

    const bubble = document.createElement('div');
    bubble.className = 'message-bubble';
    bubble.innerHTML = formatMarkdown(content);
    msgDiv.appendChild(bubble);

    // Sources
    if (sources && sources.length > 0) {
        const sourceSection = document.createElement('div');
        sourceSection.style.maxWidth = '80%';

        // Toggle button
        const toggleBtn = document.createElement('button');
        toggleBtn.className = 'sources-toggle';
        toggleBtn.innerHTML = `▸ ${sources.length} sources retrieved`;
        if (stats) {
            toggleBtn.innerHTML += ` • ${stats.total_chunks_searched} chunks searched • ${timeMs ? timeMs.toFixed(0) + 'ms' : ''}`;
        }

        const panel = document.createElement('div');
        panel.className = 'sources-panel';
        panel.style.display = 'none';

        sources.forEach(src => {
            const card = document.createElement('div');
            card.className = 'source-card';
            card.innerHTML = `
                <div class="source-header">
                    <span class="source-title">${src.section_title || src.document_title || 'Source'}</span>
                    <span class="source-score">${src.relevance_score.toFixed(3)}</span>
                </div>
                <div class="source-content">${src.content.substring(0, 250)}${src.content.length > 250 ? '...' : ''}</div>
            `;
            panel.appendChild(card);
        });

        toggleBtn.addEventListener('click', () => {
            const visible = panel.style.display !== 'none';
            panel.style.display = visible ? 'none' : 'flex';
            toggleBtn.innerHTML = toggleBtn.innerHTML.replace(visible ? '▾' : '▸', visible ? '▸' : '▾');
        });

        sourceSection.appendChild(toggleBtn);
        sourceSection.appendChild(panel);
        msgDiv.appendChild(sourceSection);
    }

    messagesContainer.appendChild(msgDiv);
    scrollToBottom();
}

function appendLoading() {
    const div = document.createElement('div');
    div.className = 'message assistant';
    div.innerHTML = `
        <div class="loading-dots">
            <span></span><span></span><span></span>
        </div>
    `;
    messagesContainer.appendChild(div);
    scrollToBottom();
    return div;
}

function scrollToBottom() {
    chatArea.scrollTop = chatArea.scrollHeight;
}

// ── Markdown Formatting ──
function formatMarkdown(text) {
    if (!text) return '';

    return text
        // Bold
        .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
        // Italic
        .replace(/\*(.*?)\*/g, '<em>$1</em>')
        // Inline code
        .replace(/`([^`]+)`/g, '<code>$1</code>')
        // Headers
        .replace(/^### (.*$)/gm, '<h4 style="margin: 12px 0 6px; font-size: 14px; color: var(--accent-light);">$1</h4>')
        .replace(/^## (.*$)/gm, '<h3 style="margin: 14px 0 8px; font-size: 15px;">$1</h3>')
        // Bullet points
        .replace(/^[-•] (.*$)/gm, '<li>$1</li>')
        .replace(/(<li>.*<\/li>\n?)+/g, '<ul style="margin: 8px 0 8px 18px;">$&</ul>')
        // Numbered lists
        .replace(/^\d+\. (.*$)/gm, '<li>$1</li>')
        // Line breaks
        .replace(/\n\n/g, '</p><p>')
        .replace(/\n/g, '<br>')
        // Wrap in paragraph
        ;
}

// ── Pipeline Animation ──
const pipelineSteps = ['embed', 'retrieve', 'rerank', 'generate'];
let pipelineTimer = null;

function animatePipeline() {
    let stepIndex = 0;
    const pipelineBox = document.getElementById('pipelineBox');
    if (!pipelineBox) return;

    resetPipeline();

    function activateStep() {
        if (stepIndex >= pipelineSteps.length) {
            clearInterval(pipelineTimer);
            return;
        }

        // Mark previous as done
        if (stepIndex > 0) {
            const prevStep = document.getElementById(`step-${pipelineSteps[stepIndex - 1]}`);
            if (prevStep) {
                prevStep.classList.remove('active');
                prevStep.classList.add('done');
            }
        }

        // Mark current as active
        const currStep = document.getElementById(`step-${pipelineSteps[stepIndex]}`);
        if (currStep) {
            currStep.classList.add('active');
        }

        stepIndex++;
    }

    activateStep(); // Start immediately
    pipelineTimer = setInterval(activateStep, 800);
}

function resetPipeline() {
    if (pipelineTimer) clearInterval(pipelineTimer);
    pipelineSteps.forEach(step => {
        const el = document.getElementById(`step-${step}`);
        if (el) {
            el.classList.remove('active', 'done');
        }
    });
}

// ── Health Check ──
async function checkHealth() {
    const dot = document.getElementById('healthDot');
    const text = document.getElementById('healthText');
    const info = document.getElementById('sysInfo');

    if (!dot || !text) return;

    try {
        const res = await fetch(`${API_BASE}/health`);
        const data = await res.json();

        dot.className = `stat-dot ${data.status}`;
        text.textContent = data.status === 'healthy' ? 'All systems online' : 'Degraded';
        if (info) {
            info.textContent = `${data.total_documents} docs • ${data.total_chunks} chunks`;
        }
    } catch (e) {
        dot.className = 'stat-dot';
        text.textContent = 'Server offline';
        if (info) info.textContent = 'Start: uvicorn app.main:app';
    }
}

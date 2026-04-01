/**
 * FinQuery Assistant — Client-side Chat Controller
 * Features: Dark/Light Mode, Pipeline Animation, RAG Querying
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
const themeToggle = document.getElementById('themeToggle');
const themeToggleIcon = document.getElementById('themeToggleIcon');
const themeToggleText = document.getElementById('themeToggleText');

// ── Initialize ──
document.addEventListener('DOMContentLoaded', () => {
    // 1. Theme Initialization
    initTheme();

    // 2. Event Listeners
    queryInput.addEventListener('keydown', handleKeyDown);
    queryInput.addEventListener('input', handleInput);
    if (themeToggle) {
        themeToggle.addEventListener('click', toggleTheme);
    }

    // 3. Health Checks
    checkHealth();
    setInterval(checkHealth, 30000);
});

// ── Theme Management ──
function initTheme() {
    const savedTheme = localStorage.getItem('finquery-theme') || 'light';
    applyTheme(savedTheme);
}

function toggleTheme() {
    const currentTheme = document.documentElement.getAttribute('data-theme');
    const newTheme = currentTheme === 'dark' ? 'light' : 'dark';
    applyTheme(newTheme);
}

function applyTheme(theme) {
    document.documentElement.setAttribute('data-theme', theme);
    localStorage.setItem('finquery-theme', theme);
    
    if (themeToggleIcon && themeToggleText) {
        if (theme === 'dark') {
            themeToggleIcon.textContent = '☀️';
            themeToggleText.textContent = 'Light Mode';
        } else {
            themeToggleIcon.textContent = '🌙';
            themeToggleText.textContent = 'Dark Mode';
        }
    }
}

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
    const charCountEl = document.getElementById('charCount');
    if (charCountEl) {
        charCountEl.textContent = `${count} / 2048`;
    }
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
    
    const charCountEl = document.getElementById('charCount');
    if (charCountEl) charCountEl.textContent = '0 / 2048';

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
                top_k: 5, // Defaulting to 5 as per new minimal UI
                similarity_threshold: 0.15,
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
        appendMessage('assistant', `⚠️ Connection Error: ${err.message}. Ensure the backend is running.`);
    }

    isLoading = false;
    sendBtn.disabled = false;
    queryInput.focus();
}

// ── Ask from suggested question ──
window.askQuestion = function(question) {
    queryInput.value = question;
    handleInput();
    sendQuery();
};

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
        sourceSection.style.width = '100%';

        // Toggle button
        const toggleBtn = document.createElement('button');
        toggleBtn.className = 'sources-toggle';
        toggleBtn.innerHTML = `<span>▸</span> ${sources.length} sources used`;
        if (stats) {
            toggleBtn.title = `${stats.total_chunks_searched} chunks searched in ${timeMs ? timeMs.toFixed(0) + 'ms' : 'N/A'}`;
        }

        const panel = document.createElement('div');
        panel.className = 'sources-panel';
        panel.style.display = 'none';

        sources.forEach(src => {
            const card = document.createElement('div');
            card.className = 'source-card';
            card.innerHTML = `
                <div class="source-header">
                    <span class="source-title">${src.section_title || src.document_title || 'Reference'}</span>
                    <span class="source-score">Relevance: ${src.relevance_score.toFixed(3)}</span>
                </div>
                <div class="source-content">${src.content.substring(0, 300)}${src.content.length > 300 ? '...' : ''}</div>
            `;
            panel.appendChild(card);
        });

        toggleBtn.addEventListener('click', () => {
            const visible = panel.style.display !== 'none';
            panel.style.display = visible ? 'none' : 'flex';
            toggleBtn.querySelector('span').textContent = visible ? '▸' : '▾';
        });

        sourceSection.appendChild(toggleBtn);
        sourceSection.appendChild(panel);
        msgDiv.appendChild(sourceSection);
    }

    messagesContainer.appendChild(msgDiv);
    
    // Smooth scroll with slight delay for animation
    setTimeout(scrollToBottom, 50);
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
        .replace(/^### (.*$)/gm, '<h4>$1</h4>')
        .replace(/^## (.*$)/gm, '<h3>$1</h3>')
        // Bullet points
        .replace(/^[-•] (.*$)/gm, '<li>$1</li>')
        .replace(/(<li>.*<\/li>\n?)+/g, '<ul>$&</ul>')
        // Numbered lists
        .replace(/^\d+\. (.*$)/gm, '<li>$1</li>')
        // Line breaks
        .replace(/\n\n/g, '</p><p>')
        .replace(/\n/g, '<br>')
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
        text.textContent = data.status === 'healthy' ? 'Systems Online' : 'Degraded';
        if (info) {
            info.textContent = `${data.total_documents} Docs • ${data.total_chunks} Chunks`;
        }
    } catch (e) {
        dot.className = 'stat-dot';
        text.textContent = 'Server Offline';
        if (info) info.textContent = 'API Unavailable';
    }
}

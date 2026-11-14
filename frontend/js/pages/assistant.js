/**
 * BoxOfficeIQ - CinePredict AI Copilot (RAG + SQL Grounded Assistant)
 * Natural language conversational intelligence grounded in cinema knowledge & SQL analytics
 */

import { sendChatMessage, getSuggestedQuestions } from '../api.js';

let messageHistory = [];

export function cleanup() {
  // Retain history across navigation
}

export async function init(container) {
  container.innerHTML = `
    <div class="chat-wrapper" style="display: flex; flex-direction: column; height: calc(100vh - var(--topbar-height) - 48px); max-width: 1040px; margin: 0 auto;">
      <!-- Header -->
      <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 14px; padding-bottom: 10px; border-bottom: 1px solid var(--border-subtle);">
        <div>
          <div style="display: flex; align-items: center; gap: 8px;">
            <h1 class="page-title" style="font-size: 1.25rem;">CinePredict AI Copilot</h1>
            <span class="badge badge--gold">RAG + SQL GROUNDED</span>
          </div>
          <p class="page-subtitle" style="margin-top: 2px;">
            Natural language intelligence grounded in 16,440 daily admissions records and cinema operations knowledge base.
          </p>
        </div>
        <button id="btn-clear-chat" class="btn btn-secondary btn-sm" title="Clear conversation history">
          <svg viewBox="0 0 16 16" width="13" height="13" fill="none" stroke="currentColor" stroke-width="2">
            <polyline points="2 4 14 4"/>
            <path d="M5 4V2H11V4"/>
            <path d="M4 6L5 14H11L12 6"/>
          </svg>
          <span>Reset Session</span>
        </button>
      </div>

      <!-- Suggested Question Pills -->
      <div class="chat-suggested" id="chat-suggested-chips" style="display: flex; flex-wrap: wrap; gap: 8px; margin-bottom: 14px;">
        <!-- Injected dynamically -->
      </div>

      <!-- Messages Scroll Area -->
      <div class="chat-messages" id="chat-messages-container" style="flex: 1; overflow-y: auto; display: flex; flex-direction: column; gap: 14px; padding-right: 6px;">
        <!-- Welcome Message -->
        <div class="chat-msg chat-msg--bot" style="display: flex; gap: 12px;">
          <div class="msg-avatar" style="width: 32px; height: 32px; border-radius: var(--radius-sm); background: var(--color-gold-dim); border: 1px solid rgba(245, 158, 11, 0.3); display: flex; align-items: center; justify-content: center; color: var(--color-gold); flex-shrink: 0; font-size: 0.9rem;">
            ✦
          </div>
          <div class="msg-bubble" style="background: var(--bg-card); border: 1px solid var(--border-subtle); border-radius: var(--radius-md); padding: 14px 18px; max-width: 80%; font-size: 0.82rem; line-height: 1.6; color: var(--color-text-high);">
            <div style="font-weight: 600; color: var(--color-gold); margin-bottom: 6px;">Welcome to CinePredict Intelligence</div>
            I am your <strong>BoxOfficeIQ Copilot</strong>. I synthesize live relational attendance metrics, LightGBM/LSTM demand forecasts, and DistilBERT sentiment signals.<br /><br />
            You can query me in natural language about cinema occupancy records, festival multipliers, weekend turnstile pacing, or technical ML model explanations.
            <div style="display: flex; align-items: center; justify-content: space-between; margin-top: 10px; padding-top: 8px; border-top: 1px solid var(--border-subtle); font-size: 0.68rem; color: var(--color-text-muted);">
              <span class="badge badge--best">AGENT READY</span>
              <span>Online Intelligence Core</span>
            </div>
          </div>
        </div>
      </div>

      <!-- Input Ribbon -->
      <div style="padding-top: 14px; border-top: 1px solid var(--border-subtle);">
        <div style="display: flex; gap: 10px; align-items: flex-end; background: var(--bg-surface); border: 1px solid var(--border-default); border-radius: var(--radius-md); padding: 8px 12px;">
          <textarea
            id="chat-user-input"
            class="form-control"
            rows="1"
            placeholder="Ask anything about cinema demand, theatre benchmarks, or forecast explanations..."
            style="background: transparent; border: none; padding: 4px 0; resize: none; min-height: 24px; max-height: 120px; line-height: 1.5; font-size: 0.84rem;"
          ></textarea>
          <button id="chat-send-btn" class="btn btn-primary btn-sm" style="height: 32px; padding: 0 14px; flex-shrink: 0;">
            <span>Send</span>
            <svg viewBox="0 0 16 16" width="12" height="12" fill="none" stroke="currentColor" stroke-width="2">
              <polygon points="2 2 14 8 2 14 4 8 2 2" fill="currentColor"/>
            </svg>
          </button>
        </div>
        <div style="display: flex; justify-content: space-between; margin-top: 6px; font-size: 0.68rem; color: var(--color-text-muted);">
          <span>Press <strong>Enter</strong> to submit, <strong>Shift + Enter</strong> for line break</span>
          <span>Dual RAG + Text-to-SQL Routing</span>
        </div>
      </div>
    </div>
  `;

  const chipsContainer = document.getElementById('chat-suggested-chips');
  const messagesContainer = document.getElementById('chat-messages-container');
  const userInput = document.getElementById('chat-user-input');
  const sendBtn = document.getElementById('chat-send-btn');
  const clearBtn = document.getElementById('btn-clear-chat');

  // Render Suggested Chips
  try {
    const questions = await getSuggestedQuestions();
    const qList = Array.isArray(questions) ? questions : (questions.questions || []);
    if (chipsContainer && qList.length > 0) {
      chipsContainer.innerHTML = qList.slice(0, 4).map(q => `
        <button type="button" class="btn btn-secondary btn-sm chip-btn" style="font-size: 0.72rem; border-radius: var(--radius-pill); padding: 4px 12px;">
          ${escapeHtml(q)}
        </button>
      `).join('');

      chipsContainer.querySelectorAll('.chip-btn').forEach(btn => {
        btn.addEventListener('click', () => {
          if (userInput) {
            userInput.value = btn.textContent.trim();
            sendMessage();
          }
        });
      });
    }
  } catch (e) {
    console.warn('Suggested questions error:', e);
  }

  // Auto-resize input
  if (userInput) {
    userInput.addEventListener('input', () => {
      userInput.style.height = 'auto';
      userInput.style.height = Math.min(userInput.scrollHeight, 120) + 'px';
    });

    userInput.addEventListener('keydown', (e) => {
      if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        sendMessage();
      }
    });
  }

  if (sendBtn) {
    sendBtn.addEventListener('click', sendMessage);
  }

  if (clearBtn) {
    clearBtn.addEventListener('click', () => {
      messageHistory = [];
      if (messagesContainer) {
        messagesContainer.innerHTML = `
          <div class="chat-msg chat-msg--bot" style="display: flex; gap: 12px;">
            <div class="msg-avatar" style="width: 32px; height: 32px; border-radius: var(--radius-sm); background: var(--color-gold-dim); border: 1px solid rgba(245, 158, 11, 0.3); display: flex; align-items: center; justify-content: center; color: var(--color-gold); flex-shrink: 0;">
              ✦
            </div>
            <div class="msg-bubble" style="background: var(--bg-card); border: 1px solid var(--border-subtle); border-radius: var(--radius-md); padding: 14px 18px; font-size: 0.82rem; color: var(--color-text-high);">
              Session cleared. What operational or demand insights can I provide for your circuit?
            </div>
          </div>
        `;
      }
    });
  }

  async function sendMessage() {
    if (!userInput) return;
    const text = userInput.value.trim();
    if (!text) return;

    userInput.value = '';
    userInput.style.height = 'auto';

    // User Message
    const userTime = new Date().toLocaleTimeString('en-US', { hour12: false, hour: '2-digit', minute: '2-digit' });
    const userMsg = document.createElement('div');
    userMsg.className = 'chat-msg chat-msg--user';
    userMsg.style.cssText = 'display: flex; justify-content: flex-end; gap: 10px;';
    userMsg.innerHTML = `
      <div style="background: var(--bg-surface-elevated); border: 1px solid var(--border-default); border-radius: var(--radius-md); padding: 12px 16px; max-width: 75%; font-size: 0.82rem; color: var(--color-text-high); line-height: 1.5;">
        ${escapeHtml(text)}
        <div style="text-align: right; font-size: 0.65rem; color: var(--color-text-muted); margin-top: 4px;">
          ${userTime}
        </div>
      </div>
    `;
    messagesContainer.appendChild(userMsg);

    // Typing Placeholder
    const typingMsg = document.createElement('div');
    typingMsg.id = 'chat-typing-indicator';
    typingMsg.style.cssText = 'display: flex; gap: 12px;';
    typingMsg.innerHTML = `
      <div style="width: 32px; height: 32px; border-radius: var(--radius-sm); background: var(--color-gold-dim); border: 1px solid rgba(245, 158, 11, 0.3); display: flex; align-items: center; justify-content: center; color: var(--color-gold); flex-shrink: 0;">
        ✦
      </div>
      <div style="background: var(--bg-card); border: 1px solid var(--border-subtle); border-radius: var(--radius-md); padding: 12px 16px; font-size: 0.78rem; color: var(--color-text-muted); display: flex; align-items: center; gap: 6px;">
        <span class="status-indicator status-indicator--checking"></span>
        <span>Synthesizing database & knowledge signals...</span>
      </div>
    `;
    messagesContainer.appendChild(typingMsg);
    messagesContainer.scrollTop = messagesContainer.scrollHeight;

    if (sendBtn) sendBtn.disabled = true;

    try {
      const response = await sendChatMessage(text, messageHistory);

      const typing = document.getElementById('chat-typing-indicator');
      if (typing) typing.remove();

      const textAnswer = response.answer || response.response || 'No response generated.';
      const citations = response.citations || response.sources || [];
      const queryType = response.query_type || 'RAG';

      messageHistory.push({ role: 'user', content: text });
      messageHistory.push({ role: 'assistant', content: textAnswer });

      let citationsHtml = '';
      if (citations && citations.length > 0) {
        citationsHtml = `
          <div style="margin-top: 10px; padding-top: 8px; border-top: 1px solid var(--border-subtle); display: flex; align-items: center; gap: 6px; flex-wrap: wrap;">
            <span style="font-size: 0.65rem; font-weight: 600; text-transform: uppercase; color: var(--color-text-muted);">Sources:</span>
            ${citations.map(c => `<span class="badge badge--default" style="font-size: 0.65rem;">${escapeHtml(c)}</span>`).join('')}
          </div>
        `;
      }

      const botTime = new Date().toLocaleTimeString('en-US', { hour12: false, hour: '2-digit', minute: '2-digit' });
      const botMsg = document.createElement('div');
      botMsg.className = 'chat-msg chat-msg--bot';
      botMsg.style.cssText = 'display: flex; gap: 12px;';
      botMsg.innerHTML = `
        <div style="width: 32px; height: 32px; border-radius: var(--radius-sm); background: var(--color-gold-dim); border: 1px solid rgba(245, 158, 11, 0.3); display: flex; align-items: center; justify-content: center; color: var(--color-gold); flex-shrink: 0; font-size: 0.9rem;">
          ✦
        </div>
        <div style="background: var(--bg-card); border: 1px solid var(--border-subtle); border-radius: var(--radius-md); padding: 14px 18px; max-width: 80%; font-size: 0.82rem; line-height: 1.6; color: var(--color-text-high);">
          <div>${formatBotText(textAnswer)}</div>
          ${citationsHtml}
          <div style="display: flex; justify-content: space-between; align-items: center; margin-top: 8px; font-size: 0.68rem; color: var(--color-text-muted);">
            <span class="badge ${queryType.toLowerCase().includes('sql') ? 'badge--multiplex' : 'badge--gold'}">${queryType}</span>
            <span>${botTime}</span>
          </div>
        </div>
      `;
      messagesContainer.appendChild(botMsg);
      messagesContainer.scrollTop = messagesContainer.scrollHeight;

    } catch (err) {
      console.error('Chat query error:', err);
      const typing = document.getElementById('chat-typing-indicator');
      if (typing) typing.remove();

      const errMsg = document.createElement('div');
      errMsg.style.cssText = 'display: flex; gap: 12px;';
      errMsg.innerHTML = `
        <div style="width: 32px; height: 32px; border-radius: var(--radius-sm); background: var(--color-rose-dim); color: var(--color-rose); display: flex; align-items: center; justify-content: center; flex-shrink: 0;">
          ✕
        </div>
        <div style="background: var(--bg-card); border: 1px solid var(--color-rose); border-radius: var(--radius-md); padding: 12px 16px; font-size: 0.8rem; color: var(--color-text-high);">
          Unable to retrieve response from cinema intelligence pipeline. Please ensure the backend is connected.
        </div>
      `;
      messagesContainer.appendChild(errMsg);
    } finally {
      if (sendBtn) sendBtn.disabled = false;
    }
  }

  function escapeHtml(str) {
    if (!str) return '';
    return String(str)
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;')
      .replace(/'/g, '&#39;');
  }

  function formatBotText(text) {
    if (!text) return '';
    // Format bold markdown (**bold**)
    let out = escapeHtml(text);
    out = out.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
    out = out.replace(/\*(.*?)\*/g, '<em>$1</em>');
    out = out.replace(/\n\n/g, '<br /><br />');
    out = out.replace(/\n/g, '<br />');
    return out;
  }
}

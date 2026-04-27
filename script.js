const API_BASE = 'http://127.0.0.1:5000';

// --- Single Global Session Manager ---
async function ensureSession() {
    let sid = localStorage.getItem('chat_session_id');
    if (!sid) {
        try {
            const res = await fetch(`${API_BASE}/get-session`);
            const data = await res.json();
            localStorage.setItem('chat_session_id', data.session_id);
            return data.session_id;
        } catch (err) {
            console.error("Session init failed", err);
            return null;
        }
    }
    return sid;
}

// Force-create a fresh session (clears stale ID first)
async function forceNewSession() {
    localStorage.removeItem('chat_session_id');
    return ensureSession();
}

async function sendUserInfo() {
    const sid = await ensureSession();
    const name = document.getElementById('user-name').value;
    const contact = document.getElementById('user-contact').value;

    if (!name || !contact) {
        alert("Please fill in both fields.");
        return;
    }

    try {
        const response = await fetch(`${API_BASE}/get_userinfo`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ session_id: sid, name: name, contact: contact })
        });

        if (response.ok) {
            document.getElementById('user-info-container').innerHTML = `
                <div style="text-align: center; color: var(--brand-primary); padding: 10px;">
                    <p style="font-weight: 600; margin: 0;">Thank you, ${name}! ✅</p>
                    <p style="font-size: 0.9rem; margin-top: 5px;">Profile synced with GraceBot.</p>
                </div>`;
        }
    } catch (err) { console.error("Info save failed", err); }
}

document.addEventListener('DOMContentLoaded', async () => {
    const viewport = document.getElementById('chat-viewport');
    const inputField = document.getElementById('user-input');
    const sendBtn = document.getElementById('send-trigger');
    const hero = document.getElementById('hero-section');
    const resetBtn = document.getElementById('reset-trigger');
    
    let isLocked = false;
    let appReady = false; // Guard: prevent input before startup completes

    let startupDone = false; // Prevent startApp from running twice

    // --- Core Session Sync ---
    async function startApp() {
        if (startupDone) return; // Guard against double-execution
        startupDone = true;

        // 1. Ensure we have a valid session
        const sid = await ensureSession();
        
        // 2. Sync history (only if we have a session)
        if (sid) {
            await syncHistory(sid);
        }

        // 3. App is ready for user input
        appReady = true;
    }

    async function syncHistory(sid) {
        if (!sid) return;

        try {
            const res = await fetch(`${API_BASE}/get-history`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ session_id: sid })
            });

            if (res.ok) {
                const { history } = await res.json();
                if (history && history.length > 0) {
                    clearHero();
                    viewport.innerHTML = ''; 
                    history.forEach(m => renderBubble(m.role === 'user' ? 'user' : 'bot', m.content, true));
                    scrollSync();
                }
            } else if (res.status === 401) {
                // Stale session — force a fresh one immediately
                await forceNewSession();
            }
        } catch (e) { console.warn("History sync bypassed", e); }
    }

    // --- UI Operations ---

    function clearHero() {
        if (hero) hero.style.display = 'none';
    }

    function renderBubble(role, text, isStatic = false) {
        const group = document.createElement('div');
        group.classList.add('msg-group', role);
        
        const bubble = document.createElement('div');
        bubble.classList.add('bubble');
        bubble.textContent = isStatic ? text : '';
        
        group.appendChild(bubble);
        viewport.appendChild(group);
        scrollSync();
        return bubble;
    }

    function showTyping() {
        const group = document.createElement('div');
        group.classList.add('msg-group', 'bot');
        group.id = 'active-indicator';
        group.innerHTML = `
            <div class="bubble">
                <div class="typing"><div class="dot"></div><div class="dot"></div><div class="dot"></div></div>
            </div>`;
        viewport.appendChild(group);
        scrollSync();
    }

    function hideTyping() {
        const el = document.getElementById('active-indicator');
        if (el) el.remove();
    }

    function scrollSync() {
        viewport.scrollTop = viewport.scrollHeight;
    }

    
    // --- Chat Logic (Fixed: no more flash/reset) ---
    async function performInquiry() {
        const prompt = inputField.value.trim();
        if (!prompt || isLocked || !appReady) return;

        isLocked = true;
        
        // 1. Ensure valid session BEFORE touching the UI
        let sid = await ensureSession();

        // 2. Now safe to update UI
        clearHero();
        renderBubble('user', prompt, true);
        inputField.value = '';
        inputField.disabled = true;
        showTyping();

        // 3. Send chat with single auto-retry on 401
        async function attemptChat(session_id, isRetry = false) {
            const response = await fetch(`${API_BASE}/stream-chat`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ prompt: prompt, session_id: session_id })
            });

            // If session rejected and this is NOT already a retry, get fresh session
            if (response.status === 401 && !isRetry) {
                console.warn("Session rejected. Getting fresh session and retrying...");
                const newSid = await forceNewSession();
                return attemptChat(newSid, true);
            }
            return response;
        }

        try {
            const response = await attemptChat(sid);

            if (!response.ok) throw new Error("Backend failed");

            hideTyping();
            const bubble = renderBubble('bot', '');
            const reader = response.body.getReader();
            const decoder = new TextDecoder();

            while (true) {
                const { value, done } = await reader.read();
                if (done) break;
                bubble.textContent += decoder.decode(value, { stream: true });
                scrollSync();
            }
        } catch (err) {
            hideTyping();
            renderBubble('bot', "A communication error occurred. Check backend connectivity.", true);
        } finally {
            isLocked = false;
            inputField.disabled = false;
            inputField.focus();
        }
    }

    // --- Listeners ---
    sendBtn.addEventListener('click', performInquiry);
    inputField.addEventListener('keydown', (e) => {
        if (e.key === 'Enter') {
            e.preventDefault(); // Prevent any implicit form submission / page reload
            performInquiry();
        }
    });
    resetBtn.addEventListener('click', () => {
        localStorage.removeItem('chat_session_id');
        window.location.reload();
    });

    // Startup
    startApp();
});
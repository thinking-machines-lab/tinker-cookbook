import { useEffect, useRef, useState } from 'react';
import { Link, useParams } from 'react-router-dom';
import { api } from '../api/client';
import { ConversationRenderer } from '../components/ConversationRenderer';
import { useUrlParam } from '../utils/useUrlParam';
import type { ConversationMessage, ContentPart } from '../api/types';

/** Wrap plain text as structured content. */
function textContent(text: string): ContentPart[] {
  return [{ type: 'text', text }];
}

interface SessionSummary {
  session_id: string;
  checkpoint_name: string;
  title: string;
  created_at: string;
  message_count: number;
}

export function ChatPage() {
  const { runId } = useParams<{ runId: string }>();
  const [checkpoint] = useUrlParam('checkpoint', '');
  const [sessionId, setSessionId] = useUrlParam('session', '');
  const [messages, setMessages] = useState<ConversationMessage[]>([]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [temperature, setTemperature] = useState(0.7);
  const [maxTokens, setMaxTokens] = useState(1024);
  const [chatEnabled, setChatEnabled] = useState<boolean | null>(null);
  const [sessions, setSessions] = useState<SessionSummary[]>([]);
  const [showJson, setShowJson] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    api.getCapabilities().then((d) => setChatEnabled(d.chat ?? false)).catch(() => setChatEnabled(false));
  }, []);

  useEffect(() => {
    if (!runId) return;
    api.getChatSessions(runId)
      .then((all) => setSessions(all.filter((s) => s.checkpoint_name === checkpoint)))
      .catch(() => setSessions([]));
  }, [runId, checkpoint]);

  useEffect(() => {
    if (!runId || !sessionId) return;
    api.getChatSession(runId, sessionId)
      .then((d) => setMessages(d.messages ?? []))
      .catch(() => {});
  }, [runId, sessionId]);

  const scrollToBottom = () => messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  useEffect(scrollToBottom, [messages]);

  const startNewChat = () => { setSessionId(''); setMessages([]); setError(null); };
  const loadSession = (sid: string) => { setSessionId(sid); setError(null); };

  const sendMessage = async () => {
    if (!input.trim() || loading || !runId || !checkpoint) return;
    const userMessage: ConversationMessage = { role: 'user', content: textContent(input.trim()) };
    const allMessages = [...messages, userMessage];
    setMessages(allMessages);
    setInput('');
    setLoading(true);
    setError(null);

    try {
      const response = await fetch(`/api/runs/${runId}/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          messages: allMessages.map((m) => ({ role: m.role, content: m.content })),
          checkpoint_name: checkpoint,
          session_id: sessionId || undefined,
          temperature, max_tokens: maxTokens,
        }),
      });
      if (!response.ok) { const err = await response.json().catch(() => ({})); throw new Error(err.detail || `HTTP ${response.status}`); }

      const reader = response.body?.getReader();
      if (!reader) throw new Error('No stream');
      const decoder = new TextDecoder();
      let streamedText = '';
      let gotError = false;

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        for (const line of decoder.decode(value).split('\n')) {
          if (!line.startsWith('data: ')) continue;
          try {
            const data = JSON.parse(line.slice(6));
            if (data.error) { setError(data.error); gotError = true; }
            else if (data.status) {
              setMessages([...allMessages, { role: 'assistant', content: textContent(`\u23f3 ${data.status}`) }]);
            }
            else if (data.content) {
              streamedText += data.content;
              // During streaming, show plain text; when done, use structured message if available
              const assistantMsg: ConversationMessage = data.message
                ? data.message
                : { role: 'assistant', content: textContent(streamedText) };
              setMessages([...allMessages, assistantMsg]);
              if (data.session_id && !sessionId) setSessionId(data.session_id);
            }
          } catch { /* skip */ }
        }
      }
      if (!streamedText && !gotError) setError('No response received');
      if (runId) api.getChatSessions(runId).then((all) => setSessions(all.filter((s) => s.checkpoint_name === checkpoint))).catch(() => {});
    } catch (e) { setError(e instanceof Error ? e.message : String(e)); }
    finally { setLoading(false); }
  };

  if (chatEnabled === null) return <div className="loading">Checking capabilities...</div>;
  if (!chatEnabled) return (
    <div>
      <div className="breadcrumb">
        <Link to="/">Dashboard</Link><span>/</span>
        <Link to={`/runs/${runId}?tab=checkpoints`}>{runId}</Link><span>/</span><span>Chat</span>
      </div>
      <div className="empty-state" style={{ marginTop: '2rem' }}>
        <div style={{ fontWeight: 600, marginBottom: '0.5rem' }}>Interactive chat requires TINKER_API_KEY</div>
        <div className="text-muted">Set <code className="mono">TINKER_API_KEY</code> when running <code className="mono">tinker-chef serve</code>.</div>
      </div>
    </div>
  );

  return (
    <div>
      <div className="breadcrumb">
        <Link to="/">Dashboard</Link><span>/</span>
        <Link to={`/runs/${runId}?tab=checkpoints`}>{runId}</Link><span>/</span>
        <span>Chat · {checkpoint}</span>
      </div>

      <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', marginBottom: '0.75rem' }}>
        <span style={{ flex: 1 }} />
        <label style={{ display: 'flex', alignItems: 'center', gap: '0.25rem', fontSize: '0.6875rem', color: 'var(--text-muted)' }}>
          temp
          <input type="number" min={0} max={2} step={0.1} value={temperature}
            onChange={(e) => setTemperature(Number(e.target.value))}
            className="mono" style={{ width: '45px', padding: '0.2rem 0.3rem', borderRadius: '4px', border: '1px solid var(--border)', background: 'var(--bg-elevated)', color: 'var(--text-primary)', fontSize: '0.6875rem' }}
          />
        </label>
        <label style={{ display: 'flex', alignItems: 'center', gap: '0.25rem', fontSize: '0.6875rem', color: 'var(--text-muted)' }}>
          max tokens
          <input type="number" min={1} max={65536} step={256} value={maxTokens}
            onChange={(e) => setMaxTokens(Number(e.target.value))}
            className="mono" style={{ width: '55px', padding: '0.2rem 0.3rem', borderRadius: '4px', border: '1px solid var(--border)', background: 'var(--bg-elevated)', color: 'var(--text-primary)', fontSize: '0.6875rem' }}
          />
        </label>
      </div>

      <div className="card" style={{ padding: 0, display: 'flex', height: 'calc(100vh - 200px)', minHeight: '350px' }}>
        <div style={{
          width: '180px', flexShrink: 0, borderRight: '1px solid var(--border)',
          display: 'flex', flexDirection: 'column', background: 'var(--bg-elevated)',
          borderRadius: '8px 0 0 8px', overflow: 'hidden',
        }}>
          <button
            onClick={startNewChat}
            style={{
              padding: '0.5rem', margin: '0.5rem', borderRadius: '6px',
              border: '1px dashed var(--border)', background: 'transparent',
              color: 'var(--accent)', cursor: 'pointer', fontSize: '0.6875rem', fontWeight: 600,
            }}
          >
            + New Chat
          </button>
          <div style={{ flex: 1, overflow: 'auto', padding: '0 0.375rem 0.375rem' }}>
            {sessions.length === 0 && (
              <div className="text-muted" style={{ fontSize: '0.625rem', padding: '0.5rem', textAlign: 'center' }}>No sessions yet</div>
            )}
            {sessions.map((s) => (
              <div
                key={s.session_id}
                onClick={() => loadSession(s.session_id)}
                style={{
                  padding: '0.375rem 0.5rem', borderRadius: '4px', cursor: 'pointer',
                  background: sessionId === s.session_id ? 'var(--accent-dim)' : 'transparent',
                  borderLeft: sessionId === s.session_id ? '2px solid var(--accent)' : '2px solid transparent',
                  marginBottom: '1px',
                }}
              >
                <div style={{
                  fontSize: '0.6875rem', fontWeight: sessionId === s.session_id ? 600 : 400,
                  color: sessionId === s.session_id ? 'var(--text-primary)' : 'var(--text-secondary)',
                  overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap',
                }}>
                  {s.title || 'Untitled'}
                </div>
                <div style={{ fontSize: '0.5625rem', color: 'var(--text-muted)' }}>
                  {s.message_count} messages
                </div>
              </div>
            ))}
          </div>
        </div>

        <div style={{ flex: 1, display: 'flex', flexDirection: 'column', minWidth: 0 }}>
          <div style={{ flex: 1, overflow: 'auto', padding: '1rem' }}>
            {messages.length === 0 && (
              <div className="empty-state" style={{ marginTop: '2rem' }}>
                <div style={{ fontSize: '0.875rem', fontWeight: 600, marginBottom: '0.25rem' }}>
                  Chat with <span className="mono" style={{ color: 'var(--accent)' }}>{checkpoint}</span>
                </div>
                <div className="text-muted" style={{ fontSize: '0.8125rem' }}>Type a message below to start.</div>
              </div>
            )}
            <ConversationRenderer messages={messages} />
            {loading && <div style={{ marginLeft: '1.5rem', color: 'var(--text-muted)', fontSize: '0.8125rem', fontStyle: 'italic', marginTop: '0.5rem' }}>Generating...</div>}
            {error && (
              <div style={{ padding: '0.5rem 0.75rem', borderRadius: '8px', background: 'var(--error-dim)', borderLeft: '3px solid var(--error)', color: 'var(--error)', fontSize: '0.8125rem', marginTop: '0.5rem' }}>
                {error}
              </div>
            )}
            <div ref={messagesEndRef} />
          </div>

          <div style={{ borderTop: '1px solid var(--border)', padding: '0.625rem 1rem', display: 'flex', gap: '0.5rem', alignItems: 'center', background: 'var(--bg-surface)' }}>
            <input
              type="text" value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={(e) => { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); sendMessage(); } }}
              placeholder="Type a message..." disabled={loading}
              style={{
                flex: 1, padding: '0.5rem 0.75rem', borderRadius: '6px',
                border: '1px solid var(--border)', background: 'var(--bg-elevated)',
                color: 'var(--text-primary)', fontSize: '0.8125rem',
                fontFamily: 'var(--font-sans)',
              }}
            />
            <button
              onClick={sendMessage} disabled={loading || !input.trim()}
              style={{
                padding: '0.5rem 1rem', borderRadius: '6px', border: 'none',
                background: loading || !input.trim() ? 'var(--bg-elevated)' : 'var(--accent)',
                color: loading || !input.trim() ? 'var(--text-muted)' : '#fff',
                cursor: loading || !input.trim() ? 'default' : 'pointer',
                fontWeight: 600, fontSize: '0.8125rem', fontFamily: 'var(--font-sans)',
              }}
            >
              Send
            </button>
            {messages.length > 0 && (
              <button
                onClick={() => setShowJson(true)}
                title="View raw JSON"
                style={{
                  padding: '0.5rem', borderRadius: '6px', border: '1px solid var(--border)',
                  background: 'transparent', color: 'var(--text-muted)', cursor: 'pointer', fontSize: '0.75rem',
                }}
              >
                {'{ }'}
              </button>
            )}
          </div>
        </div>
      </div>

      {showJson && (
        <div style={{
          position: 'fixed', inset: 0, background: 'rgba(0,0,0,0.6)', zIndex: 1000,
          display: 'flex', alignItems: 'center', justifyContent: 'center',
        }} onClick={() => setShowJson(false)}>
          <div style={{
            background: 'var(--bg-surface)', border: '1px solid var(--border)',
            borderRadius: '12px', width: '90%', maxWidth: '700px', maxHeight: '80vh',
            display: 'flex', flexDirection: 'column',
          }} onClick={(e) => e.stopPropagation()}>
            <div style={{ padding: '0.75rem 1rem', borderBottom: '1px solid var(--border)', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
              <span style={{ fontWeight: 600, fontSize: '0.875rem' }}>Raw Conversation JSON</span>
              <div style={{ display: 'flex', gap: '0.5rem' }}>
                <button
                  onClick={() => { navigator.clipboard.writeText(JSON.stringify(messages, null, 2)); }}
                  style={{ padding: '0.375rem 0.75rem', borderRadius: '6px', border: '1px solid var(--accent)', background: 'transparent', color: 'var(--accent)', cursor: 'pointer', fontSize: '0.75rem', fontWeight: 600 }}
                >
                  Copy
                </button>
                <button
                  onClick={() => setShowJson(false)}
                  style={{ padding: '0.375rem 0.75rem', borderRadius: '6px', border: '1px solid var(--border)', background: 'transparent', color: 'var(--text-muted)', cursor: 'pointer', fontSize: '0.75rem' }}
                >
                  Close
                </button>
              </div>
            </div>
            <pre style={{
              flex: 1, overflow: 'auto', padding: '1rem', margin: 0,
              fontSize: '0.75rem', lineHeight: 1.5, fontFamily: 'var(--font-mono)',
              color: 'var(--text-secondary)', background: 'var(--bg-deep)', borderRadius: '0 0 12px 12px',
            }}>
              {JSON.stringify(messages, null, 2)}
            </pre>
          </div>
        </div>
      )}
    </div>
  );
}

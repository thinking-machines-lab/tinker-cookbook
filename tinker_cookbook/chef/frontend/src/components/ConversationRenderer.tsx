/** Shared conversation renderer — handles text, thinking, images, tool calls.
 *
 * Used by RolloutDetailPage, EvalTrajectoryPage, and ChatPage.
 * Normalizes content at the boundary: accepts both string content (legacy data)
 * and structured ContentPart[] (new data from message_to_jsonable).
 */

import { useState } from 'react';
import type { ConversationMessage, ContentPart, ToolCallInfo } from '../api/types';

export { type ConversationMessage };

/** Normalize content to ContentPart[] — handles legacy string content from old data. */
function normalizeContent(content: string | ContentPart[]): ContentPart[] {
  if (typeof content === 'string') return [{ type: 'text', text: content }];
  return content;
}

interface Props {
  messages: ConversationMessage[];
  showTokenCounts?: boolean;
}

const ROLE_COLORS: Record<string, string> = {
  user: 'var(--cyan)',
  assistant: 'var(--purple)',
  system: 'var(--warning)',
  tool: 'var(--accent)',
  environment: 'var(--accent)',
  grader: '#f472b6',
};

export function ConversationRenderer({ messages, showTokenCounts }: Props) {
  return (
    <div className="conversation">
      {messages.map((msg, i) => (
        <MessageBubble key={i} message={msg} showTokenCount={showTokenCounts} />
      ))}
    </div>
  );
}

function MessageBubble({ message, showTokenCount }: { message: ConversationMessage; showTokenCount?: boolean }) {
  const [thinkingOpen, setThinkingOpen] = useState(false);
  const [collapsed, setCollapsed] = useState(false);
  const isAssistant = message.role === 'assistant';
  const color = ROLE_COLORS[message.role] ?? 'var(--text-muted)';
  const parts = normalizeContent(message.content as string | ContentPart[]);

  const textContent = parts
    .map((p) => p.text || p.thinking || '')
    .join('');
  const isLong = textContent.length > 200;
  const preview = textContent.slice(0, 120).replace(/\n/g, ' ');

  return (
    <div className={`message message-${message.role}`} style={{ marginLeft: isAssistant ? '2rem' : 0 }}>
      <div
        style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', cursor: isLong ? 'pointer' : 'default' }}
        onClick={() => isLong && setCollapsed(!collapsed)}
      >
        <span className="message-role" style={{ color }}>
          {message.role}
          {message.name && <span style={{ fontWeight: 400, marginLeft: '0.375rem' }}>({message.name})</span>}
          {isLong && (
            <span style={{ marginLeft: '0.5rem', fontSize: '0.5rem', color: 'var(--text-muted)', fontWeight: 400 }}>
              {collapsed ? '\u25b6 click to expand' : '\u25bc click to collapse'} ({textContent.length} chars)
            </span>
          )}
        </span>
        {showTokenCount && message.token_count != null && (
          <span style={{ fontSize: '0.625rem', color: 'var(--text-muted)', fontFamily: 'var(--font-mono)' }}>
            {message.token_count} tokens
          </span>
        )}
      </div>

      {collapsed ? (
        <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', fontStyle: 'italic', marginTop: '0.25rem' }}>
          {preview}...
        </div>
      ) : (
        <div className="message-content">
          {parts.map((part, i) => (
            <ContentPartView key={i} part={part} thinkingOpen={thinkingOpen} onToggleThinking={() => setThinkingOpen(!thinkingOpen)} />
          ))}
          {message.tool_calls?.map((tc, i) => (
            <ToolCallView key={`tc-${i}`} toolCall={tc} />
          ))}
          {message.unparsed_tool_calls?.map((utc, i) => (
            <div key={`utc-${i}`} className="tool-call-block" style={{ borderColor: 'var(--error)' }}>
              <div className="tool-call-label" style={{ color: 'var(--error)' }}>Unparsed Tool Call</div>
              <pre className="tool-call-code">{utc.raw_text}</pre>
              <div style={{ fontSize: '0.625rem', color: 'var(--error)', marginTop: '0.25rem' }}>{utc.error}</div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

function ContentPartView({ part, thinkingOpen, onToggleThinking }: {
  part: ContentPart;
  thinkingOpen: boolean;
  onToggleThinking: () => void;
}) {
  if (part.type === 'text' && part.text) {
    return <span>{part.text}</span>;
  }
  if (part.type === 'thinking' && part.thinking) {
    return (
      <div className="thinking-block">
        <div className="thinking-toggle" onClick={(e) => { e.stopPropagation(); onToggleThinking(); }}>
          {thinkingOpen ? '\u25bc' : '\u25b6'} Thinking
        </div>
        {thinkingOpen && <div className="thinking-content">{part.thinking}</div>}
      </div>
    );
  }
  if (part.type === 'image') {
    if (part.image && (part.image.startsWith('http') || part.image.startsWith('data:'))) {
      return (
        <div style={{ margin: '0.375rem 0' }}>
          <img
            src={part.image}
            alt="Content"
            style={{ maxWidth: '100%', maxHeight: '400px', borderRadius: '6px', border: '1px solid var(--border)' }}
          />
        </div>
      );
    }
    return <span className="tag">[Image]</span>;
  }
  // Unknown content type — render as collapsible JSON so new types don't silently disappear
  const raw = JSON.stringify(part, null, 2);
  return (
    <details style={{ margin: '0.25rem 0' }}>
      <summary className="tag" style={{ cursor: 'pointer' }}>[{part.type}]</summary>
      <pre className="mono" style={{ fontSize: '0.625rem', whiteSpace: 'pre-wrap', padding: '0.25rem', background: 'var(--bg-tertiary)', borderRadius: '4px', marginTop: '0.25rem' }}>{raw}</pre>
    </details>
  );
}

function ToolCallView({ toolCall }: { toolCall: ToolCallInfo }) {
  const [expanded, setExpanded] = useState(false);
  let args = toolCall.function.arguments;
  try { args = JSON.stringify(JSON.parse(args), null, 2); } catch { /* keep raw */ }

  return (
    <div className="tool-call-block">
      <div
        className="tool-call-label"
        style={{ cursor: 'pointer' }}
        onClick={() => setExpanded(!expanded)}
      >
        Tool: {toolCall.function.name}
        {toolCall.id && <span style={{ fontWeight: 400, marginLeft: '0.375rem', fontSize: '0.5625rem' }}>({toolCall.id})</span>}
        <span style={{ marginLeft: '0.375rem', fontSize: '0.5rem' }}>{expanded ? '\u25bc' : '\u25b6'}</span>
      </div>
      {expanded && <pre className="tool-call-code">{args}</pre>}
    </div>
  );
}

import { useEffect, useState } from 'react';
import { api } from '../api/client';
import type { LogtreeNode, LogtreeResponse } from '../api/types';

interface ConversationViewProps {
  runId: string;
  iteration: number;
  baseName?: string;
}

interface ConversationMessage {
  role: string;
  content: string | ContentPart[];
}

interface ContentPart {
  type: string;
  text?: string;
  thinking?: string;
  tool_call?: {
    function: { name: string; arguments: string };
  };
  raw_text?: string;
  error?: string;
}

/**
 * Extract conversation messages from logtree data.
 * The logtree `data` field contains `{type: "conversation", messages: [...]}`.
 */
function extractConversations(node: LogtreeNode): ConversationMessage[][] {
  const conversations: ConversationMessage[][] = [];

  if (node.data?.type === 'conversation' && Array.isArray(node.data?.messages)) {
    conversations.push(node.data.messages as ConversationMessage[]);
  }

  if (node.children) {
    for (const child of node.children) {
      if (typeof child !== 'string') {
        conversations.push(...extractConversations(child));
      }
    }
  }

  return conversations;
}

function renderContent(content: string | ContentPart[]): React.ReactElement {
  if (typeof content === 'string') {
    return <span>{content}</span>;
  }

  return (
    <>
      {content.map((part, i) => {
        if (part.type === 'text' && part.text) {
          return <span key={i}>{part.text}</span>;
        }
        if (part.type === 'thinking' && part.thinking) {
          return (
            <details key={i} style={{ margin: '0.25rem 0', padding: '0.5rem', background: 'rgba(251, 191, 36, 0.1)', borderRadius: 4, border: '1px solid rgba(251, 191, 36, 0.3)' }}>
              <summary style={{ cursor: 'pointer', fontWeight: 500, color: '#fbbf24', fontSize: '0.8125rem' }}>
                Thinking
              </summary>
              <pre style={{ marginTop: '0.25rem', fontSize: '0.8125rem', whiteSpace: 'pre-wrap', color: 'var(--text-secondary)' }}>
                {part.thinking}
              </pre>
            </details>
          );
        }
        if (part.type === 'tool_call' && part.tool_call) {
          return (
            <div key={i} style={{ margin: '0.25rem 0', padding: '0.5rem', background: 'rgba(59, 130, 246, 0.1)', borderRadius: 4, border: '1px solid rgba(59, 130, 246, 0.3)' }}>
              <span style={{ fontWeight: 500, color: '#60a5fa', fontSize: '0.8125rem' }}>
                Tool Call: {part.tool_call.function.name}
              </span>
              <pre className="mono" style={{ marginTop: '0.25rem', fontSize: '0.75rem', whiteSpace: 'pre-wrap' }}>
                {part.tool_call.function.arguments}
              </pre>
            </div>
          );
        }
        if (part.type === 'image') {
          return <span key={i} style={{ color: '#818cf8' }}>[Image]</span>;
        }
        return null;
      })}
    </>
  );
}

export function ConversationView({ runId, iteration, baseName = 'train' }: ConversationViewProps) {
  const [logtree, setLogtree] = useState<LogtreeResponse | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    api.getLogtree(runId, iteration, baseName)
      .then(setLogtree)
      .catch(() => {
        // Logtree may not exist for all iterations
      })
      .finally(() => setLoading(false));
  }, [runId, iteration, baseName]);

  if (loading) return <div className="loading">Loading conversation...</div>;
  if (!logtree) return null;

  const conversations = extractConversations(logtree.root);
  if (conversations.length === 0) return null;

  return (
    <div style={{ marginTop: '1rem' }}>
      <h3 style={{ marginBottom: '0.75rem' }}>Conversations</h3>
      {conversations.map((messages, convIdx) => (
        <div key={convIdx} className="card" style={{ marginBottom: '0.75rem' }}>
          <div className="conversation">
            {messages.map((msg, msgIdx) => (
              <div key={msgIdx} className={`message message-${msg.role}`}>
                <div className="message-role">{msg.role}</div>
                <div className="message-content">
                  {renderContent(msg.content)}
                </div>
              </div>
            ))}
          </div>
        </div>
      ))}
    </div>
  );
}

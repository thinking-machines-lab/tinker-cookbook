// Chat screen: the primary interface. A conversation sidebar, a streaming
// message thread over the chat websocket, tool calls as collapsible steps,
// and published visuals rendered inline as sandboxed iframes (the full
// gallery lives on the run's Visuals tab). `scope="run"` chats are bound to
// one run (rollout keys in answers link to the detail screen);
// `scope="global"` is the registry's cross-run chat.

import { useCallback, useEffect, useRef, useState } from "react";
import { useParams } from "react-router-dom";
import {
  apiBase,
  chatWsPath,
  getJSON,
  type AgentConfig,
  type ChatRecord,
  type ConversationSummary,
} from "../api";
import { AgentSettings } from "../components/AgentSettings";
import { MarkdownLite } from "../components/Markdown";
import { VisualFrame } from "../components/VisualFrame";
import { useApi } from "../hooks/useApi";
import { useWebSocket } from "../hooks/useWebSocket";
import { fmtRelative } from "../util";

type ChatItem =
  | { kind: "user"; text: string }
  | { kind: "assistant"; text: string }
  | {
      kind: "tool";
      id: string;
      name: string;
      arguments: Record<string, unknown>;
      result?: { isError: boolean; preview: string };
    }
  | { kind: "visual"; url: string; title: string; description: string }
  | { kind: "notice"; variant: "error" | "muted"; text: string };

/** Rebuild the thread items from a persisted JSONL transcript. */
function itemsFromRecords(records: ChatRecord[]): ChatItem[] {
  const items: ChatItem[] = [];
  const toolIndex = new Map<string, number>(); // tool_call id -> items index
  for (const record of records) {
    if (record.kind === "message") {
      if (record.role === "user") {
        items.push({ kind: "user", text: record.content ?? "" });
      } else if (record.role === "assistant") {
        if (record.content) items.push({ kind: "assistant", text: record.content });
        for (const call of record.tool_calls ?? []) {
          toolIndex.set(call.id, items.length);
          items.push({ kind: "tool", id: call.id, name: call.name, arguments: call.arguments });
        }
      } else if (record.role === "tool" && record.tool_call_id !== undefined) {
        const index = toolIndex.get(record.tool_call_id);
        const item = index !== undefined ? items[index] : undefined;
        if (item !== undefined && item.kind === "tool") {
          const content = record.content ?? "";
          items[index!] = {
            ...item,
            result: { isError: content.startsWith('{"error"'), preview: content.slice(0, 400) },
          };
        }
      }
    } else if (record.kind === "event" && record.type === "visual_published" && record.url) {
      items.push({
        kind: "visual",
        url: record.url,
        title: record.title ?? record.name ?? "visual",
        description: record.description ?? "",
      });
    }
  }
  return items;
}

/** Compact one-line summary of a tool call's arguments. */
function argsSummary(args: Record<string, unknown>): string {
  const json = JSON.stringify(args) ?? "{}";
  return json.length > 100 ? `${json.slice(0, 100)}…` : json;
}

function ToolStep({ item }: { item: Extract<ChatItem, { kind: "tool" }> }) {
  return (
    <details className="tool-step">
      <summary>
        {item.result === undefined ? (
          <span className="spinner" aria-label="running" />
        ) : (
          <span className={item.result.isError ? "tool-dot tool-dot-error" : "tool-dot"} />
        )}
        <span className="tool-name">{item.name}</span>
        <span className="tool-args-summary">{argsSummary(item.arguments)}</span>
      </summary>
      <div className="tool-step-body">
        <div className="turn-label">arguments</div>
        <pre>{JSON.stringify(item.arguments, null, 2)}</pre>
        {item.result !== undefined && (
          <>
            <div className="turn-label">{item.result.isError ? "error" : "result preview"}</div>
            <pre className={item.result.isError ? "error" : undefined}>{item.result.preview}</pre>
          </>
        )}
      </div>
    </details>
  );
}

/** Sidebar placeholder rows, sized like real conversation rows, shown
 * while the list is loading so it doesn't pop in and reflow. */
function SidebarSkeleton() {
  return (
    <>
      {[0, 1, 2, 3].map((i) => (
        <div key={i} aria-hidden="true" className="chat-convo">
          <div className="chat-convo-title">
            <span className="skeleton skeleton-line" style={{ width: `${72 - i * 9}%` }} />
          </div>
          <div className="chat-convo-time">
            <span className="skeleton skeleton-line" style={{ width: "40%" }} />
          </div>
        </div>
      ))}
    </>
  );
}

export function Chat({ scope }: { scope: "run" | "global" }) {
  const { runId } = useParams();
  const base = apiBase(runId);
  const linkRollouts = scope === "run";

  const [items, setItems] = useState<ChatItem[]>([]);
  const [activeId, setActiveId] = useState<string | null>(null);
  const [streaming, setStreaming] = useState(false);
  const [draft, setDraft] = useState("");
  const streamingRef = useRef(streaming);
  streamingRef.current = streaming;

  const config = useApi(() => getJSON<AgentConfig>("/api/agent/config"), []);
  const conversations = useApi(
    () => getJSON<{ conversations: ConversationSummary[] }>(`${base}/chats`),
    [base],
  );

  // Reset the thread when navigating to a different run's chat.
  useEffect(() => {
    setItems([]);
    setActiveId(null);
    setStreaming(false);
  }, [base]);

  const onMessage = useCallback(
    (msg: Record<string, unknown>) => {
      const type = msg.type;
      if (type === "conversation") {
        setActiveId((current) => current ?? String(msg.conversation_id));
      } else if (type === "text_delta") {
        setItems((prev) => {
          const last = prev[prev.length - 1];
          if (last !== undefined && last.kind === "assistant") {
            return [...prev.slice(0, -1), { kind: "assistant", text: last.text + String(msg.text) }];
          }
          return [...prev, { kind: "assistant", text: String(msg.text) }];
        });
      } else if (type === "tool_call") {
        setItems((prev) => [
          ...prev,
          {
            kind: "tool",
            id: String(msg.id),
            name: String(msg.name),
            arguments: (msg.arguments ?? {}) as Record<string, unknown>,
          },
        ]);
      } else if (type === "tool_result") {
        setItems((prev) =>
          prev.map((item) =>
            item.kind === "tool" && item.id === msg.id && item.result === undefined
              ? {
                  ...item,
                  result: { isError: Boolean(msg.is_error), preview: String(msg.preview ?? "") },
                }
              : item,
          ),
        );
      } else if (type === "visual_published") {
        setItems((prev) => [
          ...prev,
          {
            kind: "visual",
            url: String(msg.url),
            title: String(msg.title ?? msg.name ?? "visual"),
            description: String(msg.description ?? ""),
          },
        ]);
      } else if (type === "done") {
        setStreaming(false);
        conversations.reload();
      } else if (type === "cancelled") {
        setStreaming(false);
        setItems((prev) => [...prev, { kind: "notice", variant: "muted", text: "stopped" }]);
        conversations.reload();
      } else if (type === "error") {
        setStreaming(false);
        if (msg.code === "no_api_key") {
          config.reload(); // has_key=false surfaces the setup card below
        } else {
          setItems((prev) => [
            ...prev,
            { kind: "notice", variant: "error", text: String(msg.error ?? "chat error") },
          ]);
        }
      }
    },
    [conversations, config],
  );

  const { status, send } = useWebSocket(chatWsPath(runId), { onMessage });

  // A dropped socket kills any in-flight turn server-side.
  useEffect(() => {
    if (status === "offline" && streamingRef.current) {
      setStreaming(false);
      setItems((prev) => [
        ...prev,
        { kind: "notice", variant: "error", text: "connection lost; the turn was interrupted" },
      ]);
    }
  }, [status]);

  // Keep the thread pinned to the newest message while it streams in.
  const threadRef = useRef<HTMLDivElement>(null);
  useEffect(() => {
    const el = threadRef.current;
    if (el !== null) el.scrollTop = el.scrollHeight;
  }, [items]);

  const hasKey = config.data?.has_key ?? true; // optimistic until the config loads

  const sendMessage = () => {
    const text = draft.trim();
    if (!text || streaming || !hasKey) return;
    const message: Record<string, unknown> = { type: "user_message", text };
    if (activeId !== null) message.conversation_id = activeId;
    if (!send(message)) return; // still connecting; the composer hints at this
    setItems((prev) => [...prev, { kind: "user", text }]);
    setDraft("");
    setStreaming(true);
  };

  const stop = () => {
    send({ type: "cancel" });
  };

  const openConversation = async (conversationId: string) => {
    if (streaming) stop();
    setActiveId(conversationId);
    try {
      const payload = await getJSON<{ records: ChatRecord[] }>(`${base}/chats/${conversationId}`);
      setItems(itemsFromRecords(payload.records));
    } catch (error) {
      setItems([{ kind: "notice", variant: "error", text: (error as Error).message }]);
    }
  };

  const newChat = () => {
    if (streaming) stop();
    setActiveId(null);
    setItems([]);
    setStreaming(false);
  };

  const conversationList = conversations.data?.conversations ?? [];

  return (
    <div className="chat-layout">
      <aside className="chat-sidebar">
        <div className="chat-sidebar-label turn-label">Conversations</div>
        <button className="new-chat" onClick={newChat}>
          + New chat
        </button>
        <div className="chat-sidebar-list">
          {conversations.data === null && !conversations.error && <SidebarSkeleton />}
          {conversationList.map((conversation) => (
            <div
              key={conversation.conversation_id}
              className={
                conversation.conversation_id === activeId ? "chat-convo active" : "chat-convo"
              }
              onClick={() => void openConversation(conversation.conversation_id)}
            >
              <div className="chat-convo-title">{conversation.title || "(empty chat)"}</div>
              <div className="chat-convo-time muted">{fmtRelative(conversation.mtime)}</div>
            </div>
          ))}
          {conversations.data !== null && conversationList.length === 0 && (
            <p className="muted small chat-sidebar-empty">no conversations yet</p>
          )}
        </div>
      </aside>

      <section className="chat-main">
        <div className="chat-thread" ref={threadRef}>
          <div className="chat-column">
            {items.length === 0 && (
              <div className="chat-empty muted">
                <p>
                  Ask anything about {scope === "run" ? "this run's" : "your"} training data:
                  reward trends, odd rollouts, common failure modes. The agent queries the token
                  DB for you and can publish live-updating charts.
                </p>
              </div>
            )}
            {items.map((item, i) => {
              if (item.kind === "user") {
                return (
                  <div key={i} className="msg-user">
                    {item.text}
                  </div>
                );
              }
              if (item.kind === "assistant") {
                return (
                  <div key={i} className="msg-assistant">
                    <MarkdownLite text={item.text} linkRollouts={linkRollouts} />
                  </div>
                );
              }
              if (item.kind === "tool") return <ToolStep key={i} item={item} />;
              if (item.kind === "visual") {
                return (
                  <VisualFrame
                    key={i}
                    url={item.url}
                    title={item.title}
                    description={item.description}
                  />
                );
              }
              return (
                <p key={i} className={item.variant === "error" ? "error small" : "muted small"}>
                  {item.text}
                </p>
              );
            })}
            {streaming && <p className="muted small chat-thinking">thinking…</p>}
          </div>
        </div>

        <div className="chat-composer">
          <div className="chat-column">
            {config.data !== null && !hasKey && (
              <div className="setup-card">
                <p>
                  <strong>Set up the chat agent.</strong> Add an API key to start asking questions
                  about your training data.
                </p>
                <AgentSettings onSaved={() => config.reload()} />
              </div>
            )}
            <div className="composer-row">
              <textarea
                rows={2}
                placeholder={
                  hasKey
                    ? "Ask about this data… (Enter to send, Shift+Enter for a newline)"
                    : "add an API key above to enable chat"
                }
                disabled={!hasKey}
                value={draft}
                onChange={(event) => setDraft(event.target.value)}
                onKeyDown={(event) => {
                  if (event.key === "Enter" && !event.shiftKey) {
                    event.preventDefault();
                    sendMessage();
                  }
                }}
              />
              {streaming ? (
                <button className="primary" onClick={stop}>
                  Stop
                </button>
              ) : (
                <button
                  className="primary"
                  disabled={!hasKey || status !== "live" || !draft.trim()}
                  onClick={sendMessage}
                >
                  Send
                </button>
              )}
            </div>
            {/* Fixed-height strip in every state so the composer never moves
                when the socket connects or drops. */}
            <div className="composer-status muted small">
              {status === "connecting" && "connecting…"}
              {status === "offline" && "offline: is the viewer server running?"}
            </div>
          </div>
        </div>
      </section>
    </div>
  );
}

// Chat screen: the primary interface. A conversation sidebar, a streaming
// message thread over the chat websocket, tool calls as collapsible steps,
// and published visuals rendered inline as sandboxed iframes (the full
// gallery lives on the run's Visuals tab). `scope="run"` chats are bound to
// one run (rollout keys in answers link to the detail screen);
// `scope="global"` is the registry's cross-run chat.
//
// Turns run server-side to completion: the socket is only a *subscriber*.
// The thread is a fold over transcript records (each with a monotonically
// increasing `seq`); the same reducer handles the HTTP-loaded transcript,
// the websocket replay, and the live tail, so navigating away, reconnecting,
// or watching from a second tab all converge on the same thread.

import { useCallback, useEffect, useRef, useState } from "react";
import { useParams, useSearchParams } from "react-router-dom";
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

/** Incremental fold state over a conversation's transcript records. */
interface ThreadState {
  items: ChatItem[];
  /** tool_call id -> items index, for attaching results (idempotent). */
  toolIndex: Map<string, number>;
  /** Index of the assistant item currently being built from text deltas;
   * the segment's assistant *message* record finalizes (replaces) it. */
  openAssistant: number | null;
  /** Highest seq applied; records at or below it are duplicates. */
  lastSeq: number;
}

function newThreadState(): ThreadState {
  return { items: [], toolIndex: new Map(), openAssistant: null, lastSeq: -1 };
}

/**
 * Apply one transcript record. Handles both the current stream shape
 * (text_delta / tool_call / tool_result events interleaved with message
 * records) and pre-seq transcripts that only contain message records.
 */
function applyRecord(state: ThreadState, record: ChatRecord): void {
  if (typeof record.seq === "number") {
    if (record.seq <= state.lastSeq) return; // replay overlap: drop duplicates
    state.lastSeq = record.seq;
  }
  const items = state.items;
  if (record.kind === "message") {
    if (record.role === "user") {
      items.push({ kind: "user", text: record.content ?? "" });
      state.openAssistant = null;
    } else if (record.role === "assistant") {
      if (state.openAssistant !== null) {
        // Finalize the delta-built item with the canonical segment text.
        const open = items[state.openAssistant];
        if (open !== undefined && open.kind === "assistant" && record.content) {
          items[state.openAssistant] = { kind: "assistant", text: record.content };
        }
        state.openAssistant = null;
      } else if (record.content) {
        items.push({ kind: "assistant", text: record.content });
      }
      for (const call of record.tool_calls ?? []) {
        if (!state.toolIndex.has(call.id)) {
          state.toolIndex.set(call.id, items.length);
          items.push({ kind: "tool", id: call.id, name: call.name, arguments: call.arguments });
        }
      }
    } else if (record.role === "tool" && record.tool_call_id !== undefined) {
      const content = record.content ?? "";
      setToolResult(state, record.tool_call_id, {
        isError: content.startsWith('{"error"'),
        preview: content.slice(0, 400),
      });
    }
    return;
  }
  // kind === "event"
  if (record.type === "text_delta") {
    const text = record.text ?? "";
    const open = state.openAssistant !== null ? items[state.openAssistant] : undefined;
    if (open !== undefined && open.kind === "assistant" && state.openAssistant !== null) {
      items[state.openAssistant] = { kind: "assistant", text: open.text + text };
    } else {
      state.openAssistant = items.length;
      items.push({ kind: "assistant", text });
    }
  } else if (record.type === "tool_call" && record.id !== undefined) {
    if (!state.toolIndex.has(record.id)) {
      state.toolIndex.set(record.id, items.length);
      items.push({
        kind: "tool",
        id: record.id,
        name: record.name ?? "tool",
        arguments: record.arguments ?? {},
      });
    }
  } else if (record.type === "tool_result" && record.id !== undefined) {
    setToolResult(state, record.id, {
      isError: Boolean(record.is_error),
      preview: record.preview ?? "",
    });
  } else if (record.type === "visual_published" && record.url) {
    items.push({
      kind: "visual",
      url: record.url,
      title: record.title ?? record.name ?? "visual",
      description: record.description ?? "",
    });
  } else if (record.type === "error") {
    items.push({ kind: "notice", variant: "error", text: record.error ?? "chat error" });
  } else if (record.type === "cancelled") {
    items.push({ kind: "notice", variant: "muted", text: "stopped" });
  }
  // "done" adds nothing to the thread; the caller flips the in-flight state.
}

function setToolResult(
  state: ThreadState,
  callId: string,
  result: { isError: boolean; preview: string },
): void {
  const index = state.toolIndex.get(callId);
  const item = index !== undefined ? state.items[index] : undefined;
  // First writer wins: the tool_result event precedes the tool message
  // record, which then no-ops (and old transcripts only have the record).
  if (index !== undefined && item !== undefined && item.kind === "tool" && item.result === undefined) {
    state.items[index] = { ...item, result };
  }
}

function isTerminal(record: ChatRecord): boolean {
  return (
    record.kind === "event" &&
    (record.type === "done" || record.type === "error" || record.type === "cancelled")
  );
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

/** How often the sidebar refreshes in_flight/activity from /api/chats. */
const SIDEBAR_POLL_MS = 8000;
/** "At the bottom" tolerance for sticky autoscroll, in px. */
const STICKY_BOTTOM_PX = 40;

export function Chat({ scope }: { scope: "run" | "global" }) {
  const { runId } = useParams();
  const [searchParams, setSearchParams] = useSearchParams();
  const base = apiBase(runId);
  const linkRollouts = scope === "run";

  const [items, setItems] = useState<ChatItem[]>([]);
  const [activeId, setActiveId] = useState<string | null>(null);
  const [streaming, setStreaming] = useState(false);
  const [draft, setDraft] = useState("");

  const threadStateRef = useRef<ThreadState>(newThreadState());
  const activeIdRef = useRef<string | null>(null);
  activeIdRef.current = activeId;
  // Drop in-transit records from a previously subscribed conversation after
  // switching threads, until the server acks the new context.
  const ignoreRecordsRef = useRef(false);

  const config = useApi(() => getJSON<AgentConfig>("/api/agent/config"), []);
  const conversations = useApi(
    () => getJSON<{ conversations: ConversationSummary[] }>(`${base}/chats`),
    [base],
  );

  // --- Sticky-bottom autoscroll ---
  // Autoscroll on new content only while the user is at (within
  // STICKY_BOTTOM_PX of) the bottom. Scrolling up detaches; a "Jump to
  // latest" pill appears when content arrives while detached, and scrolling
  // back to the bottom re-sticks automatically.
  const threadRef = useRef<HTMLDivElement>(null);
  const stuckRef = useRef(true);
  const [showJump, setShowJump] = useState(false);

  const onThreadScroll = () => {
    const el = threadRef.current;
    if (el === null) return;
    const atBottom = el.scrollHeight - el.scrollTop - el.clientHeight < STICKY_BOTTOM_PX;
    stuckRef.current = atBottom;
    if (atBottom) setShowJump(false);
  };

  useEffect(() => {
    const el = threadRef.current;
    if (el === null || items.length === 0) return;
    if (stuckRef.current) {
      el.scrollTop = el.scrollHeight;
    } else {
      setShowJump(true);
    }
  }, [items, streaming]);

  const jumpToLatest = () => {
    const el = threadRef.current;
    if (el !== null) el.scrollTo({ top: el.scrollHeight, behavior: "smooth" });
    stuckRef.current = true;
    setShowJump(false);
  };

  const resetThread = useCallback(() => {
    threadStateRef.current = newThreadState();
    setItems([]);
    stuckRef.current = true;
    setShowJump(false);
  }, []);

  const ingest = useCallback(
    (records: ChatRecord[]) => {
      const state = threadStateRef.current;
      for (const record of records) applyRecord(state, record);
      setItems([...state.items]);
    },
    [],
  );

  const pushNotice = useCallback((variant: "error" | "muted", text: string) => {
    threadStateRef.current.items.push({ kind: "notice", variant, text });
    setItems([...threadStateRef.current.items]);
  }, []);

  const onMessage = useCallback(
    (msg: Record<string, unknown>) => {
      if (typeof msg.kind === "string") {
        if (ignoreRecordsRef.current) return; // stale conversation's tail
        const record = msg as unknown as ChatRecord;
        ingest([record]);
        if (isTerminal(record)) {
          setStreaming(false);
          conversations.reload();
        }
        return;
      }
      const type = msg.type;
      if (type === "conversation") {
        ignoreRecordsRef.current = false;
        setActiveId((current) => current ?? String(msg.conversation_id));
        setStreaming(true);
      } else if (type === "subscribed_conversation") {
        if (msg.conversation_id === activeIdRef.current) {
          ignoreRecordsRef.current = false;
          setStreaming(Boolean(msg.in_flight));
        }
      } else if (type === "error") {
        if (msg.code === "no_api_key") {
          config.reload(); // has_key=false surfaces the setup card below
        } else {
          pushNotice("error", String(msg.error ?? "chat error"));
        }
        if (msg.code !== "turn_in_flight") setStreaming(false);
      }
    },
    [conversations, config, ingest, pushNotice],
  );

  const { status, send } = useWebSocket(chatWsPath(runId), {
    onMessage,
    // On every (re)connect, resubscribe from the last seq we applied: the
    // server replays anything missed while offline, then tails live. An
    // in-flight turn survives disconnects untouched.
    onOpen: () => {
      const id = activeIdRef.current;
      if (id !== null) {
        send({
          type: "subscribe_conversation",
          conversation_id: id,
          after_seq: threadStateRef.current.lastSeq,
        });
      }
    },
  });

  const openConversation = useCallback(
    async (conversationId: string) => {
      setActiveId(conversationId);
      activeIdRef.current = conversationId;
      ignoreRecordsRef.current = true;
      resetThread();
      setStreaming(false);
      try {
        const payload = await getJSON<{ records: ChatRecord[] }>(
          `${base}/chats/${conversationId}`,
        );
        if (activeIdRef.current !== conversationId) return; // switched away
        ingest(payload.records);
        // Subscribe past what we loaded; the ack reports in_flight and any
        // records persisted since the load are replayed with no duplicates.
        send({
          type: "subscribe_conversation",
          conversation_id: conversationId,
          after_seq: threadStateRef.current.lastSeq,
        });
      } catch (error) {
        pushNotice("error", (error as Error).message);
      }
    },
    [base, ingest, pushNotice, resetThread, send],
  );

  // Reset when navigating to a different run's chat; ?c= opens a
  // conversation directly (e.g. from the dashboard's recent-chats card).
  useEffect(() => {
    setActiveId(null);
    activeIdRef.current = null;
    ignoreRecordsRef.current = true;
    resetThread();
    setStreaming(false);
    const requested = searchParams.get("c");
    if (requested !== null && requested !== "") {
      void openConversation(requested);
      setSearchParams({}, { replace: true }); // one-shot deep link
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [base]);

  // Keep the sidebar's in_flight badges and activity times fresh.
  useEffect(() => {
    const timer = setInterval(() => conversations.reload(), SIDEBAR_POLL_MS);
    return () => clearInterval(timer);
  }, [conversations.reload]);

  const hasKey = config.data?.has_key ?? true; // optimistic until the config loads

  const sendMessage = () => {
    const text = draft.trim();
    if (!text || streaming || !hasKey) return;
    const message: Record<string, unknown> = { type: "user_message", text };
    if (activeId !== null) message.conversation_id = activeId;
    if (!send(message)) return; // still connecting; the composer hints at this
    // The user message comes back as the turn's first transcript record.
    setDraft("");
    setStreaming(true);
    stuckRef.current = true;
    setShowJump(false);
  };

  const stop = () => {
    // The turn is server-owned, so Stop works from any subscribed client.
    const message: Record<string, unknown> = { type: "cancel" };
    if (activeId !== null) message.conversation_id = activeId;
    send(message);
  };

  const newChat = () => {
    setActiveId(null);
    activeIdRef.current = null;
    ignoreRecordsRef.current = true;
    resetThread();
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
              <div className="chat-convo-title">
                {conversation.in_flight === true && (
                  <span className="pulse-dot" title="a turn is running" />
                )}
                {conversation.title || "(empty chat)"}
              </div>
              <div className="chat-convo-time muted">
                {conversation.in_flight === true ? "running…" : fmtRelative(conversation.mtime)}
              </div>
            </div>
          ))}
          {conversations.data !== null && conversationList.length === 0 && (
            <p className="muted small chat-sidebar-empty">no conversations yet</p>
          )}
        </div>
      </aside>

      <section className="chat-main">
        <div className="chat-thread-wrap">
          <div className="chat-thread" ref={threadRef} onScroll={onThreadScroll}>
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
          {showJump && (
            <button className="jump-latest" onClick={jumpToLatest}>
              <span className="jump-latest-dot" aria-hidden="true" />
              Jump to latest
            </button>
          )}
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
              {status === "offline" &&
                (streaming
                  ? "reconnecting… the turn keeps running server-side"
                  : "offline: is the viewer server running?")}
            </div>
          </div>
        </div>
      </section>
    </div>
  );
}

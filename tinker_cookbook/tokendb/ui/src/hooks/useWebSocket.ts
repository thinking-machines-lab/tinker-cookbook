import { useCallback, useEffect, useRef, useState } from "react";
import { wsUrl } from "../api";

export type WsStatus = "connecting" | "live" | "offline";

export interface WsOptions {
  /** Handler for every parsed JSON message. Kept in a ref: changing it does not reconnect. */
  onMessage: (msg: Record<string, unknown>) => void;
  /** Message to send once the socket opens (e.g. a subscribe request). */
  sendOnOpen?: unknown;
  /** Called after every (re)connect. Kept in a ref: changing it does not
   * reconnect. Use it to resubscribe with up-to-date state (e.g. the last
   * seen transcript seq), which a static `sendOnOpen` can't carry. */
  onOpen?: () => void;
  /** Reconnect delay after a drop (ms). */
  reconnectMs?: number;
}

export interface WsHandle {
  status: WsStatus;
  /** Send a JSON message; returns false if the socket is not open. */
  send: (msg: unknown) => boolean;
}

/**
 * Maintain a websocket to `path` (null disables), reconnecting on drops.
 * Reconnects (and resubscribes) when `path` or the serialized `sendOnOpen`
 * message changes.
 */
export function useWebSocket(path: string | null, options: WsOptions): WsHandle {
  const [status, setStatus] = useState<WsStatus>("connecting");
  const onMessageRef = useRef(options.onMessage);
  onMessageRef.current = options.onMessage;
  const onOpenRef = useRef(options.onOpen);
  onOpenRef.current = options.onOpen;
  const wsRef = useRef<WebSocket | null>(null);
  const reconnectMs = options.reconnectMs ?? 2000;
  const sendOnOpenJson =
    options.sendOnOpen === undefined ? null : JSON.stringify(options.sendOnOpen);

  useEffect(() => {
    if (path === null) return;
    let closed = false;
    let timer: ReturnType<typeof setTimeout> | null = null;

    const connect = () => {
      setStatus("connecting");
      const ws = new WebSocket(wsUrl(path));
      wsRef.current = ws;
      ws.onopen = () => {
        setStatus("live");
        if (sendOnOpenJson !== null) ws.send(sendOnOpenJson);
        onOpenRef.current?.();
      };
      ws.onmessage = (event) => {
        try {
          onMessageRef.current(JSON.parse(event.data));
        } catch {
          /* ignore non-JSON frames */
        }
      };
      ws.onclose = () => {
        setStatus("offline");
        if (!closed) timer = setTimeout(connect, reconnectMs);
      };
    };
    connect();

    return () => {
      closed = true;
      if (timer !== null) clearTimeout(timer);
      wsRef.current?.close();
      wsRef.current = null;
    };
  }, [path, sendOnOpenJson, reconnectMs]);

  const send = useCallback((msg: unknown): boolean => {
    const ws = wsRef.current;
    if (ws === null || ws.readyState !== WebSocket.OPEN) return false;
    ws.send(JSON.stringify(msg));
    return true;
  }, []);

  return { status, send };
}

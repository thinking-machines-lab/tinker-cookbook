import { useEffect, useRef, useState } from "react";
import { wsUrl } from "../api";

export type WsStatus = "connecting" | "live" | "offline";

export interface WsOptions {
  /** Handler for every parsed JSON message. Kept in a ref: changing it does not reconnect. */
  onMessage: (msg: Record<string, unknown>) => void;
  /** Message to send once the socket opens (e.g. a subscribe request). */
  sendOnOpen?: unknown;
  /** Reconnect delay after a drop (ms). */
  reconnectMs?: number;
}

/**
 * Maintain a websocket to `path` (null disables), reconnecting on drops.
 * Reconnects (and resubscribes) when `path` or the serialized `sendOnOpen`
 * message changes.
 */
export function useWebSocket(path: string | null, options: WsOptions): WsStatus {
  const [status, setStatus] = useState<WsStatus>("connecting");
  const onMessageRef = useRef(options.onMessage);
  onMessageRef.current = options.onMessage;
  const reconnectMs = options.reconnectMs ?? 2000;
  const sendOnOpenJson =
    options.sendOnOpen === undefined ? null : JSON.stringify(options.sendOnOpen);

  useEffect(() => {
    if (path === null) return;
    let ws: WebSocket | null = null;
    let closed = false;
    let timer: ReturnType<typeof setTimeout> | null = null;

    const connect = () => {
      setStatus("connecting");
      ws = new WebSocket(wsUrl(path));
      ws.onopen = () => {
        setStatus("live");
        if (sendOnOpenJson !== null) ws?.send(sendOnOpenJson);
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
      ws?.close();
    };
  }, [path, sendOnOpenJson, reconnectMs]);

  return status;
}

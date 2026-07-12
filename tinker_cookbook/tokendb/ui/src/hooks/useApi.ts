import { useCallback, useEffect, useRef, useState } from "react";

export interface ApiState<T> {
  data: T | null;
  error: string | null;
  loading: boolean;
  reload: () => void;
}

/**
 * Run an async fetcher and track its data/error/loading state. The fetcher
 * re-runs whenever `deps` change; `reload()` re-runs it in place. Responses
 * from stale requests are discarded.
 */
export function useApi<T>(fetcher: () => Promise<T>, deps: unknown[]): ApiState<T> {
  const [data, setData] = useState<T | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const requestId = useRef(0);
  const fetcherRef = useRef(fetcher);
  fetcherRef.current = fetcher;

  const reload = useCallback(() => {
    const id = ++requestId.current;
    setLoading(true);
    setError(null);
    fetcherRef.current().then(
      (result) => {
        if (id !== requestId.current) return;
        setData(result);
        setLoading(false);
      },
      (err: Error) => {
        if (id !== requestId.current) return;
        setError(err.message);
        setLoading(false);
      },
    );
  }, []);

  // eslint-disable-next-line react-hooks/exhaustive-deps
  useEffect(reload, deps);
  return { data, error, loading, reload };
}

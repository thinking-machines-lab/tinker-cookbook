import { useCallback } from 'react';
import { useSearchParams } from 'react-router-dom';

/**
 * Sync a single query parameter to React state.
 *
 * Setting the value to the default removes the parameter from the URL.
 * Uses `replace: true` so back button behavior stays clean.
 */
export function useUrlParam(
  key: string,
  defaultValue: string = '',
): [string, (value: string) => void] {
  const [searchParams, setSearchParams] = useSearchParams();
  const value = searchParams.get(key) ?? defaultValue;

  const setValue = useCallback(
    (v: string) => {
      setSearchParams((prev) => {
        const next = new URLSearchParams(prev);
        if (v === defaultValue || v === '') {
          next.delete(key);
        } else {
          next.set(key, v);
        }
        return next;
      }, { replace: true });
    },
    [key, defaultValue, setSearchParams],
  );

  return [value, setValue];
}

/**
 * Like useUrlParam but for numeric values.
 */
export function useUrlParamNum(
  key: string,
  defaultValue: number | null = null,
): [number | null, (value: number | null) => void] {
  const [raw, setRaw] = useUrlParam(key, '');
  const value = raw !== '' ? Number(raw) : defaultValue;

  const setValue = useCallback(
    (v: number | null) => {
      setRaw(v != null ? String(v) : '');
    },
    [setRaw],
  );

  return [value, setValue];
}

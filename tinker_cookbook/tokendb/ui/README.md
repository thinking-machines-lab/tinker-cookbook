# Token DB viewer UI

React + TypeScript (Vite) frontend for the token DB viewer
(`python -m tinker_cookbook.tokendb.serve`).

End users never need node: the built bundle is committed as package data
under `tinker_cookbook/tokendb/static/` and served by the python CLI.

## Dev loop

Start the viewer server, then the Vite dev server (which proxies `/api`
and `/ws` to it):

```bash
# single-run mode (one run's store) ...
python -m tinker_cookbook.tokendb.serve log_path=~/runs/my-run port=7423
# ... or registry mode (dashboard of every registered run)
python -m tinker_cookbook.tokendb.serve port=7423

cd tinker_cookbook/tokendb/ui
npm install
npm run dev   # http://localhost:5173, hot reload
```

The app probes the server mode once at startup (`/api/runs` exists only in
registry mode) and routes accordingly.

## Build

```bash
cd tinker_cookbook/tokendb/ui
npm run build   # typechecks, then emits ../static/ (committed)
```

## Layout

- `src/api.ts` — typed HTTP client, server-mode detection, endpoint prefixes
- `src/App.tsx` — hash routes; per-run screens mount under both `#/run`
  (single-run mode) and `#/runs/{run_id}` (registry mode)
- `src/screens/` — one file per screen: `Dashboard`, `Feed`, `Detail`, `Search`
- `src/components/` — shared pieces: `Badge`, `Chip`, `StatCard`, `Sparkline`,
  `DataTable`, `FilterBar`, `TokenSpan`
- `src/hooks/` — `useApi` (fetch state), `useWebSocket` (reconnecting socket),
  `useDebounce`
- `src/style.css` — single stylesheet; design tokens as CSS variables in `:root`

## Screens

- `#/` — dashboard (registry mode): every registered run with liveness,
  latest iteration, row counts, recent reward, and a reward sparkline;
  live-updated over `/ws/dashboard`. In single-run mode `#/` redirects to
  the feed.
- `#/run` or `#/runs/{run_id}` — live feed: trajectory table with filters,
  websocket updates (follow toggle), reward sparkline. Superseded run
  attempts are dimmed and badged, not hidden.
- `.../rollout/{split}/{iter}/{group}/{traj}` — full transcript: per-token
  action spans colored by logprob, delta observations with an
  expand-full-context control, text vs raw-token-ID toggle, labels editor.
- `.../search` — regex and token-ID-subsequence search with per-iteration
  hit counts, plus a SELECT-only SQL console.

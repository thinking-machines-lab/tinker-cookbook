# Token DB viewer UI

React + TypeScript (Vite) frontend for the token DB viewer
(`python -m tinker_cookbook.tokendb_studio.serve`).

End users never need node: the built bundle is committed as package data
under `tinker_cookbook/tokendb_studio/static/` and served by the python CLI.

## Dev loop

Start the viewer server, then the Vite dev server (which proxies `/api`
and `/ws` to it):

```bash
# single-run mode (one run's store) ...
python -m tinker_cookbook.tokendb_studio.serve log_path=~/runs/my-run port=7423
# ... or registry mode (dashboard of every registered run)
python -m tinker_cookbook.tokendb_studio.serve port=7423

cd tinker_cookbook/tokendb_studio/ui
npm install
npm run dev   # http://localhost:5173, hot reload
```

The app probes the server mode once at startup (`/api/runs` exists only in
registry mode) and routes accordingly.

## Build

```bash
cd tinker_cookbook/tokendb_studio/ui
npm run build   # typechecks, then emits ../static/ (committed)
```

## Layout

- `src/api.ts` — typed HTTP client, server-mode detection, endpoint prefixes
- `src/App.tsx` — hash routes; per-run screens mount under both `#/run`
  (single-run mode) and `#/runs/{run_id}` (registry mode)
- `src/screens/` — one file per screen: `Dashboard`, `Chat`, `Detail`
- `src/components/` — shared pieces: `Badge`, `Chip`, `StatCard`, `Sparkline`,
  `DataTable`, `TokenSpan`, `Markdown` (tiny renderer, no dependency),
  `VisualFrame` (sandboxed visual iframes), `AgentSettings`
- `src/hooks/` — `useApi` (fetch state), `useWebSocket` (reconnecting socket
  with a `send` function)
- `src/style.css` — single stylesheet; design tokens as CSS variables in `:root`

## Screens

Chat is the primary interface: instead of reading a dense feed or writing
SQL, you ask questions and the server-side agent queries the token DB.

- `#/` — dashboard (registry mode): every registered run with liveness,
  latest iteration, row counts, recent reward, and a reward sparkline;
  live-updated over `/ws/dashboard`. Clicking a run opens that run's chat;
  a "Chat across all runs" button opens the registry-level chat at `#/chat`.
  In single-run mode `#/` redirects to `#/run/chat`.
- `#/run/chat` or `#/runs/{run_id}/chat` — chat: conversation sidebar with a
  visuals-gallery tab, streaming answers, tool calls as collapsible steps,
  published visuals inline as sandboxed iframes (maximize / open in new tab),
  and rollout keys in answers (e.g. `train/12/3/1`) linked to the detail
  screen. An inline setup card (provider / model / API key) appears when no
  key is configured; the gear in the header opens the same settings any time.
- `.../rollout/{split}/{iter}/{group}/{traj}` — full transcript: per-token
  action spans colored by logprob, delta observations with an
  expand-full-context control, text vs raw-token-ID toggle, labels editor.
  Reached from chat citations and dashboard links.

# Token DB viewer UI

Vite + vanilla TypeScript frontend for the token DB viewer
(`python -m tinker_cookbook.tokendb.serve`). No framework, no runtime
dependencies; `vite` and `typescript` are the only dev dependencies.

End users never need node: the built bundle is committed as package data
under `tinker_cookbook/tokendb/static/` and served by the python CLI.

## Dev loop

Start the viewer server against a run, then the Vite dev server (which
proxies `/api` and `/ws` to it):

```bash
python -m tinker_cookbook.tokendb.serve log_path=~/runs/my-run port=7423

cd tinker_cookbook/tokendb/ui
npm install
npm run dev   # http://localhost:5173, hot reload
```

## Build

```bash
cd tinker_cookbook/tokendb/ui
npm run build   # typechecks, then emits ../static/ (committed)
```

## Screens

- `#/` — live feed: trajectory table with filters, websocket updates
  (follow toggle), reward sparkline. Superseded run attempts are dimmed
  and badged, not hidden.
- `#/rollout/{split}/{iter}/{group}/{traj}` — full transcript: per-token
  action spans colored by logprob, delta observations with an
  expand-full-context control, text vs raw-token-ID toggle, labels editor.
- `#/search` — regex and token-ID-subsequence search with per-iteration
  hit counts, plus a SELECT-only SQL console.

# Browser acceptance checks

Use synthetic fixtures or explicitly authorized captures. Include a reversed candidate pair, a missing comparison partner, a duplicate row number in another run, an unavailable transcript, a long response, Unicode, literal HTML, and a failed request. Never use sealed-test records as convenient fixtures.

## Identity and evidence

- Open exact run, pair/example, and trajectory links in a fresh tab. Refresh and use Back/Forward. Confirm the same split, checkpoint, attempt, turn, and source text are selected.
- Sort, filter, paginate, and update the list while an inspector is open. Confirm selection stays attached to identity, not position. Missing records and hidden selections must be explicit.
- Reverse candidate presentation. Verify displayed A/B maps to the saved original identities and the label is transformed correctly. Missing or conflicting orientation must not produce a confident label.
- Compare checkpoints with different missing rows. Confirm unmatched examples are shown and unrelated examples are not paired.
- Inspect raw and rendered text. Verify whitespace, escaping, provenance, redaction notices, and decoded-versus-original labels. A string such as `<img src=x onerror=alert(1)>` must remain inert and must not trigger a network request.
- Verify absent reasoning is labeled unavailable, not synthesized or replaced with another rollout's text.
- Inspect a multi-turn tool interaction with repeated tool names, distinct call IDs, an error, and a missing result. Verify message order, call/result pairing, literal arguments and outputs, attempt boundaries, and truncation notices.

## W&B links and data boundaries

- Verify 'Open in W&B' uses the saved or user-supplied URL for the selected run, including after navigation. A missing URL should produce an explicit unavailable-link state, not a guessed destination.
- Confirm the rollout UI has no metric charts, aggregate training summaries, or embedded metric dashboards. Inspect network requests: the viewer must not fetch metric series or copy W&B data to recreate those views.
- Verify TRAIN and validation captures remain separate in lists, filters, and links. Inspect network requests: a TRAIN-only page must not prefetch evaluation data.
- Request restricted records directly through API routes, search, exports, and streams. Verify the server denies them, even if a user constructs the URL manually. Inspect responses and cached payloads for sealed data, not only visible elements.

## Interaction and rendering

- Verify the rollout interface uses [Blueprint](https://blueprintjs.com/), the prescribed default component framework. Check consistent controls, menus, and inspector styling; if embedded in a larger app, verify its styles do not leak into unrelated pages.
- Test a wide desktop viewport, a narrow mobile viewport, and 200% zoom. Long text must wrap without forcing page-wide horizontal scrolling. Code blocks may scroll independently.
- Use only the keyboard to select a run, change filters, move between trajectories, resize a splitter, open and close a drawer, and follow a deep link. Check visible focus, skip navigation, accessible names, and focus restoration.
- Reopen a previously hidden virtualized list. It must measure correctly, render rows, and reveal the selected item without trapping focus.
- Inspect diff additions/deletions in light and dark themes. Blue/red must have adequate contrast and non-color indicators. Unified or stacked mobile diffs must preserve candidate identity.
- Block lazy icon chunks and icon fonts before loading the page. Next/previous, expand, and refresh must still paint usable icons. Inspect actual SVG path geometry and a screenshot, not only node existence. Accessible labels must work without the image.

## Loading and recovery

- Test initial loading, no runs, no matches, missing capture, removed record, and server error separately. A failed fetch must not become an empty successful transcript.
- Delay a detail response, navigate elsewhere, then release it. The old response must not overwrite the new selection.
- Fail a refresh after a successful load. Last good data should remain labeled stale; retry should recover without a page reload.
- Replace or truncate a source while the viewer is open. Verify generation/revision changes clear obsolete records rather than merging incompatible histories.

## Multi-tab connection regression

Use a production build served over real HTTP/1.1, without a development server's hot-reload socket. Mock the data endpoints if needed, but let the event-stream endpoint hold a real socket. Open seven pages in one browser context on the same origin; separate contexts may not reproduce the shared connection pool.

1. Verify there is at most one active application event stream across all tabs. Count concurrent live streams, not historical attempts or reconnects.
2. Navigate and load details in every tab. Verify APIs and critical controls remain responsive and follower tabs receive fresh data through polling.
3. Close the stream-owning tab. Verify another tab takes ownership, the old socket closes, and followers continue refreshing.
4. Disable Web Locks and repeat. Expect polling in all tabs and no per-tab event streams.
5. Drop the stream, fail an API response, then recover the server. Verify reconnect/retry behavior and accurate connection/freshness indicators.
6. Include a healthy slow detail response. It should eventually render, not be aborted by a short workaround timeout.

Keep finite test deadlines and assert eventual content so hangs fail clearly. Those test deadlines are not application request-timeout recommendations. Record the browser, transport, build mode, and which failures were injected with the test results.

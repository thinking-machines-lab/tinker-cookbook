// Entry point: hash-based routing between the three screens.
//   #/                                     feed
//   #/rollout/{split}/{iter}/{group}/{traj}[?run_attempt=N]   detail
//   #/search                               search + SQL console

import "./style.css";
import { getJSON } from "./api";
import { renderDetail, type DetailParams } from "./detail";
import { renderFeed } from "./feed";
import { renderSearch } from "./search";
import { el } from "./util";

const app = document.getElementById("app")!;
const runInfo = el("span", { class: "muted small" });
const header = el("header", {}, [
  el("span", { class: "brand" }, ["Token DB"]),
  el("nav", {}, [
    el("a", { href: "#/" }, ["Feed"]),
    el("a", { href: "#/search" }, ["Search / SQL"]),
  ]),
  runInfo,
]);
const main = el("main");
app.replaceChildren(header, main);

void getJSON<{ run_id: string; run_attempt: number; context?: { model_name?: string } }>(
  "/api/run",
)
  .then((run) => {
    const model = run.context?.model_name ?? "unknown model";
    runInfo.textContent = `${model} · run ${run.run_id} · attempt ${run.run_attempt}`;
  })
  .catch(() => {
    runInfo.textContent = "run.json not found";
  });

let cleanup: () => void = () => {};

function parseDetail(path: string[], queryString: string): DetailParams | null {
  if (path.length !== 5) return null;
  const [, split, iteration, group, traj] = path;
  const params: DetailParams = {
    split: decodeURIComponent(split),
    iteration: Number(iteration),
    group_idx: Number(group),
    traj_idx: Number(traj),
  };
  const attempt = new URLSearchParams(queryString).get("run_attempt");
  if (attempt !== null) params.run_attempt = Number(attempt);
  return params;
}

function route(): void {
  cleanup();
  cleanup = () => {};
  const hash = location.hash.slice(1) || "/";
  const [pathPart, queryString = ""] = hash.split("?");
  const path = pathPart.split("/").filter(Boolean);

  if (path.length === 0) {
    cleanup = renderFeed(main);
  } else if (path[0] === "search") {
    cleanup = renderSearch(main);
  } else if (path[0] === "rollout") {
    const params = parseDetail(path, queryString);
    if (params) {
      cleanup = renderDetail(main, params);
    } else {
      main.replaceChildren(el("p", { class: "error" }, ["bad rollout URL"]));
    }
  } else {
    main.replaceChildren(el("p", { class: "error" }, [`unknown route: ${hash}`]));
  }
}

window.addEventListener("hashchange", route);
route();

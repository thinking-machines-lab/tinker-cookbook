"""Timing data API routes."""

from typing import Any

from fastapi import APIRouter, Query

from tinker_cookbook.chef.routes._helpers import require_run
from tinker_cookbook.stores import RunRegistry


def create_router(registry: RunRegistry) -> APIRouter:
    router = APIRouter(prefix="/api/runs", tags=["timing"])

    @router.get("/{run_id}/timing")
    async def get_timing(
        run_id: str,
        step_start: int | None = Query(None),
        step_end: int | None = Query(None),
    ) -> dict[str, Any]:
        require_run(registry, run_id)
        records = registry.get_training_store(run_id).read_timing()
        if step_start is not None:
            records = [r for r in records if r.get("step", 0) >= step_start]
        if step_end is not None:
            records = [r for r in records if r.get("step", 0) <= step_end]
        return {"run_id": run_id, "total_records": len(records), "records": records}

    @router.get("/{run_id}/timing/flat")
    async def get_timing_flat(
        run_id: str,
        step_start: int | None = Query(None),
        step_end: int | None = Query(None),
    ) -> dict[str, Any]:
        require_run(registry, run_id)
        spans = _flatten_spans(
            registry.get_training_store(run_id).read_timing(),
            step_start=step_start, step_end=step_end,
        )
        return {"run_id": run_id, "total_spans": len(spans), "spans": spans}

    @router.get("/{run_id}/timing/concurrency/{step}")
    async def get_concurrency(run_id: str, step: int) -> dict[str, Any]:
        require_run(registry, run_id)
        spans = _flatten_spans(registry.get_training_store(run_id).read_timing(), step=step)
        if not spans:
            return {"step": step, "spans": [], "max_concurrency": 0, "timeline": []}
        sorted_spans = sorted(spans, key=lambda s: s.get("wall_start", 0))
        events: list[tuple[float, int]] = []
        for s in sorted_spans:
            ws = s.get("wall_start", 0)
            we = s.get("wall_end", ws + s.get("duration", 0))
            events.append((ws, 1))
            events.append((we, -1))
        events.sort(key=lambda e: (e[0], e[1]))
        max_c = 0
        current = 0
        timeline: list[dict[str, Any]] = []
        for t, delta in events:
            current += delta
            max_c = max(max_c, current)
            timeline.append({"time": t, "concurrency": current})
        return {"step": step, "spans": sorted_spans, "max_concurrency": max_c, "timeline": timeline}

    @router.get("/{run_id}/timing/tree/{step}")
    async def get_timing_tree(run_id: str, step: int) -> dict[str, Any]:
        require_run(registry, run_id)
        spans = _flatten_spans(registry.get_training_store(run_id).read_timing(), step=step)
        if not spans:
            return {"step": step, "root": None}
        sorted_spans = sorted(spans, key=lambda s: (s.get("wall_start", 0), -s.get("duration", 0)))
        nodes: list[dict[str, Any]] = []
        for s in sorted_spans:
            nodes.append({
                "name": s.get("name", "?"), "duration": s.get("duration", 0),
                "wall_start": s.get("wall_start", 0), "wall_end": s.get("wall_end", 0),
                "attributes": s.get("attributes", {}), "children": [],
            })
        # Build tree by grouping spans with group_idx under their parent.
        # Strategy: separate spans into sequential (no group overlap) and
        # per-group (have group_idx). Per-group spans are organized as:
        #   parent_span → group N → [child spans with group_idx=N]
        EPS = 0.01

        # Separate spans: those with group_idx and those without
        grouped_spans: dict[int, list[dict[str, Any]]] = {}  # group_idx → spans
        ungrouped: list[dict[str, Any]] = []
        for node in nodes:
            gidx = node.get("attributes", {}).get("group_idx")
            if gidx is not None:
                if gidx not in grouped_spans:
                    grouped_spans[gidx] = []
                grouped_spans[gidx].append(node)
            else:
                ungrouped.append(node)

        # Build the ungrouped tree first (sequential spans)
        ungrouped.sort(key=lambda s: (s.get("wall_start", 0), -s.get("duration", 0)))
        root_children: list[dict[str, Any]] = []
        stack: list[dict[str, Any]] = []
        for node in ungrouped:
            while stack and stack[-1]["wall_end"] + EPS < node["wall_start"]:
                stack.pop()
            if stack and node["wall_end"] <= stack[-1]["wall_end"] + EPS:
                stack[-1]["children"].append(node)
            else:
                while stack and node["wall_end"] > stack[-1]["wall_end"] + EPS:
                    stack.pop()
                if stack:
                    stack[-1]["children"].append(node)
                else:
                    root_children.append(node)
            stack.append(node)

        # Now insert grouped spans as children of the enclosing ungrouped span
        # Each group becomes a subtree: "Group N" → [sorted child spans]
        if grouped_spans:
            # Find the ungrouped span that contains the grouped spans (usually "sampling")
            all_grouped = [s for spans in grouped_spans.values() for s in spans]
            group_start = min(s["wall_start"] for s in all_grouped)
            group_end = max(s["wall_end"] for s in all_grouped)

            def find_parent(children: list[dict[str, Any]]) -> dict[str, Any] | None:
                for child in children:
                    if child["wall_start"] <= group_start + EPS and child["wall_end"] >= group_end - EPS:
                        deeper = find_parent(child.get("children", []))
                        return deeper if deeper else child
                return None

            parent = find_parent(root_children)
            target = parent["children"] if parent else root_children

            for gidx in sorted(grouped_spans.keys()):
                spans = sorted(grouped_spans[gidx], key=lambda s: (s["wall_start"], -s["duration"]))
                # Build a mini-tree for this group
                group_node: dict[str, Any] = {
                    "name": f"group {gidx}",
                    "duration": max(s["wall_end"] for s in spans) - min(s["wall_start"] for s in spans),
                    "wall_start": min(s["wall_start"] for s in spans),
                    "wall_end": max(s["wall_end"] for s in spans),
                    "attributes": {"group_idx": gidx},
                    "children": [],
                }
                # Add spans as children, nesting by time containment within the group
                gstack: list[dict[str, Any]] = []
                for node in spans:
                    while gstack and gstack[-1]["wall_end"] + EPS < node["wall_start"]:
                        gstack.pop()
                    if gstack and node["wall_end"] <= gstack[-1]["wall_end"] + EPS:
                        gstack[-1]["children"].append(node)
                    else:
                        group_node["children"].append(node)
                    gstack.append(node)
                target.append(group_node)

            # Sort target children by wall_start
            target.sort(key=lambda s: s.get("wall_start", 0))
        all_starts = [n["wall_start"] for n in nodes]
        all_ends = [n["wall_end"] for n in nodes]
        total = max(all_ends) - min(all_starts) if all_starts else 0
        return {
            "step": step, "total_duration": total,
            "root": {
                "name": "iteration", "duration": total,
                "wall_start": min(all_starts) if all_starts else 0,
                "wall_end": max(all_ends) if all_ends else 0,
                "attributes": {}, "children": root_children,
            },
        }

    return router


def _flatten_spans(
    records: list[dict[str, Any]],
    *,
    step: int | None = None,
    step_start: int | None = None,
    step_end: int | None = None,
) -> list[dict[str, Any]]:
    """Extract flat spans from timing records, optionally filtered by step range."""
    spans: list[dict[str, Any]] = []
    for record in records:
        s = record.get("step", 0)
        if step is not None and s != step:
            continue
        if step_start is not None and s < step_start:
            continue
        if step_end is not None and s > step_end:
            continue
        if "spans" in record:
            for span in record["spans"]:
                spans.append({"step": s, **span})
        else:
            spans.append(record)
    return spans

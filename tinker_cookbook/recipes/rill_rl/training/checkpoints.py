"""Manage RILL checkpoints under a Tinker project.

The Tinker backend filters training runs by project (``GET /api/v1/training_runs?
project_id=...`` -> ``model_store.list_by_project``), but the public SDK's
``list_training_runs`` doesn't expose the ``project_id`` parameter. So we call that
endpoint through the SDK's low-level client to list *every* run in a project dynamically —
no tagging required, so any new experiment created under the project shows up automatically.
We still tag copies/runs with a human ``rill_label`` (``user_metadata``) purely for a nice
display name in the app's picker.

CLI:
    # copy a prior run's weights into the project (never-expires), tagged for the app
    python -m tinker_cookbook.recipes.rill_rl.training.checkpoints copy \\
        --project <project-id> --label exp1 \\
        --source tinker://<run-id>:train:0/weights/final

    # list the project's RILL checkpoints
    python -m tinker_cookbook.recipes.rill_rl.training.checkpoints list --project <project-id>
"""

from __future__ import annotations

import argparse

import tinker
from tinker.lib.internal_client_holder import ClientConnectionPoolType

PROJECT_TAG = "rill_project"
LABEL_TAG = "rill_label"


def _runs_in_project(project_id: str, page: int = 100) -> list:
    """All training runs under ``project_id`` via the backend's project filter.

    Calls ``GET /api/v1/training_runs?project_id=...`` through the SDK's low-level client
    (the public ``list_training_runs`` doesn't take ``project_id`` yet).
    """
    holder = tinker.ServiceClient().create_rest_client().holder
    runs: list = []
    offset = 0
    while True:

        async def _fetch(offset: int = offset):
            async def _send():
                with holder.aclient(ClientConnectionPoolType.TRAIN) as client:
                    return await client.get(
                        "/api/v1/training_runs",
                        options={
                            "params": {"project_id": project_id, "limit": page, "offset": offset}
                        },
                        cast_to=tinker.types.TrainingRunsResponse,
                    )

            return await holder.execute_with_retries(_send)

        resp = holder.run_coroutine_threadsafe(_fetch()).future().result()
        runs.extend(resp.training_runs)
        total = resp.cursor.total_count if resp.cursor else len(runs)
        if not resp.training_runs or len(runs) >= total:
            break
        offset += page
    return runs


def copy_into_project(source_weights_path: str, project_id: str, label: str) -> dict[str, str]:
    """Copy a trainable ``weights/...`` checkpoint into ``project_id``, tagged and never-expiring.

    Saves both a training checkpoint (re-trainable) and a sampler checkpoint (servable). Both
    are saved with no TTL, so they never expire. Returns the new tinker paths.
    """
    if "/weights/" not in source_weights_path:
        raise SystemExit(
            f"source must be a trainable tinker://.../weights/<name> path, got {source_weights_path!r}"
        )
    service = tinker.ServiceClient(project_id=project_id)
    training_client = service.create_training_client_from_state(
        source_weights_path,
        user_metadata={
            PROJECT_TAG: project_id,
            LABEL_TAG: label,
            "copied_from_path": source_weights_path,
        },
    )
    # ttl_seconds defaults to None == never expires.
    training_path = training_client.save_state(label).result().path
    sampler_path = training_client.save_weights_for_sampler(label).result().path
    return {"label": label, "training": training_path, "sampler": sampler_path}


def list_project_checkpoints(project_id: str) -> list[dict]:
    """Every servable (sampler) checkpoint under ``project_id``, newest first.

    Dynamic: lists all runs the backend reports for the project, so new experiments appear
    automatically. Each entry: ``{label, tinker_path, base_model, run, created_at}``.
    """
    out = []
    for r in _runs_in_project(project_id):
        cp = r.last_sampler_checkpoint
        if cp is None:
            continue  # no servable checkpoint yet
        md = r.user_metadata or {}
        out.append(
            {
                "label": md.get(LABEL_TAG) or r.training_run_id.split(":")[0][:8],
                "tinker_path": cp.tinker_path,
                "base_model": r.base_model,
                "run": r.training_run_id,
                "created_at": cp.time.isoformat(),
            }
        )
    out.sort(key=lambda e: e["created_at"], reverse=True)
    return out


def _main() -> None:
    ap = argparse.ArgumentParser(description="Manage RILL checkpoints under a Tinker project.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    cp = sub.add_parser("copy", help="copy a weights/ checkpoint into a project (never-expires)")
    cp.add_argument("--project", required=True)
    cp.add_argument("--label", required=True)
    cp.add_argument("--source", required=True, help="tinker://<run>:train:0/weights/<name>")

    ls = sub.add_parser("list", help="list a project's RILL checkpoints")
    ls.add_argument("--project", required=True)

    args = ap.parse_args()
    if args.cmd == "copy":
        res = copy_into_project(args.source, args.project, args.label)
        print(f"copied {args.label}:")
        print(f"  training: {res['training']}")
        print(f"  sampler:  {res['sampler']}")
    elif args.cmd == "list":
        for e in list_project_checkpoints(args.project):
            print(f"  {e['created_at'][:19]}  {e['label']:16} {e['tinker_path']}")


if __name__ == "__main__":
    _main()

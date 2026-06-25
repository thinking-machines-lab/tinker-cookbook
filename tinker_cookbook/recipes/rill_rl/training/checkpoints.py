"""Manage RILL checkpoints under a Tinker project.

Tinker groups runs into projects, but the SDK's run/checkpoint types don't expose the
project id, and `list_user_checkpoints` isn't project-scoped. So we tag each run with
``user_metadata`` (``rill_project`` + a human ``rill_label``) when it's created or copied,
and list a project's checkpoints by filtering runs on that tag. This is the "API" the app
uses to populate its checkpoint picker.

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

PROJECT_TAG = "rill_project"
LABEL_TAG = "rill_label"


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


def list_project_checkpoints(project_id: str, limit: int = 500) -> list[dict]:
    """Return the servable (sampler) checkpoints of runs tagged for ``project_id``.

    Each entry: ``{label, tinker_path, base_model, run, time}``, newest first.
    """
    rest = tinker.ServiceClient().create_rest_client()
    runs = []
    offset = 0
    while len(runs) < limit:
        resp = rest.list_training_runs(limit=100, offset=offset).result()
        runs.extend(resp.training_runs)
        total = resp.cursor.total_count if resp.cursor else len(runs)
        if not resp.training_runs or len(runs) >= total:
            break
        offset += 100

    out = []
    for r in runs:
        md = r.user_metadata or {}
        if md.get(PROJECT_TAG) != project_id:
            continue
        cp = r.last_sampler_checkpoint
        if cp is None:
            continue
        out.append(
            {
                "label": md.get(LABEL_TAG, r.training_run_id),
                "tinker_path": cp.tinker_path,
                "base_model": r.base_model,
                "run": r.training_run_id,
                "time": cp.time.isoformat(),
            }
        )
    out.sort(key=lambda e: e["time"], reverse=True)
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
            print(f"  {e['label']:16} {e['base_model']:20} {e['tinker_path']}")


if __name__ == "__main__":
    _main()

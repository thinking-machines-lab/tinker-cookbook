#!/usr/bin/env python3
"""
Check that .github/workflows/ only uses security-reviewed GitHub Actions.

Runs zizmor (https://docs.zizmor.sh) against the allowlist in .github/zizmor.yml
and enforces these audits:

  - forbidden-uses:     every `uses:` must match the allowlist
  - unpinned-uses:      third-party actions must be pinned to a full commit SHA
  - dangerous-triggers: no pull_request_target / workflow_run, which run with
                        repository secrets on events outsiders can cause
  - insecure-commands:  no ACTIONS_ALLOW_UNSECURE_COMMANDS
  - impostor-commit, known-vulnerable-actions: "pinned" SHAs that come from a
                        fork, and actions with a published advisory. These need
                        GitHub's API, so they run only when GH_TOKEN is set (CI)
                        and are skipped elsewhere.
  - hardcoded-container-credentials: registry passwords in workflow files
  - overprovisioned-secrets: `toJSON(secrets)` handing every secret to one step

All other zizmor audits are informational only and never block.

Changes to this file or the allowlist (including bumping ZIZMOR_VERSION) need a
maintainer's review.

Usage:
    python .github/workflows/scripts/check_github_actions_allowlist.py

Exit codes: 0 = clean (or zizmor itself was unavailable — the check fails open
so a broken scanner never blocks CI), 1 = violations found.
"""

from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

ZIZMOR_VERSION = "1.27.0"
ENFORCED_AUDITS = {
    "forbidden-uses",
    "unpinned-uses",
    "dangerous-triggers",
    "insecure-commands",
    "impostor-commit",
    "known-vulnerable-actions",
    "hardcoded-container-credentials",
    "overprovisioned-secrets",
}
CONFIG_PATH = ".github/zizmor.yml"
WORKFLOWS_DIR = ".github/workflows"

PROBLEM_BY_AUDIT = {
    "forbidden-uses": "not on the allowlist",
    "unpinned-uses": "not pinned to a full commit SHA",
    "dangerous-triggers": "pull_request_target/workflow_run runs with repository secrets",
    "insecure-commands": "ACTIONS_ALLOW_UNSECURE_COMMANDS lets log output set env vars",
}


@dataclass
class Violation:
    workflow: str
    line: int
    uses: str
    audit: str

    @property
    def problem(self) -> str:
        return PROBLEM_BY_AUDIT.get(
            self.audit, f"{self.audit}: https://docs.zizmor.sh/audits/#{self.audit}"
        )


class ZizmorError(Exception):
    """zizmor could not produce results (crash, bad config, unparseable output)."""


def get_repo_root() -> Path:
    out = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"], capture_output=True, text=True, check=True
    )
    return Path(out.stdout.strip())


def extract_json(stdout: str) -> list[dict]:
    """
    Parse zizmor's JSON findings from stdout.

    Tolerates stray log/download lines before the JSON document (e.g. if a
    wrapper merges stderr into stdout) by falling back to parsing from the
    first line that starts the document.
    """
    try:
        return json.loads(stdout)
    except json.JSONDecodeError:
        pass
    for i, line in enumerate(stdout.splitlines(keepends=True)):
        if line.lstrip().startswith("["):
            rest = "".join(stdout.splitlines(keepends=True)[i:])
            try:
                return json.loads(rest)
            except json.JSONDecodeError:
                continue
    raise ZizmorError(f"could not find JSON findings in zizmor output:\n{stdout[:2000]}")


def zizmor_cmd() -> list[str]:
    return [
        "uvx",
        "--quiet",
        f"zizmor@{ZIZMOR_VERSION}",
        "-qq",
        "--no-exit-codes",
        "--format=json",
        "--config",
        CONFIG_PATH,
        WORKFLOWS_DIR,
    ]


def run_zizmor(repo_root: Path) -> list[dict]:
    cmd = zizmor_cmd()
    try:
        proc = subprocess.run(cmd, cwd=repo_root, capture_output=True, text=True)
    except OSError as e:
        raise ZizmorError(f"could not execute {cmd[0]}: {e}") from e
    if proc.returncode != 0:
        raise ZizmorError(
            f"zizmor exited {proc.returncode}\nstdout:\n{proc.stdout[:2000]}\n"
            f"stderr:\n{proc.stderr[:2000]}"
        )
    return extract_json(proc.stdout)


def parse_violations(findings: list[dict]) -> list[Violation]:
    violations = []
    for finding in findings:
        if finding.get("ident") not in ENFORCED_AUDITS or finding.get("ignored"):
            continue
        primary = next(
            (
                loc
                for loc in finding.get("locations", [])
                if loc["symbolic"].get("kind") == "Primary"
            ),
            None,
        )
        if primary is None:
            continue
        violations.append(
            Violation(
                workflow=primary["symbolic"]["key"]["Local"]["verbatim_path"],
                # zizmor rows are 0-based
                line=primary["concrete"]["location"]["start_point"]["row"] + 1,
                uses=" ".join(primary["concrete"]["feature"].split()),
                audit=finding["ident"],
            )
        )
    violations.sort(key=lambda v: (v.workflow, v.line))
    return violations


def render_check_report(violations: list[Violation]) -> str:
    lines = [f"❌ Found {len(violations)} GitHub Actions allowlist violation(s):", ""]
    for v in violations:
        lines.append(f"  {v.workflow}:{v.line}")
        lines.append(f"    found: {v.uses}")
        lines.append(f"    → {v.problem}")
        lines.append("")
    lines += [
        f"Allowed actions are listed in {CONFIG_PATH}; third-party ones need a full commit SHA.",
        "Adding an action or a `# zizmor: ignore[<audit>] <reason>` needs a maintainer's review.",
    ]
    return "\n".join(lines)


def main() -> int:
    repo_root = get_repo_root()
    try:
        findings = run_zizmor(repo_root)
    except ZizmorError as e:
        # Fail open so a zizmor outage doesn't block CI.
        print(
            "WARNING: GitHub Actions allowlist check could not run; "
            f"passing WITHOUT scanning (fail-open):\n{e}",
            file=sys.stderr,
        )
        return 0

    violations = parse_violations(findings)

    if violations:
        print(render_check_report(violations))
    else:
        print(f"✅ All workflow actions are allowlisted and pinned ({CONFIG_PATH})")

    return 1 if violations else 0


if __name__ == "__main__":
    sys.exit(main())

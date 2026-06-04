#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Gate a ghstack `/land` on approvals and green CI.

Given the top PR of a ghstack stack, this:
  1. Reconstructs the stack by reading the `Pull-Request-resolved` trailers from
     the commits on the PR's `orig` branch.
  2. Requires every not-yet-merged PR in the stack to have at least one approval.
  3. Waits for the top PR's checks to settle and only succeeds when GitHub
     reports the PR as mergeable (clean). Fails fast on conflicts or failing
     required checks.

Relies only on GitHub-native status/check-runs, so it works for any CI setup.
"""

import argparse
import os
import re
import subprocess
import time
from typing import Any, Dict, List, Literal, Tuple

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


def classify_checks(
    statuses: List[Dict[str, Any]],
    check_runs: List[Dict[str, Any]],
) -> Tuple[Literal["pending", "failed", "success"], List[str]]:
    """Classify the head commit's combined checks without hardcoding context names."""
    failed: List[str] = []
    pending: List[str] = []

    # Legacy commit statuses (e.g. third-party integrations).
    for status in statuses:
        state = status.get("state")
        if state in ("failure", "error"):
            failed.append(status.get("context", "status"))
        elif state == "pending":
            pending.append(status.get("context", "status"))

    # GitHub Actions / check-runs.
    for run in check_runs:
        if run.get("status") != "completed":
            pending.append(run.get("name", "check"))
        elif run.get("conclusion") not in ("success", "neutral", "skipped"):
            failed.append(run.get("name", "check"))

    if failed:
        return "failed", failed
    if pending:
        return "pending", pending
    return "success", []


def main():
    parser = argparse.ArgumentParser(description="Check ghstack PR approvals and CI status")
    parser.add_argument("pr_number", type=int, help="PR number to check")
    parser.add_argument("head_ref", help="Head reference of the PR")
    parser.add_argument("repo", help="Repository in owner/repo format")
    parser.add_argument(
        "--max-wait-time",
        type=int,
        default=1800,
        help="Maximum wait time in seconds for checks to settle",
    )

    args = parser.parse_args()

    gh = requests.Session()
    gh.headers.update(
        {
            "Authorization": f"Bearer {os.environ['GITHUB_TOKEN']}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        }
    )
    # Retry transient GitHub-side 5xx / rate-limit responses. urllib3's Retry
    # ships with `requests`, so this needs no extra dependency.
    retry = Retry(
        total=5,
        backoff_factor=2.0,  # 0s, 2s, 4s, 8s, 16s
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset(["GET", "POST"]),
        raise_on_status=False,
        respect_retry_after_header=True,
    )
    adapter = HTTPAdapter(max_retries=retry)
    gh.mount("https://", adapter)
    gh.mount("http://", adapter)

    NUMBER, head_ref, REPO = args.pr_number, args.head_ref, args.repo
    MAX_WAIT_TIME = args.max_wait_time

    def must(cond: Any, msg: str):
        if not cond:
            print(msg)
            gh.post(
                f"https://api.github.com/repos/{REPO}/issues/{NUMBER}/comments",
                json={"body": f"ghstack land check failed: {msg}"},
            )
            exit(1)

    print(head_ref)
    must(
        head_ref and re.match(r"^gh/[\w-]+/\d+/head$", head_ref),
        "Not a ghstack PR",
    )
    orig_ref = head_ref.replace("/head", "/orig")

    def git_fetch_with_retry(ref: str, *, attempts: int = 5, initial_backoff: float = 2.0) -> bool:
        backoff = initial_backoff
        for attempt in range(1, attempts + 1):
            rc = os.system(f"git fetch origin {ref}")
            if rc == 0:
                return True
            print(f"git fetch origin {ref} failed (attempt {attempt}/{attempts}, exit={rc})")
            if attempt < attempts:
                print(f"   retrying in {backoff:.0f}s...")
                time.sleep(backoff)
                backoff *= 2
        return False

    print(":: Fetching newest main...")
    must(git_fetch_with_retry("main"), "Can't fetch main")
    print(":: Fetching orig branch...")
    must(git_fetch_with_retry(orig_ref), "Can't fetch orig branch")

    proc = subprocess.Popen(
        "git log FETCH_HEAD...$(git merge-base FETCH_HEAD origin/main)",
        stdout=subprocess.PIPE,
        shell=True,
    )
    out, _ = proc.communicate()
    must(proc.wait() == 0, "`git log` command failed!")

    pr_numbers = re.findall(
        r"Pull[- ]Request(?:[- ]resolved)?: https://github.com/.*?/pull/([0-9]+)",
        out.decode("utf-8"),
    )
    pr_numbers = list(map(int, pr_numbers))
    print(pr_numbers)
    must(pr_numbers and pr_numbers[0] == NUMBER, "Extracted PR numbers don't seem right!")

    # Every not-yet-merged PR in the stack needs an approval.
    print(":: Checking approvals for all PRs...")
    for n in pr_numbers:
        resp = gh.get(f"https://api.github.com/repos/{REPO}/pulls/{n}")
        must(resp.ok, f"Error checking merge status for PR #{n}!")
        if resp.json()["merged"]:
            continue
        print(f"Checking approvals for PR #{n}... ", end="")
        resp = gh.get(f"https://api.github.com/repos/{REPO}/pulls/{n}/reviews")
        must(resp.ok, f"Error getting reviews for PR #{n}!")
        has_approval = any(review["state"] == "APPROVED" for review in resp.json())
        must(has_approval, f"PR #{n} has no approvals!")
        print("APPROVED!")

    def check_pr_status(pr_number: int):
        waiting_comment_posted = False
        start_time = time.time()

        def post_success_comment():
            gh.post(
                f"https://api.github.com/repos/{REPO}/issues/{pr_number}/comments",
                json={"body": f"PR #{pr_number} checks have completed successfully!"},
            )

        while True:
            resp = gh.get(
                f"https://api.github.com/repos/{REPO}/pulls/{pr_number}",
                headers={"Accept": "application/vnd.github.v3+json"},
            )
            must(resp.ok, f"Error getting PR #{pr_number}!")
            pr_obj = resp.json()

            mergeable_state = pr_obj.get("mergeable_state", "unknown")
            if mergeable_state == "unknown":
                # GitHub is still computing mergeability; give it a moment.
                time.sleep(2)
                resp = gh.get(
                    f"https://api.github.com/repos/{REPO}/pulls/{pr_number}",
                    headers={"Accept": "application/vnd.github.v3+json"},
                )
                must(resp.ok, f"Error getting PR #{pr_number} on retry!")
                pr_obj = resp.json()
                mergeable_state = pr_obj.get("mergeable_state", "unknown")

            if mergeable_state == "unstable":
                # Non-required checks are still running (or a required one is
                # pending). Inspect the checks and wait until they settle.
                if time.time() - start_time > MAX_WAIT_TIME:
                    must(
                        False,
                        f"PR #{pr_number} stayed unstable for over "
                        f"{MAX_WAIT_TIME // 60} minutes!",
                    )

                sha = pr_obj["head"]["sha"]
                status_resp = gh.get(f"https://api.github.com/repos/{REPO}/commits/{sha}/status")
                must(status_resp.ok, f"Error getting statuses for PR #{pr_number}!")
                check_resp = gh.get(
                    f"https://api.github.com/repos/{REPO}/commits/{sha}/check-runs",
                    params={"per_page": 100},
                )
                must(check_resp.ok, f"Error getting check runs for PR #{pr_number}!")

                result, relevant = classify_checks(
                    status_resp.json().get("statuses", []),
                    check_resp.json().get("check_runs", []),
                )
                if result == "failed":
                    must(False, f"PR #{pr_number} has failing checks: {', '.join(relevant)}")
                elif result == "success":
                    post_success_comment()
                    return pr_obj

                message = f"PR #{pr_number} has pending checks: {', '.join(relevant)}"
                if not waiting_comment_posted:
                    print(f"\n{message}. Waiting for checks to settle...")
                    gh.post(
                        f"https://api.github.com/repos/{REPO}/issues/{pr_number}/comments",
                        json={"body": message},
                    )
                    waiting_comment_posted = True
                time.sleep(30)
                print(".", end="", flush=True)
                continue

            if mergeable_state == "blocked":
                must(
                    False,
                    f"PR #{pr_number} is blocked from merging (failing or missing "
                    f"required checks)! Use `/land --force <reason>` to bypass CI.",
                )
            elif mergeable_state == "dirty":
                must(False, f"PR #{pr_number} has merge conflicts that need to be resolved!")
            elif mergeable_state == "clean":
                if waiting_comment_posted:
                    post_success_comment()
                return pr_obj
            else:
                must(False, f"PR #{pr_number} is not ready to merge (state: {mergeable_state})!")

    if pr_numbers:
        print(f":: Checking status for primary PR #{pr_numbers[0]}... ", end="")
        check_pr_status(pr_numbers[0])
        print("SUCCESS!")

    print(":: All PRs are ready to be landed!")


if __name__ == "__main__":
    main()

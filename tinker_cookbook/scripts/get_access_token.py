#!/usr/bin/env python3
"""Obtain a Tinker access token (JWT) the same way the SDK does.

If your organization is set up with JWKS-based auth, you authenticate by
exchanging a short-lived credential for a Tinker JWT. This script runs the
command in the `TINKER_CREDENTIAL_CMD` environment variable to produce that
credential, then exchanges it at `/api/v1/auth/token` and prints an
`export TINKER_API_KEY=...` line you can eval.

Prerequisite: set `TINKER_CREDENTIAL_CMD` in your environment to a command
that prints a valid credential to stdout (this is the same variable the SDK
reads). The script will not run without it.

Usage:
    # Set the credential command (example), then run this script:
    export TINKER_CREDENTIAL_CMD="<command that prints your credential>"
    ./get_access_token.py

    # Apply the token to your current shell:
    eval "$(./get_access_token.py)"

Tip: combine this with `copy_checkpoint.py` for cross-org checkpoint copies.
Run it once against each org's credential to mint a source token and a
destination token, then pass the source token to the copy script:

    export TINKER_CREDENTIAL_CMD="<source-org credential command>"
    export SRC_TINKER_ACCESS_TOKEN="$(./get_access_token.py | cut -d= -f2-)"

    export TINKER_CREDENTIAL_CMD="<destination-org credential command>"
    eval "$(./get_access_token.py)"   # sets TINKER_API_KEY to the destination token

    python -m tinker_cookbook.scripts.copy_checkpoint \\
        --source-path tinker://<run-id>:train:0/weights/<name> \\
        --source-access-token "$SRC_TINKER_ACCESS_TOKEN" \\
        --destination-project-id <project-id>
"""

import json
import os
import subprocess
import sys
import urllib.request

BASE_URL = "https://tinker.thinkingmachines.dev/services/tinker-prod"


def main() -> None:
    cmd = os.environ.get("TINKER_CREDENTIAL_CMD")
    if not cmd:
        sys.exit(
            "Set TINKER_CREDENTIAL_CMD to a command that prints a credential, "
            "e.g. TINKER_CREDENTIAL_CMD='<your command>' ./get_access_token.py"
        )
    credential = subprocess.check_output(cmd, shell=True, text=True).strip()
    if not credential:
        sys.exit(f"Credential command produced nothing: {cmd!r}")

    req = urllib.request.Request(
        f"{BASE_URL}/api/v1/auth/token",
        data=b"{}",
        headers={"X-API-Key": credential, "Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req) as resp:
        jwt = json.load(resp)["jwt"]

    print(f"export TINKER_API_KEY={jwt}")


if __name__ == "__main__":
    main()

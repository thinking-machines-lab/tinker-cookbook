#!/bin/bash

# Pre-commit trufflehog scan of staged files, using the same detectors as security.yaml.

set -euo pipefail

REPO_ROOT="$(git rev-parse --show-toplevel)"

TRUFFLEHOG_VERSION="3.97.0"
CACHE_DIR="${XDG_CACHE_HOME:-${HOME}/.cache}/trufflehog/${TRUFFLEHOG_VERSION}"
TRUFFLEHOG="${CACHE_DIR}/trufflehog"

install_trufflehog() {
  local os arch platform sha256
  case "$(uname -s)" in
    Darwin) os="darwin" ;;
    Linux) os="linux" ;;
    *)
      echo "trufflehog hook: unsupported OS $(uname -s); skipping scan" >&2
      exit 0
      ;;
  esac
  case "$(uname -m)" in
    arm64 | aarch64) arch="arm64" ;;
    x86_64) arch="amd64" ;;
    *)
      echo "trufflehog hook: unsupported arch $(uname -m); skipping scan" >&2
      exit 0
      ;;
  esac
  platform="${os}_${arch}"

  # From the release's checksums.txt.
  case "$platform" in
    darwin_amd64) sha256="037e4aeb197870555ff515432bb5f1f2c98dce5f1214631a689112b5e0e4c9fd" ;;
    darwin_arm64) sha256="ad0a99bd48d6df80eabab24d11d0fd771e245fc55ed347f943cafb5e5f497c5c" ;;
    linux_amd64) sha256="62224de2f9dd7cd418800feb953760a302ed2f82a7c547fe1146a4874fb179e4" ;;
    linux_arm64) sha256="f48f57e3d4343377865b1b64653f96d381d61a7792d89d026e85524732039fde" ;;
    *)
      echo "trufflehog hook: no pinned checksum for ${platform}; skipping scan" >&2
      exit 0
      ;;
  esac

  local tarball="trufflehog_${TRUFFLEHOG_VERSION}_${platform}.tar.gz"
  local url="https://github.com/trufflesecurity/trufflehog/releases/download/v${TRUFFLEHOG_VERSION}/${tarball}"
  tmpdir="$(mktemp -d)"
  trap 'rm -rf "${tmpdir:-}"' EXIT

  echo "trufflehog hook: downloading ${tarball} (one-time setup)..." >&2
  if ! curl -fsSL --retry 3 --retry-delay 2 -o "${tmpdir}/${tarball}" "$url"; then
    # Don't block commits when offline; CI still scans.
    echo "trufflehog hook: download failed (offline?); skipping scan" >&2
    exit 0
  fi

  if ! echo "${sha256}  ${tmpdir}/${tarball}" | shasum -a 256 --check --status; then
    echo "trufflehog hook: checksum mismatch for ${tarball}; refusing to install" >&2
    exit 1
  fi

  tar -xzf "${tmpdir}/${tarball}" -C "$tmpdir" trufflehog
  mkdir -p "$CACHE_DIR"
  mv "${tmpdir}/trufflehog" "$TRUFFLEHOG"
  chmod +x "$TRUFFLEHOG"
}

if [[ ! -x "$TRUFFLEHOG" ]]; then
  install_trufflehog
fi

if [[ $# -eq 0 ]]; then
  exit 0
fi

exit_code=0
"$TRUFFLEHOG" filesystem "$@" \
  --no-verification --fail --force-skip-binaries --no-update \
  --config "${REPO_ROOT}/.github/trufflehog.yaml" \
  --include-detectors=PrivateKey,Slack,SlackWebhook,GCP,HuggingFace,AWS,Anthropic,OpenAI,WeightsAndBiases,CustomRegex \
  --filter-entropy=3 \
  --log-level=-1 || exit_code=$?

# 183 means secrets found; any other failure is the scanner's, so don't block.
if [[ "$exit_code" -ne 0 && "$exit_code" -ne 183 ]]; then
  echo "trufflehog hook: trufflehog exited ${exit_code}; skipping scan" >&2
  exit 0
fi

if [[ "$exit_code" -ne 0 ]]; then
  echo "trufflehog found possible secrets in your staged files." >&2
  echo "Remove real secrets. For a false positive, add '# trufflehog:ignore' on the same line." >&2
  exit "$exit_code"
fi

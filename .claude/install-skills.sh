#!/usr/bin/env bash
# Install tinker-cookbook Claude skills globally by symlinking into ~/.claude/skills/
#
# One-liner install (no clone needed):
#   curl -fsSL https://raw.githubusercontent.com/thinking-machines-lab/tinker-cookbook/main/.claude/install-skills.sh | bash
#
# From a local clone:
#   bash .claude/install-skills.sh
#
# Remove installed skills:
#   bash .claude/install-skills.sh --remove
#   curl -fsSL ... | bash -s -- --remove

set -euo pipefail

REPO_URL="https://github.com/thinking-machines-lab/tinker-cookbook.git"
CACHE_DIR="${HOME}/.claude/tinker-cookbook"
SKILLS_DST="${HOME}/.claude/skills"

# Layer 4 (dev-only) skills — excluded from global install
EXCLUDED=(tinker-ci tinker-contributing tinker-new-recipe tinker-manage-skills)

is_excluded() {
    local name="$1"
    for exc in "${EXCLUDED[@]}"; do
        [[ "$name" == "$exc" ]] && return 0
    done
    return 1
}

# Determine the skills source directory.
# If run from within a tinker-cookbook clone, use that. Otherwise, shallow-clone to cache.
find_skills_src() {
    # Check if we're inside a local clone (BASH_SOURCE is set when run as a file)
    if [[ -n "${BASH_SOURCE[0]:-}" && "${BASH_SOURCE[0]}" != "bash" ]]; then
        local script_dir
        script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
        local candidate="$script_dir/skills"
        if [[ -d "$candidate" ]]; then
            echo "$candidate"
            return
        fi
    fi

    # Standalone mode (e.g., piped from curl): clone or update cache
    if [[ -d "$CACHE_DIR/.git" ]]; then
        echo "Updating cached tinker-cookbook ..." >&2
        git -C "$CACHE_DIR" pull --ff-only --depth=1 -q </dev/null 2>/dev/null || true
    else
        echo "Cloning tinker-cookbook (sparse, skills only) ..." >&2
        git clone --depth=1 --filter=blob:none --sparse "$REPO_URL" "$CACHE_DIR" -q </dev/null
        git -C "$CACHE_DIR" sparse-checkout set .claude/skills </dev/null
    fi
    echo "$CACHE_DIR/.claude/skills"
}

remove_skills() {
    local skills_src="$1"
    echo "Removing tinker skill symlinks from $SKILLS_DST ..."
    local count=0
    for dir in "$skills_src"/*/; do
        local name
        name="$(basename "$dir")"
        is_excluded "$name" && continue
        local link="$SKILLS_DST/$name"
        if [[ -L "$link" ]]; then
            rm "$link"
            echo "  removed $link"
            count=$((count + 1))
        fi
    done
    echo "Removed $count symlinks."
}

install_skills() {
    local skills_src="$1"
    mkdir -p "$SKILLS_DST"
    echo "Installing tinker skills into $SKILLS_DST ..."
    local count=0
    for dir in "$skills_src"/*/; do
        local name
        name="$(basename "$dir")"
        is_excluded "$name" && continue
        local link="$SKILLS_DST/$name"
        if [[ -L "$link" ]]; then
            rm "$link"  # refresh existing symlink
        elif [[ -e "$link" ]]; then
            echo "  SKIP $name (non-symlink already exists at $link)"
            continue
        fi
        ln -s "$(cd "$dir" && pwd)" "$link"
        echo "  $name"
        count=$((count + 1))
    done
    echo "Installed $count skills. Use '/tinker-<name>' in Claude Code."
}

SKILLS_SRC="$(find_skills_src)"

if [[ "${1:-}" == "--remove" ]]; then
    remove_skills "$SKILLS_SRC"
else
    install_skills "$SKILLS_SRC"
fi

"""Utility functions for xmux"""


def find_common_prefix(strings: list[str]) -> str:
    """Find the longest common prefix among strings"""
    if not strings:
        return ""

    # Sort to compare only first and last
    sorted_strings = sorted(strings)
    first, last = sorted_strings[0], sorted_strings[-1]

    common: list[str] = []
    for i, char in enumerate(first):
        if i < len(last) and char == last[i]:
            common.append(char)
        else:
            break

    prefix = "".join(common)
    # Only use prefix if it ends at a natural boundary
    if prefix and not prefix.endswith(("/", "_", "-")):
        # Find last boundary
        for sep in ["/", "_", "-"]:
            if sep in prefix:
                prefix = prefix[: prefix.rfind(sep) + 1]
                break
        else:
            # No natural boundary, don't use prefix
            prefix = ""

    return prefix


def abbreviate_path(path: str, max_length: int = 20) -> str:
    """Abbreviate a path to fit within max_length"""
    # Common abbreviations
    replacements = {
        "learning_rate": "lr",
        "batch_size": "bs",
        "num_epochs": "ep",
        "model": "m",
        "experiment": "exp",
        "checkpoint": "ckpt",
        "validation": "val",
        "training": "train",
    }

    # Apply replacements
    result = path
    for long_form, short_form in replacements.items():
        result = result.replace(long_form, short_form)

    # If still too long, use more aggressive abbreviation
    if len(result) > max_length:
        parts = result.split("/")
        if len(parts) > 1:
            # Keep last part full, abbreviate others
            abbreviated_parts: list[str] = []
            for part in parts[:-1]:
                if len(part) > 3:
                    # Take first letter of each word/section
                    if "_" in part:
                        abbrev = "".join(word[0] for word in part.split("_"))
                    elif "-" in part:
                        abbrev = "".join(word[0] for word in part.split("-"))
                    else:
                        abbrev = part[:3]
                    abbreviated_parts.append(abbrev)
                else:
                    abbreviated_parts.append(part)

            abbreviated_parts.append(parts[-1])
            result = "/".join(abbreviated_parts)

    # Final truncation if needed
    if len(result) > max_length:
        result = "..." + result[-(max_length - 3) :]

    return result


def generate_unique_names(paths: list[str], max_length: int = 20) -> list[str]:
    """Generate unique abbreviated names for a list of paths"""
    # Find common prefix to remove
    common_prefix = find_common_prefix(paths)

    # Remove prefix and abbreviate
    names: list[str] = []
    seen_names: set[str] = set()

    for path in paths:
        # Remove common prefix
        if common_prefix and path.startswith(common_prefix):
            shortened = path[len(common_prefix) :]
        else:
            shortened = path

        # Abbreviate
        name = abbreviate_path(shortened, max_length)

        # Ensure uniqueness
        if name in seen_names:
            # Add a counter
            counter = 2
            while f"{name}-{counter}" in seen_names:
                counter += 1
            name = f"{name}-{counter}"

        seen_names.add(name)
        names.append(name)

    return names


def smart_window_name(
    log_relpath: str, session_context: list[str] | None = None, max_length: int = 20
) -> str:
    """Generate a smart window name for a single job"""
    if session_context:
        # Use context to find common patterns
        all_paths = session_context + [log_relpath]
        names = generate_unique_names(all_paths, max_length)
        return names[-1]  # Return name for the new path
    else:
        # No context, just abbreviate
        return abbreviate_path(log_relpath, max_length)


def format_status_bar_windows(window_names: list[str], max_width: int = 80) -> str:
    """Format window names for status bar display"""
    # Format: [0:ctrl] [1:name1] [2:name2] ...
    formatted: list[str] = []
    current_width = 0

    for i, name in enumerate(window_names):
        if i == 0:
            item = "[0:ctrl]"
        else:
            # Check if this is a grouped window
            if name.count("-") > 1 and name.split("-")[-1].isdigit():
                # Grouped window, add indicator
                base_name = "-".join(name.split("-")[:-1])
                item = f"[{i}:{base_name}*]"
            else:
                item = f"[{i}:{name}]"

        item_width = len(item) + 1  # +1 for space

        if current_width + item_width > max_width and formatted:
            # Would overflow, stop here
            remaining = len(window_names) - len(formatted)
            formatted.append(f"... +{remaining}")
            break

        formatted.append(item)
        current_width += item_width

    return " ".join(formatted)

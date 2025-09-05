from termcolor import colored
from tinker_cookbook.tokenizer_utils import Tokenizer


def format_colorized(
    tokens: list[int], weights: list[float], tokenizer: Tokenizer, draw_newline_arrow: bool = False
) -> str:
    """
    Colour-code text according to per-token weights.

    * Cyan text  → weight > 0
    * Yellow text  → weight = 0
    * Red text   → weight < 0

    The function minimises ANSI escape sequences by wrapping *runs* of
    like-coloured tokens, and decodes each run in a single call so that
    multi-byte or multibyte-character languages (e.g. CJK) render correctly.
    """
    if len(tokens) != len(weights):
        raise ValueError("`tokens` and `weights` must be the same length.")

    chunks, current_ids, current_color = [], [], None

    def flush_current_run():
        decoded = tokenizer.decode(current_ids)
        lines = decoded.splitlines(keepends=True)
        for line in lines:
            if draw_newline_arrow:
                line = line.replace("\n", "↵\n")
            chunks.append(colored(line, current_color))

    for tok_id, w in zip(tokens, weights, strict=True):
        if w < 0:
            color = "red"
        elif w == 0:
            color = "yellow"
        else:
            color = "green"

        # Flush when the colour changes
        if color != current_color and current_ids:
            flush_current_run()
            current_ids = []

        current_ids.append(tok_id)
        current_color = color

    flush_current_run()

    return "".join(chunks)

"""
HTML formatters for logtree.

This module provides formatter objects that encapsulate both HTML generation
and the CSS needed to style that HTML. Formatters implement the Formatter protocol
from logtree and can be logged using `logtree.log_formatter()`.
"""

import html
from dataclasses import dataclass
from typing import Any, Mapping, Sequence


@dataclass
class ConversationFormatter:
    """
    Formatter for conversation messages.

    Renders a list of messages as a styled conversation with role-based coloring.
    """

    messages: Sequence[Mapping[str, Any]]
    """List of messages, each with 'role' and 'content' keys."""

    def to_html(self) -> str:
        """Generate HTML for the conversation."""
        parts = ['<div class="lt-conversation">']
        for msg in self.messages:
            role = html.escape(msg["role"])
            content = msg["content"]
            parts.append(f'  <div class="lt-message lt-message-{role}">')
            parts.append(f'    <span class="lt-message-role">{role}:</span>')
            parts.append(f'    <span class="lt-message-content">{content}</span>')
            parts.append("  </div>")
        parts.append("</div>")
        return "\n".join(parts)

    def get_css(self) -> str:
        """Get CSS for conversation styling."""
        return CONVERSATION_CSS


# CSS for conversation formatting
CONVERSATION_CSS = """
/* Conversation formatting */
.lt-conversation {
    display: flex;
    flex-direction: column;
    gap: 0.5rem;
    margin: 0.5rem 0;
}

.lt-message {
    padding: 0.75rem;
    border-radius: 6px;
    border-left: 3px solid var(--lt-accent, #2563eb);
    background: var(--lt-bg, #f9fafb);
    line-height: 1.5;
}

.lt-message-role {
    font-weight: 600;
    color: var(--lt-accent, #2563eb);
    display: inline-block;
    min-width: 80px;
}

.lt-message-content {
    white-space: pre-wrap;
    word-wrap: break-word;
}

.lt-message-user {
    background: #e3f2fd;
    border-left-color: #1976d2;
}

.lt-message-user .lt-message-role {
    color: #1565c0;
}

.lt-message-assistant {
    background: #f3e5f5;
    border-left-color: #7b1fa2;
}

.lt-message-assistant .lt-message-role {
    color: #6a1b9a;
}

.lt-message-system {
    background: #fff3e0;
    border-left-color: #f57c00;
}

.lt-message-system .lt-message-role {
    color: #e65100;
}

.lt-message-tool {
    background: #e8f5e9;
    border-left-color: #388e3c;
}

.lt-message-tool .lt-message-role {
    color: #2e7d32;
}
"""

/** Shared conversation extraction from rollout data. */

import type { ConversationMessage, RolloutDetail } from '../api/types';

/** Extract flat conversation messages from a rollout detail.
 *
 * Priority:
 * 1. Top-level `conversation` field (schema v3)
 * 2. Per-step `_conversation` in step logs
 */
export function extractConversation(rollout: RolloutDetail): ConversationMessage[] {
  if (rollout.conversation && rollout.conversation.length > 0) {
    return rollout.conversation;
  }
  const msgs: ConversationMessage[] = [];
  for (const step of rollout.steps) {
    const conv = (step.logs as Record<string, unknown>)?._conversation;
    if (Array.isArray(conv)) {
      msgs.push(...conv as ConversationMessage[]);
    }
  }
  return msgs;
}

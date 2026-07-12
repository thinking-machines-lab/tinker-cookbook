import { logprobColor, tokenRepr } from "../util";

export type TokenViewMode = "text" | "ids";

/** One action token, colored by logprob, with a hover tooltip of id/logprob/repr. */
export function TokenSpan({
  tokenId,
  text,
  logprob,
  mode,
}: {
  tokenId: number;
  text: string | null;
  logprob: number | null;
  mode: TokenViewMode;
}) {
  const shown = mode === "ids" ? `${tokenId} ` : (text ?? String(tokenId));
  const tooltip = [
    `id ${tokenId}`,
    logprob !== null ? `logprob ${logprob.toFixed(4)}` : null,
    text !== null ? `repr ${tokenRepr(text)}` : null,
  ]
    .filter(Boolean)
    .join("\n");
  return (
    <span
      className="tok"
      title={tooltip}
      style={logprob !== null ? { color: logprobColor(logprob) } : undefined}
    >
      {shown}
    </span>
  );
}

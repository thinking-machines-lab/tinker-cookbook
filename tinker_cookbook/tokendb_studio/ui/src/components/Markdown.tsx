// Tiny markdown renderer for assistant messages: paragraphs, fenced code
// blocks, inline code, bold, headings, and bullet/numbered lists. No
// dependency; anything else renders as plain text. Rollout keys in prose
// (`split/iteration/group/traj`, e.g. train/12/3/1) become links into the
// rollout detail route when `linkRollouts` is set.

import type { ReactNode } from "react";
import { Link } from "react-router-dom";

type Block =
  | { type: "code"; text: string }
  | { type: "heading"; text: string }
  | { type: "list"; ordered: boolean; items: string[] }
  | { type: "para"; text: string };

const HEADING_RE = /^#{1,6}\s+/;
const BULLET_RE = /^\s*[-*]\s+/;
const NUMBERED_RE = /^\s*\d+[.)]\s+/;
const CITATION_RE = /\b([A-Za-z0-9_.-]+)\/(\d+)\/(\d+)\/(\d+)\b/g;

function parseBlocks(text: string): Block[] {
  const blocks: Block[] = [];
  const lines = text.split("\n");
  let i = 0;
  while (i < lines.length) {
    const line = lines[i];
    if (line.trimStart().startsWith("```")) {
      i++;
      const code: string[] = [];
      while (i < lines.length && !lines[i].trimStart().startsWith("```")) {
        code.push(lines[i]);
        i++;
      }
      i++; // closing fence (or end of text)
      blocks.push({ type: "code", text: code.join("\n") });
      continue;
    }
    if (!line.trim()) {
      i++;
      continue;
    }
    if (HEADING_RE.test(line)) {
      blocks.push({ type: "heading", text: line.replace(HEADING_RE, "") });
      i++;
      continue;
    }
    if (BULLET_RE.test(line) || NUMBERED_RE.test(line)) {
      const ordered = NUMBERED_RE.test(line);
      const marker = ordered ? NUMBERED_RE : BULLET_RE;
      const items: string[] = [];
      while (i < lines.length && marker.test(lines[i])) {
        items.push(lines[i].replace(marker, ""));
        i++;
      }
      blocks.push({ type: "list", ordered, items });
      continue;
    }
    const para: string[] = [line];
    i++;
    while (
      i < lines.length &&
      lines[i].trim() &&
      !lines[i].trimStart().startsWith("```") &&
      !HEADING_RE.test(lines[i]) &&
      !BULLET_RE.test(lines[i]) &&
      !NUMBERED_RE.test(lines[i])
    ) {
      para.push(lines[i]);
      i++;
    }
    blocks.push({ type: "para", text: para.join("\n") });
  }
  return blocks;
}

/** Plain text, with rollout keys turned into detail links (route-relative). */
function withCitations(text: string, linkRollouts: boolean, keyBase: string): ReactNode[] {
  if (!linkRollouts) return [text];
  const out: ReactNode[] = [];
  const re = new RegExp(CITATION_RE.source, "g");
  let last = 0;
  let k = 0;
  let match: RegExpExecArray | null;
  while ((match = re.exec(text)) !== null) {
    const [whole, split, iteration, group, traj] = match;
    if (match.index > last) out.push(text.slice(last, match.index));
    out.push(
      // ".." resolves to the run root (the chat screen's parent route), so
      // this navigates within the current run in both server modes.
      <Link
        key={`${keyBase}-cite${k++}`}
        relative="route"
        to={`../rollout/${encodeURIComponent(split)}/${iteration}/${group}/${traj}`}
      >
        {whole}
      </Link>,
    );
    last = match.index + whole.length;
  }
  if (last < text.length) out.push(text.slice(last));
  return out;
}

/** Inline markdown: `code`, **bold**, citations; in that nesting order. */
function inline(text: string, linkRollouts: boolean, keyBase: string): ReactNode[] {
  const out: ReactNode[] = [];
  text.split(/(`[^`\n]+`)/).forEach((piece, i) => {
    if (piece.startsWith("`") && piece.endsWith("`") && piece.length > 2) {
      out.push(<code key={`${keyBase}-code${i}`}>{piece.slice(1, -1)}</code>);
      return;
    }
    piece.split(/(\*\*[^*\n]+\*\*)/).forEach((sub, j) => {
      if (sub.startsWith("**") && sub.endsWith("**") && sub.length > 4) {
        out.push(
          <strong key={`${keyBase}-b${i}-${j}`}>
            {withCitations(sub.slice(2, -2), linkRollouts, `${keyBase}-b${i}-${j}`)}
          </strong>,
        );
      } else if (sub) {
        out.push(...withCitations(sub, linkRollouts, `${keyBase}-t${i}-${j}`));
      }
    });
  });
  return out;
}

export function MarkdownLite({
  text,
  linkRollouts = false,
}: {
  text: string;
  linkRollouts?: boolean;
}) {
  return (
    <div className="md">
      {parseBlocks(text).map((block, i) => {
        if (block.type === "code") return <pre key={i}>{block.text}</pre>;
        if (block.type === "heading")
          return <p key={i} className="md-heading">{inline(block.text, linkRollouts, `h${i}`)}</p>;
        if (block.type === "list") {
          const items = block.items.map((item, j) => (
            <li key={j}>{inline(item, linkRollouts, `l${i}-${j}`)}</li>
          ));
          return block.ordered ? <ol key={i}>{items}</ol> : <ul key={i}>{items}</ul>;
        }
        return <p key={i}>{inline(block.text, linkRollouts, `p${i}`)}</p>;
      })}
    </div>
  );
}

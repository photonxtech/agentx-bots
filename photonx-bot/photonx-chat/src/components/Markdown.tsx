import { type ReactNode } from "react";

/**
 * A tiny, dependency-free Markdown renderer for the subset the copilot emits:
 * headings, bold/italic, inline code, `[1]` citation chips, bullet and numbered
 * lists, and paragraphs. It builds React elements directly (no
 * `dangerouslySetInnerHTML`), so there is no HTML-injection surface even though
 * the text comes from an LLM.
 */

// --- inline: **bold**, *italic*, `code`, and [1] citation markers ----------
const INLINE_RE = /(\*\*[^*]+\*\*|(?<!\*)\*[^*]+\*(?!\*)|`[^`]+`|\[\d{1,2}\])/g;

function renderInline(text: string): ReactNode[] {
  const out: ReactNode[] = [];
  let last = 0;
  let key = 0;
  for (const m of text.matchAll(INLINE_RE)) {
    const tok = m[0];
    const start = m.index ?? 0;
    if (start > last) out.push(text.slice(last, start));

    if (tok.startsWith("**")) {
      out.push(<strong key={key++}>{tok.slice(2, -2)}</strong>);
    } else if (tok.startsWith("`")) {
      out.push(<code key={key++}>{tok.slice(1, -1)}</code>);
    } else if (tok.startsWith("*")) {
      out.push(<em key={key++}>{tok.slice(1, -1)}</em>);
    } else {
      // [n] citation marker → a small chip.
      out.push(
        <span className="cite" key={key++}>
          {tok.slice(1, -1)}
        </span>,
      );
    }
    last = start + tok.length;
  }
  if (last < text.length) out.push(text.slice(last));
  return out;
}

// --- block: split into headings / lists / paragraphs -----------------------
export function Markdown({ text }: { text: string }) {
  const lines = text.replace(/\r\n/g, "\n").split("\n");
  const blocks: ReactNode[] = [];
  let i = 0;
  let key = 0;

  while (i < lines.length) {
    const line = lines[i];
    const trimmed = line.trim();

    if (!trimmed) {
      i++;
      continue;
    }

    // Heading: #, ##, ###
    const heading = /^(#{1,3})\s+(.*)$/.exec(trimmed);
    if (heading) {
      const level = heading[1].length;
      const content = renderInline(heading[2]);
      blocks.push(
        level === 1 ? (
          <h3 key={key++}>{content}</h3>
        ) : level === 2 ? (
          <h4 key={key++}>{content}</h4>
        ) : (
          <h5 key={key++}>{content}</h5>
        ),
      );
      i++;
      continue;
    }

    // Bullet list: -, *, •
    if (/^[-*•]\s+/.test(trimmed)) {
      const items: ReactNode[] = [];
      while (i < lines.length && /^\s*[-*•]\s+/.test(lines[i])) {
        const item = lines[i].replace(/^\s*[-*•]\s+/, "");
        items.push(<li key={items.length}>{renderInline(item)}</li>);
        i++;
      }
      blocks.push(<ul key={key++}>{items}</ul>);
      continue;
    }

    // Numbered list: 1. 2. 3.
    if (/^\d+\.\s+/.test(trimmed)) {
      const items: ReactNode[] = [];
      while (i < lines.length && /^\s*\d+\.\s+/.test(lines[i])) {
        const item = lines[i].replace(/^\s*\d+\.\s+/, "");
        items.push(<li key={items.length}>{renderInline(item)}</li>);
        i++;
      }
      blocks.push(<ol key={key++}>{items}</ol>);
      continue;
    }

    // Paragraph: consume until a blank line or a block-start line.
    const para: string[] = [];
    while (
      i < lines.length &&
      lines[i].trim() &&
      !/^(#{1,3}\s|[-*•]\s|\d+\.\s)/.test(lines[i].trim())
    ) {
      para.push(lines[i].trim());
      i++;
    }
    blocks.push(<p key={key++}>{renderInline(para.join(" "))}</p>);
  }

  return <div className="markdown">{blocks}</div>;
}

/** A small markdown renderer. It builds DOM nodes and never assigns HTML. */

export function renderMarkdown(text, { copyLabel, onCopy }) {
  const fragment = document.createDocumentFragment();
  for (const block of splitBlocks(text || "")) {
    fragment.append(renderBlock(block, copyLabel, onCopy));
  }
  if (!fragment.childNodes.length) {
    const empty = document.createElement("p");
    fragment.append(empty);
  }
  return fragment;
}

function splitBlocks(text) {
  const lines = text.replace(/\r\n/g, "\n").split("\n");
  const blocks = [];
  let index = 0;
  while (index < lines.length) {
    const fence = lines[index].match(/^```(.*)$/);
    if (fence) {
      const body = [];
      index += 1;
      while (index < lines.length && !lines[index].startsWith("```")) {
        body.push(lines[index]);
        index += 1;
      }
      if (index < lines.length) index += 1;
      blocks.push({ type: "code", text: body.join("\n") });
      continue;
    }
    if (!lines[index].trim()) {
      index += 1;
      continue;
    }
    const heading = lines[index].match(/^(#{1,3}) (.*)$/);
    if (heading) {
      blocks.push({ type: "heading", level: heading[1].length, text: heading[2] });
      index += 1;
      continue;
    }
    if (/^[-*] /.test(lines[index])) {
      const items = [];
      while (index < lines.length && /^[-*] /.test(lines[index])) {
        items.push(lines[index].slice(2));
        index += 1;
      }
      blocks.push({ type: "ul", items });
      continue;
    }
    if (/^\d+\. /.test(lines[index])) {
      const items = [];
      while (index < lines.length && /^\d+\. /.test(lines[index])) {
        items.push(lines[index].replace(/^\d+\. /, ""));
        index += 1;
      }
      blocks.push({ type: "ol", items });
      continue;
    }
    const paragraph = [];
    while (
      index < lines.length &&
      lines[index].trim() &&
      !lines[index].startsWith("```") &&
      !/^#{1,3} /.test(lines[index]) &&
      !/^[-*] /.test(lines[index]) &&
      !/^\d+\. /.test(lines[index])
    ) {
      paragraph.push(lines[index]);
      index += 1;
    }
    blocks.push({ type: "p", text: paragraph.join(" ") });
  }
  return blocks;
}

function renderBlock(block, copyLabel, onCopy) {
  if (block.type === "code") return renderCode(block.text, copyLabel, onCopy);
  if (block.type === "heading") {
    const heading = document.createElement(`h${block.level + 2}`);
    appendInline(heading, block.text);
    return heading;
  }
  if (block.type === "ul" || block.type === "ol") {
    const list = document.createElement(block.type);
    for (const item of block.items) {
      const li = document.createElement("li");
      appendInline(li, item);
      list.append(li);
    }
    return list;
  }
  const paragraph = document.createElement("p");
  appendInline(paragraph, block.text);
  return paragraph;
}

function renderCode(text, copyLabel, onCopy) {
  const wrap = document.createElement("div");
  wrap.className = "code-block";
  const copy = document.createElement("button");
  copy.type = "button";
  copy.className = "code-copy";
  copy.textContent = copyLabel;
  copy.addEventListener("click", () => onCopy(text, copy));
  const pre = document.createElement("pre");
  const code = document.createElement("code");
  code.textContent = text;
  pre.append(code);
  wrap.append(copy, pre);
  return wrap;
}

function appendInline(parent, text) {
  const pattern = /(`[^`]+`|\*\*[^*]+\*\*|\*[^*]+\*|\[[^\]\n]+\]\(https?:\/\/[^\s)]+\))/g;
  let last = 0;
  for (const match of text.matchAll(pattern)) {
    if (match.index > last) parent.append(document.createTextNode(text.slice(last, match.index)));
    const token = match[0];
    if (token.startsWith("`")) {
      const code = document.createElement("code");
      code.textContent = token.slice(1, -1);
      parent.append(code);
    } else if (token.startsWith("**")) {
      const strong = document.createElement("strong");
      strong.textContent = token.slice(2, -2);
      parent.append(strong);
    } else if (token.startsWith("*")) {
      const em = document.createElement("em");
      em.textContent = token.slice(1, -1);
      parent.append(em);
    } else {
      const link = document.createElement("a");
      const linked = token.match(/^\[([^\]\n]+)\]\((https?:\/\/[^\s)]+)\)$/);
      link.href = linked[2];
      link.textContent = linked[1];
      link.rel = "noreferrer";
      link.target = "_blank";
      parent.append(link);
    }
    last = match.index + token.length;
  }
  if (last < text.length) parent.append(document.createTextNode(text.slice(last)));
}

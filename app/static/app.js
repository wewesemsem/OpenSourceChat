import { button, createDropdown, openDialog, textInput } from "./components.js";
import { renderMarkdown } from "./markdown.js";

const sidebar = document.querySelector("#sidebar");
const conversations = document.querySelector("#conversations");
const appTitle = document.querySelector("#app-title");
const newChatSlot = document.querySelector("#new-chat");
const titleEl = document.querySelector("#conversation-title");
const pickerSlot = document.querySelector("#model-picker");
const thread = document.querySelector("#thread");
const empty = document.querySelector("#empty");
const greeting = document.querySelector("#greeting");
const examples = document.querySelector("#examples");
const trimmed = document.querySelector("#trimmed");
const errorBox = document.querySelector("#error");
const errorText = document.querySelector("#error-text");
const retryButton = document.querySelector("#retry");
const jump = document.querySelector("#jump");
const form = document.querySelector("#composer");
const attachmentBox = document.querySelector("#attachment");
const attachButton = document.querySelector("#attach");
const fileInput = document.querySelector("#file-input");
const input = document.querySelector("#message");
const sendButton = document.querySelector("#send");
const messageLabel = document.querySelector("#message-label");
const footer = document.querySelector("#footer");
const toggle = document.querySelector("#sidebar-toggle");

const state = {
  meta: null,
  models: [],
  conversations: [],
  activeId: null,
  messages: [],
  model: null,
  sending: false,
  abort: null,
  stick: true,
  dialog: null,
  picker: null,
  attachment: null,
};

function t(key) {
  return state.meta?.strings?.[key] ?? key;
}

function url(path) {
  return `${state.meta?.api_base || ""}${path}`;
}

function pageValue(name) {
  return document.querySelector(`meta[name="${name}"]`)?.getAttribute("content") || "";
}

function apiHeaders(extra = {}) {
  const headers = { ...extra };
  const csrfHeader = pageValue("csrf-header");
  if (csrfHeader) headers[csrfHeader] = pageValue("csrf-token");
  const authToken = pageValue("auth-token");
  const authHeader = pageValue("auth-header");
  if (authToken && authHeader) {
    const scheme = pageValue("auth-scheme");
    headers[authHeader] = scheme ? `${scheme} ${authToken}` : authToken;
  }
  return headers;
}

function labelFor(name) {
  return state.models.find((item) => item.name === name)?.label || name;
}

function formatLatency(ms) {
  if (ms == null) return "";
  if (ms < 1000) return `${ms} ${t("milliseconds")}`;
  return `${(ms / 1000).toFixed(1)} ${t("seconds")}`;
}

async function load() {
  const response = await fetch("/api/meta", { headers: apiHeaders() });
  state.meta = await response.json();
  document.title = state.meta.title;
  appTitle.textContent = state.meta.title;
  footer.textContent = state.meta.footer;
  greeting.textContent = t("empty_greeting");
  messageLabel.textContent = t("message_label");
  input.placeholder = t("placeholder");
  retryButton.textContent = t("retry");
  jump.textContent = t("jump_to_latest");
  trimmed.textContent = t("trimmed");
  sendButton.textContent = t("send");
  attachButton.textContent = t("attach");
  attachButton.hidden = !state.meta.files_enabled;
  conversations.setAttribute("aria-label", t("conversations_label"));
  thread.setAttribute("aria-label", t("thread_label"));
  buildChrome();
  state.model = state.meta.default_model;
  await refreshModels();
  await refreshConversations();
  const requested = location.hash.replace("#", "");
  if (requested && state.conversations.some((item) => item.id === requested)) {
    await openConversation(requested);
  } else {
    showEmpty();
  }
  input.focus();
}

function buildChrome() {
  newChatSlot.replaceChildren(
    button(t("new_chat"), {
      onClick: () => {
        if (state.sending) return;
        showEmpty();
      },
    }),
  );
  attachButton.addEventListener("click", () => {
    if (state.sending) return;
    if (state.meta.upload_enabled) {
      const extensions = state.meta.file_extensions || [];
      fileInput.accept = extensions.map((ext) => `.${ext}`).join(",");
      fileInput.value = "";
      fileInput.click();
      return;
    }
    openPicker(null);
  });
  toggle.addEventListener("click", () => setSidebar(sidebar.classList.contains("collapsed")));
  setSidebar(!window.matchMedia("(max-width: 720px)").matches);
  state.picker = createDropdown({
    label: t("model_label"),
    statusLabels: { ready: t("ready"), offline: t("offline") },
    onSelect: (name) => {
      state.model = name;
      state.picker.setOptions(state.models, state.model);
    },
    onOpen: () => refreshModels(),
  });
  pickerSlot.replaceChildren(state.picker.root);
}

function setSidebar(open) {
  sidebar.classList.toggle("collapsed", !open);
  const label = open ? t("hide_sidebar") : t("show_sidebar");
  toggle.textContent = label;
  toggle.setAttribute("aria-expanded", String(open));
  toggle.setAttribute("aria-label", label);
}

async function refreshModels() {
  try {
    const response = await fetch(url("/api/models"), { headers: apiHeaders() });
    if (!response.ok) throw new Error("models");
    state.models = (await response.json()).models;
  } catch {
    state.models = [{ name: state.meta.default_model, label: state.meta.default_model, status: "offline" }];
  }
  if (!state.models.some((item) => item.name === state.model)) {
    state.model = state.models[0]?.name || state.meta.default_model;
  }
  state.picker?.setOptions(state.models, state.model);
}

async function refreshConversations() {
  const response = await fetch(url("/api/conversations"), { headers: apiHeaders() });
  if (!response.ok) return;
  state.conversations = (await response.json()).conversations;
}

function showEmpty() {
  state.activeId = null;
  state.messages = [];
  history.replaceState(null, "", location.pathname);
  titleEl.textContent = t("untitled");
  clearError();
  trimmed.hidden = true;
  state.attachment = null;
  renderAttachment();
  renderThread();
  renderSidebar();
  input.focus();
}

async function openConversation(id) {
  if (state.sending) return;
  const response = await fetch(url(`/api/conversations/${id}`), { headers: apiHeaders() });
  if (!response.ok) {
    showEmpty();
    return;
  }
  const payload = await response.json();
  state.activeId = id;
  state.messages = payload.messages;
  state.attachment = payload.attachment || null;
  renderAttachment();
  history.replaceState(null, "", `#${id}`);
  titleEl.textContent = payload.conversation.title;
  clearError();
  trimmed.hidden = true;
  state.stick = true;
  renderThread();
  renderSidebar();
  if (window.matchMedia("(max-width: 720px)").matches) setSidebar(false);
}

function renderSidebar() {
  conversations.replaceChildren();
  for (const item of state.conversations) {
    const row = document.createElement("li");
    row.className = `conversation${item.id === state.activeId ? " active" : ""}`;
    const open = button(item.title, {
      className: "conversation-title",
      onClick: () => openConversation(item.id),
    });
    const actions = document.createElement("div");
    actions.className = "row-actions";
    actions.append(
      button(t("rename"), {
        className: "text",
        onClick: () => startRename(row, item),
      }),
      button(t("delete"), {
        className: "text danger",
        onClick: () => askDelete(item),
      }),
    );
    row.append(open, actions);
    conversations.append(row);
  }
}

function startRename(row, item) {
  const field = textInput({
    value: item.title,
    ariaLabel: t("rename_label"),
    onCommit: async (value) => {
      const next = value.trim();
      if (!next || next === item.title) {
        renderSidebar();
        return;
      }
      const response = await fetch(url(`/api/conversations/${item.id}`), {
        method: "PATCH",
        headers: apiHeaders({ "Content-Type": "application/json" }),
        body: JSON.stringify({ title: next }),
      });
      if (response.ok && state.activeId === item.id) titleEl.textContent = next;
      await refreshConversations();
      renderSidebar();
    },
    onCancel: () => renderSidebar(),
  });
  row.replaceChildren(field);
  field.focus();
  field.select();
}

function askDelete(item) {
  state.dialog = openDialog({
    title: t("delete_title"),
    body: t("delete_body"),
    confirmLabel: t("delete"),
    cancelLabel: t("cancel"),
    onClose: () => {
      state.dialog = null;
    },
    onConfirm: async () => {
      await fetch(url(`/api/conversations/${item.id}`), { method: "DELETE", headers: apiHeaders() });
      await refreshConversations();
      if (state.activeId === item.id) showEmpty();
      else renderSidebar();
    },
  });
}

function renderExamples() {
  examples.replaceChildren();
  for (const prompt of state.meta.examples || []) {
    examples.append(
      button(prompt, {
        className: "example",
        onClick: () => {
          input.value = prompt;
          grow();
          input.focus();
        },
      }),
    );
  }
}

function renderThread() {
  const nodes = [];
  if (!state.messages.length) {
    empty.hidden = false;
    renderExamples();
    nodes.push(empty);
  } else {
    empty.hidden = true;
    let previous = null;
    state.messages.forEach((message, index) => {
      if (message.model && previous && message.model !== previous) {
        const divider = document.createElement("p");
        divider.className = "divider";
        divider.textContent = t("now_using").replace("{model}", labelFor(message.model));
        nodes.push(divider);
      }
      if (message.model) previous = message.model;
      nodes.push(renderMessage(message, index === state.messages.length - 1));
    });
  }
  nodes.push(trimmed);
  if (!errorBox.hidden) nodes.push(errorBox);
  nodes.push(jump);
  thread.replaceChildren(...nodes);
  follow(true);
}

function renderMessage(message, isLast) {
  const article = document.createElement("article");
  article.className = `message ${message.role}${message.streaming ? " streaming" : ""}`;
  if (message.role === "assistant") {
    const meta = document.createElement("p");
    meta.className = "meta";
    const latency = formatLatency(message.latency_ms);
    meta.textContent = [labelFor(message.model), latency].filter(Boolean).join(" · ");
    article.append(meta);
    if (message.streaming) {
      const plain = document.createElement("div");
      plain.className = message.content ? "plain" : "plain pending";
      plain.textContent = message.content || t("waiting");
      article.append(plain);
    } else {
      const body = document.createElement("div");
      body.className = "body";
      body.append(renderMarkdown(message.content, { copyLabel: t("copy"), onCopy: copyText }));
      article.append(body);
    }
  } else {
    const body = document.createElement("div");
    body.className = "body";
    body.textContent = message.content;
    article.append(body);
  }
  const actions = document.createElement("div");
  actions.className = "actions";
  actions.append(button(t("copy"), { className: "text", onClick: (event) => copyText(message.content, event.currentTarget) }));
  if (isLast && message.role === "assistant" && !message.streaming && message.content) {
    actions.append(button(t("regenerate"), { className: "text", onClick: () => sendMessage("", { regenerate: true }) }));
  }
  article.append(actions);
  return article;
}

function follow(force) {
  if (force || state.stick) thread.scrollTop = thread.scrollHeight;
  jump.hidden = state.stick || !state.messages.length;
}

function paintStreaming(text) {
  const node = thread.querySelector(".message.streaming .plain");
  if (!node) return;
  node.classList.remove("pending");
  node.textContent = text;
  follow(false);
}

function clearError() {
  errorBox.hidden = true;
  errorText.textContent = "";
  retryButton.hidden = false;
}

function showError(message, { retry = true } = {}) {
  errorText.textContent = message;
  retryButton.hidden = !retry;
  errorBox.hidden = false;
  renderThread();
}

function errorMessage(payload) {
  if (!payload) return t("model_error");
  if (payload.code === "unreachable") return t("unreachable");
  return payload.message || t("model_error");
}

async function readDetail(response) {
  const payload = await response.json().catch(() => ({}));
  return typeof payload.detail === "string" ? payload.detail : t("model_error");
}

async function copyText(text, control) {
  try {
    await navigator.clipboard.writeText(text);
  } catch {
    return;
  }
  const previous = control.textContent;
  control.textContent = t("copied");
  setTimeout(() => {
    control.textContent = previous;
  }, 1200);
}

async function sendMessage(content, { regenerate = false } = {}) {
  if (state.sending) return;
  clearError();
  trimmed.hidden = true;
  const model = state.model || state.meta.default_model;
  if (!regenerate) {
    state.messages.push({ role: "user", content, model, id: null });
  } else if (state.messages.at(-1)?.role === "assistant") {
    state.messages.pop();
    const user = state.messages.at(-1);
    if (user?.role === "user") user.model = model;
  }
  const assistant = { role: "assistant", content: "", model, streaming: true, id: null, latency_ms: null };
  state.messages.push(assistant);
  state.sending = true;
  state.stick = true;
  sendButton.textContent = t("stop");
  state.abort = new AbortController();
  renderThread();
  let userMessageId = null;
  try {
    const response = await fetch(url("/api/chat"), {
      method: "POST",
      headers: apiHeaders({ "Content-Type": "application/json" }),
      body: JSON.stringify({
        content: regenerate ? null : content,
        model,
        conversation_id: state.activeId,
        regenerate,
      }),
      signal: state.abort.signal,
    });
    if (!response.ok || !response.body) {
      state.messages.pop();
      if (!regenerate) state.messages.pop();
      const failure = await response.json().catch(() => ({}));
      if (failure.code === "file_over_budget") {
        offerFit(failure);
        return;
      }
      throw new Error(typeof failure.detail === "string" ? failure.detail : t("model_error"));
    }
    await readStream(response, (payload) => {
      if (payload.conversation) {
        state.activeId = payload.conversation.id;
        titleEl.textContent = payload.conversation.title;
        history.replaceState(null, "", `#${payload.conversation.id}`);
        upsertConversation(payload.conversation);
      }
      if (payload.user_message_id) {
        userMessageId = payload.user_message_id;
        const user = [...state.messages].reverse().find((item) => item.role === "user" && !item.id);
        if (user) user.id = payload.user_message_id;
      }
      if (payload.trimmed) trimmed.hidden = false;
      if (typeof payload.delta === "string" && payload.delta) {
        if (!assistant.content) assistant.content = "";
        assistant.content += payload.delta;
        paintStreaming(assistant.content);
      }
      if (payload.error) {
        const failure = new Error(errorMessage(payload.error));
        failure.partial = Boolean(assistant.content);
        throw failure;
      }
      if (payload.done) {
        assistant.streaming = false;
        assistant.id = payload.message_id;
        assistant.latency_ms = payload.latency_ms;
        assistant.model = payload.model || model;
      }
    });
    if (!assistant.content) state.messages.pop();
  } catch (error) {
    if (error.name === "AbortError") {
      await finishStop(assistant, content, userMessageId, regenerate);
      return;
    }
    assistant.streaming = false;
    if (!assistant.content) state.messages = state.messages.filter((item) => item !== assistant);
    showError(error.message || t("model_error"));
  } finally {
    state.sending = false;
    state.abort = null;
    sendButton.textContent = t("send");
    await refreshConversations();
    renderThread();
    renderSidebar();
    if (!state.dialog) input.focus();
  }
}

async function finishStop(assistant, content, userMessageId, regenerate) {
  assistant.streaming = false;
  if (assistant.content) return;
  state.messages = state.messages.filter((item) => item !== assistant);
  if (regenerate) return;
  if (userMessageId) {
    state.messages = state.messages.filter((item) => item.id !== userMessageId);
    await fetch(url(`/api/messages/${userMessageId}`), { method: "DELETE", headers: apiHeaders() });
  } else {
    state.messages.pop();
  }
  if (!state.messages.length && state.activeId) {
    await fetch(url(`/api/conversations/${state.activeId}`), {
      method: "DELETE",
      headers: apiHeaders(),
    });
    state.activeId = null;
    history.replaceState(null, "", location.pathname);
    titleEl.textContent = t("untitled");
  }
  input.value = content;
  grow();
}

function upsertConversation(conversation) {
  state.conversations = [
    conversation,
    ...state.conversations.filter((item) => item.id !== conversation.id),
  ];
  renderSidebar();
}

async function readStream(response, onEvent) {
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    const blocks = buffer.split("\n\n");
    buffer = blocks.pop() || "";
    for (const block of blocks) {
      const data = block
        .split("\n")
        .filter((line) => line.startsWith("data:"))
        .map((line) => line.slice(5).trim())
        .join("\n");
      if (!data) continue;
      onEvent(JSON.parse(data));
    }
  }
}

function grow() {
  input.style.height = "auto";
  input.style.height = `${Math.min(input.scrollHeight, 192)}px`;
}

function fill(template, values) {
  return Object.entries(values).reduce(
    (text, [key, value]) => text.replaceAll(`{${key}}`, String(value)),
    template,
  );
}

function renderAttachment() {
  attachmentBox.replaceChildren();
  if (!state.attachment) {
    attachmentBox.hidden = true;
    return;
  }
  attachmentBox.hidden = false;
  const name = document.createElement("span");
  name.textContent = state.attachment.name;
  attachmentBox.append(name);
  if (state.attachment.use_prefix) {
    const note = document.createElement("span");
    note.className = "chip-note";
    note.textContent = t("file_prefixed");
    attachmentBox.append(note);
  }
  attachmentBox.append(button(t("file_remove"), { className: "text", onClick: () => removeAttachment() }));
}

async function removeAttachment() {
  const current = state.attachment;
  if (!current || state.sending) return;
  const response = await fetch(url(`/api/attachments/${current.id}`), {
    method: "DELETE",
    headers: apiHeaders(),
  });
  if (!response.ok) return;
  state.attachment = null;
  renderAttachment();
}

function rememberAttachment(payload) {
  state.attachment = payload.attachment;
  if (payload.conversation) {
    state.activeId = payload.conversation.id;
    titleEl.textContent = payload.conversation.title;
    history.replaceState(null, "", `#${payload.conversation.id}`);
    upsertConversation(payload.conversation);
  }
  renderAttachment();
}

async function attachFile(fileId, choice) {
  const response = await fetch(url("/api/attachments"), {
    method: "POST",
    headers: apiHeaders({ "Content-Type": "application/json" }),
    body: JSON.stringify({
      file_id: fileId,
      conversation_id: state.activeId,
      model: state.model,
      choice,
    }),
  });
  const payload = await response.json().catch(() => ({}));
  if (response.status === 409 && payload.code === "file_over_budget") {
    offerFit(payload, fileId);
    return;
  }
  if (!response.ok) {
    showError(typeof payload.detail === "string" ? payload.detail : t("model_error"), { retry: false });
    return;
  }
  rememberAttachment(payload);
  await refreshConversations();
  renderSidebar();
}

function closeDialog() {
  if (!state.dialog) return;
  const current = state.dialog;
  state.dialog = null;
  current.close();
}

function openChoices({ title, body, actions, onDismiss }) {
  closeDialog();
  let acted = false;
  const backdrop = document.createElement("div");
  backdrop.className = "dialog-backdrop";
  const dialog = document.createElement("div");
  dialog.className = "dialog";
  dialog.setAttribute("role", "dialog");
  dialog.setAttribute("aria-modal", "true");
  const heading = document.createElement("h2");
  heading.id = "dialog-title";
  dialog.setAttribute("aria-labelledby", "dialog-title");
  heading.textContent = title;
  const copy = document.createElement("p");
  copy.textContent = body;
  const row = document.createElement("div");
  row.className = "dialog-actions";
  const close = () => {
    backdrop.remove();
    if (state.dialog && state.dialog.close === close) state.dialog = null;
    if (!acted && onDismiss) onDismiss();
  };
  for (const action of actions) {
    row.append(
      button(action.label, {
        className: action.className || "",
        onClick: () => {
          acted = true;
          closeDialog();
          action.onClick();
        },
      }),
    );
  }
  dialog.append(heading);
  if (body) dialog.append(copy);
  if (actions.length) dialog.append(row);
  backdrop.append(dialog);
  backdrop.addEventListener("click", (event) => {
    if (event.target === backdrop) closeDialog();
  });
  document.body.append(backdrop);
  state.dialog = { close };
  row.querySelector("button")?.focus();
}

function offerFit(payload, fileId) {
  const id = fileId || payload.file_id;
  const uploadId = payload.upload_id;
  if (!id && !uploadId) {
    showError(payload.detail || t("model_error"), { retry: false });
    return;
  }
  openChoices({
    title: t("attach_title"),
    body: payload.detail,
    onDismiss: () => {
      if (uploadId) discardUpload(uploadId);
    },
    actions: [
      {
        label: fill(t("file_use_prefix"), { budget: payload.budget_tokens }),
        onClick: () => (uploadId ? finishStored(uploadId, "prefix") : attachFile(id, "prefix")),
      },
      {
        label: t("file_index"),
        className: "text",
        onClick: () => (uploadId ? finishStored(uploadId, "index") : attachFile(id, "index")),
      },
    ],
  });
}

async function uploadAttachment(file) {
  const params = new URLSearchParams({ filename: file.name });
  if (state.activeId) params.set("conversation_id", state.activeId);
  if (state.model) params.set("model", state.model);
  const response = await fetch(url(`/api/attachments/upload?${params}`), {
    method: "POST",
    headers: apiHeaders({ "Content-Type": "application/octet-stream" }),
    body: file,
  });
  await settleAttachment(response);
}

async function finishStored(uploadId, choice) {
  const response = await fetch(url("/api/attachments"), {
    method: "POST",
    headers: apiHeaders({ "Content-Type": "application/json" }),
    body: JSON.stringify({
      upload_id: uploadId,
      conversation_id: state.activeId,
      model: state.model,
      choice,
    }),
  });
  await settleAttachment(response);
}

async function discardUpload(uploadId) {
  await fetch(url(`/api/uploads/${uploadId}`), { method: "DELETE", headers: apiHeaders() });
}

async function settleAttachment(response) {
  const payload = await response.json().catch(() => ({}));
  if (response.status === 409 && payload.code === "file_over_budget") {
    offerFit(payload);
    return;
  }
  if (!response.ok) {
    showError(typeof payload.detail === "string" ? payload.detail : t("model_error"), { retry: false });
    return;
  }
  rememberAttachment(payload);
  await refreshConversations();
  renderSidebar();
}

fileInput.addEventListener("change", () => {
  const file = fileInput.files && fileInput.files[0];
  if (file) uploadAttachment(file);
});

async function openPicker(folderId) {
  const path = folderId ? `/api/files/${folderId}` : "/api/files";
  const response = await fetch(url(path), { headers: apiHeaders() });
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) {
    showError(typeof payload.detail === "string" ? payload.detail : t("model_error"), { retry: false });
    return;
  }
  closeDialog();
  const list = document.createElement("div");
  list.className = "file-list";
  if (folderId) {
    list.append(button(t("file_back"), { className: "text", onClick: () => openPicker(payload.parent_id) }));
  }
  if (!payload.entries.length) {
    const empty = document.createElement("p");
    empty.textContent = t("file_empty_list");
    list.append(empty);
  }
  for (const entry of payload.entries) {
    list.append(
      button(entry.name, {
        className: "text",
        onClick: () => {
          if (entry.kind === "folder") openPicker(entry.id);
          else {
            closeDialog();
            attachFile(entry.id, null);
          }
        },
      }),
    );
  }
  if (payload.truncated) {
    const note = document.createElement("p");
    note.textContent = fill(t("file_list_truncated"), { count: payload.entries.length });
    list.append(note);
  }
  openChoices({ title: t("attach_title"), body: "", actions: [] });
  const dialog = document.querySelector(".dialog");
  dialog.append(list);
  list.querySelector("button")?.focus();
}

form.addEventListener("submit", (event) => {
  event.preventDefault();
  if (state.sending) return;
  const text = input.value.trim();
  if (!text) return;
  input.value = "";
  grow();
  sendMessage(text);
});

sendButton.addEventListener("click", (event) => {
  if (!state.sending) return;
  event.preventDefault();
  state.abort?.abort();
});

input.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    form.requestSubmit();
  }
});
input.addEventListener("input", grow);

retryButton.addEventListener("click", () => {
  if (state.sending) return;
  sendMessage("", { regenerate: true });
});

jump.addEventListener("click", () => {
  state.stick = true;
  follow(true);
});

thread.addEventListener("scroll", () => {
  const distance = thread.scrollHeight - thread.scrollTop - thread.clientHeight;
  state.stick = distance < 80;
  jump.hidden = state.stick || !state.messages.length;
});

document.addEventListener("keydown", (event) => {
  if (event.key !== "Escape" || event.defaultPrevented) return;
  if (state.dialog) {
    event.preventDefault();
    state.dialog.close();
    return;
  }
  if (state.picker?.isOpen()) {
    event.preventDefault();
    state.picker.close();
    return;
  }
  if (state.sending) {
    event.preventDefault();
    state.abort?.abort();
  }
});

document.addEventListener("click", (event) => {
  if (state.picker && !state.picker.root.contains(event.target)) state.picker.close();
});

load();

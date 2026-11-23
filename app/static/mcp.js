/** Customize panel for MCP servers. Uses the same buttons and dialogs as the chat. */

const TRANSPORTS = [
  ["sse", "mcp_transport_sse"],
  ["streamable_http", "mcp_transport_http"],
  ["stdio", "mcp_transport_stdio"],
];

export function mountMcp({ slot, panel, thread, composer, titleEl, t, url, headers, openDialog, button, onChange, onShow }) {
  const title = panel.querySelector("#mcp-title");
  const intro = panel.querySelector("#mcp-intro");
  const addSlot = panel.querySelector("#mcp-add");
  const form = panel.querySelector("#mcp-form");
  const formError = panel.querySelector("#mcp-form-error");
  const empty = panel.querySelector("#mcp-empty");
  const list = panel.querySelector("#mcp-list");
  const connectorTitle = panel.querySelector("#mcp-connectors-title");
  const connectorList = panel.querySelector("#mcp-connector-list");
  const label = document.querySelector("#customize-label");
  let open = false;
  let editing = null;
  let chatTitle = "";
  let poll = 0;

  label.textContent = t("customize");
  title.textContent = t("mcp_servers");
  intro.textContent = t("mcp_intro");
  empty.textContent = t("mcp_empty");
  connectorTitle.textContent = t("mcp_connectors");

  const entry = button(t("mcp_servers"), {
    className: "text",
    onClick: () => {
      if (open) close();
      else show();
    },
  });
  slot.replaceChildren(entry);
  addSlot.replaceChildren(button(t("mcp_add"), { onClick: () => openForm(null) }));
  form.addEventListener("submit", (event) => {
    event.preventDefault();
    submit();
  });

  function show() {
    open = true;
    chatTitle = titleEl.textContent;
    titleEl.textContent = t("mcp_servers");
    entry.setAttribute("aria-current", "page");
    panel.hidden = false;
    thread.hidden = true;
    composer.hidden = true;
    if (onShow) onShow();
    refresh();
  }

  function close() {
    open = false;
    editing = null;
    form.hidden = true;
    formError.hidden = true;
    entry.removeAttribute("aria-current");
    stopPoll();
    panel.hidden = true;
    thread.hidden = false;
    composer.hidden = false;
    titleEl.textContent = chatTitle || t("untitled");
  }

  function stopPoll() {
    if (poll) clearTimeout(poll);
    poll = 0;
  }

  async function refresh() {
    stopPoll();
    formError.hidden = true;
    const response = await fetch(url("/api/mcp/servers"), { headers: headers() });
    if (!response.ok) {
      showError(t("mcp_save_failed"));
      return;
    }
    const payload = await response.json();
    const connectors = payload.connectors || [];
    const ids = new Set(connectors.map((item) => item.id));
    renderConnectors(connectors);
    render((payload.servers || []).filter((server) => !ids.has(server.name)));
    if (open && connectors.some((item) => item.pending)) {
      poll = setTimeout(refresh, 1000);
    }
  }

  function renderConnectors(connectors) {
    connectorList.replaceChildren();
    for (const item of connectors) {
      const row = document.createElement("li");
      row.className = "mcp-row";
      const name = document.createElement("div");
      name.className = "mcp-name";
      const heading = document.createElement("strong");
      heading.textContent = t(item.title_key);
      const summary = document.createElement("span");
      summary.className = "mcp-status";
      summary.textContent = t(item.summary_key);
      const status = document.createElement("span");
      status.className = "mcp-status";
      const stateLabel = item.status === "connected" ? t("mcp_connected") : t("mcp_disconnected");
      status.textContent = item.detail ? `${stateLabel}. ${item.detail}` : stateLabel;
      name.append(heading, summary, status);
      if (item.notice) {
        const notice = document.createElement("p");
        notice.className = "mcp-setup";
        notice.textContent = item.notice;
        name.append(notice);
      }
      row.append(name);
      if (item.pending && item.authorization_url) {
        const link = document.createElement("a");
        link.className = "mcp-sign-in";
        link.href = item.authorization_url;
        link.target = "_blank";
        link.rel = "noopener noreferrer";
        link.textContent = t("mcp_sign_in");
        row.append(link);
      }
      if (item.installed && item.signed_in && !item.pending) {
        const toggle = button(item.enabled ? t("mcp_on") : t("mcp_off"), {
          className: "switch",
          ariaLabel: `${t(item.title_key)}, ${item.enabled ? t("mcp_on") : t("mcp_off")}`,
          onClick: () => saveServer(connectorBody(item, !item.enabled), item.id),
        });
        toggle.setAttribute("aria-pressed", String(item.enabled));
        row.append(toggle);
      } else if (!item.pending) {
        row.append(button(t("mcp_connect"), { onClick: () => connectConnector(item) }));
      }
      if (item.installed || item.pending) {
        row.append(button(t("mcp_disconnect"), {
          className: "text",
          onClick: () => confirmDisconnect(item),
        }));
      }
      connectorList.append(row);
    }
  }

  function connectorBody(item, enabled) {
    return {
      name: item.id,
      transport: item.transport,
      url: item.url || "",
      command: "",
      args: [],
      auth_type: "oauth",
      token_env: "",
      allowed_tools: item.allowed_tools || [],
      enabled,
    };
  }

  async function connectConnector(item) {
    const response = await fetch(url(`/api/mcp/connectors/${encodeURIComponent(item.id)}/connect`), {
      method: "POST",
      headers: headers(),
    });
    const payload = await response.json().catch(() => ({}));
    if (!response.ok) {
      showError(typeof payload.detail === "string" ? payload.detail : t("mcp_oauth_failed"));
      return;
    }
    if (onChange) await onChange();
    await refresh();
  }

  function confirmDisconnect(item) {
    openDialog({
      title: t("mcp_disconnect_title"),
      body: t("mcp_disconnect_body"),
      confirmLabel: t("mcp_disconnect"),
      cancelLabel: t("cancel"),
      onConfirm: () => disconnectConnector(item),
      onClose: () => {},
    });
  }

  async function disconnectConnector(item) {
    const response = await fetch(url(`/api/mcp/connectors/${encodeURIComponent(item.id)}/disconnect`), {
      method: "POST",
      headers: headers(),
    });
    if (!response.ok) {
      showError(t("mcp_save_failed"));
      return;
    }
    if (onChange) await onChange();
    await refresh();
  }

  function render(servers) {
    empty.hidden = servers.length > 0;
    list.replaceChildren();
    for (const server of servers) {
      const row = document.createElement("li");
      row.className = "mcp-row";
      const name = document.createElement("div");
      name.className = "mcp-name";
      const heading = document.createElement("strong");
      heading.textContent = server.name;
      const status = document.createElement("span");
      status.className = "mcp-status";
      const stateLabel = server.status === "connected" ? t("mcp_connected") : t("mcp_disconnected");
      status.textContent = server.detail ? `${stateLabel}. ${server.detail}` : stateLabel;
      name.append(heading, status);
      const toggle = button(server.enabled ? t("mcp_on") : t("mcp_off"), {
        className: "switch",
        ariaLabel: `${server.name}, ${server.enabled ? t("mcp_on") : t("mcp_off")}`,
        onClick: () => saveServer({ ...server, enabled: !server.enabled }, server.name),
      });
      toggle.setAttribute("aria-pressed", String(server.enabled));
      const edit = button(t("mcp_edit"), { className: "text", onClick: () => openForm(server) });
      const remove = button(t("mcp_remove"), {
        className: "text",
        onClick: () => confirmRemove(server),
      });
      row.append(name, toggle, edit, remove);
      list.append(row);
    }
  }

  function openForm(server) {
    editing = server;
    formError.hidden = true;
    form.hidden = false;
    form.replaceChildren();
    const transport = server?.transport || "streamable_http";
    form.append(
      field(t("mcp_name"), "text", "name", server?.name || ""),
      selectField(t("mcp_transport"), "transport", TRANSPORTS, transport),
      field(t("mcp_url"), "text", "url", server?.url || ""),
      field(t("mcp_command"), "text", "command", server?.command || ""),
      field(t("mcp_args"), "text", "args", (server?.args || []).join(" ")),
      selectField(t("mcp_auth"), "auth_type", [["none", "mcp_auth_none"], ["bearer", "mcp_auth_bearer"]], server?.auth_type || "none"),
      field(t("mcp_token_env"), "text", "token_env", server?.token_env || ""),
      field(t("mcp_allowed_tools"), "text", "allowed_tools", (server?.allowed_tools || []).join(", ")),
    );
    syncForm();
    form.querySelector("[name=transport]").addEventListener("change", syncForm);
    form.querySelector("[name=auth_type]").addEventListener("change", syncForm);
    const actions = document.createElement("div");
    actions.className = "dialog-actions";
    actions.append(
      button(t("cancel"), { className: "text", onClick: () => { form.hidden = true; editing = null; } }),
      button(t("mcp_save"), { onClick: submit }),
    );
    form.append(actions);
    form.querySelector("[name=name]").focus();
  }

  function syncForm() {
    const transport = form.querySelector("[name=transport]").value;
    const auth = form.querySelector("[name=auth_type]").value;
    showRow("url", transport !== "stdio");
    showRow("command", transport === "stdio");
    showRow("args", transport === "stdio");
    showRow("token_env", auth === "bearer");
  }

  function showRow(name, visible) {
    const input = form.querySelector(`[name=${name}]`);
    if (input) input.closest("label").hidden = !visible;
  }

  function field(label, type, name, value) {
    const wrap = document.createElement("label");
    wrap.textContent = label;
    const input = document.createElement("input");
    input.type = type;
    input.name = name;
    input.value = value;
    input.autocomplete = "off";
    wrap.append(input);
    return wrap;
  }

  function selectField(label, name, options, value) {
    const wrap = document.createElement("label");
    wrap.textContent = label;
    const select = document.createElement("select");
    select.name = name;
    for (const [id, key] of options) {
      const option = document.createElement("option");
      option.value = id;
      option.textContent = t(key);
      option.selected = id === value;
      select.append(option);
    }
    wrap.append(select);
    return wrap;
  }

  function readForm() {
    const value = (name) => form.querySelector(`[name=${name}]`).value.trim();
    const transport = value("transport");
    return {
      name: value("name"),
      transport,
      url: transport === "stdio" ? "" : value("url"),
      command: transport === "stdio" ? value("command") : "",
      args: transport === "stdio" ? value("args").split(/\s+/).filter(Boolean) : [],
      auth_type: value("auth_type"),
      token_env: value("auth_type") === "bearer" ? value("token_env") : "",
      allowed_tools: value("allowed_tools").split(",").map((item) => item.trim()).filter(Boolean),
      enabled: editing ? Boolean(editing.enabled) : false,
    };
  }

  async function submit() {
    const body = readForm();
    const path = editing ? `/api/mcp/servers/${encodeURIComponent(editing.name)}` : "/api/mcp/servers";
    const response = await fetch(url(path), {
      method: editing ? "PATCH" : "POST",
      headers: headers({ "Content-Type": "application/json" }),
      body: JSON.stringify(body),
    });
    const payload = await response.json().catch(() => ({}));
    if (!response.ok) {
      showError(typeof payload.detail === "string" ? payload.detail : t("mcp_save_failed"));
      return;
    }
    form.hidden = true;
    editing = null;
    if (onChange) await onChange();
    await refresh();
  }

  async function saveServer(server, name) {
    const response = await fetch(url(`/api/mcp/servers/${encodeURIComponent(name)}`), {
      method: "PATCH",
      headers: headers({ "Content-Type": "application/json" }),
      body: JSON.stringify({
        name: server.name,
        transport: server.transport,
        url: server.url || "",
        command: server.command || "",
        args: server.args || [],
        auth_type: server.auth_type || "none",
        token_env: server.token_env || "",
        allowed_tools: server.allowed_tools || [],
        enabled: Boolean(server.enabled),
      }),
    });
    const payload = await response.json().catch(() => ({}));
    if (!response.ok) {
      showError(typeof payload.detail === "string" ? payload.detail : t("mcp_save_failed"));
      return;
    }
    if (onChange) await onChange();
    await refresh();
  }

  function confirmRemove(server) {
    openDialog({
      title: t("mcp_remove_title"),
      body: t("mcp_remove_body"),
      confirmLabel: t("mcp_remove"),
      cancelLabel: t("cancel"),
      onConfirm: () => removeServer(server),
      onClose: () => {},
    });
  }

  async function removeServer(server) {
    const response = await fetch(url(`/api/mcp/servers/${encodeURIComponent(server.name)}`), {
      method: "DELETE",
      headers: headers(),
    });
    if (!response.ok) {
      showError(t("mcp_save_failed"));
      return;
    }
    if (onChange) await onChange();
    await refresh();
  }

  function showError(message) {
    formError.textContent = message;
    formError.hidden = false;
  }

  return { close, isOpen: () => open };
}

export function button(label, { className = "", onClick, ariaLabel, type = "button" } = {}) {
  const element = document.createElement("button");
  element.type = type;
  element.textContent = label;
  if (className) element.className = className;
  if (ariaLabel) element.setAttribute("aria-label", ariaLabel);
  if (onClick) element.addEventListener("click", onClick);
  return element;
}

export function textInput({ value, ariaLabel, onCommit, onCancel }) {
  const field = document.createElement("input");
  field.type = "text";
  field.className = "text-input";
  field.value = value;
  field.setAttribute("aria-label", ariaLabel);
  let settled = false;
  const commit = () => {
    if (settled) return;
    settled = true;
    onCommit(field.value);
  };
  field.addEventListener("keydown", (event) => {
    if (event.key === "Enter") {
      event.preventDefault();
      commit();
    }
    if (event.key === "Escape") {
      event.preventDefault();
      event.stopPropagation();
      settled = true;
      onCancel();
    }
  });
  field.addEventListener("blur", commit);
  return field;
}

export function createDropdown({ label, statusLabels = {}, onSelect, onOpen }) {
  const root = document.createElement("div");
  root.className = "dropdown";
  const trigger = document.createElement("button");
  trigger.type = "button";
  trigger.className = "dropdown-trigger";
  trigger.setAttribute("aria-haspopup", "listbox");
  trigger.setAttribute("aria-expanded", "false");
  const menu = document.createElement("ul");
  menu.className = "dropdown-menu";
  menu.setAttribute("role", "listbox");
  menu.hidden = true;
  root.append(trigger, menu);

  function close() {
    menu.hidden = true;
    trigger.setAttribute("aria-expanded", "false");
  }

  function open() {
    menu.hidden = false;
    trigger.setAttribute("aria-expanded", "true");
    if (onOpen) onOpen();
  }

  trigger.addEventListener("click", () => {
    if (menu.hidden) open();
    else close();
  });

  function statusText(status) {
    const text = document.createElement("span");
    text.className = "visually-hidden";
    text.textContent = statusLabels[status] || status;
    return text;
  }

  function setOptions(options, selected) {
    const current = options.find((item) => item.name === selected) || options[0];
    trigger.replaceChildren();
    if (current) {
      trigger.append(dot(current.status), document.createTextNode(current.label));
      const status = statusLabels[current.status] || current.status;
      trigger.setAttribute("aria-label", `${label}: ${current.label}, ${status}`);
    }
    menu.replaceChildren();
    for (const option of options) {
      const item = document.createElement("li");
      item.setAttribute("role", "option");
      item.setAttribute("aria-selected", String(option.name === selected));
      const choice = button("", { className: "text" });
      const status = statusLabels[option.status] || option.status;
      choice.setAttribute("aria-label", `${option.label}, ${status}`);
      choice.replaceChildren(
        dot(option.status),
        document.createTextNode(option.label),
        statusText(option.status),
      );
      choice.addEventListener("click", () => {
        onSelect(option.name);
        close();
      });
      item.append(choice);
      menu.append(item);
    }
  }

  return { root, setOptions, close, isOpen: () => !menu.hidden };
}

function dot(status) {
  const mark = document.createElement("span");
  mark.className = status === "ready" ? "dot ready" : "dot";
  mark.setAttribute("aria-hidden", "true");
  return mark;
}

export function openDialog({ title, body, confirmLabel, cancelLabel, onConfirm, onClose }) {
  const backdrop = document.createElement("div");
  backdrop.className = "dialog-backdrop";
  const dialog = document.createElement("div");
  dialog.className = "dialog";
  dialog.setAttribute("role", "dialog");
  dialog.setAttribute("aria-modal", "true");
  dialog.setAttribute("aria-labelledby", "dialog-title");
  const heading = document.createElement("h2");
  heading.id = "dialog-title";
  heading.textContent = title;
  const copy = document.createElement("p");
  copy.textContent = body;
  const actions = document.createElement("div");
  actions.className = "dialog-actions";
  let closed = false;
  const close = () => {
    if (closed) return;
    closed = true;
    backdrop.remove();
    onClose();
  };
  const cancel = button(cancelLabel, { className: "text", onClick: close });
  const confirm = button(confirmLabel, {
    onClick: () => {
      close();
      onConfirm();
    },
  });
  actions.append(cancel, confirm);
  dialog.append(heading, copy, actions);
  backdrop.append(dialog);
  backdrop.addEventListener("click", (event) => {
    if (event.target === backdrop) close();
  });
  document.body.append(backdrop);
  cancel.focus();
  return { close };
}

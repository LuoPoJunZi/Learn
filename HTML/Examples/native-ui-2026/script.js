"use strict";

const root = document.documentElement;
const supportsPopover = "showPopover" in HTMLElement.prototype;
const supportsInvoker =
  "command" in HTMLButtonElement.prototype &&
  "commandForElement" in HTMLButtonElement.prototype;

function setupPopoverFallback() {
  if (supportsPopover) {
    return;
  }

  root.classList.add("no-popover");
  const popover = document.querySelector("#anchor-note");
  const trigger = document.querySelector("[popovertarget='anchor-note']");
  const closeButton = document.querySelector("[popovertargetaction='hide']");

  trigger?.addEventListener("click", () => {
    popover?.classList.toggle("is-open");
  });
  closeButton?.addEventListener("click", () => {
    popover?.classList.remove("is-open");
  });
}

function setupInvokerFallback() {
  if (supportsInvoker) {
    return;
  }

  document.querySelectorAll("button[commandfor]").forEach((button) => {
    button.addEventListener("click", () => {
      const targetId = button.getAttribute("commandfor");
      const command = button.getAttribute("command");
      const target = targetId ? document.getElementById(targetId) : null;

      if (!(target instanceof HTMLDialogElement)) {
        return;
      }

      if (command === "show-modal" && !target.open) {
        target.showModal();
      } else if (command === "close") {
        target.close(button.value);
      } else if (command === "request-close") {
        if ("requestClose" in target) {
          target.requestClose();
        } else {
          target.close();
        }
      }
    });
  });
}

function renderSupportStatus() {
  const supportList = document.querySelector("#support-list");
  if (!supportList) {
    return;
  }

  const checks = [
    {
      name: "Invoker Commands",
      detail: "command 与 commandfor",
      supported: supportsInvoker,
    },
    {
      name: "CSS Anchor Positioning",
      detail: "position-area 与锚定浮层",
      supported: CSS.supports("position-area: block-end"),
    },
    {
      name: "Scroll-driven Animations",
      detail: "scroll() 与 view() 时间线",
      supported: CSS.supports("animation-timeline: scroll()"),
    },
    {
      name: "Popover API",
      detail: "浏览器管理的顶层非模态浮层",
      supported: supportsPopover,
    },
  ];

  supportList.replaceChildren();
  checks.forEach((check) => {
    const row = document.createElement("div");
    const term = document.createElement("dt");
    const detail = document.createElement("dd");
    const state = document.createElement("dd");

    term.textContent = check.name;
    detail.textContent = check.detail;
    state.className = "support-state";
    state.dataset.supported = String(check.supported);
    state.textContent = check.supported ? "原生支持" : "使用回退";
    row.append(term, detail, state);
    supportList.append(row);
  });
}

function reportDialogResult() {
  const dialog = document.querySelector("#release-dialog");
  const output = document.querySelector("#dialog-result");
  if (!(dialog instanceof HTMLDialogElement) || !(output instanceof HTMLOutputElement)) {
    return;
  }

  dialog.addEventListener("close", () => {
    output.value =
      dialog.returnValue === "confirm"
        ? "已确认：浏览器完成关闭并返回 confirm"
        : "已取消：对话框返回 cancel";
  });
}

setupPopoverFallback();
setupInvokerFallback();
renderSupportStatus();
reportDialogResult();

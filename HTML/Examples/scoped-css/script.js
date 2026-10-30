"use strict";

const toggle = document.querySelector("#toggle");
const panel = document.querySelector("#panel");
const support = document.querySelector("#support");
const checks = document.querySelector("#checks");
const inside = document.querySelector("#inside");
const boundary = document.querySelector("#boundary");
const outside = document.querySelector("#outside");

function supportsScopeRule() {
  try {
    const sheet = new CSSStyleSheet();
    sheet.replaceSync("@scope (.probe) { a { color: blue; } }");
    return typeof CSSScopeRule !== "undefined" && sheet.cssRules[0] instanceof CSSScopeRule;
  } catch {
    return false;
  }
}

if (toggle && panel && support && checks && inside && boundary && outside) {
  const supported = supportsScopeRule();
  support.textContent = supported
    ? "浏览器可以解析 @scope；下面会检查实际计算样式。"
    : "未检测到完整 CSSScopeRule 支持，显示基础回退样式；请人工核对链接。";

  function verify() {
    if (!supported) {
      checks.textContent = "已跳过原生作用域断言：不把回退效果当成新特性成功。";
      return;
    }
    const enabled = panel.classList.contains("scoped-panel");
    const grey = "rgb(71, 85, 105)";
    const blue = "rgb(29, 78, 216)";
    const passed = getComputedStyle(inside).color === (enabled ? blue : grey)
      && getComputedStyle(boundary).color === grey
      && getComputedStyle(outside).color === grey;
    checks.textContent = passed
      ? "3 项样式检查通过：内部、下界、外部。"
      : "样式检查未通过：请查看开发者工具中的层叠来源或用户样式覆盖。";
  }

  toggle.addEventListener("click", () => {
    const enabled = panel.classList.toggle("scoped-panel");
    toggle.setAttribute("aria-pressed", String(enabled));
    toggle.textContent = enabled ? "作用域根：已启用" : "作用域根：已移除";
    verify();
  });
  verify();
}

"use strict";

const hasTemporal = typeof globalThis.Temporal !== "undefined";

function updateSupportBanner() {
  const box = document.querySelector(".status-box");
  const msg = document.getElementById("support-message");
  if (!box || !msg) return;

  if (hasTemporal) {
    box.classList.add("is-supported");
    msg.textContent = "检测到当前浏览器原生支持 Temporal API，下方所有演示将直接调用原生 Temporal。";
  } else {
    box.classList.add("is-fallback");
    msg.textContent =
      "当前浏览器尚未内置 Temporal API：下方将展示旧 Date 对象的坑与回退提示（真实项目在全面覆盖前可使用 Intl.DateTimeFormat 或官方 polyfill）。";
  }
}

function runDateMathDemo() {
  const dateInput = document.getElementById("base-date");
  const monthsInput = document.getElementById("add-months");
  const output = document.getElementById("date-math-output");
  if (!dateInput || !monthsInput || !output) return;

  const iso = dateInput.value || "2026-01-31";
  const months = Number.parseInt(monthsInput.value, 10) || 0;
  const parts = iso.split("-").map((v) => Number.parseInt(v, 10));

  // Legacy Date comparison: mutates original object and overflows Jan 31 + 1 month into March.
  const legacyOriginal = new Date(Date.UTC(parts[0], parts[1] - 1, parts[2]));
  const legacySnapshot = legacyOriginal.toISOString().slice(0, 10);
  legacyOriginal.setUTCMonth(legacyOriginal.getUTCMonth() + months);
  const legacyAfter = legacyOriginal.toISOString().slice(0, 10);

  const lines = [
    `[旧 Date] 调用前: ${legacySnapshot}`,
    `[旧 Date] setUTCMonth(+${months}) 后: ${legacyAfter}（原对象已被原地修改，且月末可能溢出到下月）`,
  ];

  if (hasTemporal) {
    const plain = globalThis.Temporal.PlainDate.from(iso);
    const shifted = plain.add({ months });
    lines.push(`[Temporal.PlainDate] 原对象保持不变: ${plain.toString()}（month=${plain.month}）`);
    lines.push(`[Temporal.PlainDate] .add({ months: ${months} }) 新对象: ${shifted.toString()}（月末自动钳位）`);
  } else {
    lines.push("[Temporal] 当前环境不支持原生 Temporal，已跳过原生计算（不伪造结果）。");
  }

  output.textContent = lines.join("\n");
}

function runInvalidDateDemo() {
  const output = document.getElementById("invalid-date-output");
  if (!output) return;

  const silentLegacy = new Date(Date.UTC(2026, 1, 30));
  const lines = [
    `[旧 Date] new Date(Date.UTC(2026, 1, 30)) -> ${silentLegacy.toISOString().slice(0, 10)}（静默进位，未报错）`,
  ];

  if (hasTemporal) {
    try {
      globalThis.Temporal.PlainDate.from(
        { year: 2026, month: 2, day: 30 },
        { overflow: "reject" }
      );
      lines.push("[Temporal] 意外通过（不应发生）");
    } catch (error) {
      lines.push(`[Temporal] overflow:"reject" 明确拦截 -> ${error.name}: ${error.message}`);
    }
  } else {
    lines.push('[Temporal] 当前环境不支持原生 Temporal；预期在支持环境中抛出 RangeError。');
  }

  output.textContent = lines.join("\n");
}

function runTimezoneDemo() {
  const select = document.getElementById("target-tz");
  const output = document.getElementById("timezone-output");
  if (!select || !output) return;

  const tz = select.value;
  const fallbackFormatted = new Intl.DateTimeFormat("zh-CN", {
    timeZone: tz,
    dateStyle: "full",
    timeStyle: "long",
    hour12: false,
  }).format(new Date());

  const lines = [`[Intl.DateTimeFormat 通用回退] (${tz}): ${fallbackFormatted}`];

  if (hasTemporal) {
    const zoned = globalThis.Temporal.Now.zonedDateTimeISO(tz);
    lines.push(`[Temporal.ZonedDateTime] (${tz}): ${zoned.toString()}`);
  } else {
    lines.push("[Temporal] 当前环境不支持原生 Temporal，生产过渡期推荐继续使用 Intl.DateTimeFormat。");
  }

  output.textContent = lines.join("\n");
}

document.addEventListener("DOMContentLoaded", () => {
  updateSupportBanner();
  runDateMathDemo();
  runInvalidDateDemo();
  runTimezoneDemo();

  document.getElementById("run-date-math")?.addEventListener("click", runDateMathDemo);
  document.getElementById("run-invalid-date")?.addEventListener("click", runInvalidDateDemo);
  document.getElementById("run-timezone")?.addEventListener("click", runTimezoneDemo);
});

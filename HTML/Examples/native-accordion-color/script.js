"use strict";

document.addEventListener("DOMContentLoaded", () => {
  const report = document.getElementById("feature-report");
  const slider = document.getElementById("hue-slider");
  const output = document.getElementById("hue-value");

  const supportsOklch =
    typeof CSS !== "undefined" && CSS.supports("color", "oklch(0.55 0.16 250)");
  const supportsColorMix =
    typeof CSS !== "undefined" &&
    CSS.supports("color", "color-mix(in oklch, red 20%, white)");

  if (report) {
    report.textContent =
      `oklch(): ${supportsOklch ? "支持" : "不支持（已启用 sRGB 回退）"} | ` +
      `color-mix(): ${supportsColorMix ? "支持" : "不支持（已启用 sRGB 回退）"}`;
  }

  if (slider && output) {
    const applyHue = () => {
      const hue = slider.value;
      document.documentElement.style.setProperty("--brand-hue", hue);
      output.textContent = `${hue}°`;
    };
    slider.addEventListener("input", applyHue);
    applyHue();
  }
});

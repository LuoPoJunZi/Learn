"use strict";

const source = document.querySelector("#source");
const loadButton = document.querySelector("#load");
const cancelButton = document.querySelector("#cancel");
const statusText = document.querySelector("#status");
const results = document.querySelector("#results");

if (source && loadButton && cancelButton && statusText && results) {
  const loader = createLatestLoader((state) => {
    const loading = state.status === "loading";
    results.setAttribute("aria-busy", String(loading));
    cancelButton.disabled = !loading;
    statusText.textContent = state.message;
    results.replaceChildren();
    if (state.status === "success") {
      for (const course of state.data) {
        const item = document.createElement("li");
        // Remote content must be rendered as text, not interpreted as HTML.
        item.textContent = course.title;
        results.append(item);
      }
    }
  });
  loadButton.addEventListener("click", () => { void loader.load(source.value); });
  cancelButton.addEventListener("click", () => { loader.cancel(); });
}

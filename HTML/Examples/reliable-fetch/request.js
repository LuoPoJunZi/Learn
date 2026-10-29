"use strict";

// Separate transport logic from DOM updates so the same code can be tested offline.
function createLatestLoader(render, fetchImpl = fetch) {
  let generation = 0;
  let activeController = null;

  async function load(url, timeoutMs = 3000) {
    if (!Number.isFinite(timeoutMs) || timeoutMs <= 0) {
      throw new RangeError("timeoutMs must be positive and finite");
    }
    const current = ++generation;
    activeController?.abort();
    const controller = new AbortController();
    activeController = controller;
    let timedOut = false;
    const timer = setTimeout(() => {
      timedOut = true;
      controller.abort();
    }, timeoutMs);
    render({ status: "loading", message: "正在加载……" });

    try {
      const response = await fetchImpl(url, { signal: controller.signal });
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}`);
      }
      const data = await response.json();
      // Do not trust a cancelled request to stop at exactly the same time everywhere.
      if (current !== generation) return { status: "stale" };
      if (controller.signal.aborted) {
        const error = new Error("Request aborted");
        error.name = "AbortError";
        throw error;
      }
      if (!Array.isArray(data) || !data.every((item) => (
        item !== null && typeof item === "object" && typeof item.title === "string"
      ))) {
        throw new Error("数据结构错误：需要包含 title 字符串的对象数组");
      }
      const state = { status: "success", message: `已加载 ${data.length} 项`, data };
      render(state);
      return state;
    } catch (error) {
      if (current !== generation) return { status: "stale" };
      const aborted = controller.signal.aborted;
      const status = timedOut ? "timeout" : aborted ? "cancelled" : "error";
      const message = timedOut ? "请求超时，请重试" : aborted ? "已取消" : error.message;
      const state = { status, message };
      render(state);
      return state;
    } finally {
      clearTimeout(timer);
      if (current === generation) activeController = null;
    }
  }

  function cancel() {
    if (!activeController) return;
    ++generation;
    activeController.abort();
    activeController = null;
    render({ status: "cancelled", message: "已取消" });
  }

  return { load, cancel };
}

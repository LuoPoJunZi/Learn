"use strict";

const log = (id, ...msgs) => {
  const el = document.getElementById(id);
  if (!el) return;
  el.textContent = msgs
    .map((m) => (typeof m === "object" ? JSON.stringify(m, null, 2) : m))
    .join("\n");
};

const logErr = (id, msg) => {
  const el = document.getElementById(id);
  if (!el) return;
  el.textContent = msg;
};

const parseList = (v) =>
  v
    .split(",")
    .map((s) => s.trim())
    .filter(Boolean);

const hasIterHelpers =
  typeof Iterator !== "undefined" &&
  ["map", "filter", "take", "toArray"].every(
    (method) => typeof Iterator.prototype[method] === "function"
  );

let deferred = null;

const products = [
  { name: "苹果", category: "水果", price: 5, stock: 120 },
  { name: "牛奶", category: "饮品", price: 12, stock: 45 },
  { name: "香蕉", category: "水果", price: 3, stock: 0 },
  { name: "咖啡", category: "饮品", price: 28, stock: 10 },
  { name: "面包", category: "主食", price: 8, stock: 0 },
  { name: "米饭", category: "主食", price: 6, stock: 200 },
  { name: "橙汁", category: "饮品", price: 9, stock: 60 },
  { name: "葡萄", category: "水果", price: 15, stock: 30 },
];

function runSet(method) {
  if (typeof Set.prototype[method] !== "function") {
    logErr("out-set", "浏览器不支持此特性");
    return;
  }
  const inputA = document.getElementById("setA");
  const inputB = document.getElementById("setB");
  if (!inputA || !inputB) return;
  const a = new Set(parseList(inputA.value));
  const b = new Set(parseList(inputB.value));
  const result = a[method](b);
  log(
    "out-set",
    `A = {${[...a].join(", ")}}`,
    `B = {${[...b].join(", ")}}`,
    `A.${method}(B) = {${[...result].join(", ")}}`
  );
}

function createDeferred() {
  if (typeof Promise.withResolvers !== "function") {
    logErr("out-pwr", "浏览器不支持此特性");
    return;
  }
  deferred = Promise.withResolvers();
  deferred.promise.then(
    (v) =>
      log(
        "out-pwr",
        "✅ Promise 已 resolve！",
        `值: "${v}"`,
        `时间: ${new Date().toLocaleTimeString()}`
      ),
    (e) =>
      log(
        "out-pwr",
        "❌ Promise 已 reject！",
        `原因: "${e}"`,
        `时间: ${new Date().toLocaleTimeString()}`
      )
  );
  log(
    "out-pwr",
    "⏳ Promise 已创建，正在等待结果...",
    "点击 Resolve 或 Reject 按钮来决定它的命运。"
  );
}

function settleDeferred(action) {
  if (!deferred) {
    logErr("out-pwr", '请先点击"创建 Promise"');
    return;
  }
  if (action === "resolve") {
    deferred.resolve("来自按钮的成功");
  } else {
    deferred.reject("来自按钮的失败");
  }
  deferred = null;
}

function* fibonacci() {
  let a = 0n;
  let b = 1n;
  while (true) {
    yield a;
    [a, b] = [b, a + b];
  }
}

function readFibCount() {
  const input = document.getElementById("fibN");
  if (!input) return null;
  const n = Number(input.value);
  if (!Number.isInteger(n) || n < 1 || n > 50) {
    logErr("out-iter", "请输入 1 到 50 之间的整数");
    return null;
  }
  return n;
}

function runFib(mode) {
  const n = readFibCount();
  if (n === null) return;
  if (!hasIterHelpers) {
    logErr("out-iter", "浏览器不支持 Iterator Helpers");
    return;
  }
  if (mode === "plain") {
    const result = fibonacci().take(n).toArray();
    log("out-iter", `前 ${n} 项斐波那契数（惰性求值 .take(${n}).toArray()）:`, result.join(", "));
  } else if (mode === "even") {
    const result = fibonacci()
      .filter((x) => x % 2n === 0n)
      .take(n)
      .toArray();
    log("out-iter", `前 ${n} 个偶数斐波那契数 .filter(x => x%2n===0n).take(${n}):`, result.join(", "));
  } else {
    const result = fibonacci()
      .take(n)
      .map((x) => x * 2n)
      .toArray();
    log("out-iter", `前 ${n} 项 ×2  .take(${n}).map(x => x*2):`, result.join(", "));
  }
}

function runGroup(mode) {
  if (typeof Object.groupBy !== "function") {
    logErr("out-group", "浏览器不支持此特性");
    return;
  }
  let result;
  let label;
  if (mode === "category") {
    result = Object.groupBy(products, (p) => p.category);
    label = "Object.groupBy(products, p => p.category)";
  } else if (mode === "price") {
    result = Object.groupBy(products, (p) => (p.price < 10 ? "平价 (<10¥)" : "高价 (≥10¥)"));
    label = 'Object.groupBy(products, p => p.price < 10 ? "平价" : "高价")';
  } else {
    result = Object.groupBy(products, (p) => (p.stock > 0 ? "有货" : "缺货"));
    label = 'Object.groupBy(products, p => p.stock > 0 ? "有货" : "缺货")';
  }
  const lines = [label, ""];
  for (const [key, items] of Object.entries(result)) {
    lines.push(`【${key}】 → ${items.map((i) => i.name).join("、")}`);
  }
  log("out-group", ...lines);
}

document.addEventListener("DOMContentLoaded", () => {
  if (typeof Set.prototype.union !== "function") {
    const el = document.getElementById("compat-set");
    if (el) el.hidden = false;
  }
  if (typeof Promise.withResolvers !== "function") {
    const el = document.getElementById("compat-pwr");
    if (el) el.hidden = false;
  }
  if (!hasIterHelpers) {
    const el = document.getElementById("compat-iter");
    if (el) el.hidden = false;
  }
  if (typeof Object.groupBy !== "function") {
    const el = document.getElementById("compat-group");
    if (el) el.hidden = false;
  }

  document.querySelectorAll("[data-set-method]").forEach((btn) => {
    btn.addEventListener("click", () => runSet(btn.dataset.setMethod));
  });
  document.getElementById("btn-create-deferred")?.addEventListener("click", createDeferred);
  document.querySelectorAll("[data-settle]").forEach((btn) => {
    btn.addEventListener("click", () => settleDeferred(btn.dataset.settle));
  });
  document.querySelectorAll("[data-fib-mode]").forEach((btn) => {
    btn.addEventListener("click", () => runFib(btn.dataset.fibMode));
  });
  document.querySelectorAll("[data-group-mode]").forEach((btn) => {
    btn.addEventListener("click", () => runGroup(btn.dataset.groupMode));
  });
});

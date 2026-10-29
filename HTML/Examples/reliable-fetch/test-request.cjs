"use strict";

// No HTTP server or external network. Node 18+ supplies AbortController and Response.
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");
const { test } = require("node:test");
const context = vm.createContext({ AbortController, setTimeout, clearTimeout });
vm.runInContext(fs.readFileSync(path.join(__dirname, "request.js"), "utf8"), context);
const createLoader = context.createLatestLoader;
const json = (data) => new Response(JSON.stringify(data));

function waitForAbort(url, { signal }) {
  return new Promise((resolve, reject) => {
    signal.addEventListener("abort", () => {
      const error = new Error("aborted");
      error.name = "AbortError";
      reject(error);
    }, { once: true });
  });
}

test("success and empty lists", async () => {
  for (const data of [[{ title: "Python" }], []]) {
    const states = [];
    const loader = createLoader((state) => states.push(state), async () => json(data));
    const result = await loader.load("local");
    assert.equal(result.status, "success");
    assert.equal(JSON.stringify(result.data), JSON.stringify(data));
    assert.equal(states[0].status, "loading");
  }
});

test("HTTP errors must not count as success", async () => {
  const loader = createLoader(() => {}, async () => new Response("missing", { status: 404 }));
  const result = await loader.load("local");
  assert.equal(result.status, "error");
  assert.equal(result.message, "HTTP 404");
});

test("JSON parsing, schema, and network failures are handled", async () => {
  const transports = [
    async () => new Response("not JSON"),
    async () => json({ title: "not a list" }),
    async () => json([null]),
    async () => { throw new TypeError("network unavailable"); },
  ];
  for (const transport of transports) {
    const loader = createLoader(() => {}, transport);
    assert.equal((await loader.load("local")).status, "error");
  }
});

test("timeout aborts the transport", async () => {
  const loader = createLoader(() => {}, waitForAbort);
  assert.equal((await loader.load("local", 5)).status, "timeout");
});

test("explicit cancellation updates state once and ignores the old response", async () => {
  const states = [];
  const loader = createLoader((state) => states.push(state.status), waitForAbort);
  const pending = loader.load("local");
  loader.cancel();
  assert.equal((await pending).status, "stale");
  assert.deepEqual(states, ["loading", "cancelled"]);
  loader.cancel();
  assert.equal(states.length, 2);
});

test("late old response cannot overwrite new results even if transport ignores abort", async () => {
  let finishOld;
  const states = [];
  const loader = createLoader((state) => states.push(state), (url) => {
    if (url === "old") return new Promise((resolve) => { finishOld = resolve; });
    return Promise.resolve(json([{ title: "new" }]));
  });
  const oldRequest = loader.load("old");
  assert.equal((await loader.load("new")).status, "success");
  finishOld(json([{ title: "old" }]));
  assert.equal((await oldRequest).status, "stale");
  assert.equal(states.at(-1).data[0].title, "new");
  assert.equal(states.filter((state) => state.status === "success").length, 1);
});

test("late body after timeout still cannot render success", async () => {
  let finishBody;
  let sawAbort;
  const aborted = new Promise((resolve) => { sawAbort = resolve; });
  const loader = createLoader(() => {}, async (url, { signal }) => {
    signal.addEventListener("abort", sawAbort, { once: true });
    return { ok: true, json: () => new Promise((resolve) => { finishBody = resolve; }) };
  });
  const pending = loader.load("local", 5);
  await aborted;
  finishBody([{ title: "late" }]);
  assert.equal((await pending).status, "timeout");
});

test("invalid timeout is rejected before starting a request", async () => {
  const loader = createLoader(() => {}, () => { throw new Error("must not fetch"); });
  for (const timeout of [0, -1, Infinity, NaN]) {
    await assert.rejects(loader.load("local", timeout), { name: "RangeError" });
  }
});

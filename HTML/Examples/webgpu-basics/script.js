"use strict";

const statusEl = document.getElementById("support-status");
const runBtn = document.getElementById("run-btn");
const consoleEl = document.getElementById("console-output");

function log(msg) {
  if (!consoleEl) return;
  const line = document.createElement("div");
  line.textContent = `> ${msg}`;
  consoleEl.append(line);
}

async function checkSupport() {
  if (!statusEl || !runBtn) return null;
  if (!navigator.gpu) {
    statusEl.className = "status-banner status-unsupported";
    statusEl.textContent =
      "当前环境没有 WebGPU；请检查浏览器、硬件加速和安全上下文（HTTPS 或 localhost）。";
    runBtn.disabled = true;
    return null;
  }
  try {
    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) {
      throw new Error("未能找到可用的物理 GPU 适配器");
    }
    statusEl.className = "status-banner status-supported";
    statusEl.textContent = `已找到 GPU 适配器：${adapter.info?.device || "通用图形适配器"}`;
    runBtn.disabled = false;
    return adapter;
  } catch (err) {
    statusEl.className = "status-banner status-unsupported";
    statusEl.textContent = `WebGPU 初始化失败: ${err.message}`;
    runBtn.disabled = true;
    return null;
  }
}

async function runCompute(
  firstArray = new Float32Array([1.5, 3.2, 5.0, 8.4]),
  secondArray = new Float32Array([2.5, 4.8, 1.0, 1.6])
) {
  if (
    !(firstArray instanceof Float32Array) ||
    !(secondArray instanceof Float32Array) ||
    firstArray.length === 0 ||
    firstArray.length !== secondArray.length ||
    !firstArray.every(Number.isFinite) ||
    !secondArray.every(Number.isFinite)
  ) {
    throw new Error("输入必须是长度相同、非空且数值有限的 Float32Array");
  }
  if (!consoleEl || !runBtn) return [];

  consoleEl.replaceChildren();
  runBtn.disabled = true;
  log("正在申请 GPU Device 句柄...");
  let device;
  const buffers = [];
  try {
    const adapter = await navigator.gpu?.requestAdapter();
    if (!adapter) throw new Error("当前没有可用的 GPU 适配器");
    device = await adapter.requestDevice();
    device.pushErrorScope("validation");
    log(`输入向量 A: [${firstArray.join(", ")}]`);
    log(`输入向量 B: [${secondArray.join(", ")}]`);

    const bufferSize = firstArray.byteLength;
    const gpuBufferA = device.createBuffer({
      size: bufferSize,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    const gpuBufferB = device.createBuffer({
      size: bufferSize,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    const gpuBufferOut = device.createBuffer({
      size: bufferSize,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    });
    const readBuffer = device.createBuffer({
      size: bufferSize,
      usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
    });
    buffers.push(gpuBufferA, gpuBufferB, gpuBufferOut, readBuffer);

    device.queue.writeBuffer(gpuBufferA, 0, firstArray);
    device.queue.writeBuffer(gpuBufferB, 0, secondArray);
    log("数据已写入 GPU 显存 (Storage Buffer)...");

    const shaderModule = device.createShaderModule({
      code: `
        @group(0) @binding(0) var<storage, read> a: array<f32>;
        @group(0) @binding(1) var<storage, read> b: array<f32>;
        @group(0) @binding(2) var<storage, read_write> out: array<f32>;

        @compute @workgroup_size(64)
        fn main(@builtin(global_invocation_id) id: vec3<u32>) {
          let i = id.x;
          if (i < arrayLength(&a)) {
            out[i] = a[i] + b[i];
          }
        }
      `,
    });

    const computePipeline = device.createComputePipeline({
      layout: "auto",
      compute: { module: shaderModule, entryPoint: "main" },
    });

    const bindGroup = device.createBindGroup({
      layout: computePipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: gpuBufferA } },
        { binding: 1, resource: { buffer: gpuBufferB } },
        { binding: 2, resource: { buffer: gpuBufferOut } },
      ],
    });

    const commandEncoder = device.createCommandEncoder();
    const passEncoder = commandEncoder.beginComputePass();
    passEncoder.setPipeline(computePipeline);
    passEncoder.setBindGroup(0, bindGroup);
    passEncoder.dispatchWorkgroups(Math.ceil(firstArray.length / 64));
    passEncoder.end();

    commandEncoder.copyBufferToBuffer(gpuBufferOut, 0, readBuffer, 0, bufferSize);
    device.queue.submit([commandEncoder.finish()]);
    const validationError = await device.popErrorScope();
    if (validationError) throw new Error(validationError.message);
    log("GPU 指令执行完毕，正在映射回读取内存...");

    await readBuffer.mapAsync(GPUMapMode.READ);
    const copyArray = new Float32Array(readBuffer.getMappedRange().slice(0));
    log(`GPU 计算结果: [${Array.from(copyArray).map((v) => v.toFixed(2)).join(", ")}]`);
    readBuffer.unmap();
    return Array.from(copyArray);
  } finally {
    for (const buffer of buffers) buffer.destroy();
    device?.destroy();
    runBtn.disabled = false;
  }
}

checkSupport();
runBtn?.addEventListener("click", () => {
  runCompute().catch((err) => log(`运行错误: ${err.message}`));
});

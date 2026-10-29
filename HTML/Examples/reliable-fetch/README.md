# 可靠 Fetch：错误、超时、取消与旧响应覆盖

## 适合人群与目标

适合已经会写按钮事件、了解 Promise 或 `async/await` 的初学者。学完应能区分 HTTP 错误和网络错误、取消请求，并避免“先发出的慢请求覆盖后发出的快请求”。

核验日期：2026-10-29。Fetch 与 AbortController 是成熟 API，不是今年新语法。本页依据当前 MDN 指南补齐真实开发中常遗漏的可靠性处理；不依赖框架、npm 包或 CDN。

## 文件入口

- [index.html](index.html)：可操作页面。
- [request.js](request.js)：请求与状态管理。
- [page.js](page.js)：DOM 更新，使用 `textContent` 渲染数据。
- [courses.json](courses.json)：本地数据。
- [invalid.txt](invalid.txt)：故意不是合法 JSON 的测试输入。
- [test-request.cjs](test-request.cjs)：Node 标准库离线测试。

## 启动方式与安全边界

与大多数静态案例不同，本案例**不能只双击 HTML 文件**：`file://` 页面读取本地文件会受到浏览器安全限制。

在仓库根目录执行以下命令，仅将本案例目录开放给本机，不暴露整个仓库。此命令会启动本地服务，应由你手动执行，结束后按 `Ctrl+C`：

```powershell
python -m http.server 8000 --bind 127.0.0.1 --directory "HTML/Examples/reliable-fetch"
```

浏览器打开 `http://127.0.0.1:8000/`。正常数据、404 和非法 JSON 均只访问本机服务；不会请求第三方 API。不要把开发用 HTTP 服务部署到公网，也不要将 Token 放进前端脚本。

## 第一步：Fetch 成功不等于 HTTP 成功

Fetch 遇到 404/500 通常仍会返回 `Response`，所以必须先判断 `response.ok`，它代表状态码在 200–299：

```javascript
const response = await fetch(url, { signal: controller.signal });
if (!response.ok) {
  throw new Error(`HTTP ${response.status}`);
}
const data = await response.json();
```

接下来还要考虑 JSON 解析失败、返回结构不符等情况。本例只接受包含 `title` 字符串的对象数组。空数组是成功但没有内容，不应显示网络失败。

注意：通用接口可能返回 204 No Content，不能总是调用 `json()`。本案例的成功接口约定返回 JSON 数组，因此无响应体将作为解析错误处理。

## 第二步：超时与主动取消

每次请求都创建一个新 `AbortController`。设置定时器，到期执行 `abort()`；用户点击取消也会触发取消。无论成功、失败或取消，都在 `finally` 清理定时器。

这里使用 `setTimeout` 配合 controller，避免把 `AbortSignal.timeout`、`AbortSignal.any` 等更晚引入的便捷接口当作无条件可用。若项目采用新接口，应按目标浏览器做特性检测。

请求很快时来不及点取消属于正常现象：在开发者工具的 Network 面板关闭缓存并选择慢速网络，再加载后立即取消。也可选择足够慢的自定义限速触发 3 秒超时；限速效果依浏览器实现而不同，离线测试可稳定验证逻辑。

浏览器的取消不代表服务器已经停止处理业务。提交订单等写操作仍需后端实现幂等，不能靠取消按钮避免重复扣款。

## 第三步：取消之外，还要防止竞态

用户先点 A 再点 B，B 先返回、A 后返回。如果所有回调都直接改页面，最终可能显示旧的 A。

本例每次发起请求都会增加 `generation`，回调只有在序号仍是最新时才允许渲染。旧请求会被取消，但**序号校验是另一层独立保护**，可以防御取消时响应已经就绪的情况。显式取消同样推进序号，避免晚到结果把“已取消”覆盖。

页面状态包括 `loading`、`success`、`error`、`timeout` 和 `cancelled`；内部 `stale` 只表示应忽略的旧结果，不会渲染到界面。返回成功数据前还会复核取消标记，避免响应体解析结束时请求已经超时。

## 常见误区

- **用 `mode: "no-cors"` 修复跨域**：得到的 opaque 响应不能正常读取 JSON。应配置服务端 CORS，或使用同源后端，不是关闭浏览器安全设置。
- **只用 `Promise.race` 超时**：可以停止等待，但不会自动终止底层 Fetch，还需传入取消信号。
- **重复使用已取消的 signal**：signal 一旦取消不能恢复，每次请求要新建 controller。
- **用 `innerHTML` 放接口文案**：有注入风险，本例使用 `textContent`，不将数据解释成 HTML。
- **自动无限重试**：会给服务端施加负载，本例不自动重试。写操作重试还需要后端幂等支持。

## 验证步骤

1. 选择正常 JSON，看到三条课程和“已加载 3 项”。
2. 选择不存在的文件，看到 `HTTP 404`，不保留旧列表冒充本次成功。
3. 选择无效 JSON，看到解析错误而不是空白页面。
4. 限速后取消，看到“已取消”；再次加载能成功，不复用旧信号。
5. 连续点击重新加载，旧请求不能覆盖最新结果。
6. 全程用键盘操作按钮与下拉框，状态区有 `role="status"`，列表反映 `aria-busy`。

离线测试需要 Node.js 18+，不启动服务器、不访问网络，也不安装依赖。从仓库根目录运行：

```powershell
node --test HTML/Examples/reliable-fetch/test-request.cjs
```

测试覆盖成功/空列表、HTTP 错误、JSON/结构/网络错误、超时、主动取消、旧响应竞态、超时后迟到的响应体和无效超时参数。单元测试验证请求逻辑；页面视觉、键盘和辅助技术效果仍需浏览器人工验证。

## 官方资料与下一步

- [MDN：Using the Fetch API](https://developer.mozilla.org/en-US/docs/Web/API/Fetch_API/Using_Fetch)：请求、响应、取消与跨域。
- [MDN：Response.ok](https://developer.mozilla.org/en-US/docs/Web/API/Response/ok)：成功状态码范围。
- [MDN：AbortSignal](https://developer.mozilla.org/en-US/docs/Web/API/AbortSignal)：取消信号与兼容性。
- 下一步对照 [Python 结构化并发](../../../Python/Modern%20Python/README.md)，或回到 [HTML 基础](../../Basics/README.md) 补齐语法。

本文及案例为原创教学整理，未复制第三方应用源码；不是带鉴权、重试、缓存与完整可访问性审计的生产请求库。

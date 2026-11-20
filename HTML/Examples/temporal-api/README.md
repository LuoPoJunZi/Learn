# JavaScript Temporal API 日期时间实验台

## 适合人群与学习目标

适合已经学过 JavaScript 基础对象和日期处理，被旧 `Date` 对象“月份从 0 开始、原地修改、静默进位、时区混乱”坑过的初学者。

资料核验：**2026-11-20**，参考 TC39 Temporal 规范与 MDN。Temporal 已进入 ECMAScript 标准并逐步落地现代浏览器与运行时，但旧版本浏览器或受限 WebView 仍可能不支持，因此本案例包含运行时特性检测与 `Intl.DateTimeFormat` 通用回退。

完成后你应该能够：

- 解释为什么现代 JavaScript 需要 `Temporal` 来替代 1995 年遗留的 `Date`。
- 区分 `PlainDate`（纯日期）、`Instant`（时间戳瞬间）与 `ZonedDateTime`（带时区的日期时间）。
- 使用 `{ overflow: "reject" }` 拦截非法日期输入，而不是依赖旧 `Date` 的静默进位。
- 在尚未全量支持 Temporal 的环境下采用合理的特性检测与回退方案。

## 文件说明

| 文件 | 职责 |
| :--- | :--- |
| [index.html](index.html) | 语义化交互页面，包含 4 个对比区块 |
| [style.css](style.css) | 亮色高对比度布局与键盘焦点样式 |
| [script.js](script.js) | `globalThis.Temporal` 检测、不可变计算、非法日期拦截与时区转换 |

## 如何打开

双击 [index.html](index.html) 即可在浏览器中直接运行，无需 npm、打包工具或后端服务。

## 核心类型速查

| 类型 | 含义 | 适用场景 |
| :--- | :--- | :--- |
| `Temporal.PlainDate` | 不含时间与时区的纯日历日期 | 生日、账单日、日程表日期 |
| `Temporal.PlainTime` | 不含日期的纯时刻（如 `09:30:00`） | 每日闹钟、营业时间 |
| `Temporal.Instant` | 精确的全球时间点（纳秒级 UTC） | 日志时间戳、事件排序 |
| `Temporal.ZonedDateTime` | 时间点 + IANA 时区 + 日历 | 跨国会议、考虑夏令时的本地时间 |
| `Temporal.Duration` | 时间跨度（如“1 个月零 3 天”） | 订阅周期、日期加减运算 |

## 常见误区与边界

- **不要假装所有环境都已原生支持**：未检测到 `globalThis.Temporal` 时，本页面会明确标注跳过原生计算，并展示 `Intl.DateTimeFormat` 回退，不会伪造通过状态。
- **业务展示 vs 底层存储**：后台存历史事件通常用 UTC 瞬间（`Instant` / ISO 8601）；面向未来且依赖当地政策的日程（如“明年纽约早上 9 点”）应保存当地日期时间与 IANA 时区名（`America/New_York`），不要只存 `+08:00` 固定偏移。

## 权威参考与下一步

- [MDN：Temporal](https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Global_Objects/Temporal)
- [TC39 Temporal 规范文档](https://tc39.es/proposal-temporal/docs/)
- [现代 JavaScript ES2024-2026 新特性](../es2024-2026-features/README.md)
- [可靠 Fetch 与取消案例](../reliable-fetch/README.md)

# 更新日志

这个文件记录仓库中比较重要的文档结构调整，方便以后回看每一轮优化做了什么。

## 2026-09-01

### 新增

- 扩展 [Python 现代 AI 原理案例](<Python/Modern AI/README.md>)，新增贪心推测解码、分页 KV Cache 和 INT4 分组权重量化的标准库实现。
- 新增 [systemd 服务沙箱与凭据管理](Linux/docs/part2-system-admin/2.4-systemd-hardening.md)，覆盖安全审计、动态用户、文件系统限制、capabilities、credentials、验证和回滚。

### 优化

- 更新 Python、Linux、仓库首页和知识地图导航，明确区分教学模拟与生产推理实现，并把服务加固接入 VPS 学习路线。

## 2026-08-31

### 新增

- 新增 [GitHub 账号安全教程](Github/docs/part1-basics/1.4-account-security.md)，补充 2FA、Passkey、恢复码、SSH 密钥和访问令牌的区别与恢复检查。
- 新增 [Python 现代项目结构指南](Python/MODERN_PROJECT_GUIDE.md)，补充 `pyproject.toml`、`src` 布局、可编辑安装、依赖组和测试入口。
- 新增 [HTML 网页无障碍基础](HTML/Basics/1.9-accessibility-basics.md)，补充语义 HTML、图片替代文本、表单标签、键盘操作、焦点与 ARIA。
- 新增 [Linux UFW 与自动安全更新教程](Linux/docs/part3-network-ssh/3.3-firewall-updates.md)，说明远程启用防火墙的安全顺序、日志和 `unattended-upgrades` 检查。
- 新增 [Matlab 深度学习现代工作流指南](<Matlab/Neural Network/MODERN_WORKFLOW_GUIDE.md>)，解释 `train`、`trainNetwork`、`trainnet` 和 `dlnetwork` 的定位与迁移方法。
- 扩展 [Python 现代 AI 原理案例](<Python/Modern AI/README.md>)，新增 GQA、KV Cache 和稀疏 MoE Top-k 路由的标准库实现。
- 新增 [GitHub 构建证明教程](Github/docs/part5-advanced-tools/5.3-secure-releases.md) 与 [工作流模板](Github/examples/attested-release.yml)，覆盖 OIDC、Artifact Attestations、验证和不可变发布。
- 新增 [原生 UI 2026 实验台](HTML/Examples/native-ui-2026/README.md)，演示 Invoker Commands、CSS Anchor Positioning 和 Scroll-driven Animations。
- 扩展 [Matlab 现代计算案例](<Matlab/Modern Computing/README.md>)，增加 Fourier Neural Operator 频域算子与模态截断示例。

### 优化

- 更新根目录和各专题导航，将新增教程接入推荐阅读顺序，并保留已有案例与历史接口的可追溯入口。

## 2026-08-03

### 新增

- 新增 [Python 现代 AI 原理案例](<Python/Modern AI/README.md>)，使用标准库演示缩放点积注意力、BPE、迷你 RAG 检索和 LoRA 低秩更新。
- 新增 [现代 Web API 实验台](HTML/Examples/modern-web-apis/README.md)，演示 Web Components、Container Queries、Popover、View Transitions 和渐进增强。
- 新增 [Matlab 现代计算案例](<Matlab/Modern Computing/README.md>)，补充因果自注意力和物理约束损失脚本。
- 新增 [Mathematica 可运行案例](Mathematica/Examples/README.md)，补充图社区发现、符号化物理损失和轻量神经分类 `.wl` 源码。
- 新增 [GitHub Actions 自动检查教程](Github/docs/part5-advanced-tools/5.2-github-actions-ci.md) 与 Python 质量检查工作流模板。

### 优化

- 更新根目录、各主题 README、学习导读和知识地图，接通新增案例入口。
- 将 Wolfram Language 示例约定补充到代码规范和 `.editorconfig`。

## 2026-07-31

### 新增

- 新增仓库级 `.editorconfig`、`.gitattributes` 和 `.gitignore`，统一 UTF-8、换行、缩进与常见生成文件忽略规则。
- 新增 [示例代码规范](docs/CODE_STYLE.md)、[代码检查脚本](scripts/check-code.ps1) 和无第三方依赖的 Python 静态检查器。
- 新增 [Python 自动化脚本统一运行与安全说明](Python/Auto_scripts/STANDARDIZED_USAGE.md)，说明预演模式、环境变量和标准化后的安全默认值。

### 优化

- 标准化 Python 示例：补充网络超时和 UTF-8 编码，移除通配符导入，更新 PyPDF2/Pillow API，并整理过长表达式。
- 加固批量脚本：文本替换、重命名、空目录清理和文件分类默认只预演，Excel 去重默认保留原文件，账号凭证改从环境变量读取。
- 修正 HTML 示例的语言、viewport、图片替代文本、按钮类型、交互状态和 CSS 注释写法。
- 修复 Matlab 函数文件名与主函数名不一致的问题，并使用 Matlab Code Analyzer 完成全目录静态扫描。

## 2026-06-23

### 新增

- 新增 [HTML 静态网页案例集合](HTML/Examples/README.md) 和 [HTML 优秀网页项目导读](HTML/EXTERNAL_REPOSITORIES.md)，包含作品集、产品落地页、餐厅展示页、轻量数据面板、交互组件和价格页等初学者可直接打开修改的案例。
- 新增 GitHub、Linux、Markdown、Python、Matlab、Mathematica 的学习导读，重点补充学习路线、概念解释、典型场景、常见误区和外部资源。
- 新增 [30 天学习计划](docs/30_DAY_PLAN.md)、[知识地图总览](docs/KNOWLEDGE_MAP.md)、[新手术语表](docs/GLOSSARY.md) 和 [文档维护者指南](scripts/MAINTAINER_GUIDE.md)，增强仓库级学习入口和维护说明。
- 新增 GitHub、Linux、Markdown、Python、Matlab、Mathematica 常见问题文档，以及 [内容阅读指南](docs/READING_GUIDE.md)，方便新手按目标和报错快速定位内容。
- 新增 GitHub 真实场景、Linux VPS 运维、Markdown 写作进阶、Python 小项目路线、Matlab 案例阅读、Mathematica 专题学习和 [后续深入路线](docs/NEXT_STEP_ROUTES.md)，继续增强场景型学习内容。
- 新增 PR Review、VPS 基础安全、README 审查、Python 环境依赖、Matlab 数据绘图、Mathematica Notebook 和 [新手检查清单合集](docs/CHECKLISTS.md)，补齐新手执行前后的检查流程。
- 落实原文档中的悬空扩展建议：补齐 Matlab Basics 专题教程、Markdown 示例模板、Python 神经网络 CNN/框架学习入口，以及 Python 多目标优化 SPEA2/Pareto 可视化/完整算法扩展说明。
- 补强 Matlab Basics 中原先主要保留在旧笔记里的字符串/结构体/cell 和高阶绘图内容，新增独立新手专题教程。
- 落实练习和自动化脚本文档中的扩展建议：为 Python 练习补充异常处理版本，并为下载图片、删除空文件夹、批量重命名脚本补充失败记录或操作日志。

### 优化

- 更新根目录 README、各主题 README、docs 索引和学习路线，让新增学习内容能从首页和目录入口直接找到。

## 2026-06-21

### 新增

- 新增 [Python 优秀开源仓库导读](Python/EXTERNAL_REPOSITORIES.md)，介绍 walter201230/Python、TheAlgorithms/Python、geekcomputers/Python 的适合阶段和学习方式。
- 新增 Python 神经网络和多目标优化入门示例，包含梯度下降、感知机、逻辑回归、Softmax、RBF、时间序列、迷你卷积、自编码器、过拟合与正则化、模型保存加载、RNN 隐藏状态、MLP、Pareto 筛选、迷你 NSGA-II 和简化 MOEA/D，便于和 Matlab 算法目录对照学习。
- 新增 [学习记录模板](docs/STUDY_NOTES_TEMPLATE.md)，用于记录教程学习、脚本运行、报错排查和每周复盘。
- 新增 [新手常见问题排查手册](docs/TROUBLESHOOTING.md)，覆盖 Git、Python、Linux、Markdown、Matlab 常见报错和排查顺序。
- 新增 [新手练习题与自查答案](docs/PRACTICE_EXERCISES.md)，覆盖 Python、Linux、Git/GitHub 的基础练习。
- 新增 [Auto_scripts README 写作规范](Python/Auto_scripts/README_STYLE.md)，统一自动化脚本文档结构。
- 新增 [Matlab 示例运行索引](Matlab/EXAMPLE_RUN_INDEX.md)，整理多目标优化和神经网络案例的阅读入口。
- 新增 [Matlab 示例运行指南](Matlab/RUNNING_EXAMPLES.md)，说明运行案例前的路径、数据文件、工具箱和常见报错。
- 新增 [Python 自动化脚本依赖总览](Python/Auto_scripts/DEPENDENCIES.md)。
- 新增 Python 示例目录入口说明，包括图片抓取和图形小程序。
- 新增 Markdown 示例模板入口。

### 优化

- 更新仓库首页 README，补充学习路线、Mathematica 入口和维护文档入口。
- 优化 HTML Love 示例说明，补充查看方式、修改顺序和常见问题。
- 优化 Markdown 教程，补充 README 模板入口。
- 优化 Matlab、Python 目录索引，让新手更容易找到运行说明和依赖说明。

## 2026-06-18

### 新增

- 新增 [Learn 新手学习路线](docs/LEARNING_PATH.md)，按阶段说明 Markdown、GitHub、Linux、Python、HTML、Matlab、Mathematica 的学习顺序。
- 新增 [Markdown README 模板](Markdown/examples/readme-template.md)。

### 优化

- 更新主 README 的推荐入口和新手阅读顺序。
- 补充 docs 目录索引。

## 2026-06-17

### 新增

- 新增 GitHub 新手教程。
- 新增 Python 基础教程。
- 新增 Mathematica 基础教程。
- 新增 HTML 基础教程。
- 新增 Linux 新手地图。

### 优化

- 整理 Linux、Markdown、Matlab 等目录的教程结构。
- 增加文档检查脚本和文档整理规范。

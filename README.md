# nl2opt

> 把自然语言描述的线性规划（LP）/ 混合整数规划（MILP）问题，转换成**可独立验证、可闭环修复**的求解模型。

**核心主张**：LLM 负责「理解与建模」，但**不被信任** —— 每次建模都要经求解器 + 验证器复核；
失败时把「IR + 题面 + 验证错误」交回 LLM 修（每次机会最多 2 轮），而不是无脑重试。

---

## 数据集

只用一个数据集：**MIPLIB-NL** —— 223 个工业级实例，从 MIPLIB 2017 逆向构造（配套 ICML 2026 论文，
<https://github.com/zlifr/MIPLIB-NL>，许可 **CC BY-SA 4.0**）。

它的特点是 **problem–data separation**：题面只讲结构，**数值全在 `data/*.csv` 里**
（单个文件最大 187 万行）。规模：变量中位 **2,183** / 最大 283,648；约束中位 **1,388** / 最大 1,050,112。

```
data/MIPLIB-NL/dataset/<instance>/
├── instance.json   # abstract_problem（题面）+ parameters + files（CSV schema）+ optimal_value
├── model.md        # 符号化数学公式（供人读，程序不解析）
├── data/*.csv      # 实际数据（系数、时长、键）
└── solve.py        # 参考实现（Gurobi，评测时不使用）
```

### 公平性边界（不可越线）

LLM 只能看到 `instance.json` 里**除 `optimal_value` 之外**的字段，即：
`abstract_problem`（题面）+ `files`（文件名、路径、列描述）+ `parameters`（规模参数）。

| 框架刻意不读 | 原因 |
|---|---|
| `model.md` | 含正确的数学公式，就是答案 |
| `solve.py` | 参考实现 |
| `optimal_value` | 不进 prompt，只在打分时用 |
| CSV 内容（行数 / 列值 / 取值范围） | 「打开文件看一眼」应该由模型自己做 |

理由：数据在文件里是题目的一部分。框架提前替它总结，等于把这道工序的成本送掉，评测就失去区分度。

### 规模分层（按数据行数）

| 层级 | 实例数 | 代表 |
|---|---|---|
| 无数据文件可用 | **11** | `b-ball`、`ej`、`wachplan`、`chromaticindex*` |
| 声明了文件但数据缺失 | **1** | `ex1010-pi`（缺 `data/S1.csv`）|
| ≤ 50 行 | 23 | `flugpl`、`gr4x6` |
| 51–500 行 | 56 | `30n20b8`、`50v-10` |
| 501–5000 行 | 63 | `acc-tight2`、`beasleyC1` |
| > 5000 行 | **69** | `bab2`（187 万行）、`vpphard2` |

## 它解决什么问题

| 缺陷 | 本项目的机制 |
|---|---|
| LLM 输出的模型**无法独立校验** | IR 中间表示 + 独立验证器（不信任生成方，只拿 IR 重算） |
| 一次生成、无反馈 | 失败即把 IR + 题面 + 验证错误交回 LLM 重写（每次机会最多 2 轮） |
| 工业题规模大到无法逐项列举 | 符号化 IR（变量族 + 索引维 + 数据引用）→ `expand()` 读 CSV 展开 |

## 设计哲学：三权分立

| 角色 | 职责 | 实现 |
|---|---|---|
| **生成方** | 理解题面、产出 `SymbolicIR`、按失败原因重写 | LLM（DeepSeek，OpenAI 兼容协议） |
| **裁判方 ①** | 求解 | `scipy.optimize.milp` / `linprog(method="highs")`<br>进程内调用，无子进程、无容器、无代码生成 |
| **裁判方 ②** | **不依赖生成方**，只拿展开后的 IR 逐条复核 | `verifier.py`（2 项检查） |

关键点：验证器**不看 LLM 的原始输出**，只接收「展开后的字面 IR + 解向量」重新算一遍。

## 闭环流程

```
             ┌──────────── 一次「解答机会」────────────┐
UNDERSTAND ──► EXPAND ──► SOLVE ──► VERIFY ──┐
 (NL→SymbolicIR) (读CSV展平)  (HiGHS)  (2项检查) │
      ▲                                       │ 不通过
      │                                       ▼
      └────────── REWRITE ◄── IR + 题面 + 验证错误原因（最多 2 次）
                 （符号层重写 → 重新展开 → 再求解）
             └────────────────────────────────────┘
                  失败 → 整条重来（最多 2 次机会，成功即停）
```

**两个数量是分开的**（都在 `config.yaml` 的 `pipeline` 下）：

| 配置 | 默认 | 含义 |
|---|---|---|
| `attempts` | **2** | 每题最多几次**解答机会**；每次都是独立的理解 + 修复全过程，任一次成功即停止 |
| `max_rounds` | **3** | 单次机会内最多求解几轮 → **3 轮 = 2 次修复**（最后一轮失败后不再重写，那次调用没人用） |

所以单题最坏是 `2 次机会 × (1 次理解 + 2 次修复) = 6 次 LLM 调用`。

**终止条件**（单次机会内）：验证通过 / 达 `max_rounds` / 连续相同错误达阈值 / 疑似参考噪声已给过一次机会 / LLM 不可用。

**多次机会之间的取舍**：失败的尝试不会被更好的结果覆盖 —— 最终按「成功 > 有解 > 无解」挑一次输出，
同档再比与参考值的距离；token 用量按**所有尝试累加**（成本如实反映）。

**展开失败同样是「失败的一轮」，不是终止条件。** 虚构数据文件、键选错、索引对不齐、笛卡尔积爆炸
都是**建模错误**，和求解/验证失败一起进同一条修复路径（旧实现遇到展开错误直接返回，
`rounds=0` 就结束，一次笔误即判死刑）。

**没有错误分层路由。** 早期的 L1（代码层）/ L2（建模层）/ L3（理解层）/ L4（转人工）已全部移除 ——
现在只有**一条**修复路径：把 IR + 题面 + 验证失败原因交给 LLM，让它自己判断该改什么。
多一层诊断只多一次调用、多一种误判。

## 安装

需要 Python **≥ 3.10**（依赖：`pydantic` / `PyYAML` / `httpx` / `scipy` / `numpy`）。

```bash
pip install -e .          # 或 pip install -e ".[dev]" 带上 pytest
```

不安装也行 —— 把 `src/` 加进 `PYTHONPATH` 即可（下面的示例都用这种方式）：

```bash
export PYTHONPATH=src                     # bash
$env:PYTHONPATH = "$PWD\src"              # Windows PowerShell
```

LLM 需要 API Key：复制 `.env.example` 为 `.env`，填入 `DEEPSEEK_API_KEY`
（或直接设同名环境变量）。不配 Key 时，只有「自检」和「离线注入 IR」这两条路能跑。

最后准备数据（约 462 MB，**不入库**，需自行 clone）：

```bash
git clone --depth 1 https://github.com/zlifr/MIPLIB-NL data/MIPLIB-NL
```

## 快速开始

```powershell
# Windows PowerShell
# 首次使用请先执行：git config core.autocrlf false
#   —— 否则 Windows 默认的 autocrlf=true 会覆盖 .gitattributes 的行尾规则
$env:PYTHONPATH = "$PWD\src"

python scripts\run_tests.py                                        # ① 自检（零依赖，无需网络 / API Key）
python -m nl2opt list --data data\MIPLIB-NL\dataset\b-ball         # ② 看单题被喂给 LLM 的内容
python -m nl2opt list --data data\MIPLIB-NL\dataset                # ③ 整批（223 题）
```

```bash
# 需要 LLM（在 .env 或环境变量中配置 DEEPSEEK_API_KEY）
python -m nl2opt eval --data data/MIPLIB-NL/dataset/b-ball --use-llm
python -m nl2opt solve "某工厂生产两种产品……使总利润最大？" --use-llm   # 自由题面（无需 --data）
```

> `list` / `eval` 的 `--data` 是**必填**的：内置样例集已删除，没有默认数据集可退回。
> `solve` 例外 —— 不提供 `--data` 时把参数当作自由的自然语言问题（此时必须 `--use-llm`）。

## 配置

全部在 `config.yaml`，环境变量可覆盖其中一部分（见 `.env.example`）。

| 键 | 默认 | 说明 |
|---|---|---|
| `llm.model` | `deepseek-flash` | 也支持 `deepseek-v4-pro`（单价见 `llm.pricing`）|
| `llm.reasoning_effort` | **`low`** | **固定 low**：`medium` 在上千条约束的实例上会把 `max_tokens` 吃光、正文 JSON 被截断 |
| `llm.max_tokens` | `65536` | 思考与正文共享；`32768` 对最大实例是临界值 |
| `llm.timeout_seconds` | `600` | 单次 LLM 调用超时 |
| `pipeline.attempts` | **2** | 每题最多几次解答机会（成功即停）|
| `pipeline.max_rounds` | **3** | 单次机会内最多求解几轮 → 2 次修复 |
| `executor.timeout_seconds` | `120` | 单题**求解**墙钟预算（只花 CPU，不花 token）|

## 关键设计

### 1. 符号化 IR（而不是展平 IR）

数值不在题面里、规模中位 2,183 个变量 —— 靠 LLM 逐项列出既超输出预算也无意义。
因此 LLM 只输出**符号模型**：

```json
{
  "variables": [{"name": "x", "dims": ["job"], "var_type": "binary"}],
  "objective": {"sense": "minimize", "expr": {"terms": [
      {"var": "x_{j}", "coef": {"ref": {"file": "cost.csv", "column": "cost", "key_column": "job", "key_value": "{j}"}}}
  ]}},
  "constraints": [{"name": "cap_{t}", "for_each": "t",
                   "lhs": {"terms": [{"var": "x_{j}", "coef": 1.0}]}, "op": "<=",
                   "rhs": {"file": "cap.csv", "column": "capacity", "key_column": "period", "key_value": "{t}"}}]
}
```

支持：变量族 + 索引维（`list` / `csv`）、**索引算术**（`{k-1}`、`{t+1}`，越界即跳过该项）、
复合键数据引用、系数 × 引用乘积、`for_each` 的笛卡尔积与 `ForEachRows` **跨表关联**、
按分类列的**字符串值分派变量**（`bind`）、rhs 为数值 / 引用 / 变量 / 子表达式。

**这是 LLM 的唯一输出形态**。展平 IR 只由 `expand()` 产生 —— 连没有数据文件的自由题面
（`solve "…"`）也走同一条链路（给一个保证为空的 `DataStore`，用 `list` 索引 + 字面系数展开）。
早期那条「LLM 直接产展平 IR」的路径已删除：它只对手写小样例可用，对真实数据集永不成立，
留着等于维护两套契约。

### 2. 数据必须在 `expand()` 阶段解析

不能把数据引用推迟到求解层 —— `check_feasibility` 要用**字面系数**代回约束；
系数若仍是引用，验证器就失去独立核算能力，「防伪造 objective」的防线会失效。

`expand()` 还负责：**占位符校验**（无来源即报错）、**实例数上限**（默认 5 万，超限拒绝，
而不是静默降级）、**引用完整性**、**约束名去重**、**缺失实例自动补齐**（两者都记进 `assumptions`）。

### 3. 展开前先拦「虚构数据文件」

LLM 会凭空造文件名。`check_data_files()` 在展开前一次性列出「模型引用但不存在」的文件
+ 真实可用文件，避免白花一次展开和一次完整 LLM 调用。

### 4. 验证器只做两件事

| 检查 | 作用 |
|---|---|
| `check_optimality` | 解的目标值与参考值是否一致 —— **唯一判对错的检查** |
| `check_feasibility` | 解是否满足 IR 的约束 / 界 / 整数性 —— 断言「解与模型一致」 |

**能力边界（必须说清）**：保证「解 ⊨ 模型」，**不保证「模型 ⊨ 题面」**。
建模错了但自洽，`feasibility` 照样通过；只有 `optimality`（与参考值比对）能间接发现。

早期的目标一致性 / 系数接地 / 目标非空 / 语义覆盖四项检查已删除 ——
它们要么与报告值同源（恒相等）、要么依赖题面给数值（工业数据集题面没有数字）、
要么需要参考 IR（本数据集不提供）。

### 5. 重写发生在符号层

让 LLM 重写展开后的上千条约束既超输出预算、又极易引用不存在的变量
（曾出现 `IR 解析失败: 未声明的变量引用`）。因此 `rewrite_symbolic()` 只改符号模型，
改完**重新展开**；展开前查 `symbolic_degeneracy_reason`（防止把模型改空），
展开后再查 `degeneracy_reason`。

### 6. 答案泄露防护

验证失败消息里**不含参考答案数值**（只在 trajectory 的 `detail` 字段留给人看）。
否则 LLM 能「反推」出恰好算出该值的模型 —— 那是拟合答案，不是建模。

### 7. 参考答案噪声容忍

`optimal_value` 来自 MIPLIB 2017，可信度高，但加载器仍标 `trusted_reference=False`
（未逐题独立核验）。由此：

- 解可行、仅目标值不符 → 标记 `suspicious`，**仍给一次重写机会**；
- 重写后达到参考值 → `suspicious_resolved`，**计为通过**（说明原本是模型错）；
- 仍未达到 → 保持 `suspicious`，不计通过。

> 这样避免「正确解被误判 → 触发重写 → 把对改错」的恶性循环。

### 8. 思考强度固定 `low`

思考型模型的 reasoning 与正文共享 `max_tokens`。工业实例动辄上千条约束，`medium` 会把预算
吃光、正文 JSON 被截断（`finish_reason=length`），因此 `config.yaml` 把
`reasoning_effort` **固定为 `low`**，不随题调整。

代价与收益：复杂题的推理深度受限，但大实例能产出完整 IR。对本数据集来说，
「输出完整」是必要条件，「想得更细」不是。`max_tokens: 65536` 保留作最后余量。

## 目录结构

```
src/nl2opt/
├── ir.py              # 展平 IR Schema（Pydantic 严格校验）
├── symbolic.py        # 符号化 IR：变量族 / 索引维 / 数据引用 / 跨表关联 / 按值分派
├── expand.py          # 符号 → 展平：读 CSV、索引算术、占位符校验、虚构文件预检
├── verifier.py        # 独立验证器（可行性断言 + 目标值打分）
├── solver/engine.py   # scipy/HiGHS（进程内求解）
├── dataset.py         # MIPLIB-NL 加载（刻意不读 CSV、不读 model.md）
├── config.py          # 配置加载（环境变量覆盖）
├── cli.py             # list / solve / eval
├── llm/               # DeepSeek provider + fake provider（自检用）
└── pipeline/          # understand / rewrite / orchestrator / prompts
scripts/               # run_tests（零依赖自检）/ convert_miplibnl / report_cost / run_demo_rewrite
data/MIPLIB-NL/        # 223 个实例（约 462 MB，**不入库**，见「安装」一节的 clone）
artifacts/             # 评测轨迹与报告（运行时生成，不入库）
tests/                 # pytest（67 例）
```

## 已知限制（诚实报告）

| 限制 | 说明 |
|---|---|
| **展开规模上限** | 默认 5 万变量实例，超限直接拒绝（不降级）。变量中位 2,183 的题都够，但最大的 28 万变量实例不可能展开 |
| **单次输出预算** | 需要上千条约束的题仍可能撞 `max_tokens`（`reasoning_effort` 已固定 `low`，无降档空间，只能调大 `max_tokens`）|
| **非确定性** | `temperature=0` 下个别题的建模选择仍会变（连续 vs 整数）|
| **仅线性模型** | IR 不支持非线性（乘积 / 二次 / 概率连乘）|
| **验证器能力** | 只保证「解 ⊨ 模型」，不保证「模型 ⊨ 题面」；建模错但自洽时不会报错 |
| **数据缺陷** | `ex1010-pi` 缺 `data/S1.csv`；`sorrell3/4/7` 无数据声明却易诱发虚构文件名 |
| **参考值口径** | 个别实例的 `optimal_value` 与 `model.md` 单位不一致（如 `b-ball`：`model.md` 按 slot，参考值是 quarter，差 2 倍），需按题面口径换算 |

## 许可

- 项目代码：MIT
- MIPLIB-NL 数据：**CC BY-SA 4.0**（<https://github.com/zlifr/MIPLIB-NL>）

# ARC Prize 2026（ARC-AGI-2）独立工程

**人工试做（10 道训练 + 5 道评测，官方 10 色）：** [本机浏览器打开](file:///Users/Wezhang/workspace/arc-prize-2026/reports/human_preview.html) · [仓库内页面](reports/human_preview.html)

点「本机浏览器打开」即可画格子、检查答案。先看示范，再画测试输出。

GitHub：https://github.com/lhzwb2008/arc-prize-2026  
本地目录：`/Users/Wezhang/workspace/arc-prize-2026`  
Kaggle 比赛：https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-2

这不是交易/MT5 仓库的一部分。量化工程仍在 `ftmo-test`；本题单独开仓，避免两套依赖和目标搅在一起。

---

## 来龙去脉

2026-09 在看「有没有普通人能参加、实现可以很难、对错能自动验、不是外包点头」的公开竞赛。Kaggle 上最贴近的是 **ARC Prize**：主办方是非营利组织 [ARC Prize Foundation](https://arcprize.org/)（François Chollet 与 Mike Knoop），赛场在 Kaggle。

**ARC** = Abstraction and Reasoning Corpus（抽象与推理语料库），比赛名 **ARC-AGI-2**。每道题是彩色格子谜题：给你几对「输入格子 → 输出格子」，再给一张从没见过的测试输入，要求输出格子**每一个像素都完全正确**。人通常几分钟能做对；只靠背题的大模型不行。验证极简单（格子相等），实现可以非常复杂（程序合成、搜索、推理）。

本仓库目标：

1. 把题意、数据格式、交卷方式写清楚。
2. 用公开样例跑通一套 **demo → 本地打分 → 生成 submission.json → Kaggle notebook** 管道。
3. 先交一个弱但真实的 baseline，看排行榜分数，再迭代。

当前求解器（v2，`src/arc_solver/solver.py`）是纯算法、不含任何模型：在示范对上搜索能完全复现全部示范的程序，再用投票选出两次猜测。它不是冲奖金的方案，只是一个真实的起点。

## 求解器 v2 怎么工作

每道题独立求解，只用该题的 `train` 示范：

1. **整图变换搜索**：约 150 个基础算子（旋转/翻转、放大/缩小、平铺/镜像拼接、裁剪、按物体选取/保留/删除、重力、填洞、描边、分隔线网格压缩等），做深度 ≤ 2 的组合，每个组合还可再接一层从示范里学出的颜色映射。
2. **逐格规则归纳**（输入输出同尺寸时）：把每个格子的「颜色 + 局部上下文」（四/八邻域、四方向射线首个非背景色、所在行列特征、连通块大小、位置奇偶…）作为键，学一张键→输出色的表；表必须在全部示范上无冲突，且覆盖测试输入的所有格子。另有一组不看颜色只看「是否背景」的键，可泛化到示范里没出现过的颜色。
3. **逐物体重涂**：按物体属性（大小、大小排名、是否最大/最小、孔洞数、形状、是否贴边…）学属性→颜色表。
4. **补全**：对称补全（自动找镜像轴/对角线/旋转中心）和周期补全，用来填掉「遮挡色」区域；支持输出整图或只输出补好的那块。
5. **分块合成**：把输入按 2/3/4 等分（可带分隔线）拆开，学各块颜色（或是否非背景）→输出色的真值表，覆盖 AND/OR/XOR 一类题。
6. **变换拼贴**、**常量输出**。

所有能复现全部示范的程序都参与**投票**，权重按程序复杂度衰减，票数最高的两个不同输出作为 `attempt_1` / `attempt_2`。没有任何程序拟合时，退回到覆盖率最高的局部规则，再不行原样输出输入。

本地结果（2026-09-07，Apple Silicon 单进程平均 0.8 秒/题）：

| 数据 | 解出 |
|---|---|
| `data/examples/` 13 题 | 12 |
| 官方训练集 1000 题 | 159（15.9%） |
| 官方公开评测集 120 题 | **1**（0.8%） |

评测集是专门剔除了「能被暴力搜索解出的题」的，所以训练集和评测集之间差距巨大。诊断过：120 道评测题里只有 1 道存在任何能拟合全部示范的程序，也就是说瓶颈是 DSL 表达力，不是排序。要往上走得换思路（更强的程序合成 / 测试时训练），不是继续堆算子。

---

## 输入 / 输出

每道题是一个 JSON：

```json
{
  "train": [
    {"input": [[1, 0], [0, 0]], "output": [[1, 1], [1, 1]]}
  ],
  "test": [
    {"input": [[0, 0], [0, 8]]}
  ]
}
```

- 格子：二维整数数组，取值 `0–9`（10 种颜色），边长 1–30。输入和输出的高宽可以不同。
- 比赛时你只能看见 `train` 的输入输出，以及 `test` 的**输入**。
- 交卷文件必须叫 `submission.json`，每个测试输入给两次猜测 `attempt_1` / `attempt_2`，对一次即可。

Kaggle 是 **Code Competition**：必须提交 notebook，在 **断网**、12 小时时限内重跑，写出 `submission.json`。不能现场调 GPT API。拿奖金还需要开源。

截止（UTC）：报名/组队 2026-10-26，交卷 2026-11-02。

---

## 仓库结构

```
data/examples/          公开样例（含官方指南最小题 + ARC-AGI-2 训练集若干题）
data/full/              官方 ARC-AGI-2 完整数据（不进 git，见下文「官方数据」）
data/kaggle/            Kaggle 比赛数据原样下载（不进 git）
data/rules/             BARC/LARC 规则文本（SFT 用，不含评测题答案）
src/arc_solver/solver.py     求解器 v2（仅标准库，整个文件会被嵌进 notebook）
src/arc_solver/baseline.py   v1，只留作对照
scripts/run_demo.py          跑样例、出 HTML 报告
scripts/eval_full.py         在官方训练/评测集上多进程打分
scripts/make_submission.py
scripts/build_kaggle_notebook.py   生成 notebooks/kaggle_baseline.ipynb 及同内容的 .py
scripts/nvarc/                 NVARC 原版 TTT 的本地单卡端口（3090 跑公开 120 评测）
scripts/kaggle_preflight.py    提交前核对 kernel 日志：L4 + Py3.11 + 非空解码
scripts/ensemble_submit.py     模型 attempt_1 + DSL attempt_2
notebooks/kaggle_baseline.ipynb    DSL baseline notebook
notebooks/kernel-metadata.json     DSL kernel 元数据
notebooks/nvarc_2026/              NVARC Qwen3-4B TTT kernel（4×L4，钉 Python 3.11 镜像）
demo/output/                 本地产物（不进 git）
```

打分：

```bash
python3 scripts/eval_full.py --split evaluation --show-solved
python3 scripts/eval_full.py --split training --workers 8
```

---

## 本地 demo

需要 Python 3.10+，不用装科学计算库。

```bash
cd /Users/Wezhang/workspace/arc-prize-2026
python3 scripts/run_demo.py
```

会打印命中题数，并生成：

- `demo/output/report.html` — 每道题的示范、预测、标准答案
- `demo/output/demo_submission.json` — 与 Kaggle 相同结构的交卷文件

用浏览器打开 `demo/report.html` 或 `demo/output/report.html`。

v2 在示例上解出 12/13，唯一没解的是 `017c7c7b`（变色后再纵向延展）。这不代表排行榜分数。

---

## 提交 Kaggle

整条链路走 CLI，不用在网页里贴代码（Kaggle 编辑器是 iframe，浏览器自动化进不去）。

1. 比赛页登录后点 **Join Competition**，同意规则。  
   https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-2
2. 登录 CLI（OAuth，浏览器里点 Approve 后把页面上的验证码贴回终端）：

```bash
python3 -m venv .venv && source .venv/bin/activate && pip install kaggle
kaggle auth login
```

   凭据缓存在 `~/.kaggle/`，**不要**提交进 git。
3. 拉比赛数据并在本地把 notebook 完整跑一遍（用的是真实的 240 题占位测试文件，只验证格式和耗时）：

```bash
kaggle competitions download -c arc-prize-2026-arc-agi-2 -p data/kaggle && (cd data/kaggle && unzip -qo '*.zip')
python3 scripts/build_kaggle_notebook.py
ARC_INPUT_DIR=data/kaggle ARC_OUTPUT=demo/output/kaggle_submission.json python3 notebooks/kaggle_baseline_as_script.py
```

4. 推 notebook 到 Kaggle 跑（断网、挂比赛数据，都在 `notebooks/kernel-metadata.json`），等状态变 COMPLETE：

```bash
kaggle kernels push -p notebooks
kaggle kernels status wenbozhang2026/arc-agi-2-non-ml-solver-v2
kaggle kernels output wenbozhang2026/arc-agi-2-non-ml-solver-v2 -p /tmp/kaggle_out   # 看日志、确认 submission.json
```

5. 提交该版本（每天只有 1 次，交前想清楚）：

```bash
kaggle competitions submit arc-prize-2026-arc-agi-2 -k wenbozhang2026/arc-agi-2-non-ml-solver-v2 -v <版本号> -f submission.json -m "说明"
kaggle competitions submissions arc-prize-2026-arc-agi-2
```

注意：比赛数据在 Kaggle 上挂在 `/kaggle/input/competitions/arc-prize-2026-arc-agi-2/`，不是老的 `/kaggle/input/<slug>/`，notebook 里已改成递归查找。Kaggle CPU 上 240 题约 5 分钟。

### 提交记录

| 日期 | 版本 | 本地训练集 | 本地评测集 | 公开榜 |
|---|---|---|---|---|
| 2026-09-07 | DSL v2 | 159/1000 | 1/120 | **0.83**（≈1/120） |
| 2026-09-09/10 | NVARC TTT（P100 / Py3.12 环境失败） | — | — | 0.00（dummy `[[0]]`，已不选用） |
| 2026-09-10 | NVARC TTT v5 Save Version（L4×4 + Py3.11） | — | 4 题 Reload **1.0/4** | 隐藏集尚未交；公开复现约 27–33% |

公开榜 DSL 0.83 与本地评测集 1/120 一致。NVARC 的 0.00 是加速器/镜像弄错，不是模型不行。当前主路线是 NVARC 原版 TTT（先每题 LoRA，再解码），本地 120 题评测在 3090 上跑，Kaggle 只用来确认。

### NVARC 本地 120 题

3090 上独立 venv（`scripts/nvarc/requirements.txt`，torch 2.8 + Unsloth 2025.9.7），权重 `/opt/models/qwen3_4b_grids15_sft139`：

```bash
# 4 题对齐 Kaggle v5（应接近 Reload 1.0/4）
bash scripts/nvarc/run_local.sh smoke4 3 --keys 0934a4d8,135a2760,136b0064,13e47133
# 完整公开评测集（单卡约一晚）
bash scripts/nvarc/run_local.sh eval120 30
```

推 Kaggle 必须走 CLI（UI Save Version 会丢掉钉死的 3.11 镜像），加速器选 **GPU L4 ×4**：

```bash
kaggle kernels push -p notebooks/nvarc_2026 --accelerator NvidiaL4
python3 scripts/kaggle_preflight.py .venv/bin/kaggle
# 确认 COMPLETE 且 Reload 非零后，再在网页 Submit to Competition（每天 1 次）
```

隐藏榜是另外 120 道题，分布与公开评测集一致，所以预期与本地评测集同量级。每天提交一次只会得到一个总分，看不到题目和逐题对错；最终名次只看你赛末选出的最多 2 份提交，中途的低分不会拖累。

---

## 官方数据

完整数据放在 `data/full/`（已在 `.gitignore`，不进版本库），来自 [arcprize/ARC-AGI-2](https://github.com/arcprize/ARC-AGI-2) 的 `main` 分支：

```
data/full/data/training/     1000 题，公开训练集，用来开发求解器
data/full/data/evaluation/    120 题，公开评测集，只用来本地打分
```

每题一个 `<task_id>.json`，结构与上文「输入 / 输出」一致；评测集的 `test` 里带 `output`，方便本地对答案。

下载方式（本机直连 `github.com` 的 git clone 会超时，走 zip 更稳）：

```bash
curl -L -o /tmp/ARC-AGI-2-main.zip https://codeload.github.com/arcprize/ARC-AGI-2/zip/refs/heads/main
unzip -q /tmp/ARC-AGI-2-main.zip -d data && mv data/ARC-AGI-2-main data/full
```

Kaggle 上榜用的是另外两套各 120 题的隐藏集（半私有 / 私有），与这里的评测集不重合。**不要人工看评测题后把答案写进规则**，那只会在本地虚高。

## 数据版权

`data/examples/` 里除 `demo_fill_color.json` 外，任务文件与 `data/full/` 一样来自 [arcprize/ARC-AGI-2](https://github.com/arcprize/ARC-AGI-2)，Apache 2.0。

---

## 下一步

- 本地 120 题跑通 NVARC 原版 TTT，记下每题耗时和候选解，作为后续加法（DSL 填 `attempt_2`、改选解、压缩耗时）的尺子。
- 不要在 120 评测题上训练。自研 Qwen3.5 SFT/规则 TTT 在 120 题上是 0，不再作为主路线。
- 若要冲击奖金，方案需按比赛规则开源（偏 MIT-0 / CC0）。

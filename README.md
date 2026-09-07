# ARC Prize 2026（ARC-AGI-2）独立工程

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

当前 baseline 不是要冲奖金的方案。它只在训练集上拟合一小撮固定变换（旋转、翻转、平移、填洞、按物体大小涂色等）。公开的 13 道示例里本地解出了 **11/13**；Kaggle 隐藏题会难得多，首次分数很可能接近 0。这正是我们要看到的基线。

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
src/arc_solver/         本地求解器（仅标准库）
scripts/run_demo.py     跑样例、出 HTML 报告
scripts/make_submission.py
scripts/build_kaggle_notebook.py
notebooks/kaggle_baseline.ipynb   给 Kaggle 用的 notebook
demo/report.html        最近一次本地 demo 报告
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

首次跑通时（2026-09-07）本地结果：**13 题中解出 11 题**（测试格子 12/14）。未解出的是 `017c7c7b`（变色后再纵向拼接）和 `0520fde7`（左右半图做 AND）。这不代表排行榜分数。

---

## 首次提交 Kaggle

本机目前没有 `~/.kaggle/kaggle.json`，所以仓库只能把 notebook 准备好，**真正点 Submit 需要你在 Kaggle 登录并同意比赛规则**。

1. 打开比赛页，登录后点 **Join Competition**，同意规则。  
   https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-2
2. 创建 API token：https://www.kaggle.com/settings → Create New Token，把 `kaggle.json` 放到 `~/.kaggle/kaggle.json`（不要提交进 git）。
3. 安装 CLI：`pip3 install kaggle`
4. 本地确认 notebook 能写出 json：

```bash
python3 scripts/build_kaggle_notebook.py
python3 -c "import json; json.load(open('notebooks/kaggle_baseline.ipynb'))"
```

5. 到 Kaggle 新建 Notebook，挂上比赛数据，把 `notebooks/kaggle_baseline.ipynb` 贴进去（或上传）。**Internet 关掉**，Save Version → 选 Save & Run All → 完成后点 **Submit to Competition**。

首次分数只是基线。排行榜用的是你没见过的隐藏题；这个变换搜索在隐藏集上大概率很低。

---

## 数据版权

`data/examples/` 里除 `demo_fill_color.json` 外，任务文件来自 [arcprize/ARC-AGI-2](https://github.com/arcprize/ARC-AGI-2)，Apache 2.0。完整 1000/120 题请自己 clone 官方仓库，不要把评测集人工看题后写进规则。

---

## 下一步

- 在公开训练集上扩 DSL / 程序搜索，而不是对着评测集调参。
- 需要算力时再考虑 test-time 微调；Kaggle 评测断网，闭源 API 路线不可用。
- 若要冲击奖金，方案需按比赛规则开源（偏 MIT-0 / CC0）。

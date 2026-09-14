# GPU 排队任务

**正在跑、不要动：**

- NVARC v11 6×6 两遍：`run_n6x6_two.sh` → B（`eval120_n6x6_v11_b`）
- DeepSeek-V4-Pro 10 并发：`scripts/llm_bench/run_eval.py --provider dashscope --effort high`（`/opt/work/arc-chat-eval`）

**已插入、正在跑（不要杀）：**

最低全候选池搜索：`run_min_pool_search.sh`（pid 在 `eval120_search_logs/min_pool.pid`）

先 CPU 视图表（不占 GPU），再等 v11 `done`，再续 `n7_g8`。旧的 `run_gen_queue.sh` 不会自动开。

| 顺序 | 任务 | GPU？ | 说明 |
|---|---|---|---|
| 1 | CPU `oracle_view_table.py` | 否 | 已有 16×16 / 8×8A / 8×8B / 6×8 / 5×8 pickle，切 geos=1,2,4,5,6,8。看 **oracle**，不是只看总分。 |
| 2 | 等 v11 `eval120_n6x6_v11_pool/done` | — | 不杀 v11 |
| 3 | GPU 续 `eval120_search/n7_g8` | 是 | `--skip-done`，此前 62/120 |
| 4 | 再跑 CPU 表（含 7×8 → 7×5 / 7×4） | 否 | 写出 `eval120_search/min_pool_views.json` |

目标：最低 `n_train × n_geos` 使 oracle 仍接近 16×16 的 41.50，判断 Kaggle 12h 两遍是 7×5 / 8×5 / 8×4 还是必须 8×6。

粗算（4×L4、12h−20min、240 slot、L4/3090=1.61）：8×6 两遍 ≈100%；8×5 ≈89%；7×5 ≈85%；8×4 ≈78%。

---

## 不要自动重挂的旧队列

`run_gen_queue.sh`（n6x6_aug2 / lr / epochs / lora_r）**不要**接到 v11 的 `done` 上。等明确说「继续 GPU 队列」再跑。

#!/usr/bin/env python3
"""Build a standalone HTML so you can solve a handful of ARC tasks as a human.

Official 10-color palette. Demos on the left; paint the test output; check
against the hidden gold grid. Does not touch the GPU training job.

    python scripts/human_preview.py
    open reports/human_preview.html
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "full" / "data"
OUT = ROOT / "reports" / "human_preview.html"

# Mix of small / medium / size-changing training tasks.
TRAIN_IDS = [
    "d037b0a7",
    "67a3c6ac",
    "05269061",
    "f9012d9b",
    "9ddd00f0",
    "f0afb749",
    "234bbc79",
    "310f3251",
    "8dae5dfc",
    "bc4146bd",
]
# Evaluation tasks picked for being drawable on a laptop (most eval grids are 20–30).
EVAL_IDS = [
    "28a6681f",
    "dd6b8c4b",
    "3dc255db",
    "e8686506",
    "20270e3b",
]


def load(split: str, tid: str) -> dict:
    t = json.loads((DATA / split / f"{tid}.json").read_text())
    return {
        "id": tid,
        "split": split,
        "train": t["train"],
        "test": [{"input": x["input"], "output": x["output"]} for x in t["test"]],
    }


def main() -> None:
    tasks = [load("training", i) for i in TRAIN_IDS] + [load("evaluation", i) for i in EVAL_IDS]
    payload = json.dumps(tasks, separators=(",", ":"))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(HTML.replace("__TASKS__", payload), encoding="utf-8")
    print(f"wrote {OUT}  ({len(tasks)} tasks, {OUT.stat().st_size} bytes)")


HTML = r"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>ARC 人工试做 · 10 训练 + 5 评测</title>
<style>
  :root { --bg:#111; --panel:#1b1b1b; --line:#333; --fg:#eee; --muted:#9a9a9a; }
  * { box-sizing: border-box; }
  body { margin:0; font-family: ui-sans-serif, system-ui, sans-serif; background:var(--bg); color:var(--fg); }
  header { padding:16px 20px 8px; border-bottom:1px solid var(--line); }
  h1 { font-size:18px; font-weight:600; margin:0 0 6px; }
  .sub { color:var(--muted); font-size:13px; line-height:1.5; max-width:920px; }
  .layout { display:grid; grid-template-columns: 220px 1fr; min-height: calc(100vh - 90px); }
  nav { border-right:1px solid var(--line); padding:12px; overflow:auto; }
  nav h2 { font-size:11px; letter-spacing:.08em; text-transform:uppercase; color:var(--muted); margin:12px 0 6px; }
  nav button.task { display:block; width:100%; text-align:left; background:transparent; color:var(--fg);
    border:1px solid transparent; padding:7px 8px; margin:2px 0; cursor:pointer; font:inherit; border-radius:4px; }
  nav button.task:hover { background:#222; }
  nav button.task.active { border-color:#555; background:#252525; }
  nav button.task .sid { color:var(--muted); font-size:11px; }
  main { padding:16px 20px 40px; overflow:auto; }
  .legend { display:flex; gap:6px; flex-wrap:wrap; margin:10px 0 16px; }
  .sw { width:28px; height:28px; border:1px solid #444; cursor:pointer; position:relative; }
  .sw.sel { outline:2px solid #fff; outline-offset:1px; }
  .sw span { position:absolute; left:50%; top:50%; transform:translate(-50%,-50%); font-size:10px; font-weight:700;
    color:#fff; text-shadow:0 0 2px #000; }
  .pair-row { display:flex; gap:18px; flex-wrap:wrap; align-items:flex-start; margin-bottom:18px; }
  .pair { background:var(--panel); padding:10px; border:1px solid var(--line); }
  .pair .cap { color:var(--muted); font-size:12px; margin-bottom:6px; }
  .io { display:flex; align-items:center; gap:10px; }
  .arrow { color:#666; font-size:20px; }
  .arc { display:grid; gap:1px; background:#222; padding:3px; }
  .cell { width:100%; height:100%; }
  .dim { color:var(--muted); font-size:11px; margin-top:4px; }
  .tools { display:flex; gap:8px; flex-wrap:wrap; align-items:center; margin:10px 0; }
  .tools button, .tools input { font:inherit; background:#2a2a2a; color:var(--fg); border:1px solid #555;
    padding:6px 10px; cursor:pointer; border-radius:3px; }
  .tools input { width:72px; cursor:text; }
  .tools button.primary { background:#2ECC40; color:#111; border-color:#2ECC40; font-weight:600; }
  .status { min-height:22px; font-size:14px; margin:8px 0 16px; }
  .ok { color:#2ECC40; } .bad { color:#FF4136; } .hint { color:var(--muted); }
  .test-box { background:var(--panel); padding:12px; border:1px solid var(--line); }
</style>
</head>
<body>
<header>
  <h1>当人来做 ARC：10 道训练 + 5 道评测</h1>
  <div class="sub">
    数字 0–9 是 10 种固定颜色（官方配色）。先看左边示范对，归纳规则，再在测试输入右侧画出输出格子。
    评测题答案默认隐藏——先自己画，再点「检查」。官方说明：评测集人类样本正确率约 66%，每题允许两次尝试。
  </div>
</header>
<div class="layout">
  <nav id="nav"></nav>
  <main>
    <div class="legend" id="legend"></div>
    <div id="demos"></div>
    <div class="test-box">
      <div class="cap" style="color:#9a9a9a;font-size:12px;margin-bottom:8px">测试（请你来画右边）</div>
      <div class="io" id="testio"></div>
      <div class="tools">
        <button onclick="copyInput()">从输入复制</button>
        <button onclick="clearOut()">清空</button>
        <label class="hint">输出尺寸</label>
        <input id="sz" placeholder="h×w" title="例如 7x7"/>
        <button onclick="resizeOut()">改尺寸</button>
        <button class="primary" onclick="check()">检查答案</button>
        <button onclick="reveal()">显示标准答案</button>
      </div>
      <div class="status" id="status"></div>
    </div>
  </main>
</div>
<script>
const TASKS = __TASKS__;
const COLORS = {0:"#000000",1:"#0074D9",2:"#FF4136",3:"#2ECC40",4:"#FFDC00",5:"#AAAAAA",6:"#F012BE",7:"#FF851B",8:"#7FDBFF",9:"#870C25"};
const NAMES = ["黑","蓝","红","绿","黄","灰","品红","橙","青","褐"];
let idx = 0, color = 1, painting = false, showGold = false;
const answers = {}; // id -> painted grid
const attempts = {};

function clone(g){ return g.map(r => r.slice()); }
function zeros(h,w){ return Array.from({length:h},()=>Array(w).fill(0)); }
function shape(g){ return [g.length, g[0]?g[0].length:0]; }

function cellPx(g){
  const [h,w] = shape(g);
  const m = Math.max(h,w);
  if (m <= 8) return 22;
  if (m <= 12) return 16;
  if (m <= 20) return 12;
  return 9;
}

function renderGrid(g, opts){
  opts = opts || {};
  const [h,w] = shape(g);
  const px = opts.px || cellPx(g);
  const clickable = !!opts.clickable;
  const el = document.createElement("div");
  el.className = "arc";
  el.style.gridTemplateColumns = `repeat(${w},${px}px)`;
  el.style.gridAutoRows = px+"px";
  el.style.width = (w*px + 6) + "px";
  g.forEach((row,i) => row.forEach((v,j) => {
    const d = document.createElement("div");
    d.className = "cell";
    d.style.background = COLORS[v];
    d.title = v + " " + NAMES[v];
    if (clickable) {
      d.onmousedown = (e) => { e.preventDefault(); paint(i,j); };
      d.onmouseenter = () => { if (painting) paint(i,j); };
    }
    el.appendChild(d);
  }));
  const wrap = document.createElement("div");
  wrap.appendChild(el);
  const dim = document.createElement("div");
  dim.className = "dim";
  dim.textContent = h + "×" + w;
  wrap.appendChild(dim);
  return wrap;
}

function paint(i,j){
  const t = TASKS[idx];
  let g = answers[t.id];
  if (!g) return;
  if (i>=g.length || j>=g[i].length) return;
  g[i][j] = color;
  showGold = false;
  drawTest();
}

function currentPainted(){
  const t = TASKS[idx];
  if (!answers[t.id]) {
    const [h,w] = shape(t.test[0].output);
    // start blank, same size as gold so checking is fair; user can resize
    answers[t.id] = zeros(h,w);
  }
  return answers[t.id];
}

function drawLegend(){
  const box = document.getElementById("legend");
  box.innerHTML = "";
  for (let k=0;k<10;k++){
    const b = document.createElement("div");
    b.className = "sw" + (color===k ? " sel":"");
    b.style.background = COLORS[k];
    b.title = k + " " + NAMES[k];
    b.innerHTML = "<span>"+k+"</span>";
    b.onclick = () => { color = k; drawLegend(); };
    box.appendChild(b);
  }
}

function drawNav(){
  const nav = document.getElementById("nav");
  nav.innerHTML = "";
  [["训练 10 题","training"],["评测 5 题（更难）","evaluation"]].forEach(([title, split]) => {
    const h = document.createElement("h2"); h.textContent = title; nav.appendChild(h);
    TASKS.forEach((t,i) => {
      if (t.split !== split) return;
      const b = document.createElement("button");
      b.className = "task" + (i===idx ? " active":"");
      const [ih,iw] = shape(t.train[0].input);
      const [oh,ow] = shape(t.train[0].output);
      b.innerHTML = t.id.slice(0,8) + " <span class='sid'>" + t.train.length + " 示范 · " + ih+"×"+iw+"→"+oh+"×"+ow + "</span>";
      b.onclick = () => { idx=i; showGold=false; render(); };
      nav.appendChild(b);
    });
  });
}

function drawDemos(){
  const t = TASKS[idx];
  const box = document.getElementById("demos");
  box.innerHTML = "";
  t.train.forEach((p, n) => {
    const pair = document.createElement("div");
    pair.className = "pair";
    pair.innerHTML = `<div class="cap">示范 ${n+1}</div>`;
    const io = document.createElement("div");
    io.className = "io";
    io.appendChild(renderGrid(p.input));
    const ar = document.createElement("div"); ar.className="arrow"; ar.textContent="→"; io.appendChild(ar);
    io.appendChild(renderGrid(p.output));
    pair.appendChild(io);
    box.appendChild(pair);
  });
}

function drawTest(){
  const t = TASKS[idx];
  const test = t.test[0];
  const io = document.getElementById("testio");
  io.innerHTML = "";
  const left = document.createElement("div");
  const cap1 = document.createElement("div"); cap1.className="cap"; cap1.style.color="#9a9a9a"; cap1.style.fontSize="12px";
  cap1.textContent = "测试输入";
  left.appendChild(cap1);
  left.appendChild(renderGrid(test.input));
  io.appendChild(left);
  const ar = document.createElement("div"); ar.className="arrow"; ar.textContent="→"; io.appendChild(ar);
  const right = document.createElement("div");
  const cap2 = document.createElement("div"); cap2.className="cap"; cap2.style.color="#9a9a9a"; cap2.style.fontSize="12px";
  cap2.textContent = showGold ? "标准答案" : "你的输出（点格子上色，可拖拽）";
  right.appendChild(cap2);
  right.appendChild(renderGrid(showGold ? test.output : currentPainted(), {clickable: !showGold}));
  io.appendChild(right);
  const [h,w] = shape(currentPainted());
  document.getElementById("sz").value = h + "x" + w;
}

function copyInput(){
  const t = TASKS[idx];
  answers[t.id] = clone(t.test[0].input);
  showGold = false; document.getElementById("status").textContent=""; drawTest();
}
function clearOut(){
  const g = currentPainted();
  answers[TASKS[idx].id] = zeros(g.length, g[0].length);
  showGold = false; document.getElementById("status").textContent=""; drawTest();
}
function resizeOut(){
  const raw = document.getElementById("sz").value.trim().toLowerCase().replace("×","x");
  const m = raw.match(/^(\d+)\s*x\s*(\d+)$/);
  if (!m) { document.getElementById("status").innerHTML = "<span class='bad'>尺寸写成 7x7 这种</span>"; return; }
  const h=+m[1], w=+m[2];
  if (h<1||w<1||h>30||w>30) { document.getElementById("status").innerHTML = "<span class='bad'>边长 1–30</span>"; return; }
  const old = currentPainted();
  const g = zeros(h,w);
  for (let i=0;i<Math.min(h,old.length);i++)
    for (let j=0;j<Math.min(w,old[0].length);j++) g[i][j]=old[i][j];
  answers[TASKS[idx].id]=g; showGold=false; drawTest();
}
function eq(a,b){
  if (!a||!b||a.length!==b.length||a[0].length!==b[0].length) return false;
  for (let i=0;i<a.length;i++) for (let j=0;j<a[0].length;j++) if (a[i][j]!==b[i][j]) return false;
  return true;
}
function check(){
  const t = TASKS[idx];
  attempts[t.id] = (attempts[t.id]||0)+1;
  const ok = eq(currentPainted(), t.test[0].output);
  const el = document.getElementById("status");
  if (ok) el.innerHTML = "<span class='ok'>全对 · 第 "+attempts[t.id]+" 次提交</span>";
  else {
    const g=currentPainted(), gold=t.test[0].output;
    let msg = "还不对 · 第 "+attempts[t.id]+" 次";
    if (g.length!==gold.length || g[0].length!==gold[0].length)
      msg += "（尺寸应是 "+gold.length+"×"+gold[0].length+"，你画的是 "+g.length+"×"+g[0].length+"）";
    el.innerHTML = "<span class='bad'>"+msg+"</span>";
  }
}
function reveal(){ showGold = true; drawTest(); }

function render(){
  drawNav(); drawLegend(); drawDemos(); drawTest();
  document.getElementById("status").textContent = "";
}

document.body.addEventListener("mousedown", e => { if (e.button===0) painting=true; });
document.body.addEventListener("mouseup", () => painting=false);
document.body.addEventListener("mouseleave", () => painting=false);
render();
</script>
</body>
</html>
"""

if __name__ == "__main__":
    main()

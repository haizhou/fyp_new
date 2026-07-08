#!/usr/bin/env python3
"""CICADA training-ladder live dashboard (stdlib only, port 8100).

Reads (read-only, safe alongside training):
  outputs/*/trainer_log.jsonl        training progress + loss curves
  data/qa/rsft_*_r1/traces.jsonl     RSFT harvest progress
  outputs/eval/**/*.summary.json     ladder / baseline results
  nvidia-smi                         GPU line

Run:  python scripts/serve_dashboard.py   ->  http://127.0.0.1:8100
(VSCode remote auto-forwards the port.)
"""
from __future__ import annotations

import json
import subprocess
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PORT = 8100

RUNGS = [  # (display name, output dir, logical stage order)
    ("Qwen3-8B SFT",   "outputs/qwen3_8b_cicada_sft_v1"),
    ("Llama-3.1 SFT",  "outputs/llama31_8b_cicada_sft_v1"),
    ("Qwen3-8B RSFT",  "outputs/qwen3_8b_cicada_rsft_v1"),
    ("Llama-3.1 RSFT", "outputs/llama31_8b_cicada_rsft_v1"),
    ("Qwen3-8B DPO",   "outputs/qwen3_8b_cicada_dpo_v1"),
    ("Llama-3.1 DPO",  "outputs/llama31_8b_cicada_dpo_v1"),
]
HARVESTS = [
    ("Qwen RSFT harvest",  "data/qa/rsft_qwen_r1",  9267),
    ("Llama RSFT harvest", "data/qa/rsft_llama_r1", 9267),
]


def read_trainer(outdir: Path) -> dict | None:
    log = outdir / "trainer_log.jsonl"
    if not log.exists():
        return None
    rows = [json.loads(l) for l in log.read_text().splitlines() if l.strip()]
    if not rows:
        return None
    loss = [(r["current_steps"], r["loss"]) for r in rows if "loss" in r]
    ev = [(r["current_steps"], r["eval_loss"]) for r in rows if "eval_loss" in r]
    last = rows[-1]
    total = last.get("total_steps") or 1
    step = last.get("current_steps") or 0
    fresh = (time.time() - log.stat().st_mtime) < 240
    state = "done" if step >= total else ("running" if fresh else "stopped")
    return {"loss": loss, "eval": ev, "step": step, "total": total, "state": state,
            "eta": last.get("remaining_time", ""), "elapsed": last.get("elapsed_time", "")}


def gpu_line() -> list[dict]:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,utilization.gpu,memory.used,memory.total",
             "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=5).stdout
        res = []
        for ln in out.strip().splitlines():
            p = [x.strip() for x in ln.split(",")]
            if len(p) >= 4:
                res.append({"idx": p[0], "util": p[1], "used": p[2], "total": p[3]})
        return res
    except Exception:
        return []


def eval_results() -> list[dict]:
    res = []
    for f in sorted((ROOT / "outputs" / "eval").rglob("*.summary.json")):
        try:
            d = json.loads(f.read_text())
            o = d.get("overall", {})
            res.append({"name": str(f.relative_to(ROOT / "outputs" / "eval")).replace(
                "/compare_cicada.summary.json", "").replace("/compare_", " / ").replace(".summary.json", ""),
                "total": o.get("total"), "correct": o.get("correct"), "acc": o.get("accuracy")})
        except Exception:
            continue
    return res


def status() -> dict:
    rungs = []
    for name, out in RUNGS:
        t = read_trainer(ROOT / out)
        rungs.append({"name": name, **(t or {"state": "pending", "step": 0, "total": 0,
                                             "loss": [], "eval": [], "eta": "", "elapsed": ""})})
    harvests = []
    for name, d, total in HARVESTS:
        tr = ROOT / d / "traces.jsonl"
        n = sum(1 for _ in tr.open()) if tr.exists() else 0
        done = (ROOT / d / "summary.json").exists()
        harvests.append({"name": name, "n": n, "total": total,
                         "state": "done" if done else ("running" if n else "pending")})
    return {"ts": time.strftime("%H:%M:%S"), "gpu": gpu_line(),
            "rungs": rungs, "harvests": harvests, "results": eval_results()}


PAGE = r"""<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>CICADA ladder — live</title><style>
:root{--surface:#fcfcfb;--page:#f9f9f7;--ink:#0b0b0b;--ink2:#52514e;--muted:#898781;
--grid:#e1e0d9;--axis:#c3c2b7;--s1:#2a78d6;--s2:#1baf7a;--good:#0ca30c;--warn:#fab219;
--crit:#d03b3b;--ring:rgba(11,11,11,.10)}
@media(prefers-color-scheme:dark){:root{--surface:#1a1a19;--page:#0d0d0d;--ink:#fff;
--ink2:#c3c2b7;--muted:#898781;--grid:#2c2c2a;--axis:#383835;--s1:#3987e5;--s2:#199e70;
--good:#0ca30c;--ring:rgba(255,255,255,.10)}}
*{box-sizing:border-box;margin:0}body{font:14px/1.45 system-ui,-apple-system,"Segoe UI",sans-serif;
background:var(--page);color:var(--ink);padding:18px;max-width:1060px;margin:0 auto}
h1{font-size:17px;margin-bottom:2px}.sub{color:var(--muted);font-size:12px;margin-bottom:14px}
.card{background:var(--surface);border:1px solid var(--ring);border-radius:10px;padding:14px 16px;margin-bottom:12px}
.card h2{font-size:13px;color:var(--ink2);font-weight:600;margin-bottom:10px}
.row{display:flex;align-items:center;gap:10px;padding:5px 0}
.nm{width:150px;font-weight:600;font-size:13px}
.bar{flex:1;height:10px;background:var(--grid);border-radius:5px;overflow:hidden}
.bar i{display:block;height:100%;border-radius:5px;background:var(--s1)}
.pct{width:210px;text-align:right;color:var(--ink2);font-variant-numeric:tabular-nums;font-size:12px}
.tag{font-size:11px;padding:1px 8px;border-radius:9px;font-weight:600}
.t-run{background:color-mix(in srgb,var(--s1) 15%,transparent);color:var(--s1)}
.t-done{background:color-mix(in srgb,var(--good) 15%,transparent);color:var(--good)}
.t-pend{background:var(--grid);color:var(--muted)}
.t-stop{background:color-mix(in srgb,var(--crit) 15%,transparent);color:var(--crit)}
table{width:100%;border-collapse:collapse;font-size:13px}
td,th{padding:5px 8px;text-align:left;border-bottom:1px solid var(--grid)}
th{color:var(--muted);font-weight:600;font-size:11px;text-transform:uppercase;letter-spacing:.04em}
td.num{font-variant-numeric:tabular-nums;text-align:right}
.acc{font-weight:700}.gpu{display:flex;gap:14px;flex-wrap:wrap;font-size:12px;color:var(--ink2)}
.gpu b{color:var(--ink);font-variant-numeric:tabular-nums}
svg text{font:10px system-ui;fill:var(--muted)}
.leg{display:flex;gap:16px;font-size:12px;color:var(--ink2);margin:2px 0 6px}
.leg i{display:inline-block;width:14px;height:3px;border-radius:2px;vertical-align:middle;margin-right:5px}
#tip{position:fixed;pointer-events:none;background:var(--surface);border:1px solid var(--ring);
border-radius:6px;padding:5px 9px;font-size:12px;display:none;box-shadow:0 2px 8px rgba(0,0,0,.15)}
</style></head><body>
<h1>CICADA 8B student ladder — live</h1>
<div class="sub" id="ts">connecting…</div>
<div class="card"><h2>GPU</h2><div class="gpu" id="gpu"></div></div>
<div class="card"><h2>Pipeline stages</h2><div id="stages"></div></div>
<div class="card"><h2>Active training — loss (log scale)</h2>
<div class="leg"><span><i style="background:var(--s1)"></i>train loss</span>
<span><i style="background:var(--s2)"></i>eval loss</span></div>
<div id="chart"></div></div>
<div class="card"><h2>Results (accuracy)</h2><table id="res"><thead>
<tr><th>run</th><th style="text-align:right">correct</th><th style="text-align:right">n</th>
<th style="text-align:right">acc</th></tr></thead><tbody></tbody></table></div>
<div id="tip"></div>
<script>
const fmt=(x,d=1)=>x==null?"—":(+x).toFixed(d);
function bar(name,frac,state,extra){
 const tag={running:"t-run",done:"t-done",pending:"t-pend",stopped:"t-stop"}[state]||"t-pend";
 return `<div class="row"><span class="nm">${name}</span>
 <span class="tag ${tag}">${state}</span>
 <div class="bar"><i style="width:${(frac*100).toFixed(1)}%"></i></div>
 <span class="pct">${extra}</span></div>`}
function drawChart(r){
 const el=document.getElementById("chart");
 if(!r||!r.loss.length){el.innerHTML='<div class="sub">no active run</div>';return}
 const W=980,H=230,P={l:46,r:12,t:8,b:22};
 const pts=r.loss,ev=r.eval;
 const xs=pts.map(p=>p[0]).concat(ev.map(p=>p[0]));
 const ys=pts.map(p=>p[1]).concat(ev.map(p=>p[1])).filter(v=>v>0);
 const xmax=Math.max(...xs,r.total),x0=0;
 const ymin=Math.min(...ys),ymax=Math.max(...ys);
 const ly0=Math.floor(Math.log10(ymin)),ly1=Math.ceil(Math.log10(ymax));
 const X=v=>P.l+(v-x0)/(xmax-x0)*(W-P.l-P.r);
 const Y=v=>P.t+(ly1-Math.log10(v))/(ly1-ly0||1)*(H-P.t-P.b);
 let g="";
 for(let e=ly0;e<=ly1;e++){const y=Y(10**e);
  g+=`<line x1="${P.l}" y1="${y}" x2="${W-P.r}" y2="${y}" stroke="var(--grid)"/>
  <text x="${P.l-6}" y="${y+3}" text-anchor="end">1e${e}</text>`}
 for(let s=0;s<=xmax;s+=Math.max(100,Math.round(xmax/8/100)*100)){
  g+=`<text x="${X(s)}" y="${H-6}" text-anchor="middle">${s}</text>`}
 const path=a=>a.map((p,i)=>(i?"L":"M")+X(p[0]).toFixed(1)+","+Y(p[1]).toFixed(1)).join(" ");
 g+=`<path d="${path(pts)}" fill="none" stroke="var(--s1)" stroke-width="2"/>`;
 if(ev.length)g+=`<path d="${path(ev)}" fill="none" stroke="var(--s2)" stroke-width="2"/>`+
  ev.map(p=>`<circle cx="${X(p[0])}" cy="${Y(p[1])}" r="3.5" fill="var(--s2)" stroke="var(--surface)" stroke-width="2"/>`).join("");
 el.innerHTML=`<svg id="svg" viewBox="0 0 ${W} ${H}" style="width:100%">${g}
 <line id="cross" y1="${P.t}" y2="${H-P.b}" stroke="var(--axis)" stroke-dasharray="3,3" visibility="hidden"/></svg>`;
 const svg=document.getElementById("svg"),tip=document.getElementById("tip"),cross=document.getElementById("cross");
 svg.onmousemove=e=>{const b=svg.getBoundingClientRect();
  const sx=(e.clientX-b.left)/b.width*W;
  const step=x0+(sx-P.l)/(W-P.l-P.r)*(xmax-x0);
  let best=pts[0];for(const p of pts)if(Math.abs(p[0]-step)<Math.abs(best[0]-step))best=p;
  cross.setAttribute("x1",X(best[0]));cross.setAttribute("x2",X(best[0]));cross.setAttribute("visibility","visible");
  const evn=ev.filter(p=>p[0]<=best[0]).slice(-1)[0];
  tip.style.display="block";tip.style.left=(e.clientX+14)+"px";tip.style.top=(e.clientY-10)+"px";
  tip.innerHTML=`step <b>${best[0]}</b> · train <b>${best[1].toFixed(4)}</b>`+(evn?` · eval <b>${evn[1].toFixed(4)}</b>`:"");};
 svg.onmouseleave=()=>{tip.style.display="none";cross.setAttribute("visibility","hidden")}}
async function tick(){
 try{
  const d=await (await fetch("/api/status")).json();
  document.getElementById("ts").textContent="updated "+d.ts+" · refreshes every 5s";
  document.getElementById("gpu").innerHTML=d.gpu.map(g=>
   `<span>GPU${g.idx}: <b>${g.util}%</b> util · <b>${(g.used/1024).toFixed(1)}</b>/${(g.total/1024).toFixed(0)} GB</span>`).join("");
  let h="";
  for(const x of d.harvests)h+=bar(x.name,x.n/x.total,x.state,`${x.n.toLocaleString()} / ${x.total.toLocaleString()}`);
  for(const r of d.rungs)h+=bar(r.name,r.total?r.step/r.total:0,r.state,
    r.total?`${r.step}/${r.total}${r.eta?" · eta "+r.eta:""}`:"");
  document.getElementById("stages").innerHTML=h;
  const act=d.rungs.filter(r=>r.state==="running")[0]||d.rungs.filter(r=>r.loss&&r.loss.length).slice(-1)[0];
  drawChart(act);
  document.querySelector("#res tbody").innerHTML=d.results.map(r=>
   `<tr><td>${r.name}</td><td class="num">${r.correct??"—"}</td><td class="num">${r.total??"—"}</td>
   <td class="num acc">${r.acc!=null?(100*r.acc).toFixed(1)+"%":"—"}</td></tr>`).join("");
 }catch(e){document.getElementById("ts").textContent="reconnecting… ("+e.message+")"}
 setTimeout(tick,5000)}
tick();
</script></body></html>"""


class H(BaseHTTPRequestHandler):
    def do_GET(self):  # noqa: N802
        if self.path.startswith("/api/status"):
            body = json.dumps(status()).encode()
            ct = "application/json"
        else:
            body = PAGE.encode()
            ct = "text/html; charset=utf-8"
        self.send_response(200)
        self.send_header("Content-Type", ct)
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *a):  # silence access log
        pass


if __name__ == "__main__":
    print(f"dashboard: http://127.0.0.1:{PORT}")
    ThreadingHTTPServer(("127.0.0.1", PORT), H).serve_forever()

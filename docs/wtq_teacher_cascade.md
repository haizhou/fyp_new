# WTQ 教师级联收割流程(2026-07-23 定稿)

原则:免费级先行缩池,付费级只打残余;每级产出下一级的 ids 文件;教师只提名,oracle 过滤。
臂纯度:C-v5 的提名只进 C 池(金程序下游);A 池只收 composev3 / 云教师提名(答案过滤,无金程序)。

## 级联五段

| # | 段 | 采样器 | 成本 | 产出 |
|---|---|---|---|---|
| 1 | A 池 v4b 重采 | composev3(本地) | 免费 | harvest_Av4b_*.jsonl + 新零命中池 |
| 2 | 零命中提取 | — | — | /tmp/claude-1847/zerohit_v4b.ids |
| 3 | C 池自收割 | wtq_C5(本地) | 免费 | harvest_C5self_*.jsonl(仅进 C 池) |
| 4 | Job A fast 扫残池 | grok-4-1-fast(k=2) | ~$4.7 | harvest_teacher_fast.jsonl(进 A 池) |
| 5 | Job A+ reasoning 直出 | grok-4-20-reasoning(k=1, 6k tok) | 费率档待定 | harvest_teacher_reasoning.jsonl(进 A 池) |

## 命令

1-3 由门控启动器自动执行(等空卡):`scripts/wtq/run_reharvest_v4b.sh`(载荷 = `run_local_cascade.sh`)
4:`harvest_teacher.py --deployment grok-4-1-fast-non-reasoning --ids-file /tmp/claude-1847/teacher_pool.ids --k 2`
5:`harvest_teacher.py --deployment grok-4-20-reasoning --ids-file <4的残余> --k 1 --max-tokens 6000`
4/5 崩溃直接重跑同命令即续传(逐行落盘,按 id 跳过)。

## 证据链(E13c/g/h/i/j)
fast 试点 15.5%;C-v5 拯救 30%(6/20);reasoning 直出 35%(7/20);理解臂双向关死(简报对弱规划器 3/20、对强规划器 +0)。

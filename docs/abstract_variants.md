# Abstract 措辞预案(final_test 配对裁决前写好,免得赶工硬拗)

裁决实验:final_test (n=2,285, v4.1) 上 fully-local(本地 Step-1)vs hybrid(nano Step-1),
Step-2 同为 `qwen3_8b_cicada_dpo_v1`。McNemar 双侧 p<0.05 为界。

## 版本 A:superiority(若 fully-local 显著胜出)

> ... The bootstrapped system runs entirely on a single local GPU — both the understanding and
> planning stages are 8B LoRA adapters distilled through the pipeline's own verifier — and
> **outperforms its cloud-hybrid counterpart** (McNemar p=___) while exceeding its cloud
> teachers by __ points (p<1e-4), at zero marginal API cost. Abstention closes the verifier's
> blind region: ...

## 版本 B:parity(若不显著——注意这个叙事并不弱)

> ... The bootstrapped system runs entirely on a single local GPU and **matches its cloud-hybrid
> counterpart** (Δ=__ pt, n.s.) while exceeding its cloud teachers by __ points (p<1e-4) — full
> privacy and zero marginal API cost without sacrificing accuracy. Abstention closes the
> verifier's blind region: ...

去教师化的价值主张本来就是成本与私有化;反超是锦上添花,平局即胜利。

## 不受裁决影响的固定句(两版共用)

- Student-vs-teacher 双层结构(两个集合各司其职,防混用):
  "exceeds the teacher on the held-out test set (n=2,285, McNemar p=___, single teacher
  replicate); robustness of this gap to provider nondeterminism was established on the
  development set, where the student beats each of THREE teacher replicates (+35/−5, +34/−4,
  +30/−5; p≤2e-5)."
- parity 版补 CI:"Δ=__pt, 95% CI [__, __]"(配对差 CI 由不一致对数计算)——CI 窄本身就是
  parity 的实证,比裸 "n.s." 硬。
- 机制句(软化 + 模式措辞):"Across three teacher replicates, accuracy decays 6–12 points from
  the iid to the hard-composite slice; the fully-local student shows **no measurable decay
  (+0.8pt, n=136)** — with the SAME planning checkpoint, attributable to the distilled
  understanding stage."
- 命名纪律:正文一律 "hard-composite slice"(定义=难算子∪L2改写∪弃答类),不用 "OOD";
  strict holdout 只有 1 个 test-only 模板族,进 limitations;组合泛化主张由 ood_probe_v1
  (compositional OOD,预注册)承担。

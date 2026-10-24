# Paired AOPC analysis (20-alpha IG)

Method selection and IG AOPC use the updated Table 1 with 20-point Integrated Gradients (`α ∈ {0.05,…,1.00}`). Other methods use the original zero-ablation caches. Each comparison uses within-passage differences `AOPC(scope) − AOPC(baseline)`; negative differences favor the Scope. Reported p-values are two-sided paired Wilcoxon tests.

| Model | Dataset | Top Scope | Top baseline | Mean Δ [95% CI] | Wilcoxon p |
| --- | --- | --- | --- | --- | --- |
| LLaMA-3.2 1B | LAMBADA | Fisher Scope | Input × Gradient | **-0.0358 [-0.0482, -0.0235]** | **1.19e-06** |
| LLaMA-3.2 1B | IWSLT2017 DE→EN | Temperature Scope | Integrated Gradients | **-0.0490 [-0.0729, -0.0251]** | **0.000396** |
| LLaMA-3.2 3B | LAMBADA | Fisher Scope | Input × Gradient | **-0.0551 [-0.0714, -0.0388]** | **1.11e-11** |
| LLaMA-3.2 3B | IWSLT2017 DE→EN | Fisher Scope | Input × Gradient | **-0.0285 [-0.0418, -0.0151]** | **0.000391** |
| Qwen2.5 1.5B | LAMBADA | Temperature Scope | Input × Gradient | **-0.1408 [-0.1678, -0.1138]** | **3.13e-21** |
| Qwen2.5 1.5B | IWSLT2017 DE→EN | Temperature Scope | Integrated Gradients | -0.0249 [-0.0521, 0.0023] | 0.12 |
| Qwen2.5 3B | LAMBADA | Temperature Scope | Input × Gradient | **-0.1616 [-0.1870, -0.1362]** | **1.45e-32** |
| Qwen2.5 3B | IWSLT2017 DE→EN | Temperature Scope | Integrated Gradients | **-0.0813 [-0.1184, -0.0442]** | **9.29e-06** |
| Gemma-3 1B | LAMBADA | Fisher Scope | Input × Gradient | **-0.1085 [-0.1261, -0.0908]** | **3.71e-30** |
| Gemma-3 1B | IWSLT2017 DE→EN | Fisher Scope | Input × Gradient | **-0.0368 [-0.0509, -0.0227]** | **0.000112** |
| Gemma-3 4B | LAMBADA | Fisher Scope | Input × Gradient | **-0.1102 [-0.1265, -0.0938]** | **7.98e-32** |
| Gemma-3 4B | IWSLT2017 DE→EN | Fisher Scope | Integrated Gradients | **0.1447 [0.1119, 0.1775]** | **9.35e-18** |

## Summary

Significant Scope wins: **10/12**. Significant baseline wins: **1/12**. Inconclusive: **1/12**.

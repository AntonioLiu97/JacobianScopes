We thank all reviewers for their insightful comments and constructive feedback. In accordance with the reviewers' suggestions, we have added the following new analysis. We are also currently running (i) a parameter-randomization sanity check for all saliency methods and (ii) a 12-cell model–dataset AOPC sweep that replaces the zero-vector ablation with each model's reserved `<PAD>` token. We will update the rebuttal with these results as they become available.

## 1. Paired significance tests

Following the suggestions of reviewers PUGo and D5fu, we performed passage-level paired significance tests for all model–dataset combinations. Before conducting these tests, we fixed the Scope and non-Scope baseline in each cell to the methods with the best mean AOPC in the original table. We then computed `AOPC(Scope) − AOPC(baseline)` for every passage, so a negative mean paired difference favors the Scope. We report the standard Student-*t* 95% confidence interval for that mean.

| Model | Dataset | Top-performing Scope | Top-performing baseline | Mean paired difference [95% CI] | Wilcoxon *p* |
| --- | --- | --- | --- | ---: | ---: |
| LLaMA-3.2 1B | LAMBADA | Fisher | Input × Gradient | **−0.0358 [−0.0482, −0.0235]** | **1.19 × 10⁻⁶** |
| LLaMA-3.2 1B | IWSLT2017 DE→EN | Temperature | Input × Gradient | **−0.0494 [−0.0689, −0.0299]** | **9.36 × 10⁻⁵** |
| LLaMA-3.2 3B | LAMBADA | Fisher | Input × Gradient | **−0.0551 [−0.0714, −0.0388]** | **1.11 × 10⁻¹¹** |
| LLaMA-3.2 3B | IWSLT2017 DE→EN | Fisher | Input × Gradient | **−0.0285 [−0.0418, −0.0151]** | **3.91 × 10⁻⁴** |
| Qwen2.5 1.5B | LAMBADA | Temperature | Input × Gradient | **−0.1408 [−0.1678, −0.1138]** | **3.13 × 10⁻²¹** |
| Qwen2.5 1.5B | IWSLT2017 DE→EN | Temperature | Input × Gradient | −0.0252 [−0.0521, 0.0018] | 0.159 |
| Qwen2.5 3B | LAMBADA | Temperature | Input × Gradient | **−0.1616 [−0.1870, −0.1362]** | **1.45 × 10⁻³²** |
| Qwen2.5 3B | IWSLT2017 DE→EN | Temperature | Input × Gradient | **−0.0883 [−0.1247, −0.0519]** | **1.46 × 10⁻⁶** |
| Gemma-3 1B | LAMBADA | Fisher | Input × Gradient | **−0.1085 [−0.1261, −0.0908]** | **3.71 × 10⁻³⁰** |
| Gemma-3 1B | IWSLT2017 DE→EN | Fisher | Input × Gradient | **−0.0368 [−0.0509, −0.0227]** | **1.12 × 10⁻⁴** |
| Gemma-3 4B | LAMBADA | Fisher | Input × Gradient | **−0.1102 [−0.1265, −0.0938]** | **7.98 × 10⁻³²** |
| Gemma-3 4B | IWSLT2017 DE→EN | Fisher | Integrated Gradients | 0.0120 [−0.0228, 0.0468] | 0.228 |

The reported *p*-values come from two-sided paired Wilcoxon signed-rank tests applied to the passage-level differences. The 12 method pairs were fixed from the original aggregate AOPC table before conducting the paired tests; statistically significant comparisons (*p* < 0.05) are bolded.

Overall, the top-performing Jacobian Scope significantly outperforms the top-performing non-Scope baseline in **10 of 12** model–dataset combinations. No significant difference is detected in the remaining two comparisons, and there are **0 significant baseline wins**. Thus, the paired tests provide strong evidence that the improvements in the original table are reliable across model families, model sizes, and both evaluated datasets.

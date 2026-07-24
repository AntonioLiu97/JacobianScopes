We thank all reviewers for their insightful comments and constructive feedback. In accordance with the suggestions from all reviewers, we have added the following new analyses.

## 1. Paired significance tests

Following the suggestions of reviewers PUGo and D5fu, we performed passage-level paired significance tests for all model–dataset combinations. For each passage, AOPC was reconstructed by trapezoidal integration over the 5%, 10%, and 20% ablation levels; more-negative AOPC is better. Within each model–dataset combination, we compare the best-performing Jacobian Scope (Semantic, Temperature, or Fisher) with the best-performing non-Scope baseline (Input × Gradient or Integrated Gradients), as identified by the mean AOPC values in the main figure. The paired difference is `AOPC(Scope) − AOPC(baseline)`, so a negative value favors the Scope. Its 95% confidence interval was obtained by resampling passages with replacement 10,000 times while preserving the pairing. The reported *p*-value is from a two-sided paired Wilcoxon signed-rank test, with Holm correction across all 72 Scope–baseline comparisons. Statistically significant comparisons (*p* < 0.05) are bolded.

| Model | Dataset | Best Scope | Best baseline | Paired difference [95% CI] | Holm-adjusted *p* |
| --- | --- | --- | --- | ---: | ---: |
| LLaMA-3.2 1B | LAMBADA | Fisher | Input × Gradient | **−0.0358 [−0.0482, −0.0235]** | **2.96 × 10⁻⁵** |
| LLaMA-3.2 1B | IWSLT2017 DE→EN | Temperature | Input × Gradient | **−0.0494 [−0.0689, −0.0298]** | **0.00169** |
| LLaMA-3.2 3B | LAMBADA | Fisher | Input × Gradient | **−0.0551 [−0.0712, −0.0390]** | **3.89 × 10⁻¹⁰** |
| LLaMA-3.2 3B | IWSLT2017 DE→EN | Fisher | Input × Gradient | **−0.0285 [−0.0416, −0.0152]** | **0.00587** |
| Qwen2.5 1.5B | LAMBADA | Temperature | Input × Gradient | **−0.1408 [−0.1680, −0.1138]** | **1.38 × 10⁻¹⁹** |
| Qwen2.5 1.5B | IWSLT2017 DE→EN | Temperature | Input × Gradient | −0.0252 [−0.0514, 0.0020] | 0.797 |
| Qwen2.5 3B | LAMBADA | Temperature | Input × Gradient | **−0.1616 [−0.1868, −0.1366]** | **7.81 × 10⁻³¹** |
| Qwen2.5 3B | IWSLT2017 DE→EN | Temperature | Input × Gradient | **−0.0883 [−0.1242, −0.0522]** | **3.50 × 10⁻⁵** |
| Gemma-3 1B | LAMBADA | Fisher | Input × Gradient | **−0.1085 [−0.1263, −0.0909]** | **1.82 × 10⁻²⁸** |
| Gemma-3 1B | IWSLT2017 DE→EN | Fisher | Input × Gradient | **−0.0368 [−0.0514, −0.0230]** | **0.00191** |
| Gemma-3 4B | LAMBADA | Fisher | Input × Gradient | **−0.1102 [−0.1266, −0.0941]** | **4.09 × 10⁻³⁰** |
| Gemma-3 4B | IWSLT2017 DE→EN | Fisher | Integrated Gradients | 0.0120 [−0.0228, 0.0474] | 0.913 |

Overall, after multiplicity correction, the best Jacobian Scope significantly outperforms the strongest non-Scope baseline in **10 of 12** model–dataset combinations, while the baseline significantly outperforms the best Scope in **0 of 12**. The remaining two comparisons are statistically indistinguishable. Thus, the paired tests yield 10 Scope wins, 2 ties, and **0 baseline wins** across model families, model sizes, and both evaluated datasets.

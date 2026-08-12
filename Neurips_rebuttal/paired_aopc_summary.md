# Paired AOPC analysis

The Scope and baseline for each model–dataset cell are fixed to the methods with the lowest mean AOPC in the original all-passage table. Each comparison uses within-passage differences `AOPC(scope) - AOPC(baseline)`; negative differences favor the Scope. The 95% confidence interval is the standard Student-t interval for the mean paired difference. Reported p-values are from two-sided paired Wilcoxon signed-rank tests.

## all

| Model | Dataset | Top-performing Scope | Top-performing baseline | Mean paired difference | 95% CI | Wilcoxon p |
| --- | --- | --- | --- | --- | --- | --- |
| LLaMA-3.2 1B | LAMBADA | Fisher Scope | Input × Gradient | -0.0358 | [-0.0482, -0.0235] | 1.19e-06 |
| LLaMA-3.2 1B | IWSLT2017 DE→EN | Temperature Scope | Input × Gradient | -0.0494 | [-0.0689, -0.0299] | 9.36e-05 |
| LLaMA-3.2 3B | LAMBADA | Fisher Scope | Input × Gradient | -0.0551 | [-0.0714, -0.0388] | 1.11e-11 |
| LLaMA-3.2 3B | IWSLT2017 DE→EN | Fisher Scope | Input × Gradient | -0.0285 | [-0.0418, -0.0151] | 0.000391 |
| Qwen2.5 1.5B | LAMBADA | Temperature Scope | Input × Gradient | -0.1408 | [-0.1678, -0.1138] | 3.13e-21 |
| Qwen2.5 1.5B | IWSLT2017 DE→EN | Temperature Scope | Input × Gradient | -0.0252 | [-0.0521, 0.0018] | 0.159 |
| Qwen2.5 3B | LAMBADA | Temperature Scope | Input × Gradient | -0.1616 | [-0.1870, -0.1362] | 1.45e-32 |
| Qwen2.5 3B | IWSLT2017 DE→EN | Temperature Scope | Input × Gradient | -0.0883 | [-0.1247, -0.0519] | 1.46e-06 |
| Gemma-3 1B | LAMBADA | Fisher Scope | Input × Gradient | -0.1085 | [-0.1261, -0.0908] | 3.71e-30 |
| Gemma-3 1B | IWSLT2017 DE→EN | Fisher Scope | Input × Gradient | -0.0368 | [-0.0509, -0.0227] | 0.000112 |
| Gemma-3 4B | LAMBADA | Fisher Scope | Input × Gradient | -0.1102 | [-0.1265, -0.0938] | 7.98e-32 |
| Gemma-3 4B | IWSLT2017 DE→EN | Fisher Scope | Integrated Gradients | 0.0120 | [-0.0228, 0.0468] | 0.228 |

## correct_only

| Model | Dataset | Top-performing Scope | Top-performing baseline | Mean paired difference | 95% CI | Wilcoxon p |
| --- | --- | --- | --- | --- | --- | --- |
| LLaMA-3.2 1B | LAMBADA | Fisher Scope | Input × Gradient | -0.0085 | [-0.0216, 0.0045] | 0.237 |
| LLaMA-3.2 1B | IWSLT2017 DE→EN | Temperature Scope | Input × Gradient | -0.0397 | [-0.0633, -0.0160] | 0.0383 |
| LLaMA-3.2 3B | LAMBADA | Fisher Scope | Input × Gradient | -0.0247 | [-0.0419, -0.0075] | 0.00155 |
| LLaMA-3.2 3B | IWSLT2017 DE→EN | Fisher Scope | Input × Gradient | -0.0307 | [-0.0448, -0.0166] | 0.000108 |
| Qwen2.5 1.5B | LAMBADA | Temperature Scope | Input × Gradient | -0.1013 | [-0.1315, -0.0710] | 1.57e-09 |
| Qwen2.5 1.5B | IWSLT2017 DE→EN | Temperature Scope | Input × Gradient | -0.0254 | [-0.0566, 0.0057] | 0.252 |
| Qwen2.5 3B | LAMBADA | Temperature Scope | Input × Gradient | -0.1482 | [-0.1774, -0.1191] | 3.57e-21 |
| Qwen2.5 3B | IWSLT2017 DE→EN | Temperature Scope | Input × Gradient | -0.0899 | [-0.1340, -0.0458] | 3.94e-05 |
| Gemma-3 1B | LAMBADA | Fisher Scope | Input × Gradient | -0.0687 | [-0.0879, -0.0494] | 7.16e-12 |
| Gemma-3 1B | IWSLT2017 DE→EN | Fisher Scope | Input × Gradient | -0.0366 | [-0.0516, -0.0216] | 7.74e-05 |
| Gemma-3 4B | LAMBADA | Fisher Scope | Input × Gradient | -0.0766 | [-0.0936, -0.0596] | 7.28e-14 |
| Gemma-3 4B | IWSLT2017 DE→EN | Fisher Scope | Integrated Gradients | -0.0084 | [-0.0506, 0.0337] | 0.816 |

## Summary

On all passages, the top-performing Scope significantly beats the top-performing baseline in **10/12** model–dataset combinations. On correctly predicted passages, it does so in **9/12** combinations. There are **0** significant baseline wins on all passages and **0** on the correct-only subset.

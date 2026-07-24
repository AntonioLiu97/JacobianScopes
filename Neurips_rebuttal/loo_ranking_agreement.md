# Full-ranking agreement with LOO-KL interventions

Agreement is computed independently for each passage between the complete token ranking from an attribution method and the ranking induced by single-token LOO KL divergence. Values are prompt-level means with paired bootstrap 95% confidence intervals. Higher is better.

## all

### LLaMA-3.2 1B

| Method | n | Spearman ρ [95% CI] | Kendall τ-b [95% CI] | Top-10% Jaccard [95% CI] |
| --- | ---: | ---: | ---: | ---: |
| Temperature Scope | 1000 | 0.550 [0.542, 0.558] | 0.397 [0.390, 0.403] | 0.395 [0.386, 0.405] |
| Semantic Scope | 1000 | 0.562 [0.554, 0.571] | 0.406 [0.400, 0.413] | 0.388 [0.378, 0.398] |
| Input × Gradient | 1000 | 0.556 [0.547, 0.564] | 0.401 [0.394, 0.407] | 0.373 [0.364, 0.383] |
| Fisher Scope (k=4) | 1000 | 0.590 [0.582, 0.597] | 0.430 [0.423, 0.436] | 0.427 [0.418, 0.437] |
| Integrated Gradients | 1000 | 0.300 [0.290, 0.309] | 0.209 [0.202, 0.215] | 0.275 [0.266, 0.283] |
| Random | 1000 | 0.005 [-0.002, 0.012] | 0.003 [-0.002, 0.008] | 0.060 [0.057, 0.064] |

### LLaMA-3.2 3B

| Method | n | Spearman ρ [95% CI] | Kendall τ-b [95% CI] | Top-10% Jaccard [95% CI] |
| --- | ---: | ---: | ---: | ---: |
| Temperature Scope | 1000 | 0.528 [0.519, 0.537] | 0.379 [0.372, 0.386] | 0.313 [0.304, 0.323] |
| Semantic Scope | 1000 | 0.552 [0.542, 0.561] | 0.398 [0.390, 0.405] | 0.321 [0.311, 0.330] |
| Input × Gradient | 1000 | 0.535 [0.526, 0.545] | 0.384 [0.377, 0.391] | 0.297 [0.288, 0.306] |
| Fisher Scope (k=4) | 1000 | 0.575 [0.565, 0.584] | 0.417 [0.409, 0.424] | 0.345 [0.336, 0.355] |
| Integrated Gradients | 1000 | 0.321 [0.311, 0.332] | 0.224 [0.217, 0.232] | 0.178 [0.171, 0.185] |
| Random | 1000 | -0.004 [-0.011, 0.003] | -0.002 [-0.007, 0.003] | 0.057 [0.053, 0.061] |

## correct_only

### LLaMA-3.2 1B

| Method | n | Spearman ρ [95% CI] | Kendall τ-b [95% CI] | Top-10% Jaccard [95% CI] |
| --- | ---: | ---: | ---: | ---: |
| Temperature Scope | 699 | 0.535 [0.525, 0.545] | 0.385 [0.377, 0.392] | 0.390 [0.379, 0.402] |
| Semantic Scope | 699 | 0.549 [0.539, 0.558] | 0.396 [0.388, 0.403] | 0.389 [0.378, 0.400] |
| Input × Gradient | 699 | 0.543 [0.533, 0.552] | 0.390 [0.383, 0.398] | 0.373 [0.362, 0.383] |
| Fisher Scope (k=4) | 699 | 0.570 [0.560, 0.579] | 0.413 [0.405, 0.421] | 0.414 [0.403, 0.425] |
| Integrated Gradients | 699 | 0.304 [0.294, 0.315] | 0.212 [0.204, 0.220] | 0.287 [0.277, 0.297] |
| Random | 699 | 0.005 [-0.003, 0.014] | 0.004 [-0.002, 0.009] | 0.058 [0.054, 0.063] |

### LLaMA-3.2 3B

| Method | n | Spearman ρ [95% CI] | Kendall τ-b [95% CI] | Top-10% Jaccard [95% CI] |
| --- | ---: | ---: | ---: | ---: |
| Temperature Scope | 775 | 0.518 [0.508, 0.529] | 0.372 [0.364, 0.380] | 0.316 [0.306, 0.327] |
| Semantic Scope | 775 | 0.543 [0.532, 0.553] | 0.391 [0.383, 0.399] | 0.326 [0.316, 0.336] |
| Input × Gradient | 775 | 0.527 [0.516, 0.538] | 0.378 [0.369, 0.386] | 0.303 [0.293, 0.313] |
| Fisher Scope (k=4) | 775 | 0.562 [0.551, 0.572] | 0.407 [0.398, 0.415] | 0.341 [0.331, 0.352] |
| Integrated Gradients | 775 | 0.324 [0.312, 0.336] | 0.226 [0.218, 0.235] | 0.187 [0.179, 0.195] |
| Random | 775 | -0.005 [-0.013, 0.003] | -0.003 [-0.009, 0.002] | 0.056 [0.052, 0.061] |

This is an interventional ranking-agreement analysis. It complements, but does not replace, parameter-randomization sanity checks.

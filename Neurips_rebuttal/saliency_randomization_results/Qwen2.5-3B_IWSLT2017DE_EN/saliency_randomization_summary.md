# Parameter-randomization saliency sanity check

Qwen2.5 3B has 36 transformer blocks. Blocks were reinitialized cumulatively from the output side. Values are mean ± SEM across fixed, correctly predicted IWSLT2017 DE→EN prompts; lower similarity indicates greater dependence on learned parameters.

| Checkpoint | Method | Spearman ρ | Top-10% Jaccard | Random ρ floor | Random Jaccard floor |
| --- | --- | ---: | ---: | ---: | ---: |
| final_quarter | Semantic | 0.808 ± 0.006 | 0.502 ± 0.012 | 0.001 | 0.060 |
| final_quarter | Temperature | 0.840 ± 0.005 | 0.567 ± 0.012 | 0.001 | 0.056 |
| final_quarter | Fisher | 0.812 ± 0.006 | 0.501 ± 0.013 | -0.003 | 0.059 |
| final_quarter | InputXGradient | NA | NA | NA | NA |
| final_quarter | IG | NA | NA | NA | NA |
| final_half | Semantic | 0.637 ± 0.010 | 0.346 ± 0.010 | -0.004 | 0.058 |
| final_half | Temperature | 0.688 ± 0.008 | 0.410 ± 0.011 | 0.000 | 0.060 |
| final_half | Fisher | 0.644 ± 0.010 | 0.387 ± 0.012 | -0.000 | 0.059 |
| final_half | InputXGradient | NA | NA | NA | NA |
| final_half | IG | NA | NA | NA | NA |
| all_blocks | Semantic | 0.188 ± 0.010 | 0.197 ± 0.005 | 0.002 | 0.059 |
| all_blocks | Temperature | 0.159 ± 0.010 | 0.211 ± 0.005 | -0.000 | 0.058 |
| all_blocks | Fisher | 0.149 ± 0.011 | 0.183 ± 0.005 | -0.001 | 0.059 |
| all_blocks | InputXGradient | NA | NA | NA | NA |
| all_blocks | IG | NA | NA | NA | NA |

Protocol: Adebayo et al. (2018), with text-ranking metrics following Kokhlikyan et al. (2021). The target token is held fixed even if the randomized model's prediction changes.

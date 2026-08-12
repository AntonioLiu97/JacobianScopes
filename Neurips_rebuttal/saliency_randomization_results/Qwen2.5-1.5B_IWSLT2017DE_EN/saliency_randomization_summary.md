# Parameter-randomization saliency sanity check

Qwen2.5 1.5B has 28 transformer blocks. Blocks were reinitialized cumulatively from the output side. Values are mean ± SEM across fixed, correctly predicted IWSLT2017 DE→EN prompts; lower similarity indicates greater dependence on learned parameters.

| Checkpoint | Method | Spearman ρ | Top-10% Jaccard | Random ρ floor | Random Jaccard floor |
| --- | --- | ---: | ---: | ---: | ---: |
| final_quarter | Semantic | 0.827 ± 0.006 | 0.533 ± 0.013 | 0.001 | 0.061 |
| final_quarter | Temperature | 0.853 ± 0.005 | 0.568 ± 0.012 | 0.002 | 0.060 |
| final_quarter | Fisher | 0.842 ± 0.005 | 0.521 ± 0.012 | -0.001 | 0.059 |
| final_quarter | InputXGradient | NA | NA | NA | NA |
| final_quarter | IG | NA | NA | NA | NA |
| final_half | Semantic | 0.689 ± 0.009 | 0.381 ± 0.012 | -0.001 | 0.060 |
| final_half | Temperature | 0.723 ± 0.009 | 0.437 ± 0.012 | -0.000 | 0.058 |
| final_half | Fisher | 0.702 ± 0.009 | 0.376 ± 0.012 | 0.002 | 0.058 |
| final_half | InputXGradient | NA | NA | NA | NA |
| final_half | IG | NA | NA | NA | NA |
| all_blocks | Semantic | 0.142 ± 0.012 | 0.181 ± 0.006 | 0.002 | 0.058 |
| all_blocks | Temperature | 0.134 ± 0.011 | 0.204 ± 0.006 | -0.002 | 0.059 |
| all_blocks | Fisher | 0.119 ± 0.013 | 0.163 ± 0.006 | -0.001 | 0.059 |
| all_blocks | InputXGradient | NA | NA | NA | NA |
| all_blocks | IG | NA | NA | NA | NA |

Protocol: Adebayo et al. (2018), with text-ranking metrics following Kokhlikyan et al. (2021). The target token is held fixed even if the randomized model's prediction changes.

# Parameter-randomization saliency sanity check

LLaMA-3.2 3B has 28 transformer blocks. Blocks were reinitialized cumulatively from the output side. Values are mean ± SEM across fixed, correctly predicted LAMBADA prompts; lower similarity indicates greater dependence on learned parameters.

| Checkpoint | Method | Spearman ρ | Top-10% Jaccard | Random ρ floor | Random Jaccard floor |
| --- | --- | ---: | ---: | ---: | ---: |
| final_quarter | Semantic | 0.704 ± 0.011 | 0.348 ± 0.014 | 0.001 | 0.060 |
| final_quarter | Temperature | 0.737 ± 0.008 | 0.374 ± 0.013 | 0.001 | 0.058 |
| final_quarter | Fisher | 0.683 ± 0.010 | 0.336 ± 0.014 | -0.001 | 0.061 |
| final_quarter | InputXGradient | NA | NA | NA | NA |
| final_quarter | IG | NA | NA | NA | NA |
| final_half | Semantic | 0.629 ± 0.012 | 0.283 ± 0.014 | 0.001 | 0.060 |
| final_half | Temperature | 0.669 ± 0.010 | 0.304 ± 0.012 | -0.001 | 0.058 |
| final_half | Fisher | 0.621 ± 0.012 | 0.267 ± 0.013 | -0.000 | 0.059 |
| final_half | InputXGradient | NA | NA | NA | NA |
| final_half | IG | NA | NA | NA | NA |
| all_blocks | Semantic | 0.156 ± 0.013 | 0.134 ± 0.008 | 0.000 | 0.058 |
| all_blocks | Temperature | 0.181 ± 0.011 | 0.150 ± 0.008 | 0.004 | 0.061 |
| all_blocks | Fisher | 0.148 ± 0.013 | 0.136 ± 0.008 | -0.002 | 0.059 |
| all_blocks | InputXGradient | NA | NA | NA | NA |
| all_blocks | IG | NA | NA | NA | NA |

Protocol: Adebayo et al. (2018), with text-ranking metrics following Kokhlikyan et al. (2021). The target token is held fixed even if the randomized model's prediction changes.

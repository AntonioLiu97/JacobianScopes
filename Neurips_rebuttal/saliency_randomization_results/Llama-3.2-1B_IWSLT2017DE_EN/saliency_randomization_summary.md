# Parameter-randomization saliency sanity check

LLaMA-3.2 1B has 16 transformer blocks. Blocks were reinitialized cumulatively from the output side. Values are mean ± SEM across fixed, correctly predicted IWSLT2017 DE→EN prompts; lower similarity indicates greater dependence on learned parameters.

| Checkpoint | Method | Spearman ρ | Top-10% Jaccard | Random ρ floor | Random Jaccard floor |
| --- | --- | ---: | ---: | ---: | ---: |
| final_quarter | Semantic | 0.606 ± 0.011 | 0.236 ± 0.010 | 0.000 | 0.060 |
| final_quarter | Temperature | 0.608 ± 0.011 | 0.231 ± 0.008 | 0.002 | 0.059 |
| final_quarter | Fisher | 0.570 ± 0.013 | 0.197 ± 0.008 | 0.002 | 0.059 |
| final_quarter | InputXGradient | NA | NA | NA | NA |
| final_quarter | IG | NA | NA | NA | NA |
| final_half | Semantic | 0.473 ± 0.012 | 0.138 ± 0.007 | -0.003 | 0.058 |
| final_half | Temperature | 0.503 ± 0.011 | 0.139 ± 0.007 | 0.000 | 0.061 |
| final_half | Fisher | 0.449 ± 0.013 | 0.112 ± 0.006 | -0.002 | 0.059 |
| final_half | InputXGradient | NA | NA | NA | NA |
| final_half | IG | NA | NA | NA | NA |
| all_blocks | Semantic | 0.090 ± 0.012 | 0.108 ± 0.004 | 0.000 | 0.060 |
| all_blocks | Temperature | 0.106 ± 0.011 | 0.112 ± 0.005 | -0.001 | 0.059 |
| all_blocks | Fisher | 0.070 ± 0.013 | 0.093 ± 0.004 | 0.000 | 0.058 |
| all_blocks | InputXGradient | NA | NA | NA | NA |
| all_blocks | IG | NA | NA | NA | NA |

Protocol: Adebayo et al. (2018), with text-ranking metrics following Kokhlikyan et al. (2021). The target token is held fixed even if the randomized model's prediction changes.

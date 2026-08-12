# Parameter-randomization saliency sanity check

LLaMA-3.2 3B has 28 transformer blocks. Blocks were reinitialized cumulatively from the output side. Values are mean ± SEM across fixed, correctly predicted IWSLT2017 DE→EN prompts; lower similarity indicates greater dependence on learned parameters.

| Checkpoint | Method | Spearman ρ | Top-10% Jaccard | Random ρ floor | Random Jaccard floor |
| --- | --- | ---: | ---: | ---: | ---: |
| final_quarter | Semantic | 0.709 ± 0.009 | 0.315 ± 0.010 | 0.000 | 0.059 |
| final_quarter | Temperature | 0.681 ± 0.010 | 0.329 ± 0.011 | 0.000 | 0.059 |
| final_quarter | Fisher | 0.668 ± 0.010 | 0.299 ± 0.011 | 0.001 | 0.059 |
| final_quarter | InputXGradient | NA | NA | NA | NA |
| final_quarter | IG | NA | NA | NA | NA |
| final_half | Semantic | 0.623 ± 0.010 | 0.235 ± 0.008 | -0.002 | 0.059 |
| final_half | Temperature | 0.604 ± 0.012 | 0.266 ± 0.009 | 0.001 | 0.059 |
| final_half | Fisher | 0.583 ± 0.012 | 0.210 ± 0.009 | -0.000 | 0.059 |
| final_half | InputXGradient | NA | NA | NA | NA |
| final_half | IG | NA | NA | NA | NA |
| all_blocks | Semantic | 0.152 ± 0.010 | 0.168 ± 0.006 | -0.000 | 0.060 |
| all_blocks | Temperature | 0.145 ± 0.010 | 0.181 ± 0.006 | 0.001 | 0.058 |
| all_blocks | Fisher | 0.135 ± 0.010 | 0.149 ± 0.006 | -0.001 | 0.058 |
| all_blocks | InputXGradient | NA | NA | NA | NA |
| all_blocks | IG | NA | NA | NA | NA |

Protocol: Adebayo et al. (2018), with text-ranking metrics following Kokhlikyan et al. (2021). The target token is held fixed even if the randomized model's prediction changes.

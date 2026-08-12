# Parameter-randomization saliency sanity check

Qwen2.5 3B has 36 transformer blocks. Blocks were reinitialized cumulatively from the output side. Values are mean ± SEM across fixed, correctly predicted LAMBADA prompts; lower similarity indicates greater dependence on learned parameters.

| Checkpoint | Method | Spearman ρ | Top-10% Jaccard | Random ρ floor | Random Jaccard floor |
| --- | --- | ---: | ---: | ---: | ---: |
| final_quarter | Semantic | 0.819 ± 0.008 | 0.443 ± 0.014 | 0.002 | 0.059 |
| final_quarter | Temperature | 0.847 ± 0.006 | 0.525 ± 0.012 | -0.001 | 0.060 |
| final_quarter | Fisher | 0.828 ± 0.007 | 0.470 ± 0.014 | -0.001 | 0.058 |
| final_quarter | InputXGradient | NA | NA | NA | NA |
| final_quarter | IG | NA | NA | NA | NA |
| final_half | Semantic | 0.749 ± 0.009 | 0.386 ± 0.015 | -0.001 | 0.059 |
| final_half | Temperature | 0.773 ± 0.008 | 0.446 ± 0.012 | 0.002 | 0.059 |
| final_half | Fisher | 0.760 ± 0.009 | 0.397 ± 0.014 | -0.000 | 0.059 |
| final_half | InputXGradient | NA | NA | NA | NA |
| final_half | IG | NA | NA | NA | NA |
| all_blocks | Semantic | 0.234 ± 0.013 | 0.172 ± 0.009 | -0.001 | 0.059 |
| all_blocks | Temperature | 0.234 ± 0.014 | 0.187 ± 0.008 | 0.000 | 0.059 |
| all_blocks | Fisher | 0.213 ± 0.013 | 0.156 ± 0.009 | -0.001 | 0.059 |
| all_blocks | InputXGradient | NA | NA | NA | NA |
| all_blocks | IG | NA | NA | NA | NA |

Protocol: Adebayo et al. (2018), with text-ranking metrics following Kokhlikyan et al. (2021). The target token is held fixed even if the randomized model's prediction changes.

# Parameter-randomization saliency sanity check

LLaMA-3.2 1B has 16 transformer blocks. Blocks were reinitialized cumulatively from the output side. Values are mean ± SEM across fixed, correctly predicted LAMBADA prompts; lower similarity indicates greater dependence on learned parameters.

| Checkpoint | Method | Spearman ρ | Top-10% Jaccard | Random ρ floor | Random Jaccard floor |
| --- | --- | ---: | ---: | ---: | ---: |
| final_quarter | Semantic | 0.627 ± 0.012 | 0.285 ± 0.012 | 0.000 | 0.059 |
| final_quarter | Temperature | 0.650 ± 0.010 | 0.288 ± 0.011 | -0.000 | 0.059 |
| final_quarter | Fisher | 0.609 ± 0.012 | 0.272 ± 0.013 | -0.001 | 0.058 |
| final_quarter | InputXGradient | NA | NA | NA | NA |
| final_quarter | IG | NA | NA | NA | NA |
| final_half | Semantic | 0.504 ± 0.014 | 0.190 ± 0.010 | 0.001 | 0.059 |
| final_half | Temperature | 0.535 ± 0.012 | 0.194 ± 0.010 | -0.002 | 0.059 |
| final_half | Fisher | 0.492 ± 0.014 | 0.180 ± 0.010 | 0.003 | 0.060 |
| final_half | InputXGradient | NA | NA | NA | NA |
| final_half | IG | NA | NA | NA | NA |
| all_blocks | Semantic | 0.143 ± 0.016 | 0.132 ± 0.009 | 0.001 | 0.058 |
| all_blocks | Temperature | 0.152 ± 0.015 | 0.133 ± 0.008 | 0.003 | 0.058 |
| all_blocks | Fisher | 0.128 ± 0.017 | 0.125 ± 0.009 | 0.000 | 0.059 |
| all_blocks | InputXGradient | NA | NA | NA | NA |
| all_blocks | IG | NA | NA | NA | NA |

Protocol: Adebayo et al. (2018), with text-ranking metrics following Kokhlikyan et al. (2021). The target token is held fixed even if the randomized model's prediction changes.

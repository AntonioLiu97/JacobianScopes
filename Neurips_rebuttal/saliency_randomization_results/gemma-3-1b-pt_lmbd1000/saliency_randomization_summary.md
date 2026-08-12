# Parameter-randomization saliency sanity check

Gemma-3 1B has 26 transformer blocks. Blocks were reinitialized cumulatively from the output side. Values are mean ± SEM across fixed, correctly predicted LAMBADA prompts; lower similarity indicates greater dependence on learned parameters.

| Checkpoint | Method | Spearman ρ | Top-10% Jaccard | Random ρ floor | Random Jaccard floor |
| --- | --- | ---: | ---: | ---: | ---: |
| final_quarter | Semantic | 0.880 ± 0.006 | 0.587 ± 0.013 | 0.001 | 0.060 |
| final_quarter | Temperature | 0.823 ± 0.007 | 0.545 ± 0.014 | 0.001 | 0.060 |
| final_quarter | Fisher | 0.837 ± 0.007 | 0.557 ± 0.014 | 0.001 | 0.059 |
| final_half | Semantic | 0.805 ± 0.007 | 0.495 ± 0.013 | -0.004 | 0.058 |
| final_half | Temperature | 0.807 ± 0.007 | 0.470 ± 0.013 | 0.002 | 0.059 |
| final_half | Fisher | 0.794 ± 0.008 | 0.508 ± 0.014 | -0.001 | 0.059 |
| all_blocks | Semantic | 0.339 ± 0.014 | 0.257 ± 0.009 | -0.002 | 0.059 |
| all_blocks | Temperature | 0.356 ± 0.013 | 0.241 ± 0.009 | -0.003 | 0.060 |
| all_blocks | Fisher | 0.329 ± 0.014 | 0.251 ± 0.009 | -0.000 | 0.057 |

Protocol: Adebayo et al. (2018), with text-ranking metrics following Kokhlikyan et al. (2021). The target token is held fixed even if the randomized model's prediction changes.

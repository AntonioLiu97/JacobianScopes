# Parameter-randomization saliency sanity check

Gemma-3 1B has 26 transformer blocks. Blocks were reinitialized cumulatively from the output side. Values are mean ± SEM across fixed, correctly predicted IWSLT2017 DE→EN prompts; lower similarity indicates greater dependence on learned parameters.

| Checkpoint | Method | Spearman ρ | Top-10% Jaccard | Random ρ floor | Random Jaccard floor |
| --- | --- | ---: | ---: | ---: | ---: |
| final_quarter | Semantic | 0.859 ± 0.005 | 0.586 ± 0.014 | -0.002 | 0.058 |
| final_quarter | Temperature | 0.826 ± 0.006 | 0.555 ± 0.015 | 0.001 | 0.061 |
| final_quarter | Fisher | 0.829 ± 0.006 | 0.552 ± 0.014 | 0.000 | 0.059 |
| final_half | Semantic | 0.779 ± 0.007 | 0.492 ± 0.013 | -0.000 | 0.059 |
| final_half | Temperature | 0.762 ± 0.009 | 0.471 ± 0.013 | -0.002 | 0.059 |
| final_half | Fisher | 0.785 ± 0.007 | 0.490 ± 0.013 | 0.002 | 0.060 |
| all_blocks | Semantic | 0.273 ± 0.014 | 0.285 ± 0.009 | 0.002 | 0.058 |
| all_blocks | Temperature | 0.271 ± 0.015 | 0.283 ± 0.009 | -0.001 | 0.059 |
| all_blocks | Fisher | 0.275 ± 0.014 | 0.283 ± 0.010 | -0.003 | 0.058 |

Protocol: Adebayo et al. (2018), with text-ranking metrics following Kokhlikyan et al. (2021). The target token is held fixed even if the randomized model's prediction changes.

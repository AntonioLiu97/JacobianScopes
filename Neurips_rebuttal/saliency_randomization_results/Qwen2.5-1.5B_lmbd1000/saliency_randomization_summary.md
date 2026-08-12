# Parameter-randomization saliency sanity check

Qwen2.5 1.5B has 28 transformer blocks. Blocks were reinitialized cumulatively from the output side. Values are mean ± SEM across fixed, correctly predicted LAMBADA prompts; lower similarity indicates greater dependence on learned parameters.

| Checkpoint | Method | Spearman ρ | Top-10% Jaccard | Random ρ floor | Random Jaccard floor |
| --- | --- | ---: | ---: | ---: | ---: |
| final_quarter | Semantic | 0.862 ± 0.006 | 0.504 ± 0.013 | 0.000 | 0.060 |
| final_quarter | Temperature | 0.866 ± 0.005 | 0.555 ± 0.013 | 0.001 | 0.059 |
| final_quarter | Fisher | 0.857 ± 0.006 | 0.519 ± 0.014 | 0.003 | 0.059 |
| final_half | Semantic | 0.776 ± 0.008 | 0.399 ± 0.014 | -0.002 | 0.060 |
| final_half | Temperature | 0.796 ± 0.009 | 0.446 ± 0.014 | -0.002 | 0.059 |
| final_half | Fisher | 0.774 ± 0.009 | 0.413 ± 0.015 | 0.002 | 0.060 |
| all_blocks | Semantic | 0.202 ± 0.017 | 0.164 ± 0.009 | 0.001 | 0.059 |
| all_blocks | Temperature | 0.196 ± 0.016 | 0.173 ± 0.009 | 0.003 | 0.060 |
| all_blocks | Fisher | 0.182 ± 0.017 | 0.156 ± 0.009 | 0.001 | 0.058 |

Protocol: Adebayo et al. (2018), with text-ranking metrics following Kokhlikyan et al. (2021). The target token is held fixed even if the randomized model's prediction changes.

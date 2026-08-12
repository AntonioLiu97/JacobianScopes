# Parameter-randomization saliency sanity checks

Each table compares an attribution map with its original-model map as transformer blocks are cumulatively reinitialized from the output side. Values are mean ± SEM across fixed, correctly predicted prompts; lower similarity indicates greater dependence on learned parameters. `NA` marks an unfinished rerun.

## LLaMA-3.2 1B — LAMBADA (*n* = 200)

| Scope | Similarity | Original | Final quarter | Final half | All blocks |
| --- | --- | ---: | ---: | ---: | ---: |
| Semantic | Spearman ρ | 1.000 | 0.627 ± 0.012 | 0.504 ± 0.014 | 0.143 ± 0.016 |
| Semantic | Top-10% Jaccard | 1.000 | 0.285 ± 0.012 | 0.190 ± 0.010 | 0.132 ± 0.009 |
| Temperature | Spearman ρ | 1.000 | 0.650 ± 0.010 | 0.535 ± 0.012 | 0.152 ± 0.015 |
| Temperature | Top-10% Jaccard | 1.000 | 0.288 ± 0.011 | 0.194 ± 0.010 | 0.133 ± 0.008 |
| Fisher | Spearman ρ | 1.000 | 0.609 ± 0.012 | 0.492 ± 0.014 | 0.128 ± 0.017 |
| Fisher | Top-10% Jaccard | 1.000 | 0.272 ± 0.013 | 0.180 ± 0.010 | 0.125 ± 0.009 |

## LLaMA-3.2 1B — IWSLT2017 DE→EN (*n* = 200)

| Scope | Similarity | Original | Final quarter | Final half | All blocks |
| --- | --- | ---: | ---: | ---: | ---: |
| Semantic | Spearman ρ | 1.000 | 0.606 ± 0.011 | 0.473 ± 0.012 | 0.090 ± 0.012 |
| Semantic | Top-10% Jaccard | 1.000 | 0.236 ± 0.010 | 0.138 ± 0.007 | 0.108 ± 0.004 |
| Temperature | Spearman ρ | 1.000 | 0.608 ± 0.011 | 0.503 ± 0.011 | 0.106 ± 0.011 |
| Temperature | Top-10% Jaccard | 1.000 | 0.231 ± 0.008 | 0.139 ± 0.007 | 0.112 ± 0.005 |
| Fisher | Spearman ρ | 1.000 | 0.570 ± 0.013 | 0.449 ± 0.013 | 0.070 ± 0.013 |
| Fisher | Top-10% Jaccard | 1.000 | 0.197 ± 0.008 | 0.112 ± 0.006 | 0.093 ± 0.004 |

## LLaMA-3.2 3B — LAMBADA

| Scope | Similarity | Original | Final quarter | Final half | All blocks |
| --- | --- | ---: | ---: | ---: | ---: |
| Semantic | Spearman ρ | 1.000 | NA | NA | NA |
| Semantic | Top-10% Jaccard | 1.000 | NA | NA | NA |
| Temperature | Spearman ρ | 1.000 | NA | NA | NA |
| Temperature | Top-10% Jaccard | 1.000 | NA | NA | NA |
| Fisher | Spearman ρ | 1.000 | NA | NA | NA |
| Fisher | Top-10% Jaccard | 1.000 | NA | NA | NA |

## LLaMA-3.2 3B — IWSLT2017 DE→EN

| Scope | Similarity | Original | Final quarter | Final half | All blocks |
| --- | --- | ---: | ---: | ---: | ---: |
| Semantic | Spearman ρ | 1.000 | NA | NA | NA |
| Semantic | Top-10% Jaccard | 1.000 | NA | NA | NA |
| Temperature | Spearman ρ | 1.000 | NA | NA | NA |
| Temperature | Top-10% Jaccard | 1.000 | NA | NA | NA |
| Fisher | Spearman ρ | 1.000 | NA | NA | NA |
| Fisher | Top-10% Jaccard | 1.000 | NA | NA | NA |

## Qwen2.5 1.5B — LAMBADA (*n* = 200)

| Scope | Similarity | Original | Final quarter | Final half | All blocks |
| --- | --- | ---: | ---: | ---: | ---: |
| Semantic | Spearman ρ | 1.000 | 0.862 ± 0.006 | 0.776 ± 0.008 | 0.202 ± 0.017 |
| Semantic | Top-10% Jaccard | 1.000 | 0.504 ± 0.013 | 0.399 ± 0.014 | 0.164 ± 0.009 |
| Temperature | Spearman ρ | 1.000 | 0.866 ± 0.005 | 0.796 ± 0.009 | 0.196 ± 0.016 |
| Temperature | Top-10% Jaccard | 1.000 | 0.555 ± 0.013 | 0.446 ± 0.014 | 0.173 ± 0.009 |
| Fisher | Spearman ρ | 1.000 | 0.857 ± 0.006 | 0.774 ± 0.009 | 0.182 ± 0.017 |
| Fisher | Top-10% Jaccard | 1.000 | 0.519 ± 0.014 | 0.413 ± 0.015 | 0.156 ± 0.009 |

## Qwen2.5 1.5B — IWSLT2017 DE→EN (*n* = 200)

| Scope | Similarity | Original | Final quarter | Final half | All blocks |
| --- | --- | ---: | ---: | ---: | ---: |
| Semantic | Spearman ρ | 1.000 | 0.827 ± 0.006 | 0.689 ± 0.009 | 0.142 ± 0.012 |
| Semantic | Top-10% Jaccard | 1.000 | 0.533 ± 0.013 | 0.381 ± 0.012 | 0.181 ± 0.006 |
| Temperature | Spearman ρ | 1.000 | 0.853 ± 0.005 | 0.723 ± 0.009 | 0.134 ± 0.011 |
| Temperature | Top-10% Jaccard | 1.000 | 0.568 ± 0.012 | 0.437 ± 0.012 | 0.204 ± 0.006 |
| Fisher | Spearman ρ | 1.000 | 0.842 ± 0.005 | 0.702 ± 0.009 | 0.119 ± 0.013 |
| Fisher | Top-10% Jaccard | 1.000 | 0.521 ± 0.012 | 0.376 ± 0.012 | 0.163 ± 0.006 |

## Qwen2.5 3B — LAMBADA

| Scope | Similarity | Original | Final quarter | Final half | All blocks |
| --- | --- | ---: | ---: | ---: | ---: |
| Semantic | Spearman ρ | 1.000 | NA | NA | NA |
| Semantic | Top-10% Jaccard | 1.000 | NA | NA | NA |
| Temperature | Spearman ρ | 1.000 | NA | NA | NA |
| Temperature | Top-10% Jaccard | 1.000 | NA | NA | NA |
| Fisher | Spearman ρ | 1.000 | NA | NA | NA |
| Fisher | Top-10% Jaccard | 1.000 | NA | NA | NA |

## Qwen2.5 3B — IWSLT2017 DE→EN

| Scope | Similarity | Original | Final quarter | Final half | All blocks |
| --- | --- | ---: | ---: | ---: | ---: |
| Semantic | Spearman ρ | 1.000 | NA | NA | NA |
| Semantic | Top-10% Jaccard | 1.000 | NA | NA | NA |
| Temperature | Spearman ρ | 1.000 | NA | NA | NA |
| Temperature | Top-10% Jaccard | 1.000 | NA | NA | NA |
| Fisher | Spearman ρ | 1.000 | NA | NA | NA |
| Fisher | Top-10% Jaccard | 1.000 | NA | NA | NA |

## Gemma-3 1B — LAMBADA (*n* = 200)

| Scope | Similarity | Original | Final quarter | Final half | All blocks |
| --- | --- | ---: | ---: | ---: | ---: |
| Semantic | Spearman ρ | 1.000 | 0.880 ± 0.006 | 0.805 ± 0.007 | 0.339 ± 0.014 |
| Semantic | Top-10% Jaccard | 1.000 | 0.587 ± 0.013 | 0.495 ± 0.013 | 0.257 ± 0.009 |
| Temperature | Spearman ρ | 1.000 | 0.823 ± 0.007 | 0.807 ± 0.007 | 0.356 ± 0.013 |
| Temperature | Top-10% Jaccard | 1.000 | 0.545 ± 0.014 | 0.470 ± 0.013 | 0.241 ± 0.009 |
| Fisher | Spearman ρ | 1.000 | 0.837 ± 0.007 | 0.794 ± 0.008 | 0.329 ± 0.014 |
| Fisher | Top-10% Jaccard | 1.000 | 0.557 ± 0.014 | 0.508 ± 0.014 | 0.251 ± 0.009 |

## Gemma-3 1B — IWSLT2017 DE→EN (*n* = 200)

| Scope | Similarity | Original | Final quarter | Final half | All blocks |
| --- | --- | ---: | ---: | ---: | ---: |
| Semantic | Spearman ρ | 1.000 | 0.859 ± 0.005 | 0.779 ± 0.007 | 0.273 ± 0.014 |
| Semantic | Top-10% Jaccard | 1.000 | 0.586 ± 0.014 | 0.492 ± 0.013 | 0.285 ± 0.009 |
| Temperature | Spearman ρ | 1.000 | 0.826 ± 0.006 | 0.762 ± 0.009 | 0.271 ± 0.015 |
| Temperature | Top-10% Jaccard | 1.000 | 0.555 ± 0.015 | 0.471 ± 0.013 | 0.283 ± 0.009 |
| Fisher | Spearman ρ | 1.000 | 0.829 ± 0.006 | 0.785 ± 0.007 | 0.275 ± 0.014 |
| Fisher | Top-10% Jaccard | 1.000 | 0.552 ± 0.014 | 0.490 ± 0.013 | 0.283 ± 0.010 |

## Gemma-3 4B — LAMBADA

| Scope | Similarity | Original | Final quarter | Final half | All blocks |
| --- | --- | ---: | ---: | ---: | ---: |
| Semantic | Spearman ρ | 1.000 | NA | NA | NA |
| Semantic | Top-10% Jaccard | 1.000 | NA | NA | NA |
| Temperature | Spearman ρ | 1.000 | NA | NA | NA |
| Temperature | Top-10% Jaccard | 1.000 | NA | NA | NA |
| Fisher | Spearman ρ | 1.000 | NA | NA | NA |
| Fisher | Top-10% Jaccard | 1.000 | NA | NA | NA |

## Gemma-3 4B — IWSLT2017 DE→EN

| Scope | Similarity | Original | Final quarter | Final half | All blocks |
| --- | --- | ---: | ---: | ---: | ---: |
| Semantic | Spearman ρ | 1.000 | NA | NA | NA |
| Semantic | Top-10% Jaccard | 1.000 | NA | NA | NA |
| Temperature | Spearman ρ | 1.000 | NA | NA | NA |
| Temperature | Top-10% Jaccard | 1.000 | NA | NA | NA |
| Fisher | Spearman ρ | 1.000 | NA | NA | NA |
| Fisher | Top-10% Jaccard | 1.000 | NA | NA | NA |

Protocol: Adebayo et al. (2018), with text-ranking metrics following Kokhlikyan et al. (2021). The target token is held fixed even if the randomized model's prediction changes.

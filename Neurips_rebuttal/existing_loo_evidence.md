# Existing LOO-KL intervention evidence

The cached LAMBADA experiment independently ranks every token by the KL
divergence induced when that token alone is zeroed. The table reports where
each attribution method's top-ranked token appears in this interventional
LOO-KL ranking. Lower mean percentile is better; “top 5%” is the fraction of
prompts for which the attribution method's first choice is also in the top 5%
of the LOO ranking.

## LLaMA-3.2 1B

| Method | Mean LOO percentile ± SEM ↓ | Within LOO top 5% ↑ | Spearman ρ ↑ |
| --- | ---: | ---: | ---: |
| Temperature Scope | 7.96 ± 0.45 | 69.1% | 0.550 |
| Semantic Scope | 6.55 ± 0.37 | 71.5% | 0.562 |
| Fisher Scope (k=4) | **5.45 ± 0.31** | **75.0%** | **0.590** |
| Input × Gradient | 5.71 ± 0.30 | 71.8% | 0.556 |
| Integrated Gradients | 29.51 ± 0.98 | 39.3% | 0.300 |
| Random | 48.94 ± 0.91 | 6.3% | 0.005 |

## LLaMA-3.2 3B

| Method | Mean LOO percentile ± SEM ↓ | Within LOO top 5% ↑ | Spearman ρ ↑ |
| --- | ---: | ---: | ---: |
| Temperature Scope | 12.31 ± 0.58 | 56.3% | 0.528 |
| Semantic Scope | 10.18 ± 0.49 | 59.9% | 0.552 |
| Fisher Scope (k=4) | **9.40 ± 0.47** | **61.8%** | **0.575** |
| Input × Gradient | 11.66 ± 0.55 | 56.2% | 0.535 |
| Integrated Gradients | 42.82 ± 0.84 | 6.2% | 0.321 |
| Random | 48.45 ± 0.92 | 7.2% | -0.004 |

*Agreement between attribution rankings and single-token LOO-KL interventions
on 1,000 LAMBADA passages per model. Spearman ρ compares each method's complete
token ranking with the complete LOO-KL ranking; the other columns evaluate the
method's top-ranked token. Best values within each model are bold.*

These cached results support a narrow claim: the Scopes' first-ranked tokens
agree substantially better with single-token interventions than random, and IG
has markedly weaker agreement in this setup. This is complementary evidence,
not an Adebayo-style model-parameter randomization sanity check, and it does
not by itself identify attention sinks as the cause of IG's weaker agreement.

Kendall τ-b and top-10% Jaccard show the same ordering, including on the
correctly predicted subsets. Bootstrap confidence intervals and all metrics
are reported in [`loo_ranking_agreement.md`](loo_ranking_agreement.md).

Source: `paper/results/master_results.json` and cached LOO-KL result files.

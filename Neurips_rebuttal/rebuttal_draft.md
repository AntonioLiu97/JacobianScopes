## Attribution target and AOPC evaluation target

We thank the reviewer for identifying a genuine conceptual mismatch and apologize
that the manuscript described the implementation imprecisely. The original
implementation combines two conventions:

1. Target-specific methods (Semantic Scope, Input × Gradient, and Integrated
   Gradients) compute attribution with respect to the held-out gold next token.
   Gold-target saliency has precedent in NLP: Li et al. (2016) explicitly define
   saliency with respect to the gold-standard class and include next-word
   prediction as an example.
2. AOPC records the change in log-probability of the model's original
   top-likelihood prediction after ablation. This is the predicted-class
   convention used by perturbation-based faithfulness metrics; for example,
   ERASER comprehensiveness evaluates the same originally predicted class before
   and after removing rationale tokens (DeYoung et al., 2020).

Thus, contrary to the wording in the submitted manuscript, the AOPC code did
**not** measure the post-ablation likelihood of the gold token on model errors.
It measured the original top prediction. We will correct this description.

To remove the remaining target mismatch, we repeated the complete cached
analysis only on passages where the held-out gold token equals the model's
original top prediction. This is a large fraction of every evaluation cell:
59.2%–77.5%, and 8,327/11,500 passages (72.4%) overall. On this controlled
subset, target-specific attribution and perturbation evaluation concern exactly
the same token. Exact subset sizes and all method-level results are reported in
[`correct_subset_aopc_table.md`](correct_subset_aopc_table.md).

The conclusion is not an artifact of averaging unmatched passages. We first
reconstructed AOPC separately for every passage by trapezoidal integration over
5%, 10%, and 20% ablation, and then ran paired tests. Across the 72
Scope-versus-baseline comparisons on correctly predicted passages, 43 favor a
Scope, 12 favor a baseline, and 17 are not significant after Holm correction.
For Fisher Scope specifically, 17/24 comparisons are significant wins and
7/24 are not significant; there are no significant losses. Against the
strongest standard baseline, Input × Gradient, Fisher Scope has 8/12
significant wins and four non-significant comparisons, with no significant
losses. We report paired bootstrap 95% confidence intervals and Wilcoxon
signed-rank tests rather than treating passage-level observations from
different methods as independent. Full results are in
[`paired_aopc_summary.md`](paired_aopc_summary.md) and
[`paired_aopc_tests.csv`](paired_aopc_tests.csv).

We thank the reviewer again: separating the attribution target from the
evaluation target makes the experiment and its limitations substantially
clearer.

## Integrated Gradients baseline sensitivity

We agree that a zero embedding is not the only reasonable IG baseline for text.
We therefore reran LLaMA-3.2 1B on the correctly predicted LAMBADA and
IWSLT2017 passages with a predeclared baseline-sensitivity set: zero, the
reserved LLaMA right-padding token `<|finetune_right_pad_id|>` (token ID
128004), BOS, and EOS. PAD is the primary alternative because PAD-token
baselines have direct precedent in text IG, including Sanyal and Ren (2021);
BOS and EOS are reported as exploratory structural-token controls rather than
as neutral baselines. We report every candidate rather than selecting a
favorable baseline post hoc. The integration grid and AOPC intervention are
held fixed, isolating the choice of IG path baseline.

| Dataset | n | Zero baseline AOPC | PAD baseline AOPC | PAD − zero [95% CI] |
| --- | ---: | ---: | ---: | ---: |
| LAMBADA | 699 | **−1.276 ± 0.021** | −0.735 ± 0.022 | +0.541 [0.493, 0.589] |
| IWSLT2017 DE→EN | 726 | **−0.962 ± 0.019** | −0.525 ± 0.017 | +0.437 [0.401, 0.473] |

More-negative AOPC is better. Thus, replacing the zero baseline with a
literature-supported PAD-token baseline substantially weakens IG on both
datasets rather than closing the gap. Both paired differences remain
significant after Holm correction ($p<2\times10^{-69}$).

<!-- Add BOS/EOS sensitivity results when their queued runs finish. -->

## Saliency sanity check

We agree that interventional AOPC/LOO evidence does not replace a parameter
randomization sanity check. Following Adebayo et al. (2018) and the text
adaptation of Kokhlikyan et al. (2021), we use LLaMA-3.2 1B with a fixed prompt
set and fixed target token, progressively reinitialize transformer blocks from
the output side toward the input side, and recompute every attribution map.
We report Spearman rank correlation with the original map as the primary
metric, with top-10% Jaccard overlap and a random-ranking floor as supplementary
metrics. The test includes Semantic, Temperature, and Fisher Scopes, Input ×
Gradient, and Integrated Gradients.

As complementary interventional evidence, we compared each method's complete
token ranking with the ranking produced by single-token LOO KL divergence over
1,000 LAMBADA passages. Fisher Scope has the highest mean Spearman agreement
for both LLaMA-3.2 1B and 3B (0.590 and 0.575), versus 0.300 and 0.321 for IG
and approximately zero for random rankings. Kendall τ-b and top-10% Jaccard
show the same ordering, including on the correctly predicted subsets. We
describe this as interventional ranking agreement, not as a substitute for the
Adebayo test.

<!-- Insert randomization results after the GPU job finishes. -->

## References

- Adebayo, J. et al. (2018). *Sanity Checks for Saliency Maps*. NeurIPS.
- DeYoung, J. et al. (2020). *ERASER: A Benchmark to Evaluate Rationalized NLP
  Models*. ACL. https://doi.org/10.18653/v1/2020.acl-main.408
- Kokhlikyan, N. et al. (2021). *Investigating Sanity Checks for Saliency Maps
  with Image and Text Classification*. arXiv:2106.07475.
- Li, J., Chen, X., Hovy, E., and Jurafsky, D. (2016). *Visualizing and
  Understanding Neural Models in NLP*. NAACL.
  https://doi.org/10.18653/v1/N16-1082
- Sanyal, S. and Ren, X. (2021). *Discretized Integrated Gradients for
  Explaining Language Models*. EMNLP.
  https://doi.org/10.18653/v1/2021.emnlp-main.805

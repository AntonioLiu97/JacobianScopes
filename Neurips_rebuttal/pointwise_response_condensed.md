# Point-by-point Response

## Response to the Area Chair

We thank the Area Chair for identifying the two central concerns: paired significance analysis and robustness of the gains to the choice of ablation baseline. We conducted both requested analyses, and both support our central empirical claims:

First, we performed **passage-level paired tests** in all 12 model–dataset combinations. After preselecting the best Jacobian Scope and non-Scope baseline from the original aggregate table, we compared their passage-level AOPCs, reporting 95% confidence intervals and two-sided paired Wilcoxon tests. The Scope significantly outperforms the baseline in **10 of 12** combinations; the remaining two are inconclusive, with **no significant baseline advantage**. Significant gains occur across all model families, model sizes, and datasets. Detailed results and analysis appear in our response to D5fu.

Second, we **reran the complete AOPC evaluation using `<PAD>`-token ablation**. Instead of zeroing selected embeddings, we replace them with each model family's PAD embedding while keeping the corresponding positions attended. A Jacobian Scope achieves the best or tied-best mean AOPC in **all 12 combinations**, so the central Scope-versus-baseline conclusion is robust to this intervention choice. We will add these results and moderate claims where paired differences are inconclusive. Please see our response to PUGo for the full results and our rationale for using `<PAD>` as the closest consistently available neutral-masking analogue across the three model families.

In addition, we addressed each reviewer's comments and suggestions through new analyses and revisions, including:

- a parameter-randomization sanity check showing progressively decreasing attribution-map similarity as more transformer blocks are reinitialized;
- a target-aligned analysis on the 72.4% of passages where the ground-truth token equals the model's top prediction, with a Scope attaining the best mean AOPC in all 12 cells and significantly outperforming the baseline in 9 of 12 paired comparisons;
- a derivation connecting Temperature Scope to predictive entropy for non-numerical vocabularies;
- clearer motivation for gradient-based attribution in NLP and a sharper distinction between local attribution and mechanistic explanation; and
- planned improvements to the manuscript's organization, cross-references, and placement of key technical details.



## PUGo

We thank the reviewer for the thorough feedback and the helpful suggestions. Please find below our responses and requested additional experiments and analysis. (Some hyperlinks omitted due to space.)

### 1. Mask-/PAD-token ablation (Table 1 rerun)

We regenerated Table 1 with a length-preserving filler-token ablation: each selected token embedding is replaced by a model-family `<PAD>` embedding while the attention mask remains active, so sequence length and positional structure are preserved. 

Across all twelve model–dataset cells, a Jacobian Scope attains the best (or tied-best) mean AOPC in **all 12**. On LLaMA and Qwen, the original ranking is preserved, absolute AOPC magnitudes are typically more negative under PAD replacement than zeroing, and the Scope advantage over the strongest non-Scope baseline is magnified in **8 of 12** cells. The consistent advantage of the Jacobian Scopes across ablation choices strengthens our case.

The table below reports mean AOPC ± 1 SEM (trapezoidal integral over 5%, 10%, and 20% ablation; more-negative is better; the best or tied-best entry in each model–dataset column is bolded). 


| Model        | Method               | LAMBADA          | IWSLT2017 DE→EN  |
| ------------ | -------------------- | ---------------- | ---------------- |
| LLaMA-3.2 1B | Random               | −0.41 ± 0.01     | −0.36 ± 0.01     |
|              | Integrated Gradients | −1.36 ± 0.02     | −1.27 ± 0.02     |
|              | Input × Gradient     | −1.61 ± 0.02     | −1.46 ± 0.02     |
|              | Semantic Scope       | −1.63 ± 0.02     | −1.49 ± 0.02     |
|              | Temperature Scope    | **−1.69 ± 0.02** | **−1.53 ± 0.02** |
|              | Fisher Scope         | **−1.69 ± 0.02** | **−1.53 ± 0.02** |
| LLaMA-3.2 3B | Random               | −0.34 ± 0.01     | −0.29 ± 0.01     |
|              | Integrated Gradients | −0.77 ± 0.02     | −0.70 ± 0.02     |
|              | Input × Gradient     | −1.32 ± 0.02     | −1.02 ± 0.02     |
|              | Semantic Scope       | −1.35 ± 0.02     | −1.02 ± 0.02     |
|              | Temperature Scope    | −1.35 ± 0.02     | −1.00 ± 0.02     |
|              | Fisher Scope         | **−1.37 ± 0.02** | **−1.05 ± 0.02** |
| Qwen2.5 1.5B | Random               | −0.51 ± 0.02     | −0.43 ± 0.02     |
|              | Integrated Gradients | −2.06 ± 0.02     | −2.08 ± 0.03     |
|              | Input × Gradient     | −2.06 ± 0.02     | −2.09 ± 0.03     |
|              | Semantic Scope       | −2.00 ± 0.02     | −1.92 ± 0.03     |
|              | Temperature Scope    | **−2.34 ± 0.02** | **−2.20 ± 0.03** |
|              | Fisher Scope         | −2.11 ± 0.02     | −2.11 ± 0.03     |
| Qwen2.5 3B   | Random               | −0.54 ± 0.02     | −0.45 ± 0.02     |
|              | Integrated Gradients | −1.80 ± 0.02     | −1.80 ± 0.03     |
|              | Input × Gradient     | −1.81 ± 0.02     | −1.80 ± 0.03     |
|              | Semantic Scope       | −1.73 ± 0.02     | −1.58 ± 0.03     |
|              | Temperature Scope    | **−2.24 ± 0.02** | **−2.02 ± 0.02** |
|              | Fisher Scope         | −1.86 ± 0.02     | −1.85 ± 0.03     |
| Gemma-3 1B   | Random               | −0.63 ± 0.01     | −0.85 ± 0.01     |
|              | Integrated Gradients | **−0.81 ± 0.01** | −1.02 ± 0.01     |
|              | Input × Gradient     | −0.80 ± 0.01     | −1.03 ± 0.01     |
|              | Semantic Scope       | −0.80 ± 0.01     | −1.03 ± 0.02     |
|              | Temperature Scope    | **−0.81 ± 0.01** | **−1.04 ± 0.02** |
|              | Fisher Scope         | −0.80 ± 0.01     | −1.03 ± 0.02     |
| Gemma-3 4B   | Random               | −0.25 ± 0.01     | −0.14 ± 0.00     |
|              | Integrated Gradients | −0.51 ± 0.01     | −0.42 ± 0.01     |
|              | Input × Gradient     | −0.51 ± 0.01     | −0.43 ± 0.01     |
|              | Semantic Scope       | −0.51 ± 0.01     | −0.42 ± 0.01     |
|              | Temperature Scope    | **−0.54 ± 0.01** | **−0.48 ± 0.01** |
|              | Fisher Scope         | −0.52 ± 0.01     | −0.46 ± 0.01     |


We used `<PAD>` as the closest cross-family `[MASK]` analogue: LLaMA and Qwen lack `[MASK]`, while Gemma's `<mask>` has no documented cloze-training role (Gemma Team, 2024). All three families define padding, making it a consistent length-preserving non-content intervention. We avoided `<BOS>`/`<EOS>` because they may act as attention sinks and induce positional bias ([Xiao et al., 2024](https://arxiv.org/abs/2309.17453); [Gu et al., 2024](https://arxiv.org/abs/2410.10781)).

### 2. Parameter-randomization sanity check

Following Adebayo et al. (2018) and the text adaptation of Kokhlikyan et al. (2021), we fixed 200 correctly predicted passages, tokenizations, and targets per cell. Starting from pretrained attribution maps, we cumulatively reinitialized output-side blocks (final quarter, final half, then all): linear weights were redrawn from $\mathcal{N}(0,r^2)$, where $r$ is the initializer range; biases were set to zero and normalization weights to one; embeddings, LM head, and targets remained fixed. 

**Across every completed model, dataset, Scope, and metric, similarity decreases monotonically as more blocks are reinitialized.** For the sake of compute and space, below we report only the results on 1~1.5B models.

**Spearman $\rho$** correlates the complete token rankings; **top-10% Jaccard** is $|A\cap B|/|A\cup B|$ for each map's highest-scoring $\lceil0.1T\rceil$ positions. We report mean ± SEM as **Spearman / Jaccard**; lower similarity indicates stronger parameter dependence.


| Model        | Dataset         | Scope       | Original      | Final quarter                 | Final half                    | All blocks                    |
| ------------ | --------------- | ----------- | ------------- | ----------------------------- | ----------------------------- | ----------------------------- |
| LLaMA-3.2 1B | LAMBADA         | Semantic    | 1.000 / 1.000 | 0.627 ± 0.012 / 0.285 ± 0.012 | 0.504 ± 0.014 / 0.190 ± 0.010 | 0.143 ± 0.016 / 0.132 ± 0.009 |
|              |                 | Temperature | 1.000 / 1.000 | 0.650 ± 0.010 / 0.288 ± 0.011 | 0.535 ± 0.012 / 0.194 ± 0.010 | 0.152 ± 0.015 / 0.133 ± 0.008 |
|              |                 | Fisher      | 1.000 / 1.000 | 0.609 ± 0.012 / 0.272 ± 0.013 | 0.492 ± 0.014 / 0.180 ± 0.010 | 0.128 ± 0.017 / 0.125 ± 0.009 |
| LLaMA-3.2 1B | IWSLT2017 DE→EN | Semantic    | 1.000 / 1.000 | 0.606 ± 0.011 / 0.236 ± 0.010 | 0.473 ± 0.012 / 0.138 ± 0.007 | 0.090 ± 0.012 / 0.108 ± 0.004 |
|              |                 | Temperature | 1.000 / 1.000 | 0.608 ± 0.011 / 0.231 ± 0.008 | 0.503 ± 0.011 / 0.139 ± 0.007 | 0.106 ± 0.011 / 0.112 ± 0.005 |
|              |                 | Fisher      | 1.000 / 1.000 | 0.570 ± 0.013 / 0.197 ± 0.008 | 0.449 ± 0.013 / 0.112 ± 0.006 | 0.070 ± 0.013 / 0.093 ± 0.004 |
| Qwen2.5 1.5B | LAMBADA         | Semantic    | 1.000 / 1.000 | 0.862 ± 0.006 / 0.504 ± 0.013 | 0.776 ± 0.008 / 0.399 ± 0.014 | 0.202 ± 0.017 / 0.164 ± 0.009 |
|              |                 | Temperature | 1.000 / 1.000 | 0.866 ± 0.005 / 0.555 ± 0.013 | 0.796 ± 0.009 / 0.446 ± 0.014 | 0.196 ± 0.016 / 0.173 ± 0.009 |
|              |                 | Fisher      | 1.000 / 1.000 | 0.857 ± 0.006 / 0.519 ± 0.014 | 0.774 ± 0.009 / 0.413 ± 0.015 | 0.182 ± 0.017 / 0.156 ± 0.009 |
| Qwen2.5 1.5B | IWSLT2017 DE→EN | Semantic    | 1.000 / 1.000 | 0.827 ± 0.006 / 0.533 ± 0.013 | 0.689 ± 0.009 / 0.381 ± 0.012 | 0.142 ± 0.012 / 0.181 ± 0.006 |
|              |                 | Temperature | 1.000 / 1.000 | 0.853 ± 0.005 / 0.568 ± 0.012 | 0.723 ± 0.009 / 0.437 ± 0.012 | 0.134 ± 0.011 / 0.204 ± 0.006 |
|              |                 | Fisher      | 1.000 / 1.000 | 0.842 ± 0.005 / 0.521 ± 0.012 | 0.702 ± 0.009 / 0.376 ± 0.012 | 0.119 ± 0.013 / 0.163 ± 0.006 |
| Gemma-3 1B   | LAMBADA         | Semantic    | 1.000 / 1.000 | 0.880 ± 0.006 / 0.587 ± 0.013 | 0.805 ± 0.007 / 0.495 ± 0.013 | 0.339 ± 0.014 / 0.257 ± 0.009 |
|              |                 | Temperature | 1.000 / 1.000 | 0.823 ± 0.007 / 0.545 ± 0.014 | 0.807 ± 0.007 / 0.470 ± 0.013 | 0.356 ± 0.013 / 0.241 ± 0.009 |
|              |                 | Fisher      | 1.000 / 1.000 | 0.837 ± 0.007 / 0.557 ± 0.014 | 0.794 ± 0.008 / 0.508 ± 0.014 | 0.329 ± 0.014 / 0.251 ± 0.009 |
| Gemma-3 1B   | IWSLT2017 DE→EN | Semantic    | 1.000 / 1.000 | 0.859 ± 0.005 / 0.586 ± 0.014 | 0.779 ± 0.007 / 0.492 ± 0.013 | 0.273 ± 0.014 / 0.285 ± 0.009 |
|              |                 | Temperature | 1.000 / 1.000 | 0.826 ± 0.006 / 0.555 ± 0.015 | 0.762 ± 0.009 / 0.471 ± 0.013 | 0.271 ± 0.015 / 0.283 ± 0.009 |
|              |                 | Fisher      | 1.000 / 1.000 | 0.829 ± 0.006 / 0.552 ± 0.014 | 0.785 ± 0.007 / 0.490 ± 0.013 | 0.275 ± 0.014 / 0.283 ± 0.010 |


LLaMA-3.2 1B is the most sensitive: after all blocks are reinitialized, its Spearman correlations are only 0.070–0.106 on IWSLT2017 DE→EN and 0.128–0.152 on LAMBADA. Qwen2.5 1.5B falls to Spearman 0.119–0.142 on IWSLT and 0.182–0.202 on LAMBADA. Gemma-3 1B retains somewhat more similarity, but still drops substantially, to Spearman 0.271–0.275 on IWSLT and 0.329–0.356 on LAMBADA; its final Jaccard overlaps are 0.241–0.285. Thus all three Scopes depend strongly and progressively on learned model parameters rather than remaining invariant to parameter randomization.

### 3. Paired significance of AOPC gains

For each model–dataset combination, we performed a passage-level paired test between the top-performing non-Scope baseline and Jacobian Scope. Please see our response to D5fu for the full table, omitted here because of space limits.

The results show that the top-performing Jacobian Scope significantly outperforms the top-performing non-Scope baseline in **10 of 12** cells. The remaining two cells are inconclusive, with **0 significant baseline advantage**. Thus, significant gains occur across model families, model sizes, and both datasets.

### 4. Interpretations of Temperature Scope

Temperature Scope's Gaussian variance interpretation is indeed intended for the time-series in-context learning setting developed in the paper. We will explicitly state it after Eq. (7).

For unstructured non-numerical vocabularies, a continuous-token approximation is undefined, so no direct analogue of Proposition 1 exists. However, Table 1 shows that Temperature Scope is also competitive on natural-language next-token prediction. We attribute this performance to a simple fact: except in the degenerate equal-logit case, attributing inverse temperature is equivalent to attributing the entropy, and hence uncertainty, of a softmax prediction.

To see this, write $p(i;\beta)\propto e^{\beta\hat{z}(i)}$ with $\hat{z}$ fixed, and let $V(\beta)$ denote the variance of $\hat{z}(i)$ under $p(i;\beta)$. Direct differentiation of the Shannon entropy gives

$$
\begin{split}
\frac{dH}{d\beta} &= -\frac{d}{d\beta} \sum_i p(i; \beta) \log p(i;\beta) 
 &=-\beta V(\beta).
\end{split}
$$

Since $\beta>0$ and $V(\beta)>0$ for non-constant $\hat z$, the entropy is strictly decreasing in inverse temperature, from $H \to \log |\mathcal{V}|$ as $\beta \to 0$ to $H \to 0$ as $\beta \to \infty$. Therefore, the map $\beta \mapsto H$ is invertible and attributing $\beta$ is equivalent to attributing $H$.

The correspondence between $\beta$ and $H$ extends to attribution scores as well. Temperature Scope computes the radial partial derivative of $\beta$ with respect to each input $x^{(t)}$ while holding $\hat{z}$ fixed. Therefore,

$$
\frac{\partial H}{\partial x^{(t)}}=-\beta V(\beta)\frac{\partial\beta}{\partial x^{(t)}} \qquad (\hat{z}\ \text{fixed}).
$$

The prefactor is the same for every input token, so Temperature Scope and entropy attribution induce exactly the same gradient-norm token ranking. We will add this result to the revision.

## D5fu

We thank the reviewer for recognizing the novelty of distribution-level attribution, the clarity of the framework, and the usefulness of Jacobian Scopes when no single target token is naturally specified. We address the two concerns below.

### 1. Paired significance of the AOPC improvements

We agree that the absolute gains over Input × Gradient are sometimes small and that standard errors alone do not establish whether the paired differences are reliable. To address this, we performed a passage-level paired test between the top-performing non-Scope baseline and Jacobian Scope, for each model–dataset combination.

Our results show that: the top-performing Jacobian Scope significantly outperforms the top-performing non-Scope baseline in **10 of 12** model–dataset cells. The remaining two comparisons are inconclusive, with **0 significant baseline advantage**. Thus, passage-level paired testing establishes that many numerically modest improvements are statistically reliable. 

Below we report the standard 95% Student's *t* confidence interval for each mean paired difference and apply a two-sided paired Wilcoxon signed-rank test to each comparison. Statistically significant comparisons (*p* < 0.05) are bolded.


| Model        | Dataset         | Top-performing Scope | Top-performing baseline | Mean paired difference [95% CI] | Wilcoxon *p*     |
| ------------ | --------------- | -------------------- | ----------------------- | ------------------------------- | ---------------- |
| LLaMA-3.2 1B | LAMBADA         | Fisher               | Input × Gradient        | **−0.0358 [−0.0482, −0.0235]**  | **1.19 × 10⁻⁶**  |
| LLaMA-3.2 1B | IWSLT2017 DE→EN | Temperature          | Input × Gradient        | **−0.0494 [−0.0689, −0.0299]**  | **9.36 × 10⁻⁵**  |
| LLaMA-3.2 3B | LAMBADA         | Fisher               | Input × Gradient        | **−0.0551 [−0.0714, −0.0388]**  | **1.11 × 10⁻¹¹** |
| LLaMA-3.2 3B | IWSLT2017 DE→EN | Fisher               | Input × Gradient        | **−0.0285 [−0.0418, −0.0151]**  | **3.91 × 10⁻⁴**  |
| Qwen2.5 1.5B | LAMBADA         | Temperature          | Input × Gradient        | **−0.1408 [−0.1678, −0.1138]**  | **3.13 × 10⁻²¹** |
| Qwen2.5 1.5B | IWSLT2017 DE→EN | Temperature          | Input × Gradient        | −0.0252 [−0.0521, 0.0018]       | 0.159            |
| Qwen2.5 3B   | LAMBADA         | Temperature          | Input × Gradient        | **−0.1616 [−0.1870, −0.1362]**  | **1.45 × 10⁻³²** |
| Qwen2.5 3B   | IWSLT2017 DE→EN | Temperature          | Input × Gradient        | **−0.0883 [−0.1247, −0.0519]**  | **1.46 × 10⁻⁶**  |
| Gemma-3 1B   | LAMBADA         | Fisher               | Input × Gradient        | **−0.1085 [−0.1261, −0.0908]**  | **3.71 × 10⁻³⁰** |
| Gemma-3 1B   | IWSLT2017 DE→EN | Fisher               | Input × Gradient        | **−0.0368 [−0.0509, −0.0227]**  | **1.12 × 10⁻⁴**  |
| Gemma-3 4B   | LAMBADA         | Fisher               | Input × Gradient        | **−0.1102 [−0.1265, −0.0938]**  | **7.98 × 10⁻³²** |
| Gemma-3 4B   | IWSLT2017 DE→EN | Fisher               | Integrated Gradients    | 0.0120 [−0.0228, 0.0468]        | 0.228            |


If accepted, we will use the additional page to report these results and revise “consistently outperform” to distinguish the ten statistically supported gains from the two inconclusive comparisons.

### 2. Scope and impact of the contribution

We agree that Jacobian Scopes are methods for local input–output attribution and do not, by themselves, recover a model's internal reasoning process or provide a complete theory of interpretability. We will state this boundary more explicitly and avoid equating attribution with mechanistic explanation.

With these clarifications in place, we still believe the contribution is substantive: existing gradient attribution generally requires choosing a scalar target token, whereas Fisher and Temperature Scopes directly attribute properties of the **full predictive distribution**—distributional change and confidence—using one backward pass. This is useful precisely in settings where there is no privileged target token, as the reviewer notes. We will sharpen the manuscript's framing around this specific contribution and present the broader case studies as demonstrations of its applicability rather than claims to explain complete model reasoning.

## UMqk

We thank the reviewer for finding our framework interesting and for the constructive feedback. Please find below our point-by-point responses to the concerns raised.

### 1. Motivation for gradient-based attribution in NLP

We admit that our current introduction assumes, rather than motivates, gradient-based attribution. In our view, gradient-based attribution is not a complete mechanistic explanation in and of itself. Instead, it is a tool that can expose **input–output dependencies**, which, combined with further investigation and domain expertise, can help audit spurious cues and efficiently generate hypotheses for intervention. Here we offer a better motivation for the revision:

Seeking explanations from first-order structures has a long history in NLP, including first-derivative saliency ([Li et al., 2016](https://doi.org/10.18653/v1/N16-1082)), gradient-guided token substitutions ([Ebrahimi et al., 2018](https://doi.org/10.18653/v1/P18-2006)), and integrated gradients ([Sundararajan et al., 2017](https://proceedings.mlr.press/v70/sundararajan17a.html)).

Gradients are especially useful at LLM scale: one backward pass scores every input token, avoiding repeated inference for many interventions. They can also target any differentiable quantity, or what we call the explanandum, from a single logit to confidence or the full predictive distribution.

We think that for these reasons, among others, recent years have witnessed a revival of gradient-based methods in modern LLM interpretability through generation attribution ([Sarti et al., 2023](https://doi.org/10.18653/v1/2023.acl-demo.40)) and scalable localization of model behavior ([Kramár et al., 2024](https://arxiv.org/abs/2403.00745); [Syed et al., 2024](https://doi.org/10.18653/v1/2024.blackboxnlp-1.25)). Most recently, Anthropic's Jacobian lens used linearized effects on future token probabilities to identify a “J-space” whose representations were subsequently linked through intervention to reportability and flexible reasoning ([Gurnee et al., 2026](https://arxiv.org/abs/2607.15495)). While not proof that every gradient attribution is faithful, this provides strong evidence that first-order Jacobian structure can carry behaviorally meaningful information in frontier LLMs.

We will add this motivation to the introduction while clearly distinguishing efficient local attribution from mechanistic explanation, which requires additional causal analysis.

### 2. Presentation and organization

We thank the reviewer for the typographic and organizational suggestions. We will standardize references to figures, sections, and appendices throughout the manuscript. We will also use figures more economically and use the resulting space, together with the additional page available in the revision, to move important technical details—including key mathematical proofs—from the appendix into the main text.

### 3. Attribution-target alignment

Thank you for raising this subtle but important question about which next token should be the target of attribution: the ground truth or the top prediction. In our view, both conventions have precedent: [Li et al. (2016)](https://doi.org/10.18653/v1/N16-1082) define saliency with respect to the gold-standard class—i.e., the ground-truth token—and explicitly consider next-word prediction, whereas predicted-class faithfulness evaluations such as ERASER comprehensiveness track the model's original prediction ([DeYoung et al., 2020](https://doi.org/10.18653/v1/2020.acl-main.408)). For target-specific methods, such as IG, Input × Gradient, and Semantic Scope, our experiments use this held-out ground-truth token.

However, we agree with the reviewer that using the model's top prediction provides a more direct target when the objective is to explain the model's own behavior, and we will clarify these conventions in the revision.

Fortunately, the ground-truth and top-prediction targets coincide whenever the model predicts the held-out token correctly. This occurs for **8,327 of 11,500 passages (72.4%)** overall. We therefore repeated the analysis on this target-aligned subset, where attribution and perturbation evaluation necessarily concern the same token. The table reports mean AOPC ± one standard error of the mean (±1 SEM), rounded to two decimal places; more-negative values are better, and the best result in each model–dataset combination is bolded based on the unrounded means. **A Jacobian Scope still achieves the best mean AOPC in each of the 12 target-aligned model–dataset combinations.**


| Model        | Method               | LAMBADA          | IWSLT2017 DE→EN  |
| ------------ | -------------------- | ---------------- | ---------------- |
| LLaMA-3.2 1B | Random               | −0.30 ± 0.01     | −0.30 ± 0.01     |
|              | Integrated Gradients | −1.28 ± 0.02     | −0.96 ± 0.02     |
|              | Input × Gradient     | −1.52 ± 0.02     | −1.12 ± 0.02     |
|              | Semantic Scope       | **−1.54 ± 0.02** | −1.12 ± 0.02     |
|              | Temperature Scope    | −1.53 ± 0.02     | **−1.16 ± 0.02** |
|              | Fisher Scope         | −1.53 ± 0.02     | −1.14 ± 0.02     |
| LLaMA-3.2 3B | Random               | −0.25 ± 0.01     | −0.18 ± 0.01     |
|              | Integrated Gradients | −0.75 ± 0.02     | −0.59 ± 0.02     |
|              | Input × Gradient     | −1.29 ± 0.02     | −0.80 ± 0.02     |
|              | Semantic Scope       | **−1.33 ± 0.02** | −0.81 ± 0.02     |
|              | Temperature Scope    | −1.32 ± 0.02     | −0.79 ± 0.02     |
|              | Fisher Scope         | −1.31 ± 0.02     | **−0.83 ± 0.02** |
| Qwen2.5 1.5B | Random               | −0.39 ± 0.01     | −0.25 ± 0.01     |
|              | Integrated Gradients | −1.91 ± 0.02     | −1.42 ± 0.02     |
|              | Input × Gradient     | −1.92 ± 0.02     | −1.42 ± 0.02     |
|              | Semantic Scope       | −1.89 ± 0.02     | −1.28 ± 0.02     |
|              | Temperature Scope    | **−2.02 ± 0.02** | −1.45 ± 0.02     |
|              | Fisher Scope         | −1.92 ± 0.02     | **−1.45 ± 0.02** |
| Qwen2.5 3B   | Random               | −0.33 ± 0.01     | −0.20 ± 0.01     |
|              | Integrated Gradients | −1.59 ± 0.02     | −1.18 ± 0.03     |
|              | Input × Gradient     | −1.60 ± 0.02     | −1.18 ± 0.03     |
|              | Semantic Scope       | −1.55 ± 0.02     | −1.04 ± 0.03     |
|              | Temperature Scope    | **−1.75 ± 0.02** | **−1.27 ± 0.03** |
|              | Fisher Scope         | −1.59 ± 0.02     | −1.22 ± 0.03     |
| Gemma-3 1B   | Random               | −0.49 ± 0.02     | −0.41 ± 0.01     |
|              | Integrated Gradients | −0.76 ± 0.02     | −0.92 ± 0.02     |
|              | Input × Gradient     | −1.90 ± 0.02     | −1.44 ± 0.02     |
|              | Semantic Scope       | **−1.98 ± 0.02** | −1.36 ± 0.02     |
|              | Temperature Scope    | −1.98 ± 0.02     | −1.36 ± 0.02     |
|              | Fisher Scope         | −1.96 ± 0.02     | **−1.47 ± 0.02** |
| Gemma-3 4B   | Random               | −0.33 ± 0.01     | −0.18 ± 0.01     |
|              | Integrated Gradients | −1.62 ± 0.03     | −1.24 ± 0.02     |
|              | Input × Gradient     | −2.00 ± 0.03     | −1.24 ± 0.02     |
|              | Semantic Scope       | −2.06 ± 0.03     | −1.09 ± 0.02     |
|              | Temperature Scope    | −2.06 ± 0.03     | −1.04 ± 0.02     |
|              | Fisher Scope         | **−2.07 ± 0.03** | **−1.25 ± 0.02** |




### 4. Paired significance tests on the target-aligned subset

The target-aligned subset has larger error bars and more modest results. Following the suggestions from PUGo and D5fu, we also performed passage-paired significance tests on this subset. 

Similar to the paired analysis on the full dataset, the analysis on target-aligned subset show similar results: Jacobian Scope significantly outperforms the preselected baseline in **9 of 12** combinations. The remaining three comparisons are inconclusive, with **0 significant baseline wins**. The target-aligned analysis therefore leads to the same qualitative conclusion.


| Model        | Dataset         | Preselected Scope | Preselected baseline | Mean paired difference [95% CI] | Wilcoxon *p*     |
| ------------ | --------------- | ----------------- | -------------------- | ------------------------------- | ---------------- |
| LLaMA-3.2 1B | LAMBADA         | Fisher            | Input × Gradient     | −0.0085 [−0.0216, 0.0045]       | 0.237            |
| LLaMA-3.2 1B | IWSLT2017 DE→EN | Temperature       | Input × Gradient     | **−0.0397 [−0.0633, −0.0160]**  | **0.0383**       |
| LLaMA-3.2 3B | LAMBADA         | Fisher            | Input × Gradient     | **−0.0247 [−0.0419, −0.0075]**  | **0.00155**      |
| LLaMA-3.2 3B | IWSLT2017 DE→EN | Fisher            | Input × Gradient     | **−0.0307 [−0.0448, −0.0166]**  | **1.08 × 10⁻⁴**  |
| Qwen2.5 1.5B | LAMBADA         | Temperature       | Input × Gradient     | **−0.1013 [−0.1315, −0.0710]**  | **1.57 × 10⁻⁹**  |
| Qwen2.5 1.5B | IWSLT2017 DE→EN | Temperature       | Input × Gradient     | −0.0254 [−0.0566, 0.0057]       | 0.252            |
| Qwen2.5 3B   | LAMBADA         | Temperature       | Input × Gradient     | **−0.1482 [−0.1774, −0.1191]**  | **3.57 × 10⁻²¹** |
| Qwen2.5 3B   | IWSLT2017 DE→EN | Temperature       | Input × Gradient     | **−0.0899 [−0.1340, −0.0458]**  | **3.94 × 10⁻⁵**  |
| Gemma-3 1B   | LAMBADA         | Fisher            | Input × Gradient     | **−0.0687 [−0.0879, −0.0494]**  | **7.16 × 10⁻¹²** |
| Gemma-3 1B   | IWSLT2017 DE→EN | Fisher            | Input × Gradient     | **−0.0366 [−0.0516, −0.0216]**  | **7.74 × 10⁻⁵**  |
| Gemma-3 4B   | LAMBADA         | Fisher            | Input × Gradient     | **−0.0766 [−0.0936, −0.0596]**  | **7.28 × 10⁻¹⁴** |
| Gemma-3 4B   | IWSLT2017 DE→EN | Fisher            | Integrated Gradients | −0.0084 [−0.0506, 0.0337]       | 0.816            |


To align the attribution and evaluation targets while respecting both pre-existing conventions, we will update the manuscript to present results on the target-aligned subset—approximately three-quarters of all passages, for which the ground-truth token and the model's top prediction are identical—as the primary AOPC analysis. We will also report the exact subset size for every model–dataset combination.
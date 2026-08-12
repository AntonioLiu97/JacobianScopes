# Table 1 (PAD / filler-token ablation)

Regenerated from length-preserving filler-token AOPC sweeps. AOPC is the per-passage trapezoidal integral over $k\in\{5\%,10\%,20\%\}$, reported as mean ± SEM. More-negative values are better. Cells with fewer than 1000 matched examples across all three ablation fractions are shown as NA. The best or tied-best score at the displayed precision in each model×dataset column is bolded.

Filler tokens: LLaMA `<|finetune_right_pad_id|>`; Qwen shared PAD/EOS `<|endoftext|>`; Gemma `<pad>`.

| Model | Method | LAMBADA | IWSLT2017 DE→EN |
| --- | --- | --- | --- |
| LLaMA-3.2 1B | Random | -0.41 ± 0.01 | -0.36 ± 0.01 |
|  | Integrated Gradients | -1.36 ± 0.02 | -1.27 ± 0.02 |
|  | Input × Gradient | -1.61 ± 0.02 | -1.46 ± 0.02 |
|  | Semantic Scope | -1.63 ± 0.02 | -1.49 ± 0.02 |
|  | Temperature Scope | **-1.69 ± 0.02** | **-1.53 ± 0.02** |
|  | Fisher Scope | **-1.69 ± 0.02** | **-1.53 ± 0.02** |
| LLaMA-3.2 3B | Random | -0.34 ± 0.01 | -0.29 ± 0.01 |
|  | Integrated Gradients | -0.77 ± 0.02 | -0.70 ± 0.02 |
|  | Input × Gradient | -1.32 ± 0.02 | -1.02 ± 0.02 |
|  | Semantic Scope | -1.35 ± 0.02 | -1.02 ± 0.02 |
|  | Temperature Scope | -1.35 ± 0.02 | -1.00 ± 0.02 |
|  | Fisher Scope | **-1.37 ± 0.02** | **-1.05 ± 0.02** |
| Qwen2.5 1.5B | Random | -0.51 ± 0.02 | -0.43 ± 0.02 |
|  | Integrated Gradients | -2.06 ± 0.02 | -2.08 ± 0.03 |
|  | Input × Gradient | -2.06 ± 0.02 | -2.09 ± 0.03 |
|  | Semantic Scope | -2.00 ± 0.02 | -1.92 ± 0.03 |
|  | Temperature Scope | **-2.34 ± 0.02** | **-2.20 ± 0.03** |
|  | Fisher Scope | -2.11 ± 0.02 | -2.11 ± 0.03 |
| Qwen2.5 3B | Random | -0.54 ± 0.02 | -0.45 ± 0.02 |
|  | Integrated Gradients | -1.80 ± 0.02 | -1.80 ± 0.03 |
|  | Input × Gradient | -1.81 ± 0.02 | -1.80 ± 0.03 |
|  | Semantic Scope | -1.73 ± 0.02 | -1.58 ± 0.03 |
|  | Temperature Scope | **-2.24 ± 0.02** | **-2.02 ± 0.02** |
|  | Fisher Scope | -1.86 ± 0.02 | -1.85 ± 0.03 |
| Gemma-3 1B | Random | -0.63 ± 0.01 | -0.85 ± 0.01 |
|  | Integrated Gradients | **-0.81 ± 0.01** | -1.02 ± 0.01 |
|  | Input × Gradient | -0.80 ± 0.01 | -1.03 ± 0.01 |
|  | Semantic Scope | -0.80 ± 0.01 | -1.03 ± 0.02 |
|  | Temperature Scope | **-0.81 ± 0.01** | **-1.04 ± 0.02** |
|  | Fisher Scope | -0.80 ± 0.01 | -1.03 ± 0.02 |
| Gemma-3 4B | Random | -0.25 ± 0.01 | -0.14 ± 0.00 |
|  | Integrated Gradients | -0.51 ± 0.01 | -0.42 ± 0.01 |
|  | Input × Gradient | -0.51 ± 0.01 | -0.43 ± 0.01 |
|  | Semantic Scope | -0.51 ± 0.01 | -0.42 ± 0.01 |
|  | Temperature Scope | **-0.54 ± 0.01** | **-0.48 ± 0.01** |
|  | Fisher Scope | -0.52 ± 0.01 | -0.46 ± 0.01 |

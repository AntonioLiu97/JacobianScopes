# AOPC on correctly predicted passages

AOPC is reconstructed per passage using trapezoidal integration over $k\\in\\{5\\%,10\\%,20\\%\\}$, then aggregated. More-negative values are better.

## Subset sizes

| Model | Dataset | Correct | Total | Accuracy | Analyzed |
| --- | --- | --- | --- | --- | --- |
| LLaMA-3.2 1B | LAMBADA | 699 | 1000 | 69.9% | 699 |
| LLaMA-3.2 1B | IWSLT2017 DE→EN | 726 | 1000 | 72.6% | 726 |
| LLaMA-3.2 3B | LAMBADA | 775 | 1000 | 77.5% | 775 |
| LLaMA-3.2 3B | IWSLT2017 DE→EN | 754 | 1000 | 75.4% | 754 |
| Qwen2.5 1.5B | LAMBADA | 700 | 1000 | 70.0% | 700 |
| Qwen2.5 1.5B | IWSLT2017 DE→EN | 736 | 1000 | 73.6% | 736 |
| Qwen2.5 3B | LAMBADA | 725 | 1000 | 72.5% | 725 |
| Qwen2.5 3B | IWSLT2017 DE→EN | 366 | 500 | 73.2% | 366 |
| Gemma-3 1B | LAMBADA | 592 | 1000 | 59.2% | 592 |
| Gemma-3 1B | IWSLT2017 DE→EN | 735 | 1000 | 73.5% | 735 |
| Gemma-3 4B | LAMBADA | 752 | 1000 | 75.2% | 752 |
| Gemma-3 4B | IWSLT2017 DE→EN | 767 | 1000 | 76.7% | 767 |

## LLaMA-3.2 1B

| Method | LAMBADA | IWSLT2017 DE→EN |
| --- | --- | --- |
| Random | -0.296 ± 0.011 | -0.299 ± 0.011 |
| Integrated Gradients | -1.276 ± 0.021 | -0.962 ± 0.019 |
| Input × Gradient | -1.523 ± 0.020 | -1.119 ± 0.019 |
| Semantic Scope | -1.540 ± 0.020 | -1.124 ± 0.020 |
| Temperature Scope | -1.530 ± 0.020 | -1.158 ± 0.019 |
| Fisher Scope | -1.531 ± 0.020 | -1.141 ± 0.019 |

## LLaMA-3.2 3B

| Method | LAMBADA | IWSLT2017 DE→EN |
| --- | --- | --- |
| Random | -0.250 ± 0.010 | -0.182 ± 0.008 |
| Integrated Gradients | -0.754 ± 0.018 | -0.585 ± 0.017 |
| Input × Gradient | -1.287 ± 0.019 | -0.796 ± 0.020 |
| Semantic Scope | -1.326 ± 0.020 | -0.806 ± 0.020 |
| Temperature Scope | -1.324 ± 0.021 | -0.791 ± 0.020 |
| Fisher Scope | -1.312 ± 0.019 | -0.827 ± 0.020 |

## Qwen2.5 1.5B

| Method | LAMBADA | IWSLT2017 DE→EN |
| --- | --- | --- |
| Random | -0.385 ± 0.014 | -0.245 ± 0.009 |
| Integrated Gradients | -1.912 ± 0.023 | -1.424 ± 0.023 |
| Input × Gradient | -1.920 ± 0.023 | -1.421 ± 0.023 |
| Semantic Scope | -1.892 ± 0.022 | -1.284 ± 0.024 |
| Temperature Scope | -2.022 ± 0.021 | -1.446 ± 0.023 |
| Fisher Scope | -1.921 ± 0.022 | -1.447 ± 0.023 |

## Qwen2.5 3B

| Method | LAMBADA | IWSLT2017 DE→EN |
| --- | --- | --- |
| Random | -0.327 ± 0.013 | -0.198 ± 0.013 |
| Integrated Gradients | -1.591 ± 0.018 | -1.176 ± 0.028 |
| Input × Gradient | -1.601 ± 0.018 | -1.179 ± 0.028 |
| Semantic Scope | -1.551 ± 0.019 | -1.039 ± 0.031 |
| Temperature Scope | -1.749 ± 0.016 | -1.269 ± 0.027 |
| Fisher Scope | -1.590 ± 0.019 | -1.224 ± 0.028 |

## Gemma-3 1B

| Method | LAMBADA | IWSLT2017 DE→EN |
| --- | --- | --- |
| Random | -0.488 ± 0.016 | -0.406 ± 0.012 |
| Integrated Gradients | -0.756 ± 0.024 | -0.916 ± 0.020 |
| Input × Gradient | -1.896 ± 0.020 | -1.435 ± 0.019 |
| Semantic Scope | -1.982 ± 0.020 | -1.364 ± 0.020 |
| Temperature Scope | -1.981 ± 0.020 | -1.358 ± 0.019 |
| Fisher Scope | -1.964 ± 0.020 | -1.472 ± 0.019 |

## Gemma-3 4B

| Method | LAMBADA | IWSLT2017 DE→EN |
| --- | --- | --- |
| Random | -0.331 ± 0.013 | -0.180 ± 0.008 |
| Integrated Gradients | -1.622 ± 0.028 | -1.239 ± 0.020 |
| Input × Gradient | -1.997 ± 0.026 | -1.238 ± 0.021 |
| Semantic Scope | -2.064 ± 0.026 | -1.090 ± 0.022 |
| Temperature Scope | -2.060 ± 0.027 | -1.039 ± 0.021 |
| Fisher Scope | -2.073 ± 0.026 | -1.247 ± 0.021 |

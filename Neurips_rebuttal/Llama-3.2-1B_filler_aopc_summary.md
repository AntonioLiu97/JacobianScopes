# Llama-3.2-1B filler-token AOPC sweep

Partial rows are labeled by n. More-negative AOPC is better; the paired difference is filler replacement minus zeroing.

| Dataset | Method | n | Zero AOPC | Filler AOPC | Difference [95% CI] | KL@20% | Flip@20% |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| lmbd1000 | random_ablation | 1000 | -0.261 | -0.408 | -0.147 [-0.177, -0.116] | 3.924 | 70.1% |
| lmbd1000 | IG | 711 | -1.070 | -1.346 | -0.276 [-0.304, -0.248] | 9.804 | 98.7% |
| lmbd1000 | gradient_x_input | 1 | -0.355 | -0.338 | 0.017 [0.017, 0.017] | 7.941 | 100.0% |
| lmbd1000 | Semantic | 1 | -0.477 | -0.263 | 0.214 [0.214, 0.214] | 6.143 | 100.0% |
| lmbd1000 | Temperature | 1 | -0.422 | -0.328 | 0.095 [0.095, 0.095] | 5.510 | 100.0% |
| lmbd1000 | Fisher_k_1 | 1 | -0.342 | -0.476 | -0.134 [-0.134, -0.134] | 8.807 | 100.0% |
| IWSLT2017DE_EN | random_ablation | 1000 | -0.298 | -0.363 | -0.066 [-0.092, -0.039] | 3.815 | 61.6% |
| IWSLT2017DE_EN | IG | 1000 | -0.935 | -1.267 | -0.332 [-0.356, -0.308] | 10.159 | 95.7% |
| IWSLT2017DE_EN | gradient_x_input | 270 | -1.106 | -1.493 | -0.386 [-0.436, -0.337] | 11.102 | 94.4% |

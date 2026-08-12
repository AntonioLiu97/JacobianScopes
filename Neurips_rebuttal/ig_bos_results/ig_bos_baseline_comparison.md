# LLaMA-3.2 1B IG BOS-baseline sensitivity

AOPC on the same correctly predicted passages; more-negative is better. The paired difference is BOS-baseline IG minus zero-baseline IG.

| Dataset | n | Zero baseline | BOS-token baseline | Paired difference [95% CI] | Holm p |
| --- | ---: | ---: | ---: | ---: | ---: |
| LAMBADA | 699 | -1.276 ± 0.021 | -1.437 ± 0.019 | -0.161 [-0.201, -0.122] | 1.23e-14 |
| IWSLT2017 DE→EN | 726 | -0.962 ± 0.019 | -1.273 ± 0.016 | -0.310 [-0.347, -0.276] | 2.99e-49 |

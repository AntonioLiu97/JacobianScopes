# LLaMA-3.2 1B IG PAD-baseline sensitivity

AOPC on the same correctly predicted passages; more-negative is better. The paired difference is PAD-baseline IG minus zero-baseline IG.

| Dataset | n | Zero baseline | PAD-token baseline | Paired difference [95% CI] | Holm p |
| --- | ---: | ---: | ---: | ---: | ---: |
| LAMBADA | 699 | -1.276 ± 0.021 | -0.735 ± 0.022 | 0.541 [0.493, 0.589] | 1.83e-69 |
| IWSLT2017 DE→EN | 726 | -0.962 ± 0.019 | -0.525 ± 0.017 | 0.437 [0.401, 0.473] | 1.8e-75 |

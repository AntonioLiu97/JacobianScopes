# LLaMA-3.2 1B IG EOS-baseline sensitivity

AOPC on the same correctly predicted passages; more-negative is better. The paired difference is EOS-baseline IG minus zero-baseline IG.

| Dataset | n | Zero baseline | EOS-token baseline | Paired difference [95% CI] | Holm p |
| --- | ---: | ---: | ---: | ---: | ---: |
| LAMBADA | 130 | -1.256 ± 0.048 | -1.415 ± 0.051 | -0.159 [-0.254, -0.061] | 0.000344 |
| IWSLT2017 DE→EN | 726 | -0.962 ± 0.019 | -1.236 ± 0.020 | -0.273 [-0.312, -0.236] | 1.77e-38 |

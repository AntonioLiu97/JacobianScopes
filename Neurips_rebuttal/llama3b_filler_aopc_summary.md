# Llama-3.2-3B filler-token AOPC summary

Matched full-dataset prompts available at the time of summarization. More-negative AOPC is better; negative filler − zero differences favor filler replacement.

## lmbd1000

| Method | n | Zero AOPC | Filler AOPC | Filler − zero [95% CI] | KL@20% | Flip@20% |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Random | 1000 | -0.229 | -0.339 | -0.110 [-0.139, -0.082] | 3.261 | 60.6% |
| Integrated Gradients | 1000 | -0.672 | -0.765 | -0.093 [-0.107, -0.080] | 6.678 | 87.1% |
| Input × Gradient | 160 | -1.117 | -1.289 | -0.173 [-0.216, -0.128] | 8.265 | 93.1% |

## IWSLT2017DE_EN

| Method | n | Zero AOPC | Filler AOPC | Filler − zero [95% CI] | KL@20% | Flip@20% |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Random | 1000 | -0.187 | -0.295 | -0.108 [-0.131, -0.085] | 2.900 | 51.0% |
| Integrated Gradients | 1000 | -0.576 | -0.705 | -0.128 [-0.142, -0.115] | 7.132 | 84.9% |
| Input × Gradient | 1000 | -0.768 | -1.018 | -0.250 [-0.267, -0.232] | 7.739 | 89.4% |
| Semantic Scope | 650 | -0.779 | -1.013 | -0.234 [-0.256, -0.213] | 7.757 | 88.8% |

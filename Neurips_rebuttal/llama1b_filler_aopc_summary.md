# Llama-3.2-1B filler-token AOPC summary

Matched full-dataset prompts available at the time of summarization. More-negative AOPC is better; negative filler − zero differences favor filler replacement.

## lmbd1000

| Method | n | Zero AOPC | Filler AOPC | Filler − zero [95% CI] | KL@20% | Flip@20% |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Random | 1000 | -0.261 | -0.408 | -0.147 [-0.178, -0.117] | 3.924 | 70.1% |
| Integrated Gradients | 1000 | -1.095 | -1.355 | -0.260 [-0.284, -0.236] | 9.857 | 98.0% |
| Input × Gradient | 1000 | -1.284 | -1.611 | -0.327 [-0.352, -0.303] | 10.902 | 98.9% |
| Semantic Scope | 1000 | -1.298 | -1.626 | -0.329 [-0.353, -0.304] | 11.048 | 99.0% |
| Temperature Scope | 1000 | -1.317 | -1.686 | -0.369 [-0.394, -0.344] | 11.369 | 99.4% |
| Fisher Scope | 1000 | -1.320 | -1.692 | -0.372 [-0.397, -0.347] | 11.425 | 99.4% |

## IWSLT2017DE_EN

| Method | n | Zero AOPC | Filler AOPC | Filler − zero [95% CI] | KL@20% | Flip@20% |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Random | 1000 | -0.298 | -0.363 | -0.066 [-0.092, -0.039] | 3.815 | 61.6% |
| Integrated Gradients | 1000 | -0.935 | -1.267 | -0.332 [-0.356, -0.308] | 10.159 | 95.7% |
| Input × Gradient | 1000 | -1.043 | -1.460 | -0.416 [-0.443, -0.390] | 10.976 | 96.1% |
| Semantic Scope | 1000 | -1.057 | -1.489 | -0.431 [-0.459, -0.404] | 11.265 | 96.7% |
| Temperature Scope | 1000 | -1.093 | -1.534 | -0.441 [-0.468, -0.415] | 11.516 | 97.8% |
| Fisher Scope | 1000 | -1.079 | -1.529 | -0.450 [-0.477, -0.423] | 11.387 | 97.6% |

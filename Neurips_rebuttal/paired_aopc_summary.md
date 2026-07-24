# Paired AOPC analysis

Each comparison uses within-passage differences `AOPC(scope) - AOPC(baseline)`; negative differences favor the Scope. Holm correction is applied across all Scope-versus-baseline comparisons within each subset.

## all

- Comparisons: 72
- Significant in favor of a Scope: 48
- Significant in favor of a baseline: 13
- Not significant after correction: 11

| Model | Dataset | Scope | Baseline | Paired difference | 95% CI | Holm p |
| --- | --- | --- | --- | --- | --- | --- |
| LLaMA-3.2 1B | LAMBADA | Semantic Scope | Integrated Gradients | -0.2022 | [-0.2270, -0.1771] | 3.44e-45 |
| LLaMA-3.2 1B | LAMBADA | Temperature Scope | Integrated Gradients | -0.2212 | [-0.2480, -0.1947] | 1.94e-49 |
| LLaMA-3.2 1B | LAMBADA | Temperature Scope | Input × Gradient | -0.0323 | [-0.0504, -0.0144] | 0.00335 |
| LLaMA-3.2 1B | LAMBADA | Fisher Scope | Integrated Gradients | -0.2248 | [-0.2502, -0.2008] | 2.45e-59 |
| LLaMA-3.2 1B | LAMBADA | Fisher Scope | Input × Gradient | -0.0358 | [-0.0482, -0.0235] | 2.96e-05 |
| LLaMA-3.2 1B | IWSLT2017 DE→EN | Semantic Scope | Integrated Gradients | -0.1226 | [-0.1483, -0.0971] | 1.74e-17 |
| LLaMA-3.2 1B | IWSLT2017 DE→EN | Temperature Scope | Integrated Gradients | -0.1580 | [-0.1840, -0.1324] | 1.24e-28 |
| LLaMA-3.2 1B | IWSLT2017 DE→EN | Temperature Scope | Input × Gradient | -0.0494 | [-0.0689, -0.0298] | 0.00169 |
| LLaMA-3.2 1B | IWSLT2017 DE→EN | Fisher Scope | Integrated Gradients | -0.1445 | [-0.1698, -0.1191] | 3.38e-24 |
| LLaMA-3.2 1B | IWSLT2017 DE→EN | Fisher Scope | Input × Gradient | -0.0359 | [-0.0483, -0.0236] | 2.61e-06 |
| LLaMA-3.2 3B | LAMBADA | Semantic Scope | Integrated Gradients | -0.4901 | [-0.5253, -0.4553] | 1.04e-103 |
| LLaMA-3.2 3B | LAMBADA | Semantic Scope | Input × Gradient | -0.0428 | [-0.0609, -0.0247] | 7.41e-06 |
| LLaMA-3.2 3B | LAMBADA | Temperature Scope | Integrated Gradients | -0.5005 | [-0.5378, -0.4627] | 2.62e-99 |
| LLaMA-3.2 3B | LAMBADA | Temperature Scope | Input × Gradient | -0.0531 | [-0.0769, -0.0302] | 7.41e-06 |
| LLaMA-3.2 3B | LAMBADA | Fisher Scope | Integrated Gradients | -0.5024 | [-0.5385, -0.4677] | 6.75e-105 |
| LLaMA-3.2 3B | LAMBADA | Fisher Scope | Input × Gradient | -0.0551 | [-0.0712, -0.0390] | 3.89e-10 |
| LLaMA-3.2 3B | IWSLT2017 DE→EN | Semantic Scope | Integrated Gradients | -0.2002 | [-0.2306, -0.1702] | 1.54e-30 |
| LLaMA-3.2 3B | IWSLT2017 DE→EN | Temperature Scope | Integrated Gradients | -0.1834 | [-0.2139, -0.1523] | 1.69e-26 |
| LLaMA-3.2 3B | IWSLT2017 DE→EN | Fisher Scope | Integrated Gradients | -0.2205 | [-0.2511, -0.1891] | 4.26e-34 |
| LLaMA-3.2 3B | IWSLT2017 DE→EN | Fisher Scope | Input × Gradient | -0.0285 | [-0.0416, -0.0152] | 0.00587 |
| Qwen2.5 1.5B | LAMBADA | Semantic Scope | Input × Gradient | 0.0378 | [0.0193, 0.0565] | 0.00136 |
| Qwen2.5 1.5B | LAMBADA | Temperature Scope | Integrated Gradients | -0.1468 | [-0.1746, -0.1198] | 1.2e-21 |
| Qwen2.5 1.5B | LAMBADA | Temperature Scope | Input × Gradient | -0.1408 | [-0.1680, -0.1138] | 1.38e-19 |
| Qwen2.5 1.5B | LAMBADA | Fisher Scope | Integrated Gradients | -0.0609 | [-0.0806, -0.0423] | 7.69e-05 |
| Qwen2.5 1.5B | LAMBADA | Fisher Scope | Input × Gradient | -0.0549 | [-0.0742, -0.0368] | 0.0107 |
| Qwen2.5 1.5B | IWSLT2017 DE→EN | Semantic Scope | Integrated Gradients | 0.1145 | [0.0922, 0.1368] | 1.66e-18 |
| Qwen2.5 1.5B | IWSLT2017 DE→EN | Semantic Scope | Input × Gradient | 0.1154 | [0.0929, 0.1375] | 6.2e-19 |
| Qwen2.5 1.5B | IWSLT2017 DE→EN | Fisher Scope | Integrated Gradients | -0.0228 | [-0.0388, -0.0068] | 0.0107 |
| Qwen2.5 1.5B | IWSLT2017 DE→EN | Fisher Scope | Input × Gradient | -0.0219 | [-0.0371, -0.0060] | 0.0157 |
| Qwen2.5 3B | LAMBADA | Semantic Scope | Integrated Gradients | 0.0504 | [0.0322, 0.0688] | 1.86e-05 |
| Qwen2.5 3B | LAMBADA | Semantic Scope | Input × Gradient | 0.0562 | [0.0391, 0.0741] | 9.75e-07 |
| Qwen2.5 3B | LAMBADA | Temperature Scope | Integrated Gradients | -0.1674 | [-0.1940, -0.1416] | 1.01e-32 |
| Qwen2.5 3B | LAMBADA | Temperature Scope | Input × Gradient | -0.1616 | [-0.1868, -0.1366] | 7.81e-31 |
| Qwen2.5 3B | IWSLT2017 DE→EN | Semantic Scope | Integrated Gradients | 0.1007 | [0.0707, 0.1303] | 3.69e-10 |
| Qwen2.5 3B | IWSLT2017 DE→EN | Semantic Scope | Input × Gradient | 0.1030 | [0.0736, 0.1326] | 4.54e-10 |
| Qwen2.5 3B | IWSLT2017 DE→EN | Temperature Scope | Integrated Gradients | -0.0906 | [-0.1271, -0.0551] | 4.28e-05 |
| Qwen2.5 3B | IWSLT2017 DE→EN | Temperature Scope | Input × Gradient | -0.0883 | [-0.1242, -0.0522] | 3.5e-05 |
| Qwen2.5 3B | IWSLT2017 DE→EN | Fisher Scope | Integrated Gradients | -0.0446 | [-0.0663, -0.0230] | 0.00164 |
| Qwen2.5 3B | IWSLT2017 DE→EN | Fisher Scope | Input × Gradient | -0.0423 | [-0.0639, -0.0209] | 0.00126 |
| Gemma-3 1B | LAMBADA | Semantic Scope | Integrated Gradients | -0.9964 | [-1.0404, -0.9528] | 4.08e-154 |
| Gemma-3 1B | LAMBADA | Semantic Scope | Input × Gradient | -0.0739 | [-0.0898, -0.0585] | 1.05e-16 |
| Gemma-3 1B | LAMBADA | Temperature Scope | Integrated Gradients | -1.0228 | [-1.0674, -0.9786] | 1.64e-156 |
| Gemma-3 1B | LAMBADA | Temperature Scope | Input × Gradient | -0.1003 | [-0.1217, -0.0792] | 2.68e-16 |
| Gemma-3 1B | LAMBADA | Fisher Scope | Integrated Gradients | -1.0309 | [-1.0745, -0.9877] | 2.8e-157 |
| Gemma-3 1B | LAMBADA | Fisher Scope | Input × Gradient | -0.1085 | [-0.1263, -0.0909] | 1.82e-28 |
| Gemma-3 1B | IWSLT2017 DE→EN | Semantic Scope | Integrated Gradients | -0.3947 | [-0.4298, -0.3599] | 9.34e-76 |
| Gemma-3 1B | IWSLT2017 DE→EN | Semantic Scope | Input × Gradient | 0.0535 | [0.0353, 0.0717] | 3.81e-07 |
| Gemma-3 1B | IWSLT2017 DE→EN | Temperature Scope | Integrated Gradients | -0.3824 | [-0.4183, -0.3472] | 2.41e-72 |
| Gemma-3 1B | IWSLT2017 DE→EN | Temperature Scope | Input × Gradient | 0.0657 | [0.0435, 0.0877] | 3.11e-08 |
| Gemma-3 1B | IWSLT2017 DE→EN | Fisher Scope | Integrated Gradients | -0.4849 | [-0.5193, -0.4500] | 4.75e-101 |
| Gemma-3 1B | IWSLT2017 DE→EN | Fisher Scope | Input × Gradient | -0.0368 | [-0.0514, -0.0230] | 0.00191 |
| Gemma-3 4B | LAMBADA | Semantic Scope | Integrated Gradients | -0.3444 | [-0.3877, -0.3005] | 3.72e-45 |
| Gemma-3 4B | LAMBADA | Semantic Scope | Input × Gradient | -0.0706 | [-0.0889, -0.0531] | 1.83e-10 |
| Gemma-3 4B | LAMBADA | Temperature Scope | Integrated Gradients | -0.3476 | [-0.3908, -0.3054] | 8.94e-47 |
| Gemma-3 4B | LAMBADA | Temperature Scope | Input × Gradient | -0.0739 | [-0.1008, -0.0476] | 2e-08 |
| Gemma-3 4B | LAMBADA | Fisher Scope | Integrated Gradients | -0.3839 | [-0.4258, -0.3430] | 1.35e-58 |
| Gemma-3 4B | LAMBADA | Fisher Scope | Input × Gradient | -0.1102 | [-0.1266, -0.0941] | 4.09e-30 |
| Gemma-3 4B | IWSLT2017 DE→EN | Semantic Scope | Integrated Gradients | 0.1531 | [0.1176, 0.1886] | 8.69e-17 |
| Gemma-3 4B | IWSLT2017 DE→EN | Semantic Scope | Input × Gradient | 0.1154 | [0.0926, 0.1388] | 6e-20 |
| Gemma-3 4B | IWSLT2017 DE→EN | Temperature Scope | Integrated Gradients | 0.2155 | [0.1806, 0.2507] | 4.09e-30 |
| Gemma-3 4B | IWSLT2017 DE→EN | Temperature Scope | Input × Gradient | 0.1778 | [0.1514, 0.2040] | 3.46e-36 |

## correct_only

- Comparisons: 72
- Significant in favor of a Scope: 43
- Significant in favor of a baseline: 12
- Not significant after correction: 17

| Model | Dataset | Scope | Baseline | Paired difference | 95% CI | Holm p |
| --- | --- | --- | --- | --- | --- | --- |
| LLaMA-3.2 1B | LAMBADA | Semantic Scope | Integrated Gradients | -0.2639 | [-0.2958, -0.2321] | 5.02e-45 |
| LLaMA-3.2 1B | LAMBADA | Temperature Scope | Integrated Gradients | -0.2544 | [-0.2891, -0.2204] | 1.55e-35 |
| LLaMA-3.2 1B | LAMBADA | Fisher Scope | Integrated Gradients | -0.2550 | [-0.2893, -0.2221] | 5.31e-41 |
| LLaMA-3.2 1B | IWSLT2017 DE→EN | Semantic Scope | Integrated Gradients | -0.1612 | [-0.1942, -0.1283] | 1.06e-17 |
| LLaMA-3.2 1B | IWSLT2017 DE→EN | Temperature Scope | Integrated Gradients | -0.1959 | [-0.2287, -0.1635] | 7.41e-28 |
| LLaMA-3.2 1B | IWSLT2017 DE→EN | Fisher Scope | Integrated Gradients | -0.1786 | [-0.2106, -0.1463] | 1.06e-23 |
| LLaMA-3.2 1B | IWSLT2017 DE→EN | Fisher Scope | Input × Gradient | -0.0223 | [-0.0342, -0.0100] | 0.0109 |
| LLaMA-3.2 3B | LAMBADA | Semantic Scope | Integrated Gradients | -0.5718 | [-0.6135, -0.5306] | 5.89e-90 |
| LLaMA-3.2 3B | LAMBADA | Semantic Scope | Input × Gradient | -0.0385 | [-0.0595, -0.0170] | 0.00523 |
| LLaMA-3.2 3B | LAMBADA | Temperature Scope | Integrated Gradients | -0.5704 | [-0.6136, -0.5281] | 1.11e-83 |
| LLaMA-3.2 3B | LAMBADA | Temperature Scope | Input × Gradient | -0.0371 | [-0.0643, -0.0095] | 0.0186 |
| LLaMA-3.2 3B | LAMBADA | Fisher Scope | Integrated Gradients | -0.5580 | [-0.6017, -0.5165] | 2.57e-85 |
| LLaMA-3.2 3B | LAMBADA | Fisher Scope | Input × Gradient | -0.0247 | [-0.0417, -0.0079] | 0.0278 |
| LLaMA-3.2 3B | IWSLT2017 DE→EN | Semantic Scope | Integrated Gradients | -0.2210 | [-0.2578, -0.1839] | 1.19e-25 |
| LLaMA-3.2 3B | IWSLT2017 DE→EN | Temperature Scope | Integrated Gradients | -0.2059 | [-0.2429, -0.1690] | 7.14e-23 |
| LLaMA-3.2 3B | IWSLT2017 DE→EN | Fisher Scope | Integrated Gradients | -0.2420 | [-0.2789, -0.2053] | 2.14e-28 |
| LLaMA-3.2 3B | IWSLT2017 DE→EN | Fisher Scope | Input × Gradient | -0.0307 | [-0.0449, -0.0165] | 0.00291 |
| Qwen2.5 1.5B | LAMBADA | Temperature Scope | Integrated Gradients | -0.1093 | [-0.1396, -0.0802] | 3.71e-10 |
| Qwen2.5 1.5B | LAMBADA | Temperature Scope | Input × Gradient | -0.1013 | [-0.1325, -0.0711] | 5.65e-08 |
| Qwen2.5 1.5B | IWSLT2017 DE→EN | Semantic Scope | Integrated Gradients | 0.1398 | [0.1133, 0.1663] | 5.62e-19 |
| Qwen2.5 1.5B | IWSLT2017 DE→EN | Semantic Scope | Input × Gradient | 0.1369 | [0.1105, 0.1640] | 1.19e-18 |
| Qwen2.5 1.5B | IWSLT2017 DE→EN | Fisher Scope | Integrated Gradients | -0.0228 | [-0.0379, -0.0076] | 0.0109 |
| Qwen2.5 1.5B | IWSLT2017 DE→EN | Fisher Scope | Input × Gradient | -0.0257 | [-0.0406, -0.0109] | 0.00978 |
| Qwen2.5 3B | LAMBADA | Semantic Scope | Integrated Gradients | 0.0395 | [0.0183, 0.0604] | 0.023 |
| Qwen2.5 3B | LAMBADA | Semantic Scope | Input × Gradient | 0.0497 | [0.0291, 0.0706] | 0.00255 |
| Qwen2.5 3B | LAMBADA | Temperature Scope | Integrated Gradients | -0.1584 | [-0.1883, -0.1283] | 1.09e-21 |
| Qwen2.5 3B | LAMBADA | Temperature Scope | Input × Gradient | -0.1482 | [-0.1777, -0.1194] | 1.75e-19 |
| Qwen2.5 3B | IWSLT2017 DE→EN | Semantic Scope | Integrated Gradients | 0.1365 | [0.0988, 0.1743] | 9e-12 |
| Qwen2.5 3B | IWSLT2017 DE→EN | Semantic Scope | Input × Gradient | 0.1396 | [0.1034, 0.1763] | 6.14e-12 |
| Qwen2.5 3B | IWSLT2017 DE→EN | Temperature Scope | Integrated Gradients | -0.0930 | [-0.1368, -0.0499] | 0.00129 |
| Qwen2.5 3B | IWSLT2017 DE→EN | Temperature Scope | Input × Gradient | -0.0899 | [-0.1341, -0.0460] | 0.00122 |
| Qwen2.5 3B | IWSLT2017 DE→EN | Fisher Scope | Integrated Gradients | -0.0482 | [-0.0716, -0.0250] | 0.00924 |
| Qwen2.5 3B | IWSLT2017 DE→EN | Fisher Scope | Input × Gradient | -0.0451 | [-0.0692, -0.0214] | 0.00505 |
| Gemma-3 1B | LAMBADA | Semantic Scope | Integrated Gradients | -1.2266 | [-1.2819, -1.1693] | 7.37e-95 |
| Gemma-3 1B | LAMBADA | Semantic Scope | Input × Gradient | -0.0867 | [-0.1077, -0.0666] | 5.21e-12 |
| Gemma-3 1B | LAMBADA | Temperature Scope | Integrated Gradients | -1.2257 | [-1.2826, -1.1670] | 4.25e-94 |
| Gemma-3 1B | LAMBADA | Temperature Scope | Input × Gradient | -0.0858 | [-0.1131, -0.0589] | 2.76e-07 |
| Gemma-3 1B | LAMBADA | Fisher Scope | Integrated Gradients | -1.2085 | [-1.2679, -1.1512] | 3.21e-94 |
| Gemma-3 1B | LAMBADA | Fisher Scope | Input × Gradient | -0.0687 | [-0.0883, -0.0498] | 2.79e-10 |
| Gemma-3 1B | IWSLT2017 DE→EN | Semantic Scope | Integrated Gradients | -0.4475 | [-0.4898, -0.4058] | 1.98e-64 |
| Gemma-3 1B | IWSLT2017 DE→EN | Semantic Scope | Input × Gradient | 0.0714 | [0.0490, 0.0935] | 4.12e-08 |
| Gemma-3 1B | IWSLT2017 DE→EN | Temperature Scope | Integrated Gradients | -0.4416 | [-0.4842, -0.3988] | 4.77e-63 |
| Gemma-3 1B | IWSLT2017 DE→EN | Temperature Scope | Input × Gradient | 0.0773 | [0.0510, 0.1030] | 6.12e-07 |
| Gemma-3 1B | IWSLT2017 DE→EN | Fisher Scope | Integrated Gradients | -0.5555 | [-0.5967, -0.5134] | 2.71e-84 |
| Gemma-3 1B | IWSLT2017 DE→EN | Fisher Scope | Input × Gradient | -0.0366 | [-0.0519, -0.0214] | 0.00225 |
| Gemma-3 4B | LAMBADA | Semantic Scope | Integrated Gradients | -0.4420 | [-0.4946, -0.3892] | 1.31e-48 |
| Gemma-3 4B | LAMBADA | Semantic Scope | Input × Gradient | -0.0673 | [-0.0886, -0.0456] | 1.26e-05 |
| Gemma-3 4B | LAMBADA | Temperature Scope | Integrated Gradients | -0.4387 | [-0.4898, -0.3862] | 3.94e-48 |
| Gemma-3 4B | LAMBADA | Temperature Scope | Input × Gradient | -0.0640 | [-0.0958, -0.0323] | 0.000365 |
| Gemma-3 4B | LAMBADA | Fisher Scope | Integrated Gradients | -0.4513 | [-0.5030, -0.4006] | 1.5e-51 |
| Gemma-3 4B | LAMBADA | Fisher Scope | Input × Gradient | -0.0766 | [-0.0937, -0.0597] | 3.2e-12 |
| Gemma-3 4B | IWSLT2017 DE→EN | Semantic Scope | Integrated Gradients | 0.1493 | [0.1063, 0.1914] | 5.02e-11 |
| Gemma-3 4B | IWSLT2017 DE→EN | Semantic Scope | Input × Gradient | 0.1483 | [0.1210, 0.1752] | 3.91e-25 |
| Gemma-3 4B | IWSLT2017 DE→EN | Temperature Scope | Integrated Gradients | 0.2002 | [0.1579, 0.2413] | 3.35e-19 |
| Gemma-3 4B | IWSLT2017 DE→EN | Temperature Scope | Input × Gradient | 0.1991 | [0.1696, 0.2286] | 3.44e-35 |

## Best-performing Scope versus best-performing baseline

For each model–dataset combination, this section compares the Scope and non-Scope baseline with the lowest mean AOPC in the corresponding subset. Holm p-values retain the correction across all Scope-versus-baseline comparisons within that subset.

### all

| Model | Dataset | Best Scope | Scope AOPC | Best baseline | Baseline AOPC | Paired difference | 95% CI | Holm p |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| LLaMA-3.2 1B | LAMBADA | Fisher Scope | -1.320 | Input × Gradient | -1.284 | -0.0358 | [-0.0482, -0.0235] | 2.96e-05 |
| LLaMA-3.2 1B | IWSLT2017 DE→EN | Temperature Scope | -1.093 | Input × Gradient | -1.043 | -0.0494 | [-0.0689, -0.0298] | 0.00169 |
| LLaMA-3.2 3B | LAMBADA | Fisher Scope | -1.175 | Input × Gradient | -1.119 | -0.0551 | [-0.0712, -0.0390] | 3.89e-10 |
| LLaMA-3.2 3B | IWSLT2017 DE→EN | Fisher Scope | -0.797 | Input × Gradient | -0.768 | -0.0285 | [-0.0416, -0.0152] | 0.00587 |
| Qwen2.5 1.5B | LAMBADA | Temperature Scope | -1.861 | Input × Gradient | -1.720 | -0.1408 | [-0.1680, -0.1138] | 1.38e-19 |
| Qwen2.5 1.5B | IWSLT2017 DE→EN | Temperature Scope | -1.380 | Input × Gradient | -1.355 | -0.0252 | [-0.0514, 0.0020] | 0.797 |
| Qwen2.5 3B | LAMBADA | Temperature Scope | -1.554 | Input × Gradient | -1.392 | -0.1616 | [-0.1868, -0.1366] | 7.81e-31 |
| Qwen2.5 3B | IWSLT2017 DE→EN | Temperature Scope | -1.201 | Input × Gradient | -1.113 | -0.0883 | [-0.1242, -0.0522] | 3.5e-05 |
| Gemma-3 1B | LAMBADA | Fisher Scope | -1.670 | Input × Gradient | -1.561 | -0.1085 | [-0.1263, -0.0909] | 1.82e-28 |
| Gemma-3 1B | IWSLT2017 DE→EN | Fisher Scope | -1.393 | Input × Gradient | -1.356 | -0.0368 | [-0.0514, -0.0230] | 0.00191 |
| Gemma-3 4B | LAMBADA | Fisher Scope | -1.814 | Input × Gradient | -1.704 | -0.1102 | [-0.1266, -0.0941] | 4.09e-30 |
| Gemma-3 4B | IWSLT2017 DE→EN | Fisher Scope | -1.196 | Integrated Gradients | -1.208 | 0.0120 | [-0.0228, 0.0474] | 0.913 |

### correct_only

| Model | Dataset | Best Scope | Scope AOPC | Best baseline | Baseline AOPC | Paired difference | 95% CI | Holm p |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| LLaMA-3.2 1B | LAMBADA | Semantic Scope | -1.540 | Input × Gradient | -1.523 | -0.0175 | [-0.0340, -0.0005] | 0.912 |
| LLaMA-3.2 1B | IWSLT2017 DE→EN | Temperature Scope | -1.158 | Input × Gradient | -1.119 | -0.0397 | [-0.0634, -0.0162] | 0.613 |
| LLaMA-3.2 3B | LAMBADA | Semantic Scope | -1.326 | Input × Gradient | -1.287 | -0.0385 | [-0.0595, -0.0170] | 0.00523 |
| LLaMA-3.2 3B | IWSLT2017 DE→EN | Fisher Scope | -0.827 | Input × Gradient | -0.796 | -0.0307 | [-0.0449, -0.0165] | 0.00291 |
| Qwen2.5 1.5B | LAMBADA | Temperature Scope | -2.022 | Input × Gradient | -1.920 | -0.1013 | [-0.1325, -0.0711] | 5.65e-08 |
| Qwen2.5 1.5B | IWSLT2017 DE→EN | Fisher Scope | -1.447 | Integrated Gradients | -1.424 | -0.0228 | [-0.0379, -0.0076] | 0.0109 |
| Qwen2.5 3B | LAMBADA | Temperature Scope | -1.749 | Input × Gradient | -1.601 | -0.1482 | [-0.1777, -0.1194] | 1.75e-19 |
| Qwen2.5 3B | IWSLT2017 DE→EN | Temperature Scope | -1.269 | Input × Gradient | -1.179 | -0.0899 | [-0.1341, -0.0460] | 0.00122 |
| Gemma-3 1B | LAMBADA | Semantic Scope | -1.982 | Input × Gradient | -1.896 | -0.0867 | [-0.1077, -0.0666] | 5.21e-12 |
| Gemma-3 1B | IWSLT2017 DE→EN | Fisher Scope | -1.472 | Input × Gradient | -1.435 | -0.0366 | [-0.0519, -0.0214] | 0.00225 |
| Gemma-3 4B | LAMBADA | Fisher Scope | -2.073 | Input × Gradient | -1.997 | -0.0766 | [-0.0937, -0.0597] | 3.2e-12 |
| Gemma-3 4B | IWSLT2017 DE→EN | Fisher Scope | -1.247 | Integrated Gradients | -1.239 | -0.0084 | [-0.0506, 0.0325] | 1 |

### Summary and statistical interpretation

On all passages, the best Scope significantly beats the best non-Scope baseline in **10/12** model–dataset combinations. On correctly predicted passages, it does so in **9/12** combinations. There are **0** significant best-baseline wins on all passages and **0** on the correct-only subset.

The 95% CI is a bootstrap confidence interval for the mean paired difference. We resampled passages with replacement 10,000 times, preserving each passage's Scope–baseline pairing, computed the mean difference for every resample, and reported the 2.5th and 97.5th percentiles. Thus, we did perform the reviewer-suggested bootstrapping. An interval below zero supports the Scope; an interval crossing zero indicates that zero remains plausible.

The Holm p-value is the two-sided paired Wilcoxon signed-rank p-value after Holm correction across all 72 Scope-versus-baseline comparisons in the corresponding subset. A Holm p-value below 0.05 is treated as significant. The confidence interval is bootstrap-based, whereas the p-value comes from the Wilcoxon test; they are complementary rather than the same calculation.

queue: 24/24 done | running 0 | pending 0 | failed 0 [] | 02:33:06

### scarcity | Qwen2.5-7B | test Avg (%) vs. training examples per task

| label              | 100          | 250          | 500          |
|:-------------------|:-------------|:-------------|:-------------|
| LoRA               | 86.83 ± 0.08 | 87.89 ± 0.26 | 87.91 ± 0.41 |
| DoRA               | 86.62 ± 0.08 | 87.61 ± 0.56 | 88.00 ± 0.60 |
| alpha-LoRA-grad-v2 | 87.05 ± 0.20 | 87.28 ± 0.50 | 87.36 ± 0.19 |
| alpha-LoRA-theory  | 86.78 ± 0.32 | 87.58 ± 0.45 | 88.25 ± 0.30 |

gain over LoRA (points):

| label              |   100 |   250 |   500 |
|:-------------------|------:|------:|------:|
| LoRA               |  0    |  0    |  0    |
| DoRA               | -0.2  | -0.27 |  0.09 |
| alpha-LoRA-grad-v2 |  0.23 | -0.61 | -0.55 |
| alpha-LoRA-theory  | -0.05 | -0.31 |  0.34 |

### parameter overhead of alpha

| model      |   alpha |     lora |      total |   α % of trainable |   α % of total |
|:-----------|--------:|---------:|-----------:|-------------------:|---------------:|
| Qwen2.5-7B | 1390592 | 40370176 | 7657377280 |               3.44 |         0.0182 |

plots written to commonsense_reasoning/results/scarcity/plots
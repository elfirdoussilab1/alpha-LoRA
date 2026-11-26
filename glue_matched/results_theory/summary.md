queue: 100/100 done | running 0 | pending 0 | failed 0 [] | 18:20:32

### 1) Each method at its best LR (chosen on DEV): test acc (%) @ best dev, mean ± std over seeds

| method                      | rte          | mrpc         |   avg |   avg (final ckpt) |   Δ avg vs LoRA | tasks > LoRA   | seeds > LoRA   |   avg rank | best on   |
|:----------------------------|:-------------|:-------------|------:|-------------------:|----------------:|:---------------|:---------------|-----------:|:----------|
| LoRA                        | 73.57 ± 1.41 | 86.32 ± 0.73 | 79.95 |              80    |            0    | 0/2            | 0/10           |        1.5 | 1/2       |
| alpha-LoRA-theory           | 71.84 ± 0.68 | 85.00 ± 0.83 | 78.42 |              78.99 |           -1.53 | 0/2            | 2/10           |        4   | 0/2       |
| alpha-LoRA-theory_dev-probe | 71.84 ± 1.81 | 87.01 ± 0.89 | 79.43 |              79.8  |           -0.52 | 1/2            | 3/10           |        2.5 | 1/2       |
| alpha-LoRA-theory_no-update | 72.56 ± 0.56 | 85.20 ± 0.37 | 78.88 |              78.94 |           -1.07 | 0/2            | 2/10           |        2.5 | 0/2       |
| alpha-LoRA-theory_reinsert  | 72.42 ± 1.18 | 84.66 ± 0.83 | 78.54 |              78.91 |           -1.41 | 0/2            | 2/10           |        4   | 0/2       |

LR selected on dev:

| method                      |    rte |   mrpc |
|:----------------------------|-------:|-------:|
| LoRA                        | 0.0005 | 0.0002 |
| alpha-LoRA-theory           | 0.0005 | 0.0002 |
| alpha-LoRA-theory_dev-probe | 0.0005 | 0.0005 |
| alpha-LoRA-theory_no-update | 0.0005 | 0.0002 |
| alpha-LoRA-theory_reinsert  | 0.0005 | 0.0002 |

### 2) Matched hyper-parameters: test acc (%) @ best dev, at each LR

|                                         | rte          | mrpc         |   avg |
|:----------------------------------------|:-------------|:-------------|------:|
| (0.0002, 'LoRA')                        | 72.49 ± 1.96 | 86.32 ± 0.73 | 79.41 |
| (0.0002, 'alpha-LoRA-theory')           | 70.97 ± 1.85 | 85.00 ± 0.83 | 77.99 |
| (0.0002, 'alpha-LoRA-theory_dev-probe') | 70.32 ± 1.50 | 86.37 ± 0.91 | 78.35 |
| (0.0002, 'alpha-LoRA-theory_no-update') | 70.83 ± 1.70 | 85.20 ± 0.37 | 78.01 |
| (0.0002, 'alpha-LoRA-theory_reinsert')  | 71.19 ± 1.83 | 84.66 ± 0.83 | 77.92 |
| (0.0005, 'LoRA')                        | 73.57 ± 1.41 | 87.16 ± 1.31 | 80.37 |
| (0.0005, 'alpha-LoRA-theory')           | 71.84 ± 0.68 | 85.93 ± 0.57 | 78.89 |
| (0.0005, 'alpha-LoRA-theory_dev-probe') | 71.84 ± 1.81 | 87.01 ± 0.89 | 79.43 |
| (0.0005, 'alpha-LoRA-theory_no-update') | 72.56 ± 0.56 | 86.03 ± 0.44 | 79.3  |
| (0.0005, 'alpha-LoRA-theory_reinsert')  | 72.42 ± 1.18 | 85.93 ± 0.57 | 79.18 |

paired per-seed difference, alpha-LoRA-theory vs LoRA (same seed, same data, same LR):

|                  |   Δ mean (alpha-LoRA-theory − LoRA) |   Δ std | seeds won   |
|:-----------------|------------------------------------:|--------:|:------------|
| (0.0002, 'mrpc') |                               -1.32 |    1.57 | 1/5         |
| (0.0002, 'rte')  |                               -1.52 |    2.72 | 1/5         |
| (0.0005, 'mrpc') |                               -1.23 |    1.95 | 1/5         |
| (0.0005, 'rte')  |                               -1.73 |    2.11 | 1/5         |

paired per-seed difference, alpha-LoRA-theory_dev-probe vs LoRA (same seed, same data, same LR):

|                  |   Δ mean (alpha-LoRA-theory_dev-probe − LoRA) |   Δ std | seeds won   |
|:-----------------|----------------------------------------------:|--------:|:------------|
| (0.0002, 'mrpc') |                                          0.05 |    1.66 | 2/5         |
| (0.0002, 'rte')  |                                         -2.17 |    1.94 | 1/5         |
| (0.0005, 'mrpc') |                                         -0.15 |    0.54 | 1/5         |
| (0.0005, 'rte')  |                                         -1.73 |    1.48 | 0/5         |

paired per-seed difference, alpha-LoRA-theory_no-update vs LoRA (same seed, same data, same LR):

|                  |   Δ mean (alpha-LoRA-theory_no-update − LoRA) |   Δ std | seeds won   |
|:-----------------|----------------------------------------------:|--------:|:------------|
| (0.0002, 'mrpc') |                                         -1.13 |    1.09 | 1/5         |
| (0.0002, 'rte')  |                                         -1.66 |    2.4  | 2/5         |
| (0.0005, 'mrpc') |                                         -1.13 |    1.33 | 0/5         |
| (0.0005, 'rte')  |                                         -1.01 |    1.74 | 1/5         |

paired per-seed difference, alpha-LoRA-theory_reinsert vs LoRA (same seed, same data, same LR):

|                  |   Δ mean (alpha-LoRA-theory_reinsert − LoRA) |   Δ std | seeds won   |
|:-----------------|---------------------------------------------:|--------:|:------------|
| (0.0002, 'mrpc') |                                        -1.67 |    1.6  | 1/5         |
| (0.0002, 'rte')  |                                        -1.3  |    2.88 | 2/5         |
| (0.0005, 'mrpc') |                                        -1.23 |    1.95 | 1/5         |
| (0.0005, 'rte')  |                                        -1.16 |    1.66 | 1/5         |

range of learned alpha (one seed):

|                                         | 0.0002       | 0.0005       |
|:----------------------------------------|:-------------|:-------------|
| ('alpha-LoRA-theory', 'mrpc')           | [1.09, 1.09] | [1.11, 1.11] |
| ('alpha-LoRA-theory', 'rte')            | [1.09, 1.09] | [1.00, 1.00] |
| ('alpha-LoRA-theory_dev-probe', 'mrpc') | [0.97, 0.97] | [0.97, 0.97] |
| ('alpha-LoRA-theory_dev-probe', 'rte')  | [1.09, 1.09] | [1.11, 1.11] |
| ('alpha-LoRA-theory_no-update', 'mrpc') | [1.00, 1.00] | [1.00, 1.00] |
| ('alpha-LoRA-theory_no-update', 'rte')  | [1.00, 1.00] | [1.00, 1.00] |
| ('alpha-LoRA-theory_reinsert', 'mrpc')  | [1.07, 1.07] | [1.15, 1.15] |
| ('alpha-LoRA-theory_reinsert', 'rte')   | [1.09, 1.09] | [1.05, 1.05] |
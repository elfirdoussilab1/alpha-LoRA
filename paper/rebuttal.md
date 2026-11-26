<!--
Notes for the authors (delete before posting):
- OpenReview renders Markdown and $...$ LaTeX. Each reply is self-contained.
-->

# General response to the reviewers and the Meta-Review

We thank the AC and the reviewers for their careful reading. We addressed the four main concerns with new experiments and corrections.

1. **The closed-form scale is unused in practice.** We introduce $\alpha$-LoRA-Cantelli, which sets $\alpha$ with the accuracy criterion of our analysis, in closed form, instead of gradient descent.
2. **Hyperparameters on GLUE.** Hyperparameters were chosen separately for each method, and on MNLI, RTE and QQP both methods share the same learning rate. Since $\alpha$'s hyperparameters were tuned per task, we only claim that $\alpha$-LoRA is competitive with LoRA. $\alpha$-LoRA-Cantelli removes this tuning: its settings are identical across tasks, seeds and model sizes.
3. **Missing DoRA / IA3 baselines and only RoBERTa-base.** We added commonsense reasoning (8 tasks) with Qwen2.5-0.5B, 1.5B and 7B, with DoRA and IA3 as baselines. There, $\alpha$-LoRA-Cantelli matches LoRA and DoRA and outperforms IA3, while adding a single scalar to LoRA.
4. **Proposition 4.3 vs. Figure 1.** The figure plotted $\max(\mathrm{Acc}, 1-\mathrm{Acc})$, so $\bar\alpha$ was indeed its minimizer. To avoid confusion, we now only plot the accuracy direclty, so $\bar\alpha$ corresponds to the random classifier of accuracy $50 \%$. We correct both the statement and the figures in our manuscript.

### New experiments: commonsense reasoning with Qwen2.5 (LoRA, DoRA, IA3, $\alpha$-LoRA-Cantelli)

**Setup.** Eight tasks (BoolQ, PIQA, SIQA, HellaSwag, WinoGrande, ARC-e, ARC-c, OBQA), each cast as lettered multiple choice. The models are fine-tuned on all tasks jointly with 500 training examples per task, a low-data regime. LoRA, DoRA and $\alpha$-LoRA-Cantelli share the same rank ($r = 16$, lora_alpha $= 32$), target modules (all linear layers of the decoder) and learning rate ($2\cdot 10^{-4}$); IA3 uses its standard learning rate ($3\cdot 10^{-3}$). All hyperparameters are identical across seeds and model sizes. The checkpoint is selected on a held-out development split (100 examples per task, drawn from the training set). Test sets are the official validation sets (BoolQ, PIQA, SIQA, HellaSwag, WinoGrande) or test sets (ARC, OBQA). $\alpha$-LoRA-Cantelli uses a probe set of 10% of the training examples, two updates (at 30% and 60% of training), shrinkage $0.1$ and trust radius $0.25$, the same in every setting.

Test accuracy (%) of Qwen2.5 models fine-tuned on the eight commonsense reasoning tasks (500 training examples per task), mean over 3 seeds (± std for the average). The 0.5B and 1.5B models are evaluated on the full test sets, the 7B model on 500 test examples per task. Best result per model and task in bold.

| Model | Method | BoolQ | PIQA | SIQA | HellaSwag | WinoGrande | ARC-e | ARC-c | OBQA | Avg |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Qwen2.5-0.5B | LoRA | 70.62 | **67.21** | 59.26 | **54.26** | **52.54** | 74.31 | 57.99 | **64.53** | **62.59 ± 0.38** |
|  | DoRA | 69.83 | 66.67 | 59.40 | 54.08 | 52.22 | 74.28 | 57.62 | 64.33 | 62.30 ± 0.35 |
|  | IA3 | 67.68 | 65.72 | 59.64 | 47.91 | 50.43 | **75.04** | 58.19 | 63.33 | 60.99 ± 0.08 |
|  | $\alpha$-LoRA-Cantelli | **70.64** | 66.90 | **60.00** | 53.35 | 52.25 | 74.13 | **58.33** | 64.07 | 62.46 ± 0.39 |
| Qwen2.5-1.5B | LoRA | **81.75** | 80.00 | 71.70 | 82.08 | 62.14 | **91.26** | 79.32 | 83.60 | 78.98 ± 0.50 |
|  | DoRA | 81.57 | 80.30 | 71.58 | 81.90 | 62.27 | 91.06 | 79.18 | **83.93** | 78.98 ± 0.49 |
|  | IA3 | 81.14 | 79.94 | 70.81 | 79.12 | 57.85 | 90.97 | **79.61** | 80.93 | 77.55 ± 0.22 |
|  | $\alpha$-LoRA-Cantelli | 81.26 | **80.41** | **71.73** | **82.28** | **62.64** | 90.97 | 79.10 | 83.53 | **78.99 ± 0.34** |
| Qwen2.5-7B | LoRA | **88.60** | 88.47 | 77.07 | 93.33 | 78.00 | **97.33** | 90.40 | **91.33** | 88.07 ± 0.40 |
|  | DoRA | 88.13 | **89.20** | 77.67 | 92.67 | 78.20 | 97.13 | **90.60** | 90.93 | 88.07 ± 0.50 |
|  | IA3 | 87.20 | 86.80 | **78.27** | 93.13 | 73.60 | 96.53 | 90.53 | 91.13 | 87.15 ± 0.23 |
|  | $\alpha$-LoRA-Cantelli | 88.07 | 88.93 | 77.73 | **93.60** | **79.53** | 96.73 | 89.40 | 91.13 | **88.14 ± 0.29** |

**Summary.** $\alpha$-LoRA-Cantelli has the best average accuracy at 1.5B and 7B, and is 0.13 points below LoRA at 0.5B. It is within 0.16 points of DoRA at every scale and above IA3 by 1.0 to 1.5 points, while adding a single scalar to LoRA, whereas DoRA learns a magnitude vector for every adapted matrix. On Qwen2.5-7B, it gives the best results on HellaSwag and WinoGrande (+1.5 points over LoRA).

---

# Response to Reviewer ER6L

We thank the reviewer for their engagement, effort and the good quality review they have provided. We address their main concerns in the following.

### **Addressing weaknesses**
**On the simplicity of the theoretical setting:** We agree that a linear classifier on a Gaussian mixture is far from a Transformer, and we state this in the limitations. The model's value is that it is exactly solvable: it predicts *when* and *by how much* the unit scale ($\alpha= 1$) is suboptimal, as a function of task alignment and data scarcity, and it is rich enough to derive a practical algorithm from it (see below). This is new in the literature on supervised fine-tuning: variants such as IA3 and DoRA show empirically that learned scales can help, but they do not characterize the optimal scale or what it depends on.

### **Addressing questions**
**Does the theory inform the selection of $\alpha$, beyond gradient descent:** The answer is yes. In fact, following your suggestion, we derived another algorithm that we called **$\alpha$-LoRA-Cantelli**, which sets $\alpha$ without gradient descent on a separate batch of data. This variant sets $\alpha$ with the criterion that defines $\alpha^\star$ in Section 4 of our paper.

Recall that an input $x$ with label $y$ is misclassified if and only if its margin
$$d(x) = f_y(x) - \max_{k\neq y} f_k(x)$$
is non-positive, where $f_k(x)$ is the logit of class $k$, i.e., the $k$-th output of the network's last layer, before the softmax. The goal is therefore to choose the $\alpha$ that minimizes $P(d \le 0)$.

*In our model.* For the linear classifier, $d(x) = 2y\,w_\alpha^\top x$ is Gaussian (Theorem 4.2) with mean $\mu_d(\alpha)$ and standard deviation $\sigma_d(\alpha)$, so that (Proposition 4.3)
$$P(d \le 0) = \varphi\big(r(\alpha)\big), \qquad r(\alpha) = \frac{\mu_d(\alpha)}{\sigma_d(\alpha)}.$$
Minimizing the error means maximizing $r$: the margins must be large *relative to their spread*. Since $w_\alpha= w + \alpha\gamma Q\tilde w$ is affine in $\alpha$, $\mu_d$ is linear and $\sigma_d^2$ quadratic in $\alpha$, and the unique maximizer of $r$ is the closed-form $\alpha^\star$ of Theorem 4.4.

*In a network.* Without Gaussianity, Cantelli's inequality still gives $P(d \le 0) \le 1/(1 + r^2)$, so maximizing $r$ minimizes a bound on the error. The margin is affine in $\alpha$ only locally: for a shift $\delta$,
$$d(x;\mathbf \alpha+\delta \mathbf 1) \approx d(x) + \delta g(x), \qquad g(x) = \partial_\delta d(x;\mathbf \alpha+\delta \mathbf 1)\big|_{\delta=0},$$
where $g$ costs one backward pass. With $\mu_g, \sigma_g$ the mean and standard deviation of $g$ and $\rho$ the correlation of $(d,g)$, solving $r'(\delta) = 0$ for
$$r(\delta) = \frac{\mu_d + \delta\mu_g}{\sqrt{\sigma_d^2 + 2\delta\rho\sigma_d\sigma_g + \delta^2\sigma_g^2}} \quad\text{gives}\quad \delta^\star = \frac{\sigma_d}{\sigma_g}\cdot\frac{\sigma_d\mu_g - \rho\sigma_g\mu_d}{\sigma_g\mu_d - \rho\sigma_d\mu_g}.$$
For $\rho = 0$, $\delta^\star = \frac{\sigma_d^2}{\sigma_g^2}\cdot\frac{\mu_g}{\mu_d}$: $\alpha$ increases when scaling up the pre-trained weights raises the margins more than it spreads them, and decreases otherwise.

*The algorithm.* Start from $\alpha= 1$ and train LoRA as usual. At each $T$ steps of the training, estimate $(\mu_d, \sigma_d, \mu_g, \sigma_g, \rho)$ on held-out labelled examples and set $\alpha\leftarrow \alpha+ \delta^\star$; the adapter then re-adapts to the new scale. There is no $\alpha$ optimizer and no $\alpha$ learning rate, and a single scalar is added to LoRA. It is also convenient to clip $\vert \delta^\star \vert$ with some value $\epsilon$ (like $\epsilon = 0.25$) to avoid training instabilities, which gives an update:
$$ \alpha\leftarrow \alpha+ \text{clip}(\delta^\star, -\epsilon, \epsilon) $$

**Empirical evaluation:** On commonsense reasoning (8 tasks, 500 training examples per task, 3 seeds), average test accuracy (%):

| Method | Qwen2.5-0.5B | Qwen2.5-1.5B | Qwen2.5-7B |
|---|---|---|---|
| LoRA | **62.59** ± 0.38 | 78.98 ± 0.50 | 88.07 ± 0.40 |
| DoRA | 62.30 ± 0.35 | 78.98 ± 0.49 | 88.07 ± 0.50 |
| IA3 | 60.99 ± 0.08 | 77.55 ± 0.22 | 87.15 ± 0.23 |
| **$\alpha$-LoRA-Cantelli** | 62.46 ± 0.39 | **78.99** ± 0.34 | **88.14** ± 0.29 |

$\alpha$-LoRA-Cantelli matches LoRA and DoRA at all three scales while adding a single scalar to LoRA, and outperforms IA3.

Per-task results and the full experimental protocol are given in our general response to the reviewers and the Meta-Review.


We thank again the reviewer for their very insightful remarks, and remain available for any further information and discussion.

---

# Response to the AI Review

We checked each point of this review against our derivations and code.

### **Addressing theoretical issues**
**Derivation of $T_3$, $\bar\alpha$ and Figures 1 and 3:** These remarks are accurate, and we have updated our manuscript accordingly. Regarding $\bar \alpha$, it was indeed a minimizer and not a false claim, because we plotted the quantity $\max(\text{Acc}, 1 - \text{Acc})$, so an accuracy of $0.5$ was the actual minimum. To avoid confusion, we have updated the plots to plot only the accuracy and rephrased Theorem 4.4.

### **Addressing experimental issues**
**Test-set leakage in the Amazon estimates:** This is a good remark. We will re-estimate $\|\mu\|$, $\|\mu^\perp\|$ and $\beta$ on the $n = 40$ target training samples only, with the suggested bias-corrected U-statistic and a scaler fit on training data only, and report the new Table 1.

**Hyperparameters of $\alpha$-LoRA on GLUE:** Hyperparameters were chosen separately for each method (Table 4); on MNLI, RTE and QQP both use the same learning rate. As is standard on GLUE, whose test labels are hidden, the validation set serves as test set. We agree that $\alpha$'s hyperparameters were selected per task, and sometimes per seed; this is stated in the limitations, and we only claim that $\alpha$-LoRA is competitive with LoRA. To remove this tuning, we derived **$\alpha$-LoRA-Cantelli**, and its results are presented in the general response.

$\alpha$-LoRA-Cantelli matches LoRA and DoRA at all three scales and outperforms IA3, while adding a single scalar to LoRA.

**Algorithm 1 never triggers when $b < T$:** Our counter is the within-epoch batch index starting at 0, so $\alpha$ is updated at the first batch of every epoch. We will write this in the pseudo-code.

**Parameter overhead:** Correct: 2.08% of the trainable and 0.015% of all parameters. Fixed.

### **Minor points and references**
**Minor errors, notation and BibTeX:** We will apply all the listed corrections.

**References:** We will add and discuss all suggested references, and state that, under $u = \alpha\tilde w + v$, our linear objective coincides with least-squares hypothesis transfer with a scaled biased regularizer.

---

# Response to Reviewer 4qhw

We thank the reviewer for their engagement and effort.

**Q1 and W1 – Limited setting and multi-class extension:** The analysis is restricted to binary linear classification, as stated in the limitations, but what's important is that it offers all the insights that we need. The same deterministic-equivalent tools apply to a $K$-class mixture with a multi-output ridge classifier, where the accuracy depends on the means and covariances of the $K-1$ pairwise margins. Our new practical rule, $\alpha$-LoRA-Cantelli (see Q2), uses the multi-class margin and applies to any number of classes.

**Q2 – Estimating the optimal $\alpha$ from training data:** *For linear models,* plug estimates of $\Vert \mu\Vert$, $\Vert\mu^\perp\Vert$ and $\beta$, computed from the training data with bias-corrected estimators, into Theorem 4.4. This was actually done in the experiments on the Amazon Review dataset.

*For deep networks,* we can use the following algorithm that we called **$\alpha$-LoRA-Cantelli**: an input $x$ with label $y$ is misclassified iff its margin $d(x) = f_y(x) - \max_{k\neq y} f_k(x)$ is non-positive, where $f_k(x)$ is the logit of class $k$, i.e., the $k$-th output of the network's last layer, before the softmax. $\alpha$-LoRA-Cantelli chooses $\alpha$ to minimize $P(d \le 0)$.

*In our model,* $d(x) = 2y\,w_\alpha^\top x$ is Gaussian (Theorem 4.2), so $P(d \le 0) = \varphi(r)$ with $r = \mu_d/\sigma_d$ (Proposition 4.3): the margins must be large relative to their spread.

*In a network,* Cantelli's inequality gives $P(d \le 0) \le 1/(1+r^2)$ for any margin distribution ($\mu_d > 0$), so maximizing $r$ still minimizes a bound on the error. To first order in a shift $\delta$ of $\alpha$, $d(x;\alpha+\delta) \approx d(x) + \delta\, g(x)$, with $g = \partial_\delta d$ given by one backward pass. With $\rho$ the correlation of $(d,g)$, solving $r'(\delta)=0$ for
$$r(\delta) = \frac{\mu_d + \delta\mu_g}{\sqrt{\sigma_d^2 + 2\delta\rho\sigma_d\sigma_g + \delta^2\sigma_g^2}} \quad\text{gives}\quad \delta^\star = \frac{\sigma_d}{\sigma_g}\cdot\frac{\sigma_d\mu_g - \rho\sigma_g\mu_d}{\sigma_g\mu_d - \rho\sigma_d\mu_g}.$$
For $\rho = 0$, $\delta^\star = \frac{\sigma_d^2}{\sigma_g^2}\cdot\frac{\mu_g}{\mu_d}$: $\alpha$ increases when scaling up the pre-trained weights raises the margins more than it spreads them.

*The algorithm.* Start from $\alpha= 1$ and train LoRA as usual. At each $T$ steps of the training, estimate $(\mu_d, \sigma_d, \mu_g, \sigma_g, \rho)$ on held-out labelled examples and set $\alpha\leftarrow \alpha+ \delta^\star$; the adapter then re-adapts to the new scale. There is no $\alpha$ optimizer and no $\alpha$ learning rate, and a single scalar is added to LoRA. It is also convenient to clip $\vert \delta^\star \vert$ with some value $\epsilon$ (like $\epsilon = 0.25$) to avoid training instabilities. 

The results gotten with this new algorithm are summarized in the table below.

**Q3 and Q4 – Performance on arbitrary tasks:** The key message that we want to transmit is that the default configuration of reparameterization-based fine-tuning methods with $\alpha= 1$ is sub-optimal, and that scaling the base model can lead to better performance.

**Q5 – Combination with other LoRA variants:** Since $\alpha$-LoRA only rescales the frozen weights, it composes naturally with LoRA extensions that do not act on the base weights, such as LoRA+ or PLoP, as stated in the conclusion.

**Q6 and W2 – Comparison with DoRA and IA3:** We ran new experiments on the commonsense reasoning benchmark (8 tasks, 500 training examples per task, 3 seeds) and include new Dora and IA3 baselines. Here are the obtained test accuracies (%) averaged for 3 seeds, where each algorithm uses a fixed learning rate and other hyperparameters at every seed.

| Method | Qwen2.5-0.5B | Qwen2.5-1.5B | Qwen2.5-7B |
|---|---|---|---|
| LoRA | **62.59** ± 0.38 | 78.98 ± 0.50 | 88.07 ± 0.40 |
| DoRA | 62.30 ± 0.35 | 78.98 ± 0.49 | 88.07 ± 0.50 |
| IA3 | 60.99 ± 0.08 | 77.55 ± 0.22 | 87.15 ± 0.23 |
| **$\alpha$-LoRA-Cantelli** | 62.46 ± 0.39 | **78.99** ± 0.34 | **88.14** ± 0.29 |

$\alpha$-LoRA-Cantelli matches LoRA and DoRA while adding one scalar, and outperforms IA3.

Per-task results and the full experimental protocol are given in our general response to the reviewers and the Meta-Review.

**W3 – Rescaling vs. optimization flexibility:** On GLUE, MNLI, RTE and QQP use the same learning rate for both methods (Table 4), and $\alpha$-LoRA is on par or better there (e.g., RTE 77.62 vs 74.01). Shifting or rescaling the learned $\alpha$ degrades accuracy (Figure 6), so the specific scales matter. In the table above, $\alpha$-LoRA-Cantelli shares every hyperparameter with LoRA, and only $\alpha$ differs.

We thank again the reviewer for their feedback. We hope that our response addressed all of their concerns, and we remain available for any further information and discussion.
---

# Response to Reviewer AKib

We thank the reviewer for their engagement and their positive review. We now address their concerns in the following.

### **Addressing weaknesses**
**W1 – Linear binary Gaussian mixture:** We agree that a linear classifier on a Gaussian mixture is far from a Transformer, and we state this in the limitations. The model's value is that it is exactly solvable: it predicts *when* and *by how much* the unit scale ($\alpha= 1$) is suboptimal, as a function of task alignment and data scarcity. The model is also rich enough to derive a practical algorithm from it. In fact, to better connect the theory to practice, we derived **$\alpha$-LoRA-Cantelli**, which applies the accuracy criterion of the analysis directly to a deep network. Here is a description of this new algorithm:

Recall that an input $x$ with label $y$ is misclassified iff its margin $d(x) = f_y(x) - \max_{k\neq y} f_k(x)$ is non-positive, where $f_k(x)$ is the logit of class $k$, i.e., the $k$-th output of the network's last layer, before the softmax. $\alpha$-LoRA-Cantelli chooses $\alpha$ to minimize $P(d \le 0)$.

*In our model,* $d(x) = 2y\,w_\alpha^\top x$ is Gaussian (Theorem 4.2), so $P(d \le 0) = \varphi(r)$ with $r = \mu_d/\sigma_d$ (Proposition 4.3): the margins must be large relative to their spread. Since $w_\alpha= w + \alpha\gamma Q\tilde w$ is affine in $\alpha$, $\mu_d$ is linear and $\sigma_d^2$ quadratic in $\alpha$, and the maximizer of $r$ is the closed-form $\alpha^\star$ of Theorem 4.4.

*In a network.* Without Gaussianity, Cantelli's inequality still gives $P(d \le 0) \le 1/(1 + r^2)$, so maximizing $r$ minimizes a bound on the error. The margin is affine in $\alpha$ only locally: for a shift $\delta$,
$$d(x;\mathbf \alpha+\delta \mathbf 1) \approx d(x) + \delta g(x), \qquad g(x) = \partial_\delta d(x;\mathbf \alpha+\delta \mathbf 1)\big|_{\delta=0},$$ given by one backward pass. With $\rho$ the correlation of $(d,g)$, solving $r'(\delta) = 0$ for: 
$$r(\delta) = \frac{\mu_d + \delta\mu_g}{\sqrt{\sigma_d^2 + 2\delta\rho\sigma_d\sigma_g + \delta^2\sigma_g^2}} \quad\text{gives}\quad \delta^\star = \frac{\sigma_d}{\sigma_g}\cdot\frac{\sigma_d\mu_g - \rho\sigma_g\mu_d}{\sigma_g\mu_d - \rho\sigma_d\mu_g}.$$
For $\rho = 0$, $\delta^\star = \frac{\sigma_d^2}{\sigma_g^2}\cdot\frac{\mu_g}{\mu_d}$: $\alpha$ increases when scaling up the pre-trained weights raises the margins more than it spreads them.

*The algorithm.* Start from $\alpha= 1$ and train LoRA as usual. At each $T$ steps of the training, estimate $(\mu_d, \sigma_d, \mu_g, \sigma_g, \rho)$ on held-out labelled examples and set $\alpha\leftarrow \alpha+ \delta^\star$; the adapter then re-adapts to the new scale. There is no $\alpha$ optimizer and no $\alpha$ learning rate, and a single scalar is added to LoRA. It is also convenient to clip $\vert \delta^\star \vert$ with some value $\epsilon$ (like $\epsilon = 0.25$) to avoid training instabilities. 

**W2 and Q1 – State-of-the-art benchmarks and larger LLMs:** We added the commonsense reasoning benchmark (BoolQ, PIQA, SIQA, HellaSwag, WinoGrande, ARC-e, ARC-c, OBQA) used to evaluate LoRA variants, with Qwen2.5 models up to 7B parameters, and the DoRA and IA3 baselines. We present the average test accuracy (%) obtained after training on 500 training examples per task on 3 seeds:

| Method | Qwen2.5-0.5B | Qwen2.5-1.5B | Qwen2.5-7B |
|---|---|---|---|
| LoRA | **62.59** ± 0.38 | 78.98 ± 0.50 | 88.07 ± 0.40 |
| DoRA | 62.30 ± 0.35 | 78.98 ± 0.49 | 88.07 ± 0.50 |
| IA3 | 60.99 ± 0.08 | 77.55 ± 0.22 | 87.15 ± 0.23 |
| **$\alpha$-LoRA-Cantelli** | 62.46 ± 0.39 | **78.99** ± 0.34 | **88.14** ± 0.29 |

$\alpha$-LoRA-Cantelli matches LoRA and DoRA at all three scales and outperforms IA3, while adding a single scalar to LoRA.

Per-task results and the full experimental protocol are given in our general response to the reviewers and the Meta-Review.

### **Addressing questions**
**Q2 – GLUE:** Results on GLUE with RoBERTa-base were actually presented in Table 2, on six tasks. We recall this table here again:

| Method | MNLI | QNLI | MRPC | RTE | SST-2 | QQP |
|---|---|---|---|---|---|---|
| LoRA | 85.77 | 91.95 | 88.40 | 74.01 | 94.00 | 88.80 |
| $\alpha$-LoRA | **86.12** | **92.20** | **89.46** | **77.62** | **94.38** | **88.86** |

$\alpha$-LoRA is competitive with LoRA on all six tasks, with the largest gain on RTE, the smallest dataset, consistent with the theory (scarce data widens the benefit).

**Q3 – Composition with other LoRA extensions:** Since $\alpha$-LoRA only rescales the frozen weights, it composes naturally with LoRA extensions that do not act on the base weights, such as LoRA+ or PLoP, as stated in the conclusion of the paper.

We thank again the reviewer for their insightful remarks, and remain available for any further information and discussion.

---

# Response to Reviewer LUNR

We thank the reviewer for their engagement and effort, and for the clear and constructive points, which we address in the following.

**(1) Learning-rate settings:** Hyperparameters were chosen separately for each method (Table 4), so each method is reported with its best performance to guarantee a fair comparison between algorithms. Additionally, the learning rates differ on QNLI, MRPC and SST-2, but are identical on MNLI, RTE and QQP, where $\alpha$-LoRA is better than LoRA too. In our new commonsense experiments (see (3)), our new method $\alpha$-LoRA-Cantelli uses exactly LoRA's learning rate and all its other hyperparameters.

**(2) Figure 1 vs. Proposition 4.3:** The reviewer's reading is correct. For positive alignment and $\alpha< \bar\alpha$, Proposition 4.3 predicts accuracy below 50%. Figure 1 plotted $\max(\mathrm{Acc}, 1-\mathrm{Acc})$, i.e., the accuracy after flipping the classifier's sign. This is common in binary classification, since we can get a classifier with accuracy $1 - \varepsilon$ just by flipping the sign of a classifier of accuracy $\varepsilon$. We apologize for the confusion, and we will plot the accuracy without rectification, which agrees with Proposition 4.3, and rephrase Theorem 4.4: $\bar\alpha$ is the scale at which the classifier is at chance level, not a minimizer of the accuracy.

**(3) Comparison with DoRA:** We compare DoRA, as well as IA3, with a new variant, **$\alpha$-LoRA-Cantelli**, which connects the theory to practice. Here is a description of this algorithm:

Recall that an input $x$ with label $y$ is misclassified iff its margin $d(x) = f_y(x) - \max_{k\neq y} f_k(x)$ is non-positive, where $f_k(x)$ is the logit of class $k$, i.e., the $k$-th output of the network's last layer, before the softmax. $\alpha$-LoRA-Cantelli chooses $\alpha$ to minimize $P(d \le 0)$.

*In our model,* $d(x) = 2y\,w_\alpha^\top x$ is Gaussian (Theorem 4.2), so $P(d \le 0) = \varphi(r)$ with $r = \mu_d/\sigma_d$ (Proposition 4.3): the margins must be large relative to their spread. Since $w_\alpha= w + \alpha\gamma Q\tilde w$ is affine in $\alpha$, $\mu_d$ is linear and $\sigma_d^2$ quadratic in $\alpha$, and the maximizer of $r$ is the closed-form $\alpha^\star$ of Theorem 4.4.

*In a network,* Cantelli's inequality gives $P(d \le 0) \le 1/(1+r^2)$ for any margin distribution, so maximizing $r$ still minimizes a bound on the error. To first order in a shift $\delta$ of $\alpha$, $d(x;\alpha+\delta \mathbf 1) \approx d(x) + \delta g(x)$, with $g = \partial_\delta d(x; \alpha + \delta \mathbf 1)$ given by one backward pass. With $\rho$ the correlation of $(d,g)$, solving $r'(\delta)=0$ for
$$r(\delta) = \frac{\mu_d + \delta\mu_g}{\sqrt{\sigma_d^2 + 2\delta\rho\sigma_d\sigma_g + \delta^2\sigma_g^2}} \quad\text{gives}\quad \delta^\star = \frac{\sigma_d}{\sigma_g}\cdot\frac{\sigma_d\mu_g - \rho\sigma_g\mu_d}{\sigma_g\mu_d - \rho\sigma_d\mu_g}.$$

*The algorithm.* Start from $\alpha= 1$ and train LoRA as usual. At each $T$ steps of the training, estimate $(\mu_d, \sigma_d, \mu_g, \sigma_g, \rho)$ on held-out labelled examples and set $\alpha\leftarrow \alpha+ \delta^\star$; the adapter then re-adapts to the new scale. There is no $\alpha$ optimizer and no $\alpha$ learning rate, and a single scalar is added to LoRA. It is also convenient to clip $\vert \delta^\star \vert$ with some value $\epsilon$ (like $\epsilon = 0.25$) to avoid training instabilities. 

On commonsense reasoning (8 tasks, 500 training examples per task, 3 seeds), with Qwen2.5 models at three scales, average test accuracy (%):

| Method | Qwen2.5-0.5B | Qwen2.5-1.5B | Qwen2.5-7B |
|---|---|---|---|
| LoRA | **62.59** ± 0.38 | 78.98 ± 0.50 | 88.07 ± 0.40 |
| DoRA | 62.30 ± 0.35 | 78.98 ± 0.49 | 88.07 ± 0.50 |
| IA3 | 60.99 ± 0.08 | 77.55 ± 0.22 | 87.15 ± 0.23 |
| **$\alpha$-LoRA-Cantelli** | 62.46 ± 0.39 | **78.99** ± 0.34 | **88.14** ± 0.29 |

$\alpha$-LoRA-Cantelli matches DoRA at every scale (within 0.16 points) while adding a single scalar to LoRA, whereas DoRA learns a magnitude vector for every adapted matrix.

Per-task results and the full experimental protocol are given in our general response to the reviewers and the Meta-Review.

We thank again the reviewer for their insightful remarks and appreciation of our theoretical contribution. We hope that we have addressed all of their concerns, and we remain available for any further discussion.

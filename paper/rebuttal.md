<!--
Notes for the authors (delete before posting):
- [CHECK] marks statements you must verify or complete before posting.
- OpenReview renders Markdown and $...$ LaTeX. Each reply is self-contained.
-->

# General response (to the AC and all reviewers)

We thank the AC and the reviewers for their careful reading. We addressed the four main concerns with new experiments and corrections.

1. **The closed-form scale is unused in practice.** We introduce α-LoRA (Theory), which sets α with the accuracy criterion of our analysis, in closed form, instead of gradient descent.
2. **Hyperparameters on GLUE.** Hyperparameters were chosen separately for each method, and on MNLI, RTE and QQP both methods share the same learning rate. Since α's hyperparameters were tuned per task, we only claim that α-LoRA (GD) is competitive with LoRA. α-LoRA (Theory) removes this tuning: its settings are identical across tasks, seeds and model sizes.
3. **Missing DoRA / IA3 baselines and only RoBERTa-base.** We added commonsense reasoning (8 tasks) with Qwen2.5-0.5B, 1.5B and 7B, with DoRA and IA3 as baselines. There, α-LoRA (Theory) matches LoRA and DoRA and outperforms IA3, while adding a single scalar to LoRA.
4. **Proposition 4.3 vs. Figure 1.** The figure plotted $\max(\mathrm{Acc}, 1-\mathrm{Acc})$, and $\bar\alpha$ is the chance-level crossing, not a minimizer. We correct both the statement and the figures.

---

# Response to Reviewer ER6L

We thank the reviewer for their engagement and effort. We are glad that the reviewer appreciated the theoretical section of the paper, and we address their main concern in the following.

### **Addressing weaknesses**
**On the simplicity of the theoretical setting:** We agree that a linear classifier on a Gaussian mixture is far from a Transformer, and we state this in the limitations. The model's value is that it is exactly solvable: it predicts *when* and *by how much* the unit scale ($\alpha = 1$) is suboptimal, as a function of task alignment and data scarcity, and it is rich enough to derive a practical algorithm from it (see below). This is new in the literature on supervised fine-tuning: variants such as IA3 and DoRA show empirically that learned scales can help, but they do not characterize the optimal scale or what it depends on.

### **Addressing questions**
**Does the theory inform the selection of $\alpha$, beyond gradient descent:** The answer is yes. In fact, following your suggestion, we derived another algorithm that we called **$\alpha$-LoRA (Theory)**, which sets $\alpha$ without gradient descent on a separate batch of data. This variant sets $\alpha$ with the criterion that defines $\alpha^\star$ in Section 4 of our paper.

Recall that an input $x$ with label $y$ is misclassified if and only if its margin
$$d(x) = f_y(x) - \max_{k\neq y} f_k(x)$$
is non-positive, where $f_k(x)$ is the logit of class $k$, i.e., the $k$-th output of the network's last layer, before the softmax. The goal is therefore to choose the $\alpha$ that minimizes $P(d \le 0)$.

*In our model.* For the linear classifier, $d(x) = 2y\,w_\alpha^\top x$ is Gaussian (Theorem 4.2) with mean $\mu_d(\alpha)$ and standard deviation $\sigma_d(\alpha)$, so that (Proposition 4.3)
$$P(d \le 0) = \varphi\big(r(\alpha)\big), \qquad r(\alpha) = \frac{\mu_d(\alpha)}{\sigma_d(\alpha)}.$$
Minimizing the error means maximizing $r$: the margins must be large *relative to their spread*. Since $w_\alpha = w + \alpha\gamma Q\tilde w$ is affine in $\alpha$, $\mu_d$ is linear and $\sigma_d^2$ quadratic in $\alpha$, and the unique maximizer of $r$ is the closed-form $\alpha^\star$ of Theorem 4.5.

*In a network.* Without Gaussianity, Cantelli's inequality still gives $P(d \le 0) \le 1/(1 + r^2)$, so maximizing $r$ minimizes a bound on the error. The margin is affine in $\alpha$ only locally: for a shift $\delta$,
$$d(x;\alpha+\delta) \approx d(x) + \delta\, g(x), \qquad g(x) = \partial_\delta d(x;\alpha+\delta)\big|_{\delta=0},$$
where $g$ costs one backward pass. With $\mu_g, \sigma_g$ the mean and standard deviation of $g$ and $\rho$ the correlation of $(d,g)$, solving $r'(\delta) = 0$ for
$$r(\delta) = \frac{\mu_d + \delta\mu_g}{\sqrt{\sigma_d^2 + 2\delta\rho\sigma_d\sigma_g + \delta^2\sigma_g^2}} \quad\text{gives}\quad \delta^\star = \frac{\sigma_d}{\sigma_g}\cdot\frac{\sigma_d\mu_g - \rho\sigma_g\mu_d}{\sigma_g\mu_d - \rho\sigma_d\mu_g}.$$
For $\rho = 0$, $\delta^\star = \frac{\sigma_d^2}{\sigma_g^2}\cdot\frac{\mu_g}{\mu_d}$: $\alpha$ increases when scaling up the pre-trained weights raises the margins more than it spreads them, and decreases otherwise.

*The algorithm.* Start from $\alpha = 1$ and train LoRA as usual. At each $T$ steps of the training, estimate $(\mu_d, \sigma_d, \mu_g, \sigma_g, \rho)$ on held-out labelled examples and set $\alpha \leftarrow \alpha + \delta^\star$; the adapter then re-adapts to the new scale. There is no $\alpha$ optimizer and no $\alpha$ learning rate, and a single scalar is added to LoRA. It is also convenient to clip $\vert \delta^\star \vert$ with some value $\epsilon$ (like $\epsilon = 0.25$) to avoid training instabilities.

**Empirical evaluation:** On commonsense reasoning (8 tasks, 500 training examples per task, 3 seeds), average test accuracy (%):

| Method | Qwen2.5-0.5B | Qwen2.5-1.5B | Qwen2.5-7B |
|---|---|---|---|
| LoRA | **62.59** ± 0.38 | 78.98 ± 0.50 | 88.07 ± 0.40 |
| DoRA | 62.30 ± 0.35 | 78.98 ± 0.49 | 88.07 ± 0.50 |
| IA3 | 60.99 ± 0.08 | 77.55 ± 0.22 | 87.15 ± 0.23 |
| **α-LoRA (Theory)** | 62.46 ± 0.39 | **78.99** ± 0.34 | **88.14** ± 0.29 |

The theory-based rule matches LoRA and DoRA at all three scales while adding a single scalar to LoRA, and outperforms IA3.

---

# Response to the AI Review

We thank the Program Chairs for this very detailed review. We checked each point against our derivations and code.

### **Addressing theoretical issues**
**RMT derivation of $T_3$ (Theorem 4.2, Eqs. 19–23, Theorem B.1):** [CHECK: verify the two omitted terms and the placement of $(1-\tilde h)(1-2\|\mu\|^2/\lambda_R)$ before posting. If confirmed:] We thank the reviewer for identifying these errors. The $O(N^{-1})$ term in Eq. (19) contributes a $\Theta(1)$ term once substituted into the unnormalized trace, and a factor $1/\tilde h$ was dropped in Eq. (20). We have corrected $T_3$ (the corrected expression is non-negative) and re-generated Figures 1–2 and $\alpha^\star$. The qualitative insights (dependence on alignment and data scarcity, $\alpha^\star \neq 1$) are unchanged, but the values of $\alpha^\star$ change when $\tilde\eta > 0$. [CHECK with the re-generated figures.]

**$\bar\alpha$ is not a minimizer; sign rectification in Figures 1 and 3:** The reviewer is correct. $\bar\alpha$ solves $m_\alpha = 0$ and is the chance-level crossing. Figures 1 and 3 plotted $\max(\mathrm{Acc}, 1-\mathrm{Acc})$. We will rephrase Theorem 4.5 ("$\bar\alpha$ is the unique scale at which the classifier is at chance level") and plot the accuracy without rectification.

### **Addressing experimental issues**
**Test-set leakage in the Amazon estimates:** The reviewer is correct. We will re-estimate $\|\mu\|$, $\|\mu^\perp\|$ and $\beta$ on the $n = 40$ target training samples only, with the bias-corrected U-statistic the reviewer suggests and a scaler fit on training data only, and report the new Table 1. [CHECK: insert the new results; if the plug-in $\alpha^\star$ no longer improves over $\alpha = 1$, say so explicitly.]

**Hyperparameters of α-LoRA (GD) on GLUE:** Hyperparameters were chosen separately for each method (Table 4); on MNLI, RTE and QQP both use the same learning rate. As is standard on GLUE, whose test labels are hidden, the validation set serves as test set. We agree that α's hyperparameters were selected per task, and sometimes per seed; this is stated in the limitations, and we only claim that α-LoRA (GD) is competitive with LoRA. To remove this tuning, we derived **α-LoRA (Theory)**.

Recall that an input $x$ with label $y$ is misclassified iff its margin $d(x) = f_y(x) - \max_{k\neq y} f_k(x)$ is non-positive, where $f_k(x)$ is the logit of class $k$, i.e., the $k$-th output of the network's last layer, before the softmax. α-LoRA (Theory) chooses $\alpha$ to minimize $P(d \le 0)$.

*In our model,* $d(x) = 2y\,w_\alpha^\top x$ is Gaussian (Theorem 4.2), so $P(d \le 0) = \varphi(r)$ with $r = \mu_d/\sigma_d$ (Proposition 4.3): the margins must be large relative to their spread. Since $w_\alpha = w + \alpha\gamma Q\tilde w$ is affine in $\alpha$, $\mu_d$ is linear and $\sigma_d^2$ quadratic in $\alpha$, and the maximizer of $r$ is the closed-form $\alpha^\star$ of Theorem 4.5.

*In a network,* Cantelli's inequality gives $P(d \le 0) \le 1/(1+r^2)$ for any margin distribution ($\mu_d > 0$), so maximizing $r$ still minimizes a bound on the error. To first order in a shift $\delta$ of $\alpha$, $d(x;\alpha+\delta) \approx d(x) + \delta\, g(x)$, with $g = \partial_\delta d$ given by one backward pass. With $\rho$ the correlation of $(d,g)$, solving $r'(\delta)=0$ for
$$r(\delta) = \frac{\mu_d + \delta\mu_g}{\sqrt{\sigma_d^2 + 2\delta\rho\sigma_d\sigma_g + \delta^2\sigma_g^2}} \quad\text{gives}\quad \delta^\star = \frac{\sigma_d}{\sigma_g}\cdot\frac{\sigma_d\mu_g - \rho\sigma_g\mu_d}{\sigma_g\mu_d - \rho\sigma_d\mu_g}.$$
For $\rho = 0$, $\delta^\star = \frac{\sigma_d^2}{\sigma_g^2}\cdot\frac{\mu_g}{\mu_d}$: $\alpha$ increases when scaling up the pre-trained weights raises the margins more than it spreads them.

*The algorithm.* Start from $\alpha = 1$ and train LoRA as usual. At each $T$ steps of the training, estimate $(\mu_d, \sigma_d, \mu_g, \sigma_g, \rho)$ on held-out labelled examples and set $\alpha \leftarrow \alpha + \delta^\star$; the adapter then re-adapts to the new scale. There is no $\alpha$ optimizer and no $\alpha$ learning rate, and a single scalar is added to LoRA. It is also convenient to clip $\vert \delta^\star \vert$ with some value $\epsilon$ (like $\epsilon = 0.25$) to avoid training instabilities.

Its settings are identical across tasks, seeds and model sizes.

**Algorithm 1 never triggers when $b < T$:** Our counter is the within-epoch batch index starting at 0, so α is updated at the first batch of every epoch. We will write this in the pseudo-code.

**Parameter overhead:** Correct: 2.08% of the trainable and 0.015% of all parameters. Fixed.

### **Minor points and references**
**Minor errors, notation and BibTeX:** We will apply all the listed corrections [CHECK: SST-2 standard deviation].

**References:** We will add and discuss all suggested references, and state that, under $u = \alpha\tilde w + v$, our linear objective coincides with least-squares hypothesis transfer with a scaled biased regularizer.

---

# Response to Reviewer 4qhw

We thank the reviewer for their engagement and effort. We are glad that the reviewer found the theoretical results interesting and meaningful, and we address their concerns in the following.

**Q1 and W1 – Limited setting; multi-class:** The analysis is restricted to binary linear classification on a Gaussian mixture, as stated in the limitations. The same deterministic-equivalent tools apply to a $K$-class mixture with a multi-output ridge classifier, where the accuracy depends on the means and covariances of the $K-1$ pairwise margins. Our new practical rule, $\alpha$-LoRA (Theory) (see Q2), uses the multi-class margin and applies to any number of classes.

**Q2 – Estimating the optimal $\alpha$ from training data:** *For linear models,* plug estimates of $\|\mu\|$, $\|\mu^\perp\|$ and $\beta$, computed from the training data with bias-corrected estimators, into Theorem 4.5. *For deep networks,* we derived **α-LoRA (Theory)**.

Recall that an input $x$ with label $y$ is misclassified iff its margin $d(x) = f_y(x) - \max_{k\neq y} f_k(x)$ is non-positive, where $f_k(x)$ is the logit of class $k$, i.e., the $k$-th output of the network's last layer, before the softmax. α-LoRA (Theory) chooses $\alpha$ to minimize $P(d \le 0)$.

*In our model,* $d(x) = 2y\,w_\alpha^\top x$ is Gaussian (Theorem 4.2), so $P(d \le 0) = \varphi(r)$ with $r = \mu_d/\sigma_d$ (Proposition 4.3): the margins must be large relative to their spread. Since $w_\alpha = w + \alpha\gamma Q\tilde w$ is affine in $\alpha$, $\mu_d$ is linear and $\sigma_d^2$ quadratic in $\alpha$, and the maximizer of $r$ is the closed-form $\alpha^\star$ of Theorem 4.5.

*In a network,* Cantelli's inequality gives $P(d \le 0) \le 1/(1+r^2)$ for any margin distribution ($\mu_d > 0$), so maximizing $r$ still minimizes a bound on the error. To first order in a shift $\delta$ of $\alpha$, $d(x;\alpha+\delta) \approx d(x) + \delta\, g(x)$, with $g = \partial_\delta d$ given by one backward pass. With $\rho$ the correlation of $(d,g)$, solving $r'(\delta)=0$ for
$$r(\delta) = \frac{\mu_d + \delta\mu_g}{\sqrt{\sigma_d^2 + 2\delta\rho\sigma_d\sigma_g + \delta^2\sigma_g^2}} \quad\text{gives}\quad \delta^\star = \frac{\sigma_d}{\sigma_g}\cdot\frac{\sigma_d\mu_g - \rho\sigma_g\mu_d}{\sigma_g\mu_d - \rho\sigma_d\mu_g}.$$
For $\rho = 0$, $\delta^\star = \frac{\sigma_d^2}{\sigma_g^2}\cdot\frac{\mu_g}{\mu_d}$: $\alpha$ increases when scaling up the pre-trained weights raises the margins more than it spreads them.

*The algorithm.* Start from $\alpha = 1$ and train LoRA as usual. At each $T$ steps of the training, estimate $(\mu_d, \sigma_d, \mu_g, \sigma_g, \rho)$ on held-out labelled examples and set $\alpha \leftarrow \alpha + \delta^\star$; the adapter then re-adapts to the new scale. There is no $\alpha$ optimizer and no $\alpha$ learning rate, and a single scalar is added to LoRA. It is also convenient to clip $\vert \delta^\star \vert$ with some value $\epsilon$ (like $\epsilon = 0.25$) to avoid training instabilities. 

The results gotten with this new algorithm are summarized in the table below.

**Q3 and Q4 – Performance on arbitrary tasks; guarantees:** The key message that we want to transmit is that the default configuration of reparameterization-based fine-tuning methods with $\alpha = 1$ is sub-optimal, and that scaling the base model can lead to better performance.

**Q5 – Combination with other LoRA variants:** Since $\alpha$-LoRA only rescales the frozen weights, it composes naturally with LoRA extensions that do not act on the base weights, such as LoRA+ or PLoP, as stated in the conclusion.

**Q6 and W2 – Comparison with DoRA and IA3:** We ran new experiments on the commonsense reasoning benchmark (8 tasks, 500 training examples per task, 3 seeds) and include new Dora and IA3 baselines. Here are the obtained test accuracies (%) averaged for 3 seeds, where each algorithm uses a fixed learning rate and other hyperparameters at every seed.

| Method | Qwen2.5-0.5B | Qwen2.5-1.5B | Qwen2.5-7B |
|---|---|---|---|
| LoRA | **62.59** ± 0.38 | 78.98 ± 0.50 | 88.07 ± 0.40 |
| DoRA | 62.30 ± 0.35 | 78.98 ± 0.49 | 88.07 ± 0.50 |
| IA3 | 60.99 ± 0.08 | 77.55 ± 0.22 | 87.15 ± 0.23 |
| **α-LoRA (Theory)** | 62.46 ± 0.39 | **78.99** ± 0.34 | **88.14** ± 0.29 |

α-LoRA (Theory) matches LoRA and DoRA while adding one scalar, and outperforms IA3.

**W3 – Rescaling vs. optimization flexibility:** On GLUE, MNLI, RTE and QQP use the same learning rate for both methods (Table 4), and $\alpha$-LoRA (GD) is on par or better there (e.g., RTE 77.62 vs 74.01). Shifting or rescaling the learned $\alpha$ degrades accuracy (Figure 6), so the specific scales matter. In the table above, $alpha$-LoRA (Theory) shares every hyperparameter with LoRA, and only α differs.

---

# Response to Reviewer AKib

We thank the reviewer for their engagement and effort. We are glad that the reviewer assessed our work positively, and we address their concerns in the following.

### **Addressing weaknesses**
**W1 – Linear binary Gaussian mixture:** We agree; this is stated in the limitations. To connect the theory to practice, we derived **α-LoRA (Theory)**, which applies the accuracy criterion of the analysis directly to a network.

Recall that an input $x$ with label $y$ is misclassified iff its margin $d(x) = f_y(x) - \max_{k\neq y} f_k(x)$ is non-positive, where $f_k(x)$ is the logit of class $k$, i.e., the $k$-th output of the network's last layer, before the softmax. α-LoRA (Theory) chooses $\alpha$ to minimize $P(d \le 0)$.

*In our model,* $d(x) = 2y\,w_\alpha^\top x$ is Gaussian (Theorem 4.2), so $P(d \le 0) = \varphi(r)$ with $r = \mu_d/\sigma_d$ (Proposition 4.3): the margins must be large relative to their spread. Since $w_\alpha = w + \alpha\gamma Q\tilde w$ is affine in $\alpha$, $\mu_d$ is linear and $\sigma_d^2$ quadratic in $\alpha$, and the maximizer of $r$ is the closed-form $\alpha^\star$ of Theorem 4.5.

*In a network,* Cantelli's inequality gives $P(d \le 0) \le 1/(1+r^2)$ for any margin distribution ($\mu_d > 0$), so maximizing $r$ still minimizes a bound on the error. To first order in a shift $\delta$ of $\alpha$, $d(x;\alpha+\delta) \approx d(x) + \delta\, g(x)$, with $g = \partial_\delta d$ given by one backward pass. With $\rho$ the correlation of $(d,g)$, solving $r'(\delta)=0$ for
$$r(\delta) = \frac{\mu_d + \delta\mu_g}{\sqrt{\sigma_d^2 + 2\delta\rho\sigma_d\sigma_g + \delta^2\sigma_g^2}} \quad\text{gives}\quad \delta^\star = \frac{\sigma_d}{\sigma_g}\cdot\frac{\sigma_d\mu_g - \rho\sigma_g\mu_d}{\sigma_g\mu_d - \rho\sigma_d\mu_g}.$$
For $\rho = 0$, $\delta^\star = \frac{\sigma_d^2}{\sigma_g^2}\cdot\frac{\mu_g}{\mu_d}$: $\alpha$ increases when scaling up the pre-trained weights raises the margins more than it spreads them.

*The algorithm.* Start from $\alpha = 1$ and train LoRA as usual. At each $T$ steps of the training, estimate $(\mu_d, \sigma_d, \mu_g, \sigma_g, \rho)$ on held-out labelled examples and set $\alpha \leftarrow \alpha + \delta^\star$; the adapter then re-adapts to the new scale. There is no $\alpha$ optimizer and no $\alpha$ learning rate, and a single scalar is added to LoRA. It is also convenient to clip $\vert \delta^\star \vert$ with some value $\epsilon$ (like $\epsilon = 0.25$) to avoid training instabilities. 

**W2 / Q1 – State-of-the-art benchmarks; larger LLMs:** We added the commonsense reasoning benchmark used to evaluate LoRA variants such as DoRA (BoolQ, PIQA, SIQA, HellaSwag, WinoGrande, ARC-e, ARC-c, OBQA), with Qwen2.5 models up to 7B parameters, and the DoRA and IA3 baselines. Average test accuracy (%), 500 training examples per task, 3 seeds:

| Method | Qwen2.5-0.5B | Qwen2.5-1.5B | Qwen2.5-7B |
|---|---|---|---|
| LoRA | **62.59** ± 0.38 | 78.98 ± 0.50 | 88.07 ± 0.40 |
| DoRA | 62.30 ± 0.35 | 78.98 ± 0.49 | 88.07 ± 0.50 |
| IA3 | 60.99 ± 0.08 | 77.55 ± 0.22 | 87.15 ± 0.23 |
| **α-LoRA (Theory)** | 62.46 ± 0.39 | **78.99** ± 0.34 | **88.14** ± 0.29 |

α-LoRA (Theory) matches LoRA and DoRA at all three scales and outperforms IA3, while adding a single scalar to LoRA.

### **Addressing questions**
**Q2 – GLUE:** Our GLUE results with RoBERTa-base are in Table 2, on six tasks. Test accuracy (%):

| Method | MNLI | QNLI | MRPC | RTE | SST-2 | QQP |
|---|---|---|---|---|---|---|
| LoRA | 85.77 | 91.95 | 88.40 | 74.01 | 94.00 | 88.80 |
| α-LoRA (GD) | **86.12** | **92.20** | **89.46** | **77.62** | **94.38** | **88.86** |

α-LoRA (GD) is competitive with LoRA on all six tasks, with the largest gain on RTE, the smallest dataset, consistent with the theory (scarce data widens the benefit).

**Q3 – Align-LoRA:** Since $\alpha$-LoRA only rescales the frozen weights, it composes naturally with LoRA extensions that do not act on the base weights, such as LoRA+ or PLoP, as stated in the conclusion.

---

# Response to Reviewer LUNR

We thank the reviewer for their engagement and effort, and for the clear and constructive points, which we address in the following.

**(1) Learning-rate settings:** Hyperparameters were chosen separately for each method (Table 4), so each method is reported with its best performance to guarantee. fair comparison between algorothms. Additionally, the learning rates differ on QNLI, MRPC and SST-2, but are identical on MNLI, RTE and QQP, where $\alpha$-LoRA is on par with or better than LoRA. In our new commonsense experiments (see (3)), our new method $\alpha$-LoRA (Theory) uses exactly LoRA's learning rate and all its other hyperparameters.

**(2) Figure 1 vs. Proposition 4.3:** The reviewer's reading is correct. For positive alignment and $\alpha < \bar\alpha$, Proposition 4.3 predicts accuracy below 50%. Figure 1 plotted $\max(\mathrm{Acc}, 1-\mathrm{Acc})$, i.e., the accuracy after flipping the classifier's sign. This is common in binary classification, since we can get a classifier with accuracy $1 - \varepsilon$ just by flipping the sign of a classifier of accuracy $\varepsilon$. We apologize for the confusion, and we will plot the accuracy without rectification, which agrees with Proposition 4.3, and rephrase Theorem 4.5: $\bar\alpha$ is the scale at which the classifier is at chance level, not a minimizer of the accuracy.

**(3) Comparison with DoRA:** We compare DoRA, as well as IA3, with a new variant, **α-LoRA (Theory)**, which sets $\alpha$ with the criterion of our analysis.

In fact, an input $x$ with label $y$ is misclassified iff its margin $d(x) = f_y(x) - \max_{k\neq y} f_k(x)$ is non-positive, where $f_k(x)$ is the logit of class $k$, i.e., the $k$-th output of the network's last layer, before the softmax. α-LoRA (Theory) chooses $\alpha$ to minimize $P(d \le 0)$.

*In our model,* $d(x) = 2y\,w_\alpha^\top x$ is Gaussian (Theorem 4.2), so $P(d \le 0) = \varphi(r)$ with $r = \mu_d/\sigma_d$ (Proposition 4.3): the margins must be large relative to their spread. Since $w_\alpha = w + \alpha\gamma Q\tilde w$ is affine in $\alpha$, $\mu_d$ is linear and $\sigma_d^2$ quadratic in $\alpha$, and the maximizer of $r$ is the closed-form $\alpha^\star$ of Theorem 4.5.

*In a network,* Cantelli's inequality gives $P(d \le 0) \le 1/(1+r^2)$ for any margin distribution ($\mu_d > 0$), so maximizing $r$ still minimizes a bound on the error. To first order in a shift $\delta$ of $\alpha$, $d(x;\alpha+\delta) \approx d(x) + \delta\, g(x)$, with $g = \partial_\delta d$ given by one backward pass. With $\rho$ the correlation of $(d,g)$, solving $r'(\delta)=0$ for
$$r(\delta) = \frac{\mu_d + \delta\mu_g}{\sqrt{\sigma_d^2 + 2\delta\rho\sigma_d\sigma_g + \delta^2\sigma_g^2}} \quad\text{gives}\quad \delta^\star = \frac{\sigma_d}{\sigma_g}\cdot\frac{\sigma_d\mu_g - \rho\sigma_g\mu_d}{\sigma_g\mu_d - \rho\sigma_d\mu_g}.$$
For $\rho = 0$, $\delta^\star = \frac{\sigma_d^2}{\sigma_g^2}\cdot\frac{\mu_g}{\mu_d}$: $\alpha$ increases when scaling up the pre-trained weights raises the margins more than it spreads them.

*The algorithm.* Start from $\alpha = 1$ and train LoRA as usual. At each $T$ steps of the training, estimate $(\mu_d, \sigma_d, \mu_g, \sigma_g, \rho)$ on held-out labelled examples and set $\alpha \leftarrow \alpha + \delta^\star$; the adapter then re-adapts to the new scale. There is no $\alpha$ optimizer and no $\alpha$ learning rate, and a single scalar is added to LoRA. It is also convenient to clip $\vert \delta^\star \vert$ with some value $\epsilon$ (like $\epsilon = 0.25$) to avoid training instabilities. 

On commonsense reasoning (8 tasks, 500 training examples per task, 3 seeds), with Qwen2.5 models at three scales, average test accuracy (%):

| Method | Qwen2.5-0.5B | Qwen2.5-1.5B | Qwen2.5-7B |
|---|---|---|---|
| LoRA | **62.59** ± 0.38 | 78.98 ± 0.50 | 88.07 ± 0.40 |
| DoRA | 62.30 ± 0.35 | 78.98 ± 0.49 | 88.07 ± 0.50 |
| IA3 | 60.99 ± 0.08 | 77.55 ± 0.22 | 87.15 ± 0.23 |
| **α-LoRA (Theory)** | 62.46 ± 0.39 | **78.99** ± 0.34 | **88.14** ± 0.29 |

$\alpha$-LoRA (Theory) matches DoRA at every scale (within 0.16 points) while adding a single scalar to LoRA, whereas DoRA learns a magnitude vector for every adapted matrix.

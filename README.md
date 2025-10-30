# $\alpha$-LoRA: Effective Fine-tuning via Base Model Rescaling
This is the official code repository of the paper: $\alpha$-LoRA: Effective Fine-tuning via Base Model Rescaling, currently under review at ICLR 2026.

## Abstract
Fine-tuning has proven to be highly effective in adapting pre-trained models to perform better on new desired tasks with minimal data samples. Among the most widely used approaches are reparameterization methods, which update a target module by augmenting its frozen weight matrix with an additional trainable weight matrix. The most prominent example is Low Rank Adaption (LoRA), which gained significant attention in recent years. In this paper, we introduce a new class of reparametrization methods for transfer learning, designed to enhance the generalization ability of fine-tuned models. We establish the effectiveness of our approach in a high-dimensional binary classification setting using tools from Random Matrix Theory, and further validate our theoretical findings through more realistic experiments, such as fine-tuning large language models.

## Paper figures:
All the figures presented in the paper can be found in the folder [results-plot](results-plot/).

## Reproducing figures:
* Run the file [accuracy_alpha](accuracy_alpha.py) to reproduce Figure 1.
* Reproduction of Figures 2, 4 and 5 is provided in the notebook named [simulations](simulations.ipynb).
* Run the file [accuracy_alpha_amazon](accuracy_alpha_amazon.py) to reproduce Figure 3.

## Reproducing the values of tables:
* Run the file [accuracy_comparison_amazon](accuracy_comparison_amazon.py) to reproduce the experiments of Table 1.
* Run the file [fine_tune_glue](fine_tune_glue.py) to reproduce the experiments of Table 2. Note that you should pick the right hyperparameters which are all described in Appendix E of the paper.

-----

# DeepFeatureProxyVariable
Code for "
Deep Proxy Causal Learning and its Application to Confounded Bandit Policy Evaluation" (ICLR2023) (https://arxiv.org/abs/2106.03907)

## Update from Neurips [Camera Ready version](https://proceedings.neurips.cc/paper/2021/file/dcf3219715a7c9cd9286f19db46f2384-Paper.pdf)

### Structure Change of DFPV
As found in [Kompa et al. 2022](https://arxiv.org/pdf/2205.09824.pdf), our old code may suffer from numerical instability in the Demand design experiment. This occurs because we applied ReLU activation at the final layer of features and they can all be zero during in the training. We updated our code to remove this final activation, which not only resolve numerical instability but also slightly increases the performance

![images](misc/Comparison_of_having_ReLU.png)

To ensure reproducibility, we keep the old neural net structure in ``src/models/DFPV/nn_structure/nn_structure_for_demand_deprecated.py``. You can replace ``src/models/DFPV/nn_structure/nn_structure_for_demand.py`` with it to reproduce the results in ICLR paper.

### Update on dSprite experiment

**We  thank Olawale  Salaudeen for alerting us to this issue.**

In the proceedings version of this document, we employed a different experimental dSprite setting. 
We used the  structural function 

$$ f_{\mathrm{struct}}(A) =\frac{\|BA\|_2^2 - 5000}{1000} $$

where each element of the matrix $B \in \mathbb{R}^{10\times4096}$ was generated from $Unif(0.0, 1.0)$, and the outcome was generated as 

$$ Y = \frac{1}{12} (\mathrm{posY}-0.5) f_{\mathrm{struct}}(A) + \varepsilon, \varepsilon\sim\mathcal{N}(0, 0.5). $$

However, this had the following limitations. Sprite images $A$ have a small number of pixels with value $1$, and many pixels with value $0$. Each entry of $BA$ thus represents a sum of uniformly distributed independent random variables from $B$ corresponding to the nonzero entries of $A$. For this reason, the position of $A$ is very difficult to recover from $BA$, since this would require memorizing the specific sum of uniform random values for each sprite position. In practice, the structural function effectively appears as a constant function with additional noise due to $\|BA\|_2^2$.

To deal with this limitation, we introduced new structural function given as

$$ f_{\mathrm{struct}}(a) = \frac{(\mathrm{vec}({B})^\top a)^2 - 3000}{500}$$

where each element of the matrix ${B} \in \mathbb{R}^{64\times 64}$ is given as $B_{ij} = |32-j| / 32$. The outcome is generated as 

$$ Y = \frac{1}{12} (\mathrm{posY}-0.5) f_{\mathrm{struct}}(A) + \varepsilon$$

which refects the position of sprite images $A$. To run the old dsprite experiment, please specify ``dsprite_org`` in the config files.

### Update on KPV and PMMR (09.13.2024)

A implementation now uses more stable formulation for both methods are found in [this paper](https://arxiv.org/pdf/2308.04585).
Keep the original code as `deprecated`


## How to Run codes?

1. Install all dependencies
   ```
   pip install -r requirements.txt
   ```
2. Create empty directories for logging
   ```
   mkdir logs
   mkdir dumps
   ```
3. Run codes
   ```
   python main.py <path-to-configs> <problem_setting>
   ```
   `<problem_setting>` can be selected from `ate` and `ope`, which corresponds to ate experiments and policy evaluation experiments in the paper. Make sure to input the corresponding config file to each setting. The result can be found in `dumps` folder. You can run in parallel by specifing  `-t` option.

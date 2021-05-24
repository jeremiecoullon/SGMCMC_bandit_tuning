# Stochastic Gradient MCMC with Multi-Armed Bandit Tuning

_By Jeremie Coullon, Leah South, and Christoper Nemeth_

**Install:**
- Install packages in `requirements.txt`
- Run a Jupyter server; the experiments and figures are in the notebooks in the `experiments` directory

## figures

### Logistic regression

- `LR_2d_gridsearch.ipynb`: reproduces figure 2 in paper. the 2D grid search with 9 SH parameters overlayed
- `LR_KSD_curves`: runs the KSD curves and reproduces figure 1


### PMF

- `PMF-KSD_curves.ipynb`: reproduces figure 3

### NN

- `NN-KSD_curves.ipynb`: reproduces the KSD curves in figure 4
- `NN-ECE_curves.ipynb`: reproduces the ECE curves in figure 4
- `NN-uncertainty_metrics.ipynb`: reproduces the OOD plot in figure 7

### Other

- `SH_tuning_tests.ipynb`: reproduces figures 5 and 6

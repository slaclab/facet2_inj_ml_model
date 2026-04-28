# FACET II Model

This repository contains the model files corresponding to the FACET II NN surrogate model. 

## Model Description

The model was trained on IMPACT-T simulation data to predict beam properties at 241 using injector PVs for LCLS. 

These are the input scalar PVs upstream of 241. 
Feature columns:

- CQ10121:b1_gradient
- GUNF:rf_field_scale
- GUNF:theta0_deg
- SOL10111:solenoid_field_scale
- SQ10122:b1_gradient
- distgen:t_dist:sigma_t:value
- distgen:total_charge:value

The model consumes the 7 simulator-space inputs above and outputs a full 6x6 covariance matrix. Targets are the 21 lower-triangular Cholesky factors (cov_chol_0..20); the network reconstructs C = L @ L^T internally. Loss is computed in per-element normalized covariance space.

## Model architecture
7 inputs -> 21 Cholesky outputs -> 6x6 covariance output:

```Linear(7 -> 100), ELU
Linear(100 -> 200), ELU, Dropout(0.05)
Linear(200 -> 200), ELU, Dropout(0.05)
Linear(200 -> 300), ELU, Dropout(0.05)
Linear(300 -> 300), ELU, Dropout(0.05)
Linear(300 -> 200), ELU, Dropout(0.05)
Linear(200 -> 100), ELU, Dropout(0.05)
Linear(100 -> 100), ELU
Linear(100 -> 100), ELU
Linear(100 -> 21) -> build L (lower triangular) -> C = L @ L^T
```

### Normalization
Before training, the code computes:

- x_mean, x_std from the training split inputs
- y_mean, y_std from the training split Cholesky targets
- cov_mean, cov_std from the reconstructed training split covariance matrices

The loss is then applied in normalized covariance space so each covariance element contributes on a comparable scale.

## Usage 
After doing pip install of this repository
```
>>> from facet2_model import load_model
>>> model = load_model()
>>> print(model.evaluate({"SOLN:IN10:121:BACT": 0.43}))
```


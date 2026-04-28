# FACET II Injector ML Model

This repository contains the model files corresponding to the FACET II NN surrogate model. 

It includes both inference assets and the packaged training implementation used to retrain the covariance model.

## Model Description

The model was trained on IMPACT-T simulation data to predict beam properties at 241 using PVs for FACET. 

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
>>> from from facet2_inj_ml_model import load_model
>>> model = load_model()
>>> print(model.evaluate({"SOLN:IN10:121:BACT": 0.43}))
```

## Training Modules

There are two similarly named modules in this repository with different purposes:

- `facet2_inj_ml_model/training.py` is the actual training implementation.
- `train.py` at the repository root is a compatibility shim for serialized PyTorch artifacts that still reference `train.CovarianceSurrogateModel`.

The root `train.py` should not be treated as the main training script. It only re-exports symbols from `facet2_inj_ml_model.training` so legacy `torch.load` resolution continues to work.

## Retraining

Retraining can be run from this repository if you already have prepared CSV splits. The training entrypoint is `facet2_inj_ml_model.training`.

Example:

```bash
python -m facet2_inj_ml_model.training \
	--train-csv path/to/dataset-train.csv \
	--val-csv path/to/dataset-val.csv \
	--test-csv path/to/dataset-test.csv
```

Expected data requirements:

- The input features are the seven PV columns listed above.
- The targets are 21 lower-triangular Cholesky columns named `cov_chol_0` through `cov_chol_20`.
- The repository does not include dataset generation or preprocessing code for producing these CSV files from raw simulation data.

If your previous training workflow created those CSVs in a separate repository, that preprocessing step is still required before retraining here.


# Notebook guide

## Shared workflow

The precipitation notebooks load prepared NPZ tensors, apply the saved permutation,
take samples `0:500` for training and samples `500:` for testing, predict two output
channels, and compare predictions against observations. Plot labels identify
these outputs as one- and two-month lead precipitation. Raw-forecast comparisons
usually use input channel `lead + 1`. Most case-map loops select initialization
months May through July; the full test-set mean and skill plots do not apply that
seasonal filter.

The new path block in the first code cell locates the project root and defines
`DATA_DIR`, `WEIGHTS_DIR`, and `RESULTS_DIR`. It replaces the archived absolute
`/home/afahad/nb/precip_SA` paths. Sample ordering, model definitions, training
settings, notebook cell order, and saved outputs are preserved. Changed paths
have not been used to rerun the experiments.

## GEOS_ML_precip.ipynb — baseline

- Uses three precipitation input channels without active standardization.
- Builds Conv2D layers with 32, 64, and 128 filters, batch normalization, and a
  linear two-channel output; optimizes mean squared error with Adam.
- Fits for up to 500 epochs, batch size 16, and validation fraction 0.15, saving
  the best validation checkpoint.
- Computes gridpoint RMSE and temporal Pearson correlation for the second output,
  then plots mean fields, selected cases, biases, and gradient saliency.
- Late diagnostic cells are exploratory: one uses missing `sal.npz` and undefined
  `lon1`/`lat1`; its stored output includes a NameError. Another case loop was
  interrupted. A presentation panel uses a different raw input channel from the
  usual `lead + 1` convention. Review channel alignment before reusing that panel.

## GEOS_ML_precip2.ipynb — augmented CNN

- Loads precipitation plus auxiliary arrays described in source comments as
  GEOS-S2S-3 VP and M2 VP. Each auxiliary array replaces NaNs with its global mean.
- Standardizes each input group and the target, then concatenates seven channels.
- Uses Conv2D filters 32, 64, 128, and 256, batch normalization, and a linear
  two-channel output with Adam/MSE.
- The training/checkpoint cell is commented out; its example settings are
  100 epochs, batch size 100, validation fraction 0.1. Prediction loads
  `best_model_weights2.h5` unconditionally.
- Inverts target scaling for precipitation comparison. Computes second-lead RMSE
  and correlation, draws skill/bias/case maps, and explores gradient saliency.
- `original_X` applies precipitation-group scaling to all seven channels; only
  its first three channels regain the intended original units. Saliency inputs
  are also built from this array rather than the standardized training inputs.
  Review the normalization before interpreting auxiliary-channel diagnostics.

## GEOS_ML_precip3_unet.ipynb — deep-supervision experiment

- Uses the same augmented seven-channel inputs and normalization as notebook 2.
- Contains a sequential CNN, a commented-out encoder/decoder U-Net, and an active
  functional CNN that overwrites `model` with intermediate and final outputs.
- The active network uses filters 32, 64, 128, 256, 512, then 256; both output
  heads have two linear channels. Their MSE losses have weights 0.5 and 1.0.
- Training requests 300 epochs, batch size 32, validation fraction 0.1; predictions
  select the final head using `Yp_norm_test[1]`. Saved loss curves are preserved.
- Later cells compare mean fields/cases and explore saliency. The saliency code
  was inherited from a single-output workflow; explicitly select the intended
  head and use correctly standardized inputs before relying on its gradients.
- Check multi-output target handling with the chosen TensorFlow/Keras version.
  The filename does not establish that the saved figures came from a U-Net run.

## Mapping notebooks

`notebooks/exploratory/Untitled.ipynb` explores a Plotly US county choropleth and
Leafmap global-wind rasters/velocity layers. It downloads public sample data and
exports `mymap.html`. The archive's HTML is preserved in `artifacts/`; it is a
mapping experiment, not a precipitation-model result. Stored errors include
a missing Basemap installation and a Leafmap tile-client attribute error.

`notebooks/exploratory/Untitled1.ipynb` sketches NetCDF wind-grid conversion to
GeoJSON points (`output.geojson`) and xarray interpolation. Its initial cell
references `data` before that variable is created in a later cell. Reorder/fix
those exploratory cells before running from a clean kernel. Run mapping notebooks
from their own folder and install `requirements-mapping.txt` as needed.

## Interpretation and reproducibility

Normalization in the augmented experiments is fitted before the train/test split,
which allows test information into preprocessing. The random sample split also
does not establish independent held-out years or seasons. Refit transformations
on training data and use a suitable time-based evaluation before making claims
about forecast generalization. No fresh training, numerical validation, or
architecture comparison has been performed during repository organization.

The saved figures lack complete dataset metadata and unit labels. Treat them as
archived exploratory diagnostics. Gridpoint RMSE is
`sqrt(mean((prediction - observation)**2, axis=0))`; correlation is calculated over
test samples at each grid cell. Difference maps use ML minus raw GEOS: negative
RMSE differences and positive correlation differences indicate improvement for
that saved comparison. No overall performance statistic is inferred here.

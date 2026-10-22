# South Asian Monsoon Precipitation Bias Correction

Machine-learning experiments for correcting **GEOS-S2S-3 precipitation forecasts
over South Asia** at one- and two-month leads. The notebooks map gridded forecast
inputs to observed precipitation with convolutional neural networks (CNNs), compare
raw and corrected forecasts using RMSE, temporal correlation, and bias maps, and
explore input sensitivity with gradient saliency.

This repository organizes the research notebooks and saved outputs supplied in
`precip_SA.tar`. Training data and trained weights were **not included**. The plots
below are original saved notebook outputs, exported without rerunning training.
The archive does not identify the observation product, complete data period, or
precipitation units, so these are exploratory diagnostics rather than a fully
reproducible benchmark.

## Project contents

```text
.
├── notebooks/
│   ├── GEOS_ML_precip.ipynb          # Three-channel precipitation CNN baseline
│   ├── GEOS_ML_precip2.ipynb         # Seven-channel CNN with auxiliary VP inputs
│   ├── GEOS_ML_precip3_unet.ipynb    # Deep-supervision CNN; U-Net code is commented
│   └── exploratory/
│       ├── Untitled.ipynb           # Plotly/Leafmap wind-mapping experiments
│       └── Untitled1.ipynb          # NetCDF-to-GeoJSON and interpolation sketch
├── scripts/export_readme_figures.py # Exports saved notebook PNGs; no model execution
├── figures/                        # README plots, source manifest, original GEOSML.png
├── data/                           # Saved permutation and input-data documentation
├── models/                         # Checkpoint documentation; weights supplied separately
├── artifacts/mymap.html            # Archived interactive wind-map export
├── docs/NOTEBOOK_GUIDE.md           # Workflow, model details, and known limitations
├── requirements.txt                # Inferred core dependencies; not a tested lockfile
└── requirements-mapping.txt         # Optional mapping-notebook dependencies
```

## What the notebooks and script do

| File | Purpose and behavior |
| --- | --- |
| [GEOS_ML_precip.ipynb](notebooks/GEOS_ML_precip.ipynb) | Baseline CNN: three input channels, Conv2D filters 32/64/128, two linear precipitation outputs, Adam/MSE. Includes training, prediction, RMSE/correlation, mean/case maps, and saliency experiments. |
| [GEOS_ML_precip2.ipynb](notebooks/GEOS_ML_precip2.ipynb) | Adds auxiliary inputs labeled GEOS-S2S-3 VP and M2 VP in the code, standardizes input groups/targets, and uses a seven-channel CNN with filters 32/64/128/256. Training is commented out; prediction requires separately supplied weights or enabling training. |
| [GEOS_ML_precip3_unet.ipynb](notebooks/GEOS_ML_precip3_unet.ipynb) | Explores deeper models. The active final model is a CNN with intermediate/final output heads and weighted MSE losses; the U-Net implementation is commented out. Includes saved training curves and forecast/saliency plots. |
| [Untitled.ipynb](notebooks/exploratory/Untitled.ipynb) | Separate visualization experiments: US county choropleth, sample global-wind NetCDF, Leafmap raster/velocity layers, and HTML export. |
| [Untitled1.ipynb](notebooks/exploratory/Untitled1.ipynb) | Draft conversion of a wind grid to GeoJSON points plus xarray interpolation; needs cell-order fixes before clean execution. |
| [export_readme_figures.py](scripts/export_readme_figures.py) | Extracts four selected PNG outputs from the notebooks and writes [manifest.json](figures/manifest.json) with source cell/output indices and SHA-256 hashes. Uses only the Python standard library. |

See the [notebook guide](docs/NOTEBOOK_GUIDE.md) for training settings, evaluation
logic, and known issues in the archived research code.

## Which version to use for handoff

**Start with [GEOS_ML_precip2.ipynb](notebooks/GEOS_ML_precip2.ipynb)** for a
research handoff. It is the recommended starting point because it combines the
seven-channel input workflow with saved RMSE, correlation, bias, and case-map
diagnostics. It is a notebook, not a standalone prediction script or a packaged
deployment model. Its weights and data must be supplied separately.

| Version | Role | Handoff recommendation |
| --- | --- | --- |
| `GEOS_ML_precip.ipynb` | Three-channel baseline | Use for understanding the simpler model and as a baseline comparison. |
| **`GEOS_ML_precip2.ipynb`** | Seven-channel augmented CNN | **Primary handoff notebook** for reviewing the existing forecast-correction workflow and saved skill diagnostics. |
| `GEOS_ML_precip3_unet.ipynb` | Most advanced experiment in the supplied numbered sequence: a CNN with intermediate/final prediction heads | Use for continuing model development. The active model is not a U-Net; U-Net code is commented out. |

**Latest experimental version:** notebook 3 in the numbered sequence. This is an
ordering inferred from the filenames and model progression, not a release date
or a documented production designation. **Best-performing version:** not
established by this archive. There is no controlled side-by-side evaluation of
all three models, complete evaluation metadata, or supplied checkpoints that
would justify naming an accuracy winner. Notebook 2 is recommended for handoff
readiness, not proven superior predictive performance.

For the person taking over:

1. Read this README, [data requirements](data/README.md),
   [checkpoint requirements](models/README.md), and the
   [notebook guide](docs/NOTEBOOK_GUIDE.md).
2. Open `notebooks/GEOS_ML_precip2.ipynb` in Jupyter and select the project kernel.
   Supply `ml_data_pr.npz`, `ml_data1.npz`, `ml_data2.npz`, and the compatible
   `best_model_weights2.h5` checkpoint. Keep the included sample permutation.
3. Run the path/import, loading, normalization, permutation, model-definition,
   and split cells in order, then load weights and run prediction/evaluation.
   If retraining, enable the commented training cell and resolve the environment
   and checkpoint compatibility notes first.
4. Validate normalization, channel alignment, and the temporal evaluation split
   before treating the results as a reproducible benchmark. Carry notebook 3
   forward as a separate development experiment and compare it on the same
   held-out data before selecting a best model.

For a **documentation-only handoff**, run
`python3 scripts/export_readme_figures.py`. This refreshes the four README plots
from saved notebook outputs; it does not train a model or run new forecasts.

## Saved plots

All four plots below come directly from embedded notebook outputs. Source cell
indices are zero-based and recorded in the [figure manifest](figures/manifest.json).
The separate [GEOSML.png](figures/GEOSML.png) is preserved as supplied; its exact
generating cell was not established.

### Forecast skill at two-month lead

![Raw GEOS and CNN-corrected forecast RMSE and correlation, with difference maps](figures/cnn_skill.png)

From `GEOS_ML_precip2.ipynb`, cell 43. Top row: raw forecast RMSE, corrected RMSE,
and corrected-minus-raw RMSE. Bottom row: raw correlation, corrected correlation,
and their difference. Negative RMSE differences and positive correlation
differences indicate improvement in this saved comparison. These are gridpoint
diagnostics over the notebook's test samples, not a separately validated score.

### Mean precipitation and bias at two-month lead

![Two-month-lead mean raw forecast, CNN forecast, observations, and forecast biases](figures/cnn_mean_lead2.png)

From `GEOS_ML_precip2.ipynb`, cell 46. The upper panels compare test-set mean raw
GEOS precipitation, corrected precipitation, and observations. Lower panels show
forecast-minus-observation mean bias; values closer to zero indicate less bias.

### Example forecast and observed precipitation

![Example two-month-lead forecast comparison with raw and corrected bias maps](figures/cnn_case.png)

From `GEOS_ML_precip2.ipynb`, cell 40. A saved individual-case comparison at the
second lead: raw forecast, corrected forecast, observations, and their biases.
The saved figure has no date label; no date is inferred here.

### Training and validation loss

![Saved training and validation loss over 300 epochs](figures/training_loss.png)

From `GEOS_ML_precip3_unet.ipynb`, cell 27. This archived curve shows the training
and validation loss histories. Their widening gap warrants review of
generalization; the curve alone does not establish predictive skill or a U-Net
result.

## Setup and use

View the saved outputs on GitHub without installing anything. To work locally:

```bash
git clone https://github.com/afahadabdullah/monsoon_bias_MAP24.git
cd monsoon_bias_MAP24
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m ipykernel install --user --name monsoon-bias --display-name "Monsoon bias"
python -m jupyterlab
```

Select the `Monsoon bias` kernel and open a notebook under `notebooks/`. The
dependency lists were inferred from imports; no original environment lock was
provided and this installation has not been validated for training. Archived
core notebook metadata reports Python 3.12.8, but does not record package versions.
Match TensorFlow/Keras to any supplied checkpoints; see [checkpoint notes](models/README.md)
for the legacy `.h5` filenames and modern Keras suffix requirements.

Before executing model cells:

1. Supply the datasets listed in [data/README.md](data/README.md). Required keys
   include `X`, `Y`, `lon`, `lat`, `date`, and auxiliary `X1` arrays. The model code
   assumes a 91 × 101 grid, three or seven input channels, and two target channels.
2. Put weights in `models/`, or enable/configure training before prediction.
   Notebook 2's unconditional weight load requires an existing compatible checkpoint.
3. Start Jupyter inside this checkout. The first code cell resolves paths from the
   repository root. Optional `PRECIP_DATA_DIR` and `PRECIP_WEIGHTS_DIR` environment
   variables, set before starting Jupyter, point to external data/weight folders.
4. Review the [known notebook issues](docs/NOTEBOOK_GUIDE.md) before using Run All.
   Saved outputs remain available even when dependencies or external inputs are missing.

Generated training figures go to ignored `results/`; input datasets and weights
are also ignored. The saved `data/random_numbers.npz` permutation stays tracked.
Mapping notebooks use a separate optional dependency list:

```bash
python -m pip install -r requirements-mapping.txt
```

To refresh only the README figures from saved outputs, without data, TensorFlow,
or notebook execution:

```bash
python3 scripts/export_readme_figures.py
```

## Scientific and reproducibility notes

The experiments permute samples, use the first 500 for training, and use the
remainder for testing. In the augmented notebooks, scaling statistics are computed
before splitting, allowing test information into preprocessing. This random split
does not demonstrate skill on independent years. Refit scaling on training data
and evaluate with a suitable temporal split before drawing generalization claims.

Some late plotting and saliency cells depend on missing variables/files, inconsistent
channel choices, or assumptions about model outputs. The mapping experiments also
contain saved errors. These issues are documented in the [notebook guide](docs/NOTEBOOK_GUIDE.md);
organization and path changes do not resolve the underlying research questions.
No model training or numerical reproduction was performed during this cleanup.

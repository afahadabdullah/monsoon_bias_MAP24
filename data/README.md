# Data inputs

Only `random_numbers.npz` was supplied. It contains the saved sample permutation
under the key `random_numbers`; keep it to reproduce the archived sample ordering.

The following arrays must be supplied separately, either here or in the directory
selected by `PRECIP_DATA_DIR`:

| File | Required keys | Used by |
| --- | --- | --- |
| `ml_data_pr.npz` | `X`, `Y`, `lon`, `lat`, `date` | All three precipitation notebooks |
| `ml_data1.npz` | `X1`, `lon1`, `lat1` | Augmented CNN and deep-supervision notebooks; comments label this GEOS-S2S-3 VP |
| `ml_data2.npz` | `X1` | Augmented CNN and deep-supervision notebooks; comments label this M2 VP |
| `sal.npz` | `sal` | Optional late plotting cell in the baseline notebook |

The code assumes channel-last tensors: `(sample, 91, 101, channel)`. Baseline `X`
has three channels and `Y` has two. The augmented notebooks concatenate three
precipitation channels and auxiliary VP channels into seven channels. Arrays
must already share sample and spatial dimensions; no preprocessing or regridding
pipeline was included. The archive does not specify the observation product,
date range, full channel definitions, or units. Confirm these with the data owner
before scientific interpretation. Inputs are ignored by Git except the permutation.

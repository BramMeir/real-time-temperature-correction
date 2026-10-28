# src/models/jacobs2024

The gap-filling methods of Jacobs et al. (2024), which the paper compares with RegSARIMA as a retrospective
gap-filler (`src/scripts/jacobs_comparison_experiment.py`). The code is theirs, from
[amberJ99/Gapfilling_debiasingERA5](https://github.com/amberJ99/Gapfilling_debiasingERA5) (commit `1cf8b70`), under the
MIT licence in `LICENSE`.

| File | Upstream file |
|---|---|
| `gf_techniques.py` | `GFtechniques.py`: linear interpolation, raw ERA5, and the regression (LR) and mean-bias (MB, TVar) debiasing of ERA5 over a learning period around the gap |
| `evaluation_gf_techniques.py` | `Evaluation_GFtechniques.py`: placing a gap in a series and evaluating the techniques on it |

The code is unchanged apart from two fixes for current pandas, neither of which changes a result: the hourly
frequency is written `"h"` instead of the removed `"H"`, and in `GF_debmodelTvar` a column assignment replaces a
chained `replace(..., inplace=True)`.

The comparison uses `GF_linint`, `GF_fillmodel`, `GF_debmodelReg`, `GF_debmodelMeanbias` and `makegap`.

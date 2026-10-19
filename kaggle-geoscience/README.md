# Three Geoscience Tasks: Results and Code

[Open on Kaggle](https://www.kaggle.com/code/giswlh/three-geoscience-tasks-results-and-code) · [Download ipynb](three-geoscience-tasks-results-and-code.ipynb)

A combined notebook for Soil Grain Size, CarbonGlobe and ClimX, by giswlh / Longhao Wang, with AI assistance.

The notebook includes a dated official-results table, development evidence, fixed failed-experiment ledger, the complete train-only soil texture/sqrt-mass baseline, CarbonGlobe40year features and site-grouped training core, and a small real historical ClimX daily-field quantile core. It distinguishes official scores, development validation and bounded implementation checks. These educational cores are not complete reproductions of every best-score producer. At the snapshot0/3official public top10% goals have been achieved; final acceptance is unverified.

## Run

Install `requirements.txt`, open the ipynb and Run All. Under Kaggle all3competition attachments are read-only. The soil audit and small ClimX historical check run automatically when labelled/historical inputs exist; otherwise archived real results are displayed and missing inputs are stated. No data are generated to replace missing inputs. Attach CarbonGlobe labelled training data separately and opt in to its training cell; Kaggle competition test arrays are never used as training data.

Local input overrides: `SOIL_DATA_ROOT`, `CLIMX_TRAIN_ROOT`, `CARBON_TRAIN_ROOT`. No authentication token is required by the notebook. CPU is default; optional Torch CUDA code is explicit. No raw datasets, images, private model weights, test predictions or credentials are published. No automatic Kaggle competition submissions.

`public_evidence.json` stores sanitized aggregate evidence. `manifest.json` records SHA256 values for exact source synchronization; numerical-library versions may change floating-point outputs slightly. The original soil notebook completed on Kaggle with four protocol EMDs approximately37.9462/40.2714/46.5478/40.4786; these are training-only development scores, not official36.40650.

## Sources and data rights

See the notebook for links/credit to the competition hosts, YogaNukko(texture/SAM), Nomannic(sqrt-mass), Amirhossein Karimiee(CarbonGlobe baseline) and maverickss26(ClimX distributional emulator). Code here is an educational implementation of the stated kernels, not a redistribution of their model weights. Competition datasets retain their own licenses and access restrictions. Preserve source attribution when extending these ideas, and report domain failures as well as gains.

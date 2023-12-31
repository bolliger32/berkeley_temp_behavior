This code replicates figures and tables for:

```text
Almås, I., Auffhammer, M., Bold, T., Bolliger, I., et al., “Destructive Behavior, Judgment, and Economic Decision-making under Thermal Stress.” Cambridge, MA. NBER Working Paper. http://www.nber.org/papers/w25785.pdf.
```

The following environmentally-related figures and tables are created in this repository:
- Fig. 1
- Fig. A.2.1
- Fig. A.2.2
<!--TODO: See if these figs are referenced elsewhere - Fig. A.4.1
- Fig. B.2.1
- Fig. B.2.2 -->

To ensure replicability, we recommend you create a [conda](https://docs.conda.io/en/latest/miniconda.html) environment to match that used to run this code for the publication. To do so, install [conda](https://docs.conda.io/en/latest/miniconda.html) and run the following from the root directory of this repository:

```bash
conda env create -f environment/environment.yml
conda activate temp-behavior-env
```

Once you are in the conda environment, there are two notebooks to run:
1. `concatenate_meas.ipynb` loads and processes raw data.
2. `publication_figures_and_tables.ipynb` creates figures and tables from the processed data

Note that to run this code you must have access to the [raw data folder on Dropbox](https://www.dropbox.com/sh/g30v78bux6adw1g/AAB-Lw2MPw44hc0BsWFCFzK_a?dl=0). You will need to change the ``home_dir`` parameter at the top of the notebooks to point to this folder once you have downloaded it.

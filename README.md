# Predictive Performance Precision Analysis in Medicine: Identification of low-confidence predictions at patient and profile levels (MED3pa I)

This repository is to generate results of the article "Predictive Performance Precision Analysis in Medicine: 
Identification of low-confidence predictions at patient and profile levels (MED3pa I)": https://doi.org/10.1101/2025.08.22.25334254.

The MED3PA package used in this work is available at this link: https://github.com/MEDomicsLab/MED3pa

The MED3PA package can be installed from pypi: https://med3pa.readthedocs.io/en/latest/installation.html

## 1. Data availability
This study uses three types of datasets: 

- **Simulated Data**
A synthetic dataset is generated in this repository, using the 'generated_simulated_dataset.py' script in the datasets/simulated_dataset folder.
- **Public Clinical Datasets** (In-hospital mortality task)
  - **eICU Collaborative Research Database (eICU)** – Requires credentialed access through [PhysioNet](https://physionet.org/).
  - **MIMIC-IV** – Also requires credentialed access through [PhysioNet](https://physionet.org/).  
  Users must complete the required training and data use agreements to access these datasets.
- **Private Clinical Dataset** (One Year mortality task, POYM)
    Due to regulations safeguarding patient privacy, this dataset cannot be shared. However, a synthetic dataset is publicly available on https://zenodo.org/doi/10.5281/zenodo.12954672.

## 2. Study recreation
To reproduce experiments and figures from the paper:
1. **Install requirements**
First install the requirements under *Python 3.12.4* as following:
```
$ pip install -r requirements.txt
```
2. **Generate simulated Data**
```
$ python -m datasets.simulated_dataset.generate_simulated_dataset
```

3. **Run experiments**
```
$ python -m experiments.simulated_dataset
```
This will generate results in the `results` folder.

# GC-MS Spectrum-Based Compound Identification and Out-of-Library Detection

This repository contains Python scripts for evaluating GC-MS mass spectrum analysis methods for compound identification, clustering, and out-of-library detection under noisy conditions.

The project focuses on robust analysis of electron ionization GC-MS spectra, where measured spectra may be affected by electronic noise, ion-counting variability, and spurious peaks. The goal is to evaluate whether mass spectra can be reliably compared, retrieved, clustered, and screened even when the query spectrum does not perfectly match the reference library.

## Project Summary

GC-MS library searching is commonly used to identify compounds by comparing an acquired mass spectrum with reference spectra. However, real measurement environments may introduce noise, intensity fluctuations, and non-analyte peaks. In addition, some query compounds may not be represented in the reference library.

This repository provides research code for evaluating three approaches:

1. Tolerance-based heuristic screening
2. Distance-based spectrum comparison and clustering
3. Dirichlet Process Mixture Model-based clustering and visualization

The methods were designed to examine how robustly GC-MS spectra can be compared and grouped under different noise conditions, with a focus on reliable compound identification and potential out-of-library detection.

## Research Focus

This project addresses the following questions:

* Can a noisy query spectrum be correctly matched to its reference spectrum?
* How do different distance metrics perform under various noise conditions?
* Can clustering reduce the search space while preserving identification accuracy?
* Can Bayesian nonparametric clustering help identify spectra that may not belong to existing reference groups?

## Overview

This repository includes three analysis scripts:

| File                                  | Description                                                                      |
| ------------------------------------- | -------------------------------------------------------------------------------- |
| `heuristic_screening.py`              | Evaluates a tolerance-based heuristic screening method                           |
| `distribution_distance_method.py`     | Evaluates distance-based compound retrieval and clustering methods               |
| `dirichlet_process_mixture_method.py` | Performs Dirichlet Process Mixture Model-based clustering and UMAP visualization |

## Key Features

* Evaluation of GC-MS spectrum analysis under multiple noise conditions
* Spectrum comparison using distributional and geometric distance metrics
* Compound-level retrieval accuracy evaluation
* Cluster-level retrieval and aggregation analysis
* Dirichlet Process Mixture Model-based clustering
* UMAP-based visualization of clustering results
* CPU/GPU execution support through command-line arguments

## Methods

### 1. Heuristic Screening

`heuristic_screening.py`

This script evaluates whether a noisy query spectrum can be uniquely matched to the correct reference spectrum using intensity tolerance-based filtering.

The method checks whether peak intensities in a noisy query spectrum fall within a predefined tolerance range of reference spectra. It is used as a baseline approach to examine the limitations of simple rule-based screening under noisy measurement conditions.

Noise types:

* Gaussian noise
* Poisson noise
* Spurious peak noise

Main outputs:

* Accuracy by noise type
* Accuracy by tolerance value

Example:

```bash
python heuristic_screening.py --excel-path data.xlsx --n-trials 50 --seed 42
```

Save results as CSV:

```bash
python heuristic_screening.py --excel-path data.xlsx --output-csv heuristic_results.csv
```

---

### 2. Distribution Distance Method

`distribution_distance_method.py`

This script evaluates mass spectrum comparison methods based on distributional and geometric distances.

Mass spectra are compared after being represented as normalized intensity distributions. The script evaluates whether distance-based similarity measures can retrieve the correct compound or assign a noisy query spectrum to an appropriate cluster.

Distance metrics:

* Hellinger distance
* Square-root Jensen-Shannon divergence
* Cosine distance
* Hybrid distance

The script evaluates:

* Compound-level retrieval accuracy
* Cluster-level accuracy
* Cluster aggregation performance

Example:

```bash
python distribution_distance_method.py --excel-path data.xlsx --n-trials 50
```

Run on CPU:

```bash
python distribution_distance_method.py --excel-path data.xlsx --device cpu
```

Save results as CSV:

```bash
python distribution_distance_method.py --excel-path data.xlsx --output-csv distance_results.csv
```

---

### 3. Dirichlet Process Mixture Method

`dirichlet_process_mixture_method.py`

This script performs Dirichlet Process Mixture Model-based clustering for GC-MS spectra.

The method is used to group spectra without fixing the number of clusters in advance. This is useful for analyzing whether newly added spectra are assigned to existing groups or form separate clusters, which can support out-of-library detection.

The script also supports UMAP visualization of clustering results.

Example:

```bash
python dirichlet_process_mixture_method.py --base-path base_data.xlsx
```

Run with additional new data:

```bash
python dirichlet_process_mixture_method.py --base-path base_data.xlsx --new-paths new_data.xlsx
```

Skip UMAP visualization:

```bash
python dirichlet_process_mixture_method.py --base-path base_data.xlsx --skip-plot
```

Run on CPU:

```bash
python dirichlet_process_mixture_method.py --base-path base_data.xlsx --device cpu
```

## Input Data Format

The input Excel file should contain the following columns:

| Column      | Description             |
| ----------- | ----------------------- |
| `Name`      | Compound or sample name |
| `m/z`       | Mass-to-charge ratio    |
| `Intensity` | Peak intensity          |

For `distribution_distance_method.py`, the Excel file must also contain the following column:

| Column | Description                                                              |
| ------ | ------------------------------------------------------------------------ |
| `Form` | Chemical formula or class label used to determine the number of clusters |

## Installation

Install the required packages:

```bash
pip install -r requirements.txt
```

## Project Structure

```text
mass_spectrum_ool_detection/
├── heuristic_screening.py
├── distribution_distance_method.py
├── dirichlet_process_mixture_method.py
├── README.md
├── requirements.txt
└── .gitignore
```

## Data Availability

The original GC-MS Excel data files are not included in this repository because they may contain private, research-related, institution-owned, or licensed reference library information.

To run the scripts, prepare an Excel file in the required format described above and pass the file path using the command-line arguments.

Example:

```bash
python heuristic_screening.py --excel-path your_data.xlsx
python distribution_distance_method.py --excel-path your_data.xlsx
python dirichlet_process_mixture_method.py --base-path your_base_data.xlsx
```

## Notes

This repository is intended to provide the implementation structure and analysis workflow for GC-MS spectrum-based compound identification, clustering, and out-of-library detection research.

Because the original reference spectra are not publicly distributed, numerical results may depend on the user-provided dataset, preprocessing procedure, and experimental configuration.

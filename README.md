# GC-MS Spectrum Analysis Methods

This repository contains Python scripts for evaluating GC-MS mass spectrum analysis methods.

The project focuses on compound identification and clustering using mass spectrum data under noisy conditions.

## Overview

This repository includes three analysis scripts:

| File | Description |
|---|---|
| `heuristic_screening.py` | Evaluates a tolerance-based heuristic screening method |
| `distribution_distance_method.py` | Evaluates distance-based compound retrieval and clustering methods |
| `dirichlet_process_mixture_method.py` | Performs Dirichlet Process Mixture Model-based clustering and UMAP visualization |

## Methods

### 1. Heuristic Screening

`heuristic_screening.py`

This script evaluates whether a noisy query spectrum can be uniquely matched to the correct reference spectrum using intensity tolerance-based filtering.

Noise types:
- Gaussian noise
- Poisson noise
- Spurious peak noise

Main outputs:
- Accuracy by noise type
- Accuracy by tolerance value

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

Distance metrics:
- Hellinger distance
- Square-root Jensen-Shannon divergence
- Cosine distance
- Hybrid distance

The script evaluates:
- Compound-level retrieval accuracy
- Cluster-level accuracy
- Cluster aggregation performance

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

It also supports UMAP visualization of clustering results.

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

| Column | Description |
|---|---|
| `Name` | Compound or sample name |
| `m/z` | Mass-to-charge ratio |
| `Intensity` | Peak intensity |

For `distribution_distance_method.py`, the Excel file must also contain the following column:

| Column | Description |
|---|---|
| `Form` | Chemical formula or class label used to determine the number of clusters |

## Installation

Install the required packages:

```bash
pip install -r requirements.txt
```

## Project Structure

```text
gc-ms-spectrum-analysis/
├── heuristic_screening.py
├── distribution_distance_method.py
├── dirichlet_process_mixture_method.py
├── README.md
├── requirements.txt
└── .gitignore
```

## Notes

The original GC-MS Excel data files are not included in this repository because they may contain private, research-related, or institution-owned information.

To run the scripts, prepare an Excel file in the required format described above and pass the file path using the command-line arguments.

Example:

```bash
python heuristic_screening.py --excel-path your_data.xlsx
python distribution_distance_method.py --excel-path your_data.xlsx
python dirichlet_process_mixture_method.py --base-path your_base_data.xlsx
```

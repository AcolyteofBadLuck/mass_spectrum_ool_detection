# GC-MS Spectrum-Based Compound Identification, Clustering, and Potential Out-of-Library Analysis

This repository contains research code for evaluating GC-MS mass spectrum screening, distance-based retrieval, hierarchical clustering, and Bayesian nonparametric clustering under controlled noise conditions.

The project focuses on electron ionization GC-MS spectra that may be affected by electronic noise, ion-counting variability, and spurious peaks. It examines how these perturbations affect reference-spectrum retention, exact-reference retrieval, cluster-level routing, and posterior cluster formation.

## Research Highlights

* The associated study evaluated the proposed analysis workflow using 15,697 reference EI GC-MS spectra.
* Robustness was examined under Gaussian noise, Poisson variability, and spurious-peak perturbations.
* Multiple distributional, geometric, and weighted dot-product distance measures were compared.
* Hierarchical clustering was used to evaluate whether noisy queries could be routed to the cluster containing the generating spectrum.
* In the associated out-of-library-oriented analysis, 444 spectra whose molecular formulas were absent from the base reference set were examined using DPMM clustering.
* Singleton formation was interpreted as a potential warning signal rather than a definitive declaration of out-of-library status.

The original spectral data are not included because they may contain institution-owned, research-related, or licensed reference-library information.

## Project Summary

GC-MS library searching identifies compounds by comparing an acquired mass spectrum with spectra stored in a reference library. Its reliability may be affected by measurement noise, intensity variation, spurious peaks, and incomplete library coverage.

This repository provides three complementary analysis scripts:

1. Tolerance-based heuristic screening
2. Distance-based spectrum retrieval and hierarchical clustering
3. Dirichlet-multinomial Dirichlet Process Mixture Model clustering and visualization

The heuristic method evaluates whether the generating reference spectrum remains in a candidate set after tolerance-based screening.

The distance-based method evaluates exact-reference retrieval, cluster-level retrieval, and query-to-cluster aggregation under simulated perturbations.

The Dirichlet Process Mixture Model examines posterior cluster structures for a base reference set and for datasets containing newly added spectra. Its results may support the examination of spectra that are not adequately represented by existing reference groups, but they should not be interpreted as definitive out-of-library classifications.

## Research Questions

This project addresses the following questions:

* Does the generating reference spectrum remain in the candidate set after tolerance-based screening of a noisy query?
* How do different spectrum distance measures perform under multiple perturbation conditions?
* Can a noisy query be routed to the hierarchical cluster containing its generating reference spectrum?
* How do minimum, mean, and maximum query-to-cluster aggregation affect cluster assignment?
* How does the posterior cluster structure change when new spectra are analyzed together with a base reference set?

## Repository Overview

| File                                  | Description                                                                                    |
| ------------------------------------- | ---------------------------------------------------------------------------------------------- |
| `heuristic_screening.py`              | Evaluates tolerance-based candidate screening under simulated perturbations                    |
| `distribution_distance_method.py`     | Evaluates exact-reference retrieval, hierarchical clustering, and query-to-cluster aggregation |
| `dirichlet_process_mixture_method.py` | Performs Dirichlet-multinomial DPMM clustering and UMAP visualization                          |

## Key Features

* Evaluation under Gaussian, Poisson, and spurious-peak perturbations
* Tolerance-based reference-spectrum candidate screening
* Comparison of distributional, geometric, hybrid, and weighted dot-product distances
* Exact-reference retrieval accuracy evaluation
* Cluster-level retrieval accuracy evaluation
* Minimum, mean, and maximum query-to-cluster aggregation
* Dirichlet Process Mixture Model-based clustering
* Analysis of newly added spectra using base cluster assignments as initialization
* UMAP visualization of cluster centroids and newly added spectra
* Optional CSV export for the heuristic and distance-based analyses
* CPU and GPU execution options for the PyTorch-based scripts

## Methods

### 1. Heuristic Screening

`heuristic_screening.py`

This script evaluates a tolerance-based screening method for noisy query spectra.

The positive-intensity peaks of a query spectrum are examined in descending order of intensity. At each peak, reference spectra whose intensities fall outside the specified tolerance range are removed from the candidate set.

The screening process stops when no more than one candidate remains or when all positive query peaks have been examined.

For each trial, the result is counted as correct when the generating reference spectrum remains in the final candidate set. The candidate set may contain more than one spectrum; therefore, the reported value represents **target-retention accuracy**, not unique-identification accuracy.

Noise conditions:

* Gaussian noise
* Poisson noise
* Spurious-peak noise

Default settings:

| Parameter                       | Default values |
| ------------------------------- | -------------- |
| Intensity tolerances            | 0.1, 0.2, 0.3  |
| Gaussian standard deviations    | 1.0, 2.0, 3.0  |
| Number of spurious peaks        | 1, 2, 3        |
| Repeated trials per spectrum    | 50             |
| Maximum spurious-peak intensity | 100            |

Main outputs:

* Target-retention accuracy by noise condition
* Target-retention accuracy by noise parameter
* Target-retention accuracy by tolerance value

Example:

```bash
python heuristic_screening.py \
    --excel-path data.xlsx \
    --n-trials 50 \
    --seed 42
```

Specify tolerance values:

```bash
python heuristic_screening.py \
    --excel-path data.xlsx \
    --tolerances 0.1 0.2 0.3
```

Save the results as CSV:

```bash
python heuristic_screening.py \
    --excel-path data.xlsx \
    --output-csv heuristic_results.csv
```

---

### 2. Distribution Distance Method

`distribution_distance_method.py`

This script evaluates spectrum retrieval and hierarchical clustering using multiple distance measures.

For each simulated noisy query, the script calculates its distance from every reference spectrum. The nearest reference spectrum is then used to evaluate both exact-reference retrieval and cluster-level retrieval.

Distance measures:

* Hellinger distance
* Square-root Jensen-Shannon divergence
* Cosine distance
* Hybrid distance
* Stein-Scott weighted dot-product distance

The hybrid distance combines square-root Jensen-Shannon divergence and cosine distance. Its weight is selected through a grid search over candidate values.

By default, the weight search uses the average exact-reference retrieval accuracy under the following perturbation settings:

* Gaussian noise with standard deviation 2
* Poisson noise
* Two spurious peaks

The default number of candidate weights is 21, corresponding to values from 0 to 1 at intervals of 0.05.

#### Hierarchical clustering

A separate hierarchical clustering result is constructed for each distance measure using complete linkage.

For the controlled clustering evaluation, the number of distinct values in the `Form` column is used as the target number of clusters. The `Form` values are not directly used as cluster assignments.

#### Evaluation outputs

The script evaluates:

* **Exact-reference retrieval accuracy**: whether the nearest reference spectrum is the generating spectrum
* **Cluster-level retrieval accuracy**: whether the nearest reference spectrum belongs to the same generated cluster as the target spectrum
* **Query-to-cluster aggregation accuracy**: whether the cluster selected from aggregated spectrum distances is the target spectrum's cluster

Query-to-cluster aggregation methods:

* Minimum distance
* Mean distance
* Maximum distance

Noise conditions:

* Gaussian noise with standard deviations 1, 2, and 3
* Poisson noise
* One, two, and three spurious peaks

Example:

```bash
python distribution_distance_method.py \
    --excel-path data.xlsx \
    --n-trials 50
```

Run on CPU:

```bash
python distribution_distance_method.py \
    --excel-path data.xlsx \
    --device cpu
```

Change the number of hybrid-weight candidates:

```bash
python distribution_distance_method.py \
    --excel-path data.xlsx \
    --n-lambda 21
```

Save the aggregation and retrieval results as CSV:

```bash
python distribution_distance_method.py \
    --excel-path data.xlsx \
    --output-csv distance_results.csv
```

The selected hybrid weight and its grid-search accuracy values are printed to the console. They are not included in the output CSV.

---

### 3. Dirichlet Process Mixture Method

`dirichlet_process_mixture_method.py`

This script performs Dirichlet-multinomial Dirichlet Process Mixture Model clustering using Chinese Restaurant Process Gibbs sampling.

The number of clusters is not specified in advance. Instead, cluster assignments and the number of occupied clusters are inferred during posterior sampling.

The Dirichlet prior is estimated before the Gibbs updates, while the concentration parameter is re-estimated during posterior sampling.

#### Base analysis

The base analysis constructs a posterior clustering structure from a reference dataset.

After Gibbs sampling, cluster labels are aligned across posterior samples. The final assignment for each spectrum is obtained from the posterior mode after burn-in.

#### Analysis of newly added spectra

One or more new datasets can be analyzed together with the base reference spectra.

For each new file, the new spectra are concatenated with the base spectra. The base cluster assignments are used as part of the initial clustering configuration, and each file supplied through `--new-paths` is processed separately against the same base analysis.

The resulting cluster structure and UMAP visualization can support the examination of whether newly added spectra appear near existing reference groups or form distinct posterior groups.

Formation of a separate or singleton cluster may provide a useful warning signal that a spectrum is not adequately represented by the base reference set. However, the current script does not produce a definitive OOL classification or an OOL probability.

#### Outputs

The command-line script provides:

* Number of final clusters for the base analysis
* Number of final clusters after analyzing each new dataset
* UMAP-based density visualization of cluster centroids
* UMAP visualization of cluster centroids and newly added spectra

The current command-line implementation does not save individual cluster assignments or OOL decisions as a CSV file.

Default sampling settings:

| Analysis            | Gibbs iterations | Burn-in |
| ------------------- | ---------------: | ------: |
| Base reference data |           15,000 |   5,000 |
| Newly added data    |              500 |     100 |

Run the base analysis:

```bash
python dirichlet_process_mixture_method.py \
    --base-path base_data.xlsx
```

Run with additional data:

```bash
python dirichlet_process_mixture_method.py \
    --base-path base_data.xlsx \
    --new-paths new_data.xlsx
```

Analyze multiple new files:

```bash
python dirichlet_process_mixture_method.py \
    --base-path base_data.xlsx \
    --new-paths new_data_1.xlsx new_data_2.xlsx
```

Change the sampling settings:

```bash
python dirichlet_process_mixture_method.py \
    --base-path base_data.xlsx \
    --base-iters 15000 \
    --base-burn-in 5000 \
    --new-iters 500 \
    --new-burn-in 100
```

Skip UMAP visualization:

```bash
python dirichlet_process_mixture_method.py \
    --base-path base_data.xlsx \
    --skip-plot
```

Run on CPU:

```bash
python dirichlet_process_mixture_method.py \
    --base-path base_data.xlsx \
    --device cpu
```

## Input Data Format

The input Excel files must use long format, with each row representing one peak from one spectrum.

Required columns:

| Column      | Description                                                   |
| ----------- | ------------------------------------------------------------- |
| `Name`      | Identifier used to group multiple peak rows into one spectrum |
| `m/z`       | Nominal mass-to-charge value                                  |
| `Intensity` | Peak intensity                                                |

Example:

| Name         | m/z | Intensity |
| ------------ | --: | --------: |
| Spectrum_001 |  43 |       100 |
| Spectrum_001 |  57 |        62 |
| Spectrum_001 |  71 |        18 |
| Spectrum_002 |  41 |        74 |
| Spectrum_002 |  55 |       100 |

### Input requirements

* Each unique `Name` value is interpreted as one spectrum.
* Different spectra, including replicate measurements, must use different `Name` values.
* The `m/z` values should be positive nominal-mass integers.
* Intensities should be non-negative numeric values.
* Each combination of `Name` and `m/z` should appear at most once because duplicate peak rows are not aggregated by the scripts.
* Spectra with zero total or maximum intensity may cause an error during normalization.

The heuristic and distance-based scripts round `m/z` values to the nearest integer bin. The DPMM script converts `m/z` values to integers directly. Positive integer `m/z` values should therefore be used for consistent behavior across all three scripts.

For analysis with newly added spectra, the maximum `m/z` value in a new dataset must not exceed the maximum `m/z` dimension established by the base dataset.

### Additional column for the distance-based method

`distribution_distance_method.py` additionally requires:

| Column | Description                                                            |
| ------ | ---------------------------------------------------------------------- |
| `Form` | Chemical formula label used to determine the target number of clusters |

Each spectrum identified by `Name` should be associated with one consistent `Form` value.

The `Form` column determines only the target number of clusters in the controlled hierarchical-clustering evaluation. It does not provide the resulting cluster labels.

## Installation

Install the required packages:

```bash
pip install -r requirements.txt
```

A CUDA-capable PyTorch installation can be used for the distance-based and DPMM scripts. CPU execution is available through the `--device cpu` argument.

The heuristic script uses NumPy and runs on CPU.

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

The original GC-MS Excel data files are not included in this repository because they may contain private, institution-owned, research-related, or licensed reference-library information.

To run the scripts, prepare compatible Excel files using the format described above.

```bash
python heuristic_screening.py --excel-path your_data.xlsx

python distribution_distance_method.py --excel-path your_data.xlsx

python dirichlet_process_mixture_method.py \
    --base-path your_base_data.xlsx
```

Because the original reference spectra are not publicly distributed, the exact numerical results associated with the research cannot be reproduced using this repository alone.

Results obtained from other datasets may vary according to:

* Reference-library composition
* Data preprocessing
* Noise settings
* Hybrid-distance weight
* Clustering configuration
* Gibbs-sampling settings
* Software and hardware environment

## Reproducibility Notes

* `heuristic_screening.py` provides a `--seed` argument for its NumPy-based noise simulations.
* `distribution_distance_method.py` does not currently provide a command-line seed option for its PyTorch-based noise simulations.
* `dirichlet_process_mixture_method.py` does not currently provide a command-line seed option for Gibbs sampling.
* The UMAP visualization uses a fixed random state of 42.
* Results from the distance-based and DPMM scripts may therefore vary between runs.
* Package-version, CUDA, hardware, and numerical differences may also affect the results.

## Scope and Interpretation

This repository provides research implementations and analysis workflows for GC-MS spectrum screening, distance-based retrieval, hierarchical clustering, and Bayesian nonparametric clustering.

The code is intended primarily for:

* Inspection of the research implementation
* Methodological comparison
* Adaptation to compatible GC-MS datasets
* Research and portfolio documentation

The repository should not be interpreted as:

* A production-ready GC-MS compound identification system
* A validated standalone OOL classifier
* A tool that provides definitive OOL decisions from a single clustering result

Potential OOL interpretation should be based on additional validation and domain-specific evidence rather than cluster formation alone.

## Related Publication

This repository accompanies the following peer-reviewed article:

Minsu Son, Hyoju Kim, Hyungjun Kim, Youngho Jin, and Jaeoh Kim (2026). [Reliable Identification and Out-of-Library Detection in Mass Spectra](https://doi.org/10.1002/cem.70174). *Journal of Chemometrics*, **40**(8), e70174. https://doi.org/10.1002/cem.70174

If you use this code in your research, please cite:

```bibtex
@article{son2026reliable,
  author  = {Son, Minsu and Kim, Hyoju and Kim, Hyungjun and Jin, Youngho and Kim, Jaeoh},
  title   = {Reliable Identification and Out-of-Library Detection in Mass Spectra},
  journal = {Journal of Chemometrics},
  year    = {2026},
  volume  = {40},
  number  = {8},
  pages   = {e70174},
  doi     = {10.1002/cem.70174},
  url     = {https://doi.org/10.1002/cem.70174}
}
```

## License

No open-source license is currently specified. Please contact the repository owner before reusing or redistributing the code.

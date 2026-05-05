import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import trange


DEFAULT_TOLERANCES = (0.1, 0.2, 0.3)
DEFAULT_GAUSSIAN_SIGMAS = (1.0, 2.0, 3.0)
DEFAULT_SPURIOUS_NUM_PEAKS = (1, 2, 3)


def normalize_to_100(spectrum):
    spectrum = np.asarray(spectrum, dtype=np.float32)
    max_value = spectrum.max()

    if max_value <= 0:
        raise ValueError("Zero-maximum spectrum encountered")

    normalized = np.round((spectrum / max_value) * 100)
    normalized[normalized < 0] = 0

    return normalized.astype(np.float32)


def find_unique_candidate(query, library, tolerance):
    query = np.asarray(query, dtype=np.float32)
    library = np.asarray(library, dtype=np.float32)

    peak_order = np.argsort(query)[::-1]
    candidate_mask = np.ones(library.shape[0], dtype=bool)

    for mz_idx in peak_order:
        query_intensity = query[mz_idx]

        if query_intensity == 0:
            current_mask = library[:, mz_idx] == 0
        else:
            lower = query_intensity * (1 - tolerance)
            upper = query_intensity * (1 + tolerance)
            lower, upper = sorted((lower, upper))
            current_mask = (library[:, mz_idx] >= lower) & (library[:, mz_idx] <= upper)

        candidate_mask &= current_mask
        n_candidates = int(candidate_mask.sum())

        if n_candidates == 1:
            return int(np.flatnonzero(candidate_mask)[0])
        if n_candidates == 0:
            return -1

    return -1


def add_gaussian_noise(spectrum, sigma, rng):
    noisy = spectrum + rng.normal(0, sigma, size=spectrum.shape)
    return normalize_to_100(noisy)


def add_poisson_noise(spectrum, rng):
    noisy = rng.poisson(lam=spectrum).astype(np.float32)
    return normalize_to_100(noisy)


def add_spurious_peak(spectrum, num_peaks, rng, max_intensity=100.0):
    if num_peaks < 0:
        raise ValueError("num_peaks must be non-negative")
    if num_peaks > spectrum.shape[0]:
        raise ValueError("num_peaks cannot exceed the spectrum dimension")
    if max_intensity < 0:
        raise ValueError("max_intensity must be non-negative")

    noisy = spectrum.copy()

    if num_peaks == 0:
        return normalize_to_100(noisy)

    mz_indices = rng.choice(spectrum.shape[0], size=num_peaks, replace=False)
    intensities = rng.uniform(0, max_intensity, size=num_peaks)

    noisy[mz_indices] += intensities
    return normalize_to_100(noisy)


def read_ms_excel(excel_path):
    return pd.read_excel(excel_path)


def preprocess_ms_dataframe(ms):
    required_columns = {"Name", "m/z", "Intensity"}
    missing_columns = required_columns - set(ms.columns)

    if missing_columns:
        raise ValueError(f"Missing required columns: {sorted(missing_columns)}")

    processed = ms.copy()
    names = processed["Name"].drop_duplicates().reset_index(drop=True)
    name_to_idx = {name: idx for idx, name in enumerate(names)}

    processed["row_idx"] = processed["Name"].map(name_to_idx)
    processed["m/z_idx"] = (np.round(processed["m/z"]) - 1).astype("Int64")

    valid_rows = processed["row_idx"].notna() & processed["m/z_idx"].notna()
    processed = processed.loc[valid_rows].copy()

    if processed.empty:
        raise ValueError("No valid mass spectrum rows were found")

    return processed


def build_spectrum_matrix(ms):
    row_indices = ms["row_idx"].values.astype(np.int64)
    mz_indices = ms["m/z_idx"].values.astype(np.int64)
    intensities = ms["Intensity"].values.astype(np.float32)

    n_samples = int(row_indices.max() + 1)
    dimension = int(mz_indices.max() + 1)

    matrix = np.zeros((n_samples, dimension), dtype=np.float32)
    matrix[row_indices, mz_indices] = intensities

    return matrix


def load_ms_spectrum_matrix(excel_path):
    ms = read_ms_excel(excel_path)
    processed_ms = preprocess_ms_dataframe(ms)
    spectrum_matrix = build_spectrum_matrix(processed_ms)

    return processed_ms, spectrum_matrix


def evaluate_single_trial(query, library, target_idx, tolerances):
    return {
        tolerance: int(find_unique_candidate(query, library, tolerance) == target_idx)
        for tolerance in tolerances
    }


def summarize_accuracy(correct_counts, n_samples, n_trials):
    denominator = n_samples * n_trials
    return {
        tolerance: count / denominator
        for tolerance, count in correct_counts.items()
    }


def evaluate_unique_candidate_accuracy(data, noise_fn, tolerances=DEFAULT_TOLERANCES, n_trials=50):
    n_samples = data.shape[0]
    correct_counts = {tolerance: 0 for tolerance in tolerances}

    for sample_idx in trange(n_samples, desc="Processing rows"):
        for _ in range(n_trials):
            noisy_query = noise_fn(data[sample_idx])
            trial_result = evaluate_single_trial(
                query=noisy_query,
                library=data,
                target_idx=sample_idx,
                tolerances=tolerances,
            )

            for tolerance, is_correct in trial_result.items():
                correct_counts[tolerance] += is_correct

    return summarize_accuracy(correct_counts, n_samples=n_samples, n_trials=n_trials)


def make_result_frame(noise_type, parameter_name, parameter_value, accuracy_by_tolerance):
    rows = []

    for tolerance, accuracy in accuracy_by_tolerance.items():
        rows.append(
            {
                "noise_type": noise_type,
                "parameter_name": parameter_name,
                "parameter_value": parameter_value,
                "tolerance": tolerance,
                "accuracy": accuracy,
            }
        )

    return pd.DataFrame(rows)


def run_gaussian_noise_experiments(data, sigmas, tolerances, n_trials, rng):
    result_frames = []

    for sigma in sigmas:
        accuracy = evaluate_unique_candidate_accuracy(
            data=data,
            noise_fn=lambda x, sigma=sigma: add_gaussian_noise(x, sigma=sigma, rng=rng),
            tolerances=tolerances,
            n_trials=n_trials,
        )
        result_frames.append(
            make_result_frame(
                noise_type="gaussian",
                parameter_name="sigma",
                parameter_value=sigma,
                accuracy_by_tolerance=accuracy,
            )
        )

    return pd.concat(result_frames, ignore_index=True)


def run_poisson_noise_experiment(data, tolerances, n_trials, rng):
    accuracy = evaluate_unique_candidate_accuracy(
        data=data,
        noise_fn=lambda x: add_poisson_noise(x, rng=rng),
        tolerances=tolerances,
        n_trials=n_trials,
    )

    return make_result_frame(
        noise_type="poisson",
        parameter_name=None,
        parameter_value=None,
        accuracy_by_tolerance=accuracy,
    )


def run_spurious_peak_experiments(data, num_peaks_list, tolerances, n_trials, rng, max_intensity=100.0):
    result_frames = []

    for num_peaks in num_peaks_list:
        accuracy = evaluate_unique_candidate_accuracy(
            data=data,
            noise_fn=lambda x, num_peaks=num_peaks: add_spurious_peak(
                x,
                num_peaks=num_peaks,
                rng=rng,
                max_intensity=max_intensity,
            ),
            tolerances=tolerances,
            n_trials=n_trials,
        )
        result_frames.append(
            make_result_frame(
                noise_type="spurious_peak",
                parameter_name="num_peaks",
                parameter_value=num_peaks,
                accuracy_by_tolerance=accuracy,
            )
        )

    return pd.concat(result_frames, ignore_index=True)


def run_all_experiments(
    data,
    tolerances=DEFAULT_TOLERANCES,
    n_trials=50,
    gaussian_sigmas=DEFAULT_GAUSSIAN_SIGMAS,
    spurious_num_peaks=DEFAULT_SPURIOUS_NUM_PEAKS,
    max_spurious_intensity=100.0,
    seed=None,
):
    rng = np.random.default_rng(seed)

    result_frames = [
        run_gaussian_noise_experiments(
            data=data,
            sigmas=gaussian_sigmas,
            tolerances=tolerances,
            n_trials=n_trials,
            rng=rng,
        ),
        run_poisson_noise_experiment(
            data=data,
            tolerances=tolerances,
            n_trials=n_trials,
            rng=rng,
        ),
        run_spurious_peak_experiments(
            data=data,
            num_peaks_list=spurious_num_peaks,
            tolerances=tolerances,
            n_trials=n_trials,
            rng=rng,
            max_intensity=max_spurious_intensity,
        ),
    ]

    return pd.concat(result_frames, ignore_index=True)


def print_accuracy_results(results):
    for (noise_type, parameter_name, parameter_value), group in results.groupby(
        ["noise_type", "parameter_name", "parameter_value"], dropna=False
    ):
        if pd.isna(parameter_name):
            title = f"{noise_type}_result"
        else:
            title = f"{noise_type}_{parameter_name}_{parameter_value:g}_result"

        print(title)
        for _, row in group.sort_values("tolerance").iterrows():
            print(f"acc_{int(row['tolerance'] * 100)}: {row['accuracy'] * 100:.2f}%")
        print()


def parse_args():
    parser = argparse.ArgumentParser(description="Heuristic screening analysis for mass spectra.")
    parser.add_argument(
        "--excel-path",
        type=Path,
        required=True,
        help="Path to the Excel file used in the original notebook.",
    )
    parser.add_argument("--n-trials", type=int, default=50, help="Number of trials for repeated evaluations.")
    parser.add_argument(
        "--tolerances",
        type=float,
        nargs="+",
        default=list(DEFAULT_TOLERANCES),
        help="Tolerance values used to screen unique candidates.",
    )
    parser.add_argument(
        "--gaussian-sigmas",
        type=float,
        nargs="+",
        default=list(DEFAULT_GAUSSIAN_SIGMAS),
        help="Gaussian noise standard deviations to evaluate.",
    )
    parser.add_argument(
        "--spurious-num-peaks",
        type=int,
        nargs="+",
        default=list(DEFAULT_SPURIOUS_NUM_PEAKS),
        help="Numbers of spurious peaks to evaluate.",
    )
    parser.add_argument(
        "--max-spurious-intensity",
        type=float,
        default=100.0,
        help="Maximum intensity for added spurious peaks.",
    )
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility.")
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=None,
        help="Optional path to save the result table as a CSV file.",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    _, data = load_ms_spectrum_matrix(args.excel_path)
    results = run_all_experiments(
        data=data,
        tolerances=tuple(args.tolerances),
        n_trials=args.n_trials,
        gaussian_sigmas=tuple(args.gaussian_sigmas),
        spurious_num_peaks=tuple(args.spurious_num_peaks),
        max_spurious_intensity=args.max_spurious_intensity,
        seed=args.seed,
    )

    print_accuracy_results(results)

    if args.output_csv is not None:
        results.to_csv(args.output_csv, index=False)
        print(f"Saved results to {args.output_csv}")


if __name__ == "__main__":
    main()

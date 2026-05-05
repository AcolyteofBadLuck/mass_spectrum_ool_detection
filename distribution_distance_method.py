import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from tqdm import trange


DEFAULT_GAUSSIAN_SIGMAS = (1.0, 2.0, 3.0)
DEFAULT_SPURIOUS_NUM_PEAKS = (1, 2, 3)
DEFAULT_SEARCH_NOISES = ("gaussian_sigma_2", "poisson", "spurious_peak_2")
DEFAULT_AGGREGATION_METHODS = ("min", "mean", "max")
DEFAULT_METRICS = ("hel", "jsd", "csd", "hyb")


def hellinger_distance_matrix(query, library, eps=1e-12):
    query_norm = query / (query.sum() + eps)
    library_norm = library / (library.sum(dim=1, keepdim=True) + eps)

    query_expanded = query_norm.expand_as(library_norm)
    bc = torch.clamp(torch.sum(torch.sqrt(query_expanded * library_norm + eps), dim=1), eps, 1.0)

    return torch.sqrt(1.0 - bc)


def hellinger_pdist(data, eps=1e-12):
    normalized = torch.sqrt(data / (data.sum(dim=1, keepdim=True) + eps))
    bc = torch.clamp(torch.matmul(normalized, normalized.T), 0.0, 1.0)
    dist_matrix = torch.sqrt(1.0 - bc)

    return squareform(dist_matrix.cpu().numpy(), checks=False)


def sqrt_jsd_matrix(query, library, eps=1e-12):
    query_norm = query / (query.sum() + eps)
    library_norm = library / (library.sum(dim=1, keepdim=True) + eps)

    query_expanded = query_norm.expand_as(library_norm)
    midpoint = 0.5 * (query_expanded + library_norm)

    kl_query = torch.sum(query_expanded * torch.log2((query_expanded + eps) / (midpoint + eps)), dim=1)
    kl_library = torch.sum(library_norm * torch.log2((library_norm + eps) / (midpoint + eps)), dim=1)

    return torch.sqrt(0.5 * kl_query + 0.5 * kl_library)


def sqrt_jsd_pdist(data, eps=1e-12, batch_size=256):
    normalized = data / (data.sum(dim=1, keepdim=True) + eps)
    n_samples = normalized.shape[0]

    entropy = -torch.sum(normalized * torch.log2(normalized + eps), dim=1)
    dist_matrix = torch.zeros((n_samples, n_samples), dtype=torch.float32, device=data.device)

    for start in range(0, n_samples, batch_size):
        end = min(start + batch_size, n_samples)

        batch = normalized[start:end]
        midpoint = 0.5 * (batch.unsqueeze(1) + normalized.unsqueeze(0))
        midpoint_entropy = -torch.sum(midpoint * torch.log2(midpoint + eps), dim=2)

        jsd = midpoint_entropy - 0.5 * (entropy[start:end].unsqueeze(1) + entropy.unsqueeze(0))
        dist_matrix[start:end] = torch.sqrt(torch.clamp(jsd, min=0.0))

    dist_matrix = 0.5 * (dist_matrix + dist_matrix.T)
    return squareform(dist_matrix.cpu().numpy(), checks=False)


def cosine_distance_matrix(query, library, eps=1e-12):
    numerator = torch.matmul(library, query)

    query_norm = torch.norm(query) + eps
    library_norm = torch.norm(library, dim=1) + eps
    denominator = query_norm * library_norm

    cosine_similarity = torch.clamp(numerator / denominator, 0.0, 1.0)
    return torch.arccos(cosine_similarity) / torch.pi * 2


def cosine_pdist(data, eps=1e-12):
    normalized = data / (torch.norm(data, dim=1, keepdim=True) + eps)
    cosine_similarity = torch.clamp(torch.matmul(normalized, normalized.T), 0.0, 1.0)
    dist_matrix = torch.arccos(cosine_similarity) / torch.pi * 2

    return squareform(dist_matrix.cpu().numpy(), checks=False)


def hybrid_distance(x, y, lamb=0.5, d=1):
    return (lamb * (x**d) + (1 - lamb) * (y**d)) ** (1 / d)


def normalize_spectrum(spectrum):
    denom = spectrum.sum()

    if denom == 0:
        raise ValueError("Zero-sum spectrum encountered")

    return spectrum / denom


def add_gaussian_noise(spectrum, sigma):
    noise = torch.normal(mean=0.0, std=sigma, size=spectrum.shape, device=spectrum.device)
    noisy = torch.clamp(spectrum + noise, min=0)

    return normalize_spectrum(noisy)


def add_poisson_noise(spectrum):
    noisy = torch.poisson(spectrum).to(torch.float32)

    return normalize_spectrum(noisy)


def add_spurious_peak(spectrum, num_peaks, max_intensity=100.0):
    if num_peaks < 0:
        raise ValueError("num_peaks must be non-negative")
    if num_peaks > spectrum.shape[0]:
        raise ValueError("num_peaks cannot exceed the spectrum dimension")
    if max_intensity < 0:
        raise ValueError("max_intensity must be non-negative")

    noise = torch.zeros(spectrum.shape[0], device=spectrum.device)

    if num_peaks > 0:
        mz_indices = torch.randperm(spectrum.shape[0], device=spectrum.device)[:num_peaks]
        intensities = max_intensity * torch.rand(num_peaks, device=spectrum.device)
        noise[mz_indices] = intensities

    noisy = spectrum + noise
    return normalize_spectrum(noisy)


def make_noise_functions(max_spurious_intensity=100.0):
    return {
        "gaussian_sigma_1": lambda x: add_gaussian_noise(x, sigma=1),
        "gaussian_sigma_2": lambda x: add_gaussian_noise(x, sigma=2),
        "gaussian_sigma_3": lambda x: add_gaussian_noise(x, sigma=3),
        "poisson": lambda x: add_poisson_noise(x),
        "spurious_peak_1": lambda x: add_spurious_peak(
            x,
            num_peaks=1,
            max_intensity=max_spurious_intensity,
        ),
        "spurious_peak_2": lambda x: add_spurious_peak(
            x,
            num_peaks=2,
            max_intensity=max_spurious_intensity,
        ),
        "spurious_peak_3": lambda x: add_spurious_peak(
            x,
            num_peaks=3,
            max_intensity=max_spurious_intensity,
        ),
    }


def compute_metric_distances(query, library, lamb):
    jsd = sqrt_jsd_matrix(query, library)
    csd = cosine_distance_matrix(query, library)

    return {
        "hel": hellinger_distance_matrix(query, library),
        "jsd": jsd,
        "csd": csd,
        "hyb": hybrid_distance(jsd, csd, lamb),
    }


def aggregate_cluster_distances(dist, cluster_labels, n_cluster, method):
    if method == "min":
        cluster_dist = torch.full((n_cluster,), float("inf"), device=dist.device)
        cluster_dist.scatter_reduce_(0, cluster_labels, dist, reduce="amin")
        return cluster_dist

    if method == "mean":
        cluster_sum = torch.zeros(n_cluster, device=dist.device)
        cluster_count = torch.zeros(n_cluster, device=dist.device)

        cluster_sum.scatter_add_(0, cluster_labels, dist)
        cluster_count.scatter_add_(0, cluster_labels, torch.ones_like(dist))

        return cluster_sum / cluster_count.clamp(min=1)

    if method == "max":
        cluster_dist = torch.full((n_cluster,), float("-inf"), device=dist.device)
        cluster_dist.scatter_reduce_(0, cluster_labels, dist, reduce="amax")
        return cluster_dist

    raise ValueError(f"Unknown aggregation method: {method}")


def evaluate_lambda_single_noise(data, noise_fn, n_lambda=21, n_trials=50):
    n_samples = data.shape[0]
    results = []

    for sample_idx in trange(n_samples, desc="Processing rows"):
        correct_counts = torch.zeros(n_lambda, device=data.device)

        for _ in range(n_trials):
            noisy_query = noise_fn(data[sample_idx])
            jsd = sqrt_jsd_matrix(noisy_query, data)
            csd = cosine_distance_matrix(noisy_query, data)

            for lambda_idx in range(n_lambda):
                lamb = lambda_idx / (n_lambda - 1)
                hyb = hybrid_distance(jsd, csd, lamb)

                if int(torch.argmin(hyb).item()) == sample_idx:
                    correct_counts[lambda_idx] += 1

        results.append(correct_counts)

    results = torch.stack(results).cpu().numpy()
    return results.mean(axis=0) / n_trials


def tune_hybrid_lambda(data, noise_names=DEFAULT_SEARCH_NOISES, n_lambda=21, n_trials=50):
    noise_functions = make_noise_functions()
    result_by_noise = []

    for noise_name in noise_names:
        accuracy = evaluate_lambda_single_noise(
            data=data,
            noise_fn=noise_functions[noise_name],
            n_lambda=n_lambda,
            n_trials=n_trials,
        )
        result_by_noise.append(accuracy)

    accuracy_avg = np.mean(np.vstack(result_by_noise), axis=0)
    best_lambda = int(accuracy_avg.argmax()) / (n_lambda - 1)

    return best_lambda, accuracy_avg


def evaluate_aggregation_single_trial(query, library, target_idx, lamb, n_cluster, cluster_labels, cluster_labels_gpu):
    metric_distances = compute_metric_distances(query, library, lamb)
    trial_result = {}

    for metric_name, dist in metric_distances.items():
        true_cluster = cluster_labels[metric_name][target_idx]
        gpu_labels = cluster_labels_gpu[metric_name] - 1

        for aggregation_method in DEFAULT_AGGREGATION_METHODS:
            cluster_dist = aggregate_cluster_distances(
                dist=dist,
                cluster_labels=gpu_labels,
                n_cluster=n_cluster,
                method=aggregation_method,
            )
            pred_cluster = int(cluster_dist.argmin().item()) + 1
            result_name = f"{metric_name}_{aggregation_method}_acc"
            trial_result[result_name] = int(pred_cluster == true_cluster)

    return trial_result


def evaluate_cluster_aggregation_accuracy(data, n_cluster, lamb, cluster_labels, cluster_labels_gpu, noise_fn, n_trials=50):
    n_samples = data.shape[0]
    correct_counts = {
        f"{metric}_{aggregation}_acc": 0
        for aggregation in DEFAULT_AGGREGATION_METHODS
        for metric in DEFAULT_METRICS
    }

    for sample_idx in trange(n_samples, desc="Processing rows"):
        for _ in range(n_trials):
            noisy_query = noise_fn(data[sample_idx])
            trial_result = evaluate_aggregation_single_trial(
                query=noisy_query,
                library=data,
                target_idx=sample_idx,
                lamb=lamb,
                n_cluster=n_cluster,
                cluster_labels=cluster_labels,
                cluster_labels_gpu=cluster_labels_gpu,
            )

            for result_name, is_correct in trial_result.items():
                correct_counts[result_name] += is_correct

    return summarize_accuracy(correct_counts, n_samples=n_samples, n_trials=n_trials)


def evaluate_retrieval_single_trial(query, library, target_idx, lamb, cluster_labels):
    metric_distances = compute_metric_distances(query, library, lamb)
    trial_result = {}

    for metric_name, dist in metric_distances.items():
        pred_idx = int(dist.argmin().item())
        true_cluster = cluster_labels[metric_name][target_idx]
        pred_cluster = cluster_labels[metric_name][pred_idx]

        trial_result[f"{metric_name}_cluster_acc"] = int(pred_cluster == true_cluster)
        trial_result[f"{metric_name}_predict_acc"] = int(pred_idx == target_idx)

    return trial_result


def evaluate_retrieval_accuracy(data, lamb, cluster_labels, noise_fn, n_trials=50):
    n_samples = data.shape[0]
    correct_counts = {
        f"{metric}_{result_type}_acc": 0
        for metric in DEFAULT_METRICS
        for result_type in ("cluster", "predict")
    }

    for sample_idx in trange(n_samples, desc="Processing rows"):
        for _ in range(n_trials):
            noisy_query = noise_fn(data[sample_idx])
            trial_result = evaluate_retrieval_single_trial(
                query=noisy_query,
                library=data,
                target_idx=sample_idx,
                lamb=lamb,
                cluster_labels=cluster_labels,
            )

            for result_name, is_correct in trial_result.items():
                correct_counts[result_name] += is_correct

    return summarize_accuracy(correct_counts, n_samples=n_samples, n_trials=n_trials)


def summarize_accuracy(correct_counts, n_samples, n_trials):
    denominator = n_samples * n_trials
    return {
        result_name: count / denominator
        for result_name, count in correct_counts.items()
    }


def make_result_frame(experiment_name, result_type, accuracy_by_metric):
    rows = []

    for metric_name, accuracy in accuracy_by_metric.items():
        rows.append(
            {
                "experiment": experiment_name,
                "result_type": result_type,
                "metric": metric_name,
                "accuracy": accuracy,
            }
        )

    return pd.DataFrame(rows)


def read_ms_excel(excel_path):
    return pd.read_excel(excel_path)


def preprocess_ms_dataframe(ms):
    required_columns = {"Name", "m/z", "Intensity", "Form"}
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


def build_spectrum_matrix(ms, device):
    row_indices = torch.from_numpy(ms["row_idx"].values.astype(np.int64)).to(device)
    mz_indices = torch.from_numpy(ms["m/z_idx"].values.astype(np.int64)).to(device)
    intensities = torch.from_numpy(ms["Intensity"].values.astype(np.float32)).to(device)

    n_samples = int(ms["row_idx"].max() + 1)
    dimension = int(ms["m/z_idx"].max() + 1)

    matrix = torch.zeros((n_samples, dimension), dtype=torch.float32, device=device)
    matrix[row_indices, mz_indices] = torch.round(intensities)

    return matrix


def load_ms_spectrum_matrix(excel_path, device):
    ms = read_ms_excel(excel_path)
    processed_ms = preprocess_ms_dataframe(ms)
    spectrum_matrix = build_spectrum_matrix(processed_ms, device=device)

    return processed_ms, spectrum_matrix


def build_distance_matrices(data, lamb):
    jsd_dist = sqrt_jsd_pdist(data)
    csd_dist = cosine_pdist(data)

    return {
        "hel": hellinger_pdist(data),
        "jsd": jsd_dist,
        "csd": csd_dist,
        "hyb": hybrid_distance(jsd_dist, csd_dist, lamb),
    }


def get_n_clusters_from_form(ms):
    form = ms[["Name", "Form"]].drop_duplicates().reset_index(drop=True)["Form"]
    return len(pd.unique(form))


def build_clusters(ms, data, lamb, linkage_method="complete"):
    n_cluster = get_n_clusters_from_form(ms)
    distance_matrices = build_distance_matrices(data, lamb)

    cluster_labels = {}
    for metric_name, dist in distance_matrices.items():
        z = linkage(dist, method=linkage_method)
        cluster_labels[metric_name] = fcluster(z, t=n_cluster, criterion="maxclust")

    cluster_labels_gpu = {
        metric_name: torch.tensor(labels, dtype=torch.long, device=data.device)
        for metric_name, labels in cluster_labels.items()
    }

    return n_cluster, cluster_labels, cluster_labels_gpu


def run_aggregation_experiments(data, n_cluster, lamb, cluster_labels, cluster_labels_gpu, n_trials):
    noise_functions = make_noise_functions()
    experiment_noise_names = {
        "g_metric_acc": "gaussian_sigma_2",
        "p_metric_acc": "poisson",
        "s_metric_acc": "spurious_peak_2",
    }
    result_frames = []

    for experiment_name, noise_name in experiment_noise_names.items():
        accuracy = evaluate_cluster_aggregation_accuracy(
            data=data,
            n_cluster=n_cluster,
            lamb=lamb,
            cluster_labels=cluster_labels,
            cluster_labels_gpu=cluster_labels_gpu,
            noise_fn=noise_functions[noise_name],
            n_trials=n_trials,
        )
        print_accuracy_results(experiment_name, accuracy)
        result_frames.append(
            make_result_frame(
                experiment_name=experiment_name,
                result_type="aggregation",
                accuracy_by_metric=accuracy,
            )
        )

    return pd.concat(result_frames, ignore_index=True)


def run_noise_robustness_experiments(data, lamb, cluster_labels, n_trials):
    noise_functions = make_noise_functions()
    experiment_noise_names = {
        "g_sigma_1_result": "gaussian_sigma_1",
        "g_sigma_2_result": "gaussian_sigma_2",
        "g_sigma_3_result": "gaussian_sigma_3",
        "p_result": "poisson",
        "s_1_result": "spurious_peak_1",
        "s_2_result": "spurious_peak_2",
        "s_3_result": "spurious_peak_3",
    }
    result_frames = []

    for experiment_name, noise_name in experiment_noise_names.items():
        accuracy = evaluate_retrieval_accuracy(
            data=data,
            lamb=lamb,
            cluster_labels=cluster_labels,
            noise_fn=noise_functions[noise_name],
            n_trials=n_trials,
        )
        print_accuracy_results(experiment_name, accuracy)
        result_frames.append(
            make_result_frame(
                experiment_name=experiment_name,
                result_type="retrieval",
                accuracy_by_metric=accuracy,
            )
        )

    return pd.concat(result_frames, ignore_index=True)


def print_accuracy_results(result_name, result_dict):
    print(result_name)
    for key, value in result_dict.items():
        print(f"{key}: {value * 100:.2f}%")
    print()


def parse_args():
    parser = argparse.ArgumentParser(description="Distance-based analysis for mass spectra.")
    parser.add_argument(
        "--excel-path",
        type=Path,
        required=True,
        help="Path to the Excel file used in the original notebook.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help='Torch device to use, e.g. "cuda" or "cpu".',
    )
    parser.add_argument("--n-trials", type=int, default=50, help="Number of trials for repeated evaluations.")
    parser.add_argument("--n-lambda", type=int, default=21, help="Number of lambda candidates for grid search.")
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=None,
        help="Optional path to save the result table as a CSV file.",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    ms, data = load_ms_spectrum_matrix(args.excel_path, device=args.device)
    lamb, lambda_accuracy = tune_hybrid_lambda(
        data=data,
        n_lambda=args.n_lambda,
        n_trials=args.n_trials,
    )

    print("acc_avg:", lambda_accuracy)
    print("lambda:", lamb)

    n_cluster, cluster_labels, cluster_labels_gpu = build_clusters(ms, data, lamb)
    print("n_cluster:", n_cluster)

    aggregation_results = run_aggregation_experiments(
        data=data,
        n_cluster=n_cluster,
        lamb=lamb,
        cluster_labels=cluster_labels,
        cluster_labels_gpu=cluster_labels_gpu,
        n_trials=args.n_trials,
    )
    retrieval_results = run_noise_robustness_experiments(
        data=data,
        lamb=lamb,
        cluster_labels=cluster_labels,
        n_trials=args.n_trials,
    )
    results = pd.concat([aggregation_results, retrieval_results], ignore_index=True)

    if args.output_csv is not None:
        results.to_csv(args.output_csv, index=False)
        print(f"Saved results to {args.output_csv}")


if __name__ == "__main__":
    main()

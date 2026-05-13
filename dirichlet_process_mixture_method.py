import argparse
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import umap.umap_ as umap
from scipy.optimize import linear_sum_assignment
from torch.special import gammaln, polygamma, psi
from tqdm import trange


def load_spectrum_matrix(path, device="cuda", reference_dim=None):
    ms = pd.read_excel(path)
    ms["idx"] = ms.groupby("Name", sort=False).ngroup()

    idx = torch.from_numpy(ms["idx"].values.astype(np.int64)).to(device)
    mz = torch.from_numpy(ms["m/z"].values.astype(np.int64)).to(device) - 1
    intensity = torch.from_numpy(ms["Intensity"].values.astype(np.float32)).to(device)

    n_samples = int(ms["idx"].max() + 1)
    dim = int(ms["m/z"].max()) if reference_dim is None else reference_dim

    reshaped = torch.zeros((n_samples, dim), dtype=torch.float32, device=device)
    reshaped[idx, mz] = intensity

    return reshaped


def filter_active_features(matrix):
    return matrix[:, matrix.max(dim=0).values > 0]


def concat_new_with_base(new_matrix, base_matrix):
    return torch.cat([new_matrix, base_matrix], dim=0)


def log_dirichlet_multinomial(y, r):
    s_r = r.sum()
    n = y.sum()
    return gammaln(s_r) - gammaln(s_r + n) + (gammaln(r + y) - gammaln(r)).sum()


def log_dirichlet_multinomial_vec(y, r):
    s_r = r.sum(1)
    n = y.sum()
    return gammaln(s_r) - gammaln(s_r + n) + (gammaln(r + y) - gammaln(r)).sum(1)


def estimate_dirichlet_prior(data, r, device, tol=1e-4, max_iter=1000):
    eps_r = 1e-6
    old_log_p = torch.tensor(torch.nan, dtype=torch.float32, device=device)

    for _ in range(max_iter):
        r_s = torch.sum(r)
        num = (psi(data + r) - psi(r)).sum(0)
        den = (psi(data.sum(1) + r_s) - psi(r_s)).sum()

        r = torch.clamp(r * num / den, eps_r, torch.inf)

        log_p = sum(log_dirichlet_multinomial(x, r) for x in data)
        if torch.isfinite(old_log_p) and abs(log_p - old_log_p) < tol:
            break
        old_log_p = log_p

    return r


def estimate_alpha(n, k, device, alpha=1.0, tol=1e-6, max_iter=1000):
    alpha = torch.tensor(alpha, dtype=torch.float32, device=device)

    for _ in range(max_iter):
        f = k / alpha + psi(alpha) - psi(alpha + n)
        fp = -k / alpha**2 + polygamma(1, alpha) - polygamma(1, alpha + n)
        alpha_new = alpha - f / fp

        if alpha_new <= 0:
            alpha_new = 1e-8
        if abs(alpha_new - alpha) < tol:
            break

        alpha = alpha_new
    return alpha


def crp_gibbs(data, c_init, device, max_iters=1000):
    n, data_dim = data.shape

    z = c_init.clone()
    n_k = torch.bincount(z).tolist()
    k = len(n_k)
    r = estimate_dirichlet_prior(data, r=torch.ones(data_dim, dtype=torch.float32, device=device), device=device)

    res = torch.zeros((n, max_iters), dtype=torch.int32, device="cpu")

    for iter_idx in trange(max_iters, desc="Sampling"):
        alpha = estimate_alpha(n, k, device)

        for i in range(n):
            c_i = int(z[i].item())
            n_k[c_i] -= 1
            z[i] = -1

            if n_k[c_i] == 0:
                last = k - 1
                if c_i != last:
                    z[z == last] = c_i
                    n_k[c_i] = n_k[last]
                n_k.pop()
                k -= 1

            log_probs = torch.empty(k + 1, device=device)

            if k > 0:
                    mask = z != -1
                    
                    y_sum = torch.zeros((k, data_dim), dtype=torch.float32, device=device)
                    y_sum.index_add_(0, z[mask].long(), data[mask])
                    
                    counts = torch.tensor(n_k, dtype=torch.float32, device=device)
                    
                    log_prior_existing = torch.log(counts)
                    log_like_existing = log_dirichlet_multinomial_vec(data[i], r + y_sum)
                    log_probs[:k] = log_prior_existing + log_like_existing

            log_probs[k] = torch.log(alpha) + log_dirichlet_multinomial(data[i], r)

            probs = torch.softmax(log_probs - log_probs.max(), dim=0)
            c_new = int(torch.multinomial(probs, 1).item())

            if c_new == k:
                z[i] = k
                n_k.append(1)
                k += 1
            else:
                z[i] = c_new
                n_k[c_new] += 1

        res[:, iter_idx] = z.to(torch.int32).cpu()

    return res


def _relabel_to_compact(z):
    uniq, inv = torch.unique(z, sorted=True, return_inverse=True)
    return inv.to(z.dtype), int(uniq.numel())


def align_labels_by_hungarian(res, burn_in):
    orig_device = res.device
    z = res.to("cpu", dtype=torch.int64).clone()
    n, t = z.shape
    assert burn_in < t

    ref_iter = burn_in
    canon, _ = _relabel_to_compact(z[:, ref_iter])
    z[:, ref_iter] = canon

    canon_fixed = canon.clone()
    k_c = int(canon_fixed.max().item()) + 1
    next_label_id = k_c

    for iter_idx in range(burn_in, t):
        if iter_idx == ref_iter:
            continue

        zt, k_z = _relabel_to_compact(z[:, iter_idx])

        flat_idx = zt * k_c + canon_fixed
        cost = torch.bincount(flat_idx, minlength=k_z * k_c).view(k_z, k_c)

        r_ind, c_ind = linear_sum_assignment((-cost).numpy())

        map_vec = torch.full((k_z,), -1, dtype=torch.int64)
        map_vec[torch.as_tensor(r_ind, dtype=torch.int64)] = torch.as_tensor(c_ind, dtype=torch.int64)

        unmatched = torch.where(map_vec < 0)[0]
        if len(unmatched) > 0:
            new_ids = torch.arange(next_label_id, next_label_id + len(unmatched), dtype=torch.int64)
            map_vec[unmatched] = new_ids
            next_label_id += len(unmatched)

        z[:, iter_idx] = map_vec[zt]

    aligned_res = z.to(orig_device)
    post_res = aligned_res[:, burn_in:]
    final_clusters = torch.mode(post_res, dim=1).values
    final_clusters, _ = _relabel_to_compact(final_clusters)

    return final_clusters, aligned_res


def l2_normalize(x, eps=1e-12):
    n = np.linalg.norm(x, axis=1, keepdims=True)
    return x / (n + eps)


def centroids_and_sizes(data_cpu, clusters_cpu, normalize_centroid=True):
    unique_clusters = torch.unique(clusters_cpu)
    unique_clusters, _ = torch.sort(unique_clusters)

    centroids = []
    sizes = []
    for cid in unique_clusters:
        mask = clusters_cpu == cid
        sizes.append(mask.sum().item())
        c = data_cpu[mask].mean(dim=0)
        centroids.append(c)

    centroids = torch.stack(centroids).cpu().numpy()
    sizes = np.asarray(sizes, dtype=np.int64)

    if normalize_centroid:
        centroids = l2_normalize(centroids)

    return unique_clusters.cpu().numpy(), centroids, sizes


def fix_kde_pdf_artifacts(ax):
    for col in ax.collections:
        col.set_edgecolor("face")
        col.set_linewidth(0.0)
        col.set_antialiased(False)
        col.set_rasterized(True)


def visualize_umap_clusters(
    data_tensor,
    cluster_tensor,
    n_new=None,
    n_neighbors=15,
    min_dist=0.1,
    metric="cosine",
    random_state=42,
    figsize=(7, 6),
    centroid_size=30,
    new_point_size=50,
):
    data_cpu = data_tensor.detach().to("cpu")
    clusters_cpu = cluster_tensor.detach().to("cpu").to(torch.int64)

    x = data_cpu.numpy()
    x = l2_normalize(x)

    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric=metric,
        random_state=random_state,
    )
    embedding_all = reducer.fit_transform(x)

    _, centroids, cluster_sizes = centroids_and_sizes(
        data_cpu,
        clusters_cpu,
        normalize_centroid=True,
    )
    embedding_centroids = reducer.transform(centroids)

    plt.figure(figsize=figsize)
    ax = sns.kdeplot(
        x=embedding_centroids[:, 0],
        y=embedding_centroids[:, 1],
        fill=True,
        cmap="viridis",
        bw_adjust=0.5,
        levels=20,
        thresh=0.05,
        linewidths=0,
    )
    fix_kde_pdf_artifacts(ax)
    plt.xlabel("UMAP-1")
    plt.ylabel("UMAP-2")
    plt.tight_layout()
    plt.show()

    idx = np.argsort(cluster_sizes)

    plt.figure(figsize=figsize)
    sc = plt.scatter(
        embedding_centroids[idx, 0],
        embedding_centroids[idx, 1],
        s=centroid_size,
        c=np.log1p(cluster_sizes[idx]),
        cmap="plasma",
        alpha=0.8,
        edgecolors="none",
        label="Cluster centroids",
    )

    if n_new is not None and n_new > 0:
        embedding_new = embedding_all[:n_new]
        plt.scatter(
            embedding_new[:, 0],
            embedding_new[:, 1],
            s=new_point_size,
            c="red",
            marker="x",
            alpha=0.9,
            label=f"Newly added spectra (n={n_new})",
        )
        plt.legend(loc="best", frameon=True)

    plt.xlabel("UMAP-1")
    plt.ylabel("UMAP-2")
    plt.colorbar(sc, label="log(1+cluster size)")
    plt.tight_layout()
    plt.show()

    return {
        "embedding_all": embedding_all,
        "embedding_centroids": embedding_centroids,
        "cluster_sizes": cluster_sizes,
    }


def run_base_analysis(base_path, device, max_iters, burn_in):
    reshaped_ms = load_spectrum_matrix(base_path, device=device)
    data = filter_active_features(reshaped_ms)

    res = crp_gibbs(
        data=data,
        c_init=torch.zeros(data.shape[0], dtype=torch.int32, device=device),
        device=device,
        max_iters=max_iters,
    )
    final_clusters, aligned_res = align_labels_by_hungarian(res=res, burn_in=burn_in)

    unique, counts = torch.unique(final_clusters, return_counts=True)
    print(f"base_n_cluster: {len(unique)}")

    return reshaped_ms, data, final_clusters, aligned_res


def run_new_analysis(new_path, base_matrix, base_clusters, device, max_iters, burn_in):
    reshaped_ms = load_spectrum_matrix(new_path, device=device, reference_dim=base_matrix.shape[1])
    data = filter_active_features(concat_new_with_base(reshaped_ms, base_matrix))

    c_init = torch.cat(
        [
            torch.zeros(reshaped_ms.shape[0], dtype=torch.int32, device=device),
            base_clusters.to(device).to(torch.int32),
        ],
        dim=0,
    )

    res = crp_gibbs(data=data, c_init=c_init, device=device, max_iters=max_iters)
    final_clusters, aligned_res = align_labels_by_hungarian(res=res, burn_in=burn_in)

    unique, counts = torch.unique(final_clusters, return_counts=True)
    print(f"{new_path.stem}_n_cluster: {len(unique)}")

    return reshaped_ms, data, final_clusters, aligned_res


def main():
    parser = argparse.ArgumentParser(description="DPMM-based clustering for EI GC-MS spectra.")
    parser.add_argument("--base-path", type=Path, required=True, help="Path to the base Excel file.")
    parser.add_argument("--new-paths", type=Path, nargs="*", default=[], help="Path(s) to new Excel file(s).")
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help='Torch device to use, e.g. "cuda" or "cpu".',
    )
    parser.add_argument("--base-iters", type=int, default=15000, help="Number of Gibbs iterations for base data.")
    parser.add_argument("--base-burn-in", type=int, default=5000, help="Burn-in for base clustering.")
    parser.add_argument("--new-iters", type=int, default=500, help="Number of Gibbs iterations for new data.")
    parser.add_argument("--new-burn-in", type=int, default=100, help="Burn-in for new clustering.")
    parser.add_argument("--skip-plot", action="store_true", help="Skip UMAP visualization.")
    args = parser.parse_args()

    base_matrix, base_data, base_clusters, base_aligned_res = run_base_analysis(
        base_path=args.base_path,
        device=args.device,
        max_iters=args.base_iters,
        burn_in=args.base_burn_in,
    )

    if not args.skip_plot:
        visualize_umap_clusters(data_tensor=base_data, cluster_tensor=base_clusters)

    new_results = []
    for new_path in args.new_paths:
        reshaped_ms, data, final_clusters, aligned_res = run_new_analysis(
            new_path=new_path,
            base_matrix=base_matrix,
            base_clusters=base_clusters,
            device=args.device,
            max_iters=args.new_iters,
            burn_in=args.new_burn_in,
        )
        new_results.append((reshaped_ms, data, final_clusters, aligned_res))

        if not args.skip_plot:
            visualize_umap_clusters(
                data_tensor=data,
                cluster_tensor=final_clusters,
                n_new=reshaped_ms.shape[0],
            )


if __name__ == "__main__":
    main()

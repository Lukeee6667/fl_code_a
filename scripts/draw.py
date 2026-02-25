import os
import re
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


def parse_list(text):
    if not text:
        return []
    items = [x.strip() for x in text.split(",") if x.strip()]
    return [float(x) for x in items]


def extract_lambda(lines, key):
    pattern = re.compile(rf"{key}\s*[:=]\s*([0-9.]+)")
    for line in lines:
        match = pattern.search(line)
        if match:
            return float(match.group(1))
    return None
 

def extract_num_corrupt(lines):
    pattern = re.compile(r"num_corrupt\s*[:=]\s*([0-9]+)")
    for line in lines:
        match = pattern.search(line)
        if match:
            return int(match.group(1))
    return 0


def main():
    log_path = Path("/root/autodl-tmp/fl_code_a/logs/cifar10/attack_badnet_ar_0.27/defense_origin_alignins_4metrics/2026-02-25-10-30_noniid(0.5)_pr(0.5)/2026-02-25-10-30_noniid(0.5)_pr(0.5).log")
    out_dir = Path("/root/autodl-tmp/fl_code_a/pic_7")
    if not log_path.is_file():
        raise FileNotFoundError(f"log not found: {log_path}")

    lines = log_path.read_text(errors="ignore").splitlines()

    lambda_s = extract_lambda(lines, "lambda_s")
    lambda_c = extract_lambda(lines, "lambda_c")
    lambda_g = extract_lambda(lines, "lambda_g")
    lambda_mean_cos = extract_lambda(lines, "lambda_mean_cos")
    num_corrupt = extract_num_corrupt(lines)

    if lambda_s is None:
        lambda_s = 1
    if lambda_c is None:
        lambda_c = 1
    if lambda_g is None:
        lambda_g = 1.5
    if lambda_mean_cos is None:
        lambda_mean_cos = 1.5

    round_data = {}

    patterns_basic = {
        "tda": re.compile(r"Round\s+(\d+)\s+TDA:\s+\[(.*)\]"),
        "mpsa": re.compile(r"Round\s+(\d+)\s+MPSA:\s+\[(.*)\]"),
        "mz_tda": re.compile(r"Round\s+(\d+)\s+MZ-score of TDA:\s+\[(.*)\]"),
        "mz_mpsa": re.compile(r"Round\s+(\d+)\s+MZ-score of MPSA:\s+\[(.*)\]"),
    }
    patterns_extra = {
        "grad_norm": re.compile(r"Round\s+(\d+)\s+Grad Norm:\s+\[(.*)\]"),
        "mean_cos": re.compile(r"Round\s+(\d+)\s+Mean Cos:\s+\[(.*)\]"),
        "mz_grad_norm": re.compile(r"Round\s+(\d+)\s+MZ-score of Grad Norm:\s+\[(.*)\]"),
        "mz_mean_cos": re.compile(r"Round\s+(\d+)\s+MZ-score of Mean Cos:\s+\[(.*)\]"),
    }

    for line in lines:
        for key, pattern in patterns_basic.items():
            m = pattern.search(line)
            if m:
                rnd = int(m.group(1))
                values = parse_list(m.group(2))
                round_data.setdefault(rnd, {})[key] = values
        for key, pattern in patterns_extra.items():
            m = pattern.search(line)
            if m:
                rnd = int(m.group(1))
                values = parse_list(m.group(2))
                round_data.setdefault(rnd, {})[key] = values

    out_dir.mkdir(parents=True, exist_ok=True)

    def plot_metric(ax, values, ben_idx, mal_idx, title, lambda_val, y_label):
        ax.scatter(ben_idx, values[ben_idx], s=18, color="blue", label="Benign")
        if len(mal_idx) > 0:
            ax.scatter(mal_idx, values[mal_idx], s=28, color="red", marker="x", label="Malicious")
        med = np.median(values)
        std = np.std(values)
        if std > 0:
            lower = med - lambda_val * std
            upper = med + lambda_val * std
            ax.axhline(lower, color="green", linestyle="-", linewidth=1.5)
            ax.axhline(upper, color="green", linestyle="-", linewidth=1.5)
        else:
            ax.axhline(med, color="green", linestyle="-", linewidth=1.5)
        ax.set_title(title)
        ax.set_xlabel("Client Index")
        ax.set_ylabel(y_label)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize=8)

    def plot_mz(ax, values, ben_idx, mal_idx, title, lambda_val, y_label):
        ax.scatter(ben_idx, values[ben_idx], s=18, color="blue", label="Benign")
        if len(mal_idx) > 0:
            ax.scatter(mal_idx, values[mal_idx], s=28, color="red", marker="x", label="Malicious")
        ax.axhline(lambda_val, color="green", linestyle="-", linewidth=1.5, label="Threshold")
        ax.set_title(title)
        ax.set_xlabel("Client Index")
        ax.set_ylabel(y_label)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize=8)

    rounds_basic = sorted(r for r, data in round_data.items() if all(k in data for k in patterns_basic))
    rounds_extra = sorted(r for r, data in round_data.items() if all(k in data for k in patterns_extra))

    for rnd in rounds_basic:
        data = round_data[rnd]
        fig, axes = plt.subplots(2, 2, figsize=(16, 10), constrained_layout=True)

        tda = np.array(data["tda"])
        mpsa = np.array(data["mpsa"])
        mz_tda = np.array(data["mz_tda"])
        mz_mpsa = np.array(data["mz_mpsa"])
        client_indices = np.arange(len(tda))
        mal_idx = client_indices[:num_corrupt]
        ben_idx = client_indices[num_corrupt:]

        plot_metric(axes[0, 0], tda, ben_idx, mal_idx, "TDA Distribution", lambda_c, "TDA Value")
        plot_metric(axes[0, 1], mpsa, ben_idx, mal_idx, "MPSA Distribution", lambda_s, "MPSA Value")
        plot_mz(axes[1, 0], mz_tda, ben_idx, mal_idx, "TDA MZ-score Distribution", lambda_c, "MZ-score Value")
        plot_mz(axes[1, 1], mz_mpsa, ben_idx, mal_idx, "MPSA MZ-score Distribution", lambda_s, "MZ-score Value")

        fig.suptitle(f"Round {rnd} - Client Metrics Analysis", fontsize=16)
        fig.savefig(out_dir / f"round_{rnd:03d}.png", dpi=150)
        plt.close(fig)

    for rnd in rounds_extra:
        data = round_data[rnd]
        fig, axes = plt.subplots(2, 2, figsize=(16, 10), constrained_layout=True)

        grad_norm = np.array(data["grad_norm"])
        mean_cos = np.array(data["mean_cos"])
        mz_grad_norm = np.array(data["mz_grad_norm"])
        mz_mean_cos = np.array(data["mz_mean_cos"])
        client_indices = np.arange(len(grad_norm))
        mal_idx = client_indices[:num_corrupt]
        ben_idx = client_indices[num_corrupt:]

        plot_metric(axes[0, 0], grad_norm, ben_idx, mal_idx, "Grad Norm Distribution", lambda_g, "Grad Norm Value")
        plot_metric(axes[0, 1], mean_cos, ben_idx, mal_idx, "Mean Cos Distribution", lambda_mean_cos, "Mean Cos Value")
        plot_mz(axes[1, 0], mz_grad_norm, ben_idx, mal_idx, "Grad Norm MZ-score Distribution", lambda_g, "MZ-score Value")
        plot_mz(axes[1, 1], mz_mean_cos, ben_idx, mal_idx, "Mean Cos MZ-score Distribution", lambda_mean_cos, "MZ-score Value")

        fig.suptitle(f"Round {rnd} - Additional Client Metrics Analysis", fontsize=16)
        fig.savefig(out_dir / f"round_{rnd:03d}_extra.png", dpi=150)
        plt.close(fig)

    print(f"Saved {len(rounds_basic)} round plots to {out_dir}")
    if rounds_extra:
        print(f"Saved {len(rounds_extra)} extra round plots to {out_dir}")


if __name__ == "__main__":
    main()

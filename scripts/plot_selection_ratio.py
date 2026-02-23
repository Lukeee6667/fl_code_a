import argparse
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_int_from_log(text, key):
    match = re.search(rf"{key}:\s*(\d+)", text)
    return int(match.group(1)) if match else None


def extract_selected_indices(text):
    round_pattern = re.compile(r"Round\s+(\d+)\s+TDA:")
    selected_pattern = re.compile(r"selected update index:\s*\[(.*?)\]")
    lines = text.splitlines()
    current_round = None
    selected = []
    for line in lines:
        round_match = round_pattern.search(line)
        if round_match:
            current_round = int(round_match.group(1))
        selected_match = selected_pattern.search(line)
        if selected_match and current_round is not None:
            raw = selected_match.group(1).strip()
            if raw:
                indices = [int(x.strip()) for x in raw.split(",") if x.strip()]
            else:
                indices = []
            selected.append((current_round, indices))
    return selected


def build_ratios(selected, num_corrupt, num_agents):
    rounds = []
    malicious_selected_ratio = []
    benign_selected_ratio = []
    total_malicious = max(num_corrupt, 0)
    total_benign = max(num_agents - num_corrupt, 0)
    for round_id, indices in selected:
        malicious_selected = sum(1 for x in indices if x < num_corrupt)
        benign_selected = sum(1 for x in indices if x >= num_corrupt)
        mal_ratio = malicious_selected / total_malicious if total_malicious > 0 else 0
        ben_ratio = benign_selected / total_benign if total_benign > 0 else 0
        rounds.append(round_id)
        malicious_selected_ratio.append(mal_ratio)
        benign_selected_ratio.append(ben_ratio)
    pairs = sorted(zip(rounds, malicious_selected_ratio, benign_selected_ratio))
    if not pairs:
        return [], [], []
    rounds, malicious_selected_ratio, benign_selected_ratio = zip(*pairs)
    return rounds, malicious_selected_ratio, benign_selected_ratio


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("log_path")
    parser.add_argument("--output", default=None)
    parser.add_argument("--title", default="Client Selection Ratio Over Rounds")
    args = parser.parse_args()

    log_path = Path(args.log_path)
    text = log_path.read_text(errors="ignore")
    num_corrupt = parse_int_from_log(text, "num_corrupt") or 0
    num_agents = parse_int_from_log(text, "num_agents") or 0
    selected = extract_selected_indices(text)
    rounds, malicious_ratio, benign_ratio = build_ratios(selected, num_corrupt, num_agents)
    if not rounds:
        raise SystemExit("No selected update index lines found in log")

    plt.figure(figsize=(8, 4))
    plt.plot(rounds, malicious_ratio, label="Malicious selected ratio")
    plt.plot(rounds, benign_ratio, label="Benign selected ratio")
    plt.ylim(0, 1)
    plt.xlabel("Round")
    plt.ylabel("Selected ratio")
    plt.title(args.title)
    plt.legend()
    plt.tight_layout()

    if args.output:
        out_path = Path(args.output)
    else:
        out_path = log_path.parent / "selection_ratio.png"
    plt.savefig(out_path)
    print(out_path)


if __name__ == "__main__":
    main()

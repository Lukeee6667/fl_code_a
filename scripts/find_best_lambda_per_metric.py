import argparse
import re
from pathlib import Path


def parse_int(text, key):
    match = re.search(rf"{key}\s*:\s*(\d+)", text)
    return int(match.group(1)) if match else None


def parse_round_metric(text, metric_name):
    pattern = re.compile(rf"Round\s+(\d+)\s+MZ-score of {re.escape(metric_name)}:\s*\[(.*?)\]")
    results = {}
    for match in pattern.finditer(text):
        round_id = int(match.group(1))
        raw = match.group(2).strip()
        if not raw:
            values = []
        else:
            values = [float(x.strip()) for x in raw.split(",") if x.strip()]
        results[round_id] = values
    return results


def quantile(sorted_vals, q):
    if not sorted_vals:
        return None
    if q <= 0:
        return sorted_vals[0]
    if q >= 1:
        return sorted_vals[-1]
    idx = q * (len(sorted_vals) - 1)
    lo = int(idx)
    hi = min(lo + 1, len(sorted_vals) - 1)
    frac = idx - lo
    return sorted_vals[lo] * (1 - frac) + sorted_vals[hi] * frac


def build_candidates(values, steps=50):
    sorted_vals = sorted(values)
    if not sorted_vals:
        return []
    qs = [i / steps for i in range(steps + 1)]
    candidates = [quantile(sorted_vals, q) for q in qs]
    max_val = sorted_vals[-1]
    candidates.append(max_val + 1e-6)
    return sorted(set(candidates))


def evaluate_round(values, num_corrupt, threshold):
    if not values:
        return None
    n = len(values)
    benign_selected = [i for i, v in enumerate(values) if v < threshold]
    benign_total = max(n - num_corrupt, 0)
    malicious_total = max(min(num_corrupt, n), 0)
    correct = sum(1 for i in benign_selected if i >= num_corrupt)
    wrong = sum(1 for i in benign_selected if i < num_corrupt)
    tpr = correct / benign_total if benign_total > 0 else 0.0
    fpr = wrong / malicious_total if malicious_total > 0 else 0.0
    return tpr, fpr, correct, wrong


def best_threshold(metric_map, num_corrupt, steps):
    rounds = sorted(metric_map.keys())
    all_vals = [v for r in rounds for v in metric_map[r]]
    candidates = build_candidates(all_vals, steps=steps)
    best = None
    best_score = None
    for threshold in candidates:
        tprs = []
        fprs = []
        correct_total = 0
        wrong_total = 0
        for r in rounds:
            result = evaluate_round(metric_map[r], num_corrupt, threshold)
            if result is None:
                continue
            tpr, fpr, correct, wrong = result
            tprs.append(tpr)
            fprs.append(fpr)
            correct_total += correct
            wrong_total += wrong
        if not tprs:
            continue
        avg_tpr = sum(tprs) / len(tprs)
        avg_fpr = sum(fprs) / len(fprs)
        score = avg_tpr - avg_fpr
        if best_score is None or score > best_score:
            best_score = score
            best = (threshold, avg_tpr, avg_fpr, correct_total, wrong_total, score)
    return best


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("log_path")
    parser.add_argument("--steps", type=int, default=50)
    args = parser.parse_args()

    text = Path(args.log_path).read_text(errors="ignore")
    num_corrupt = parse_int(text, "num_corrupt")
    if num_corrupt is None:
        raise SystemExit("num_corrupt not found in log")

    metrics = {
        "lambda_s (MPSA)": parse_round_metric(text, "MPSA"),
        "lambda_c (TDA)": parse_round_metric(text, "TDA"),
        "lambda_g (Grad Norm)": parse_round_metric(text, "Grad Norm"),
        "lambda_mean_cos (Mean Cos)": parse_round_metric(text, "Mean Cos"),
    }

    best_thresholds = {}
    for name, metric_map in metrics.items():
        if not metric_map:
            raise SystemExit(f"No data for {name}")
        best = best_threshold(metric_map, num_corrupt, steps=args.steps)
        if best is None:
            raise SystemExit(f"No valid threshold for {name}")
        threshold, avg_tpr, avg_fpr, correct_total, wrong_total, score = best
        best_thresholds[name] = threshold
        print(f"{name}: {threshold}")
        print(f"avg_tpr: {avg_tpr:.4f}")
        print(f"avg_fpr: {avg_fpr:.4f}")
        print(f"correct_total: {correct_total}")
        print(f"wrong_total: {wrong_total}")
        print(f"score: {score:.4f}")
        print("-" * 20)

    # Joint Evaluation (Voting 3/4)
    print("Joint Evaluation (Voting 3/4) with best thresholds:")
    rounds = sorted(metrics["lambda_s (MPSA)"].keys())
    total_correct = 0
    total_wrong = 0
    total_benign_rounds = 0
    total_malicious_rounds = 0
    
    for r in rounds:
        # Check if round exists in all metrics
        if not all(r in metrics[m] for m in metrics):
            continue
            
        votes = []
        # Collect votes from all 4 metrics
        # Order: MPSA, TDA, Grad Norm, Mean Cos (order doesn't matter for sum)
        for name in metrics:
            vals = metrics[name][r]
            thresh = best_thresholds[name]
            votes.append([1 if v < thresh else 0 for v in vals])
        
        # Sum votes per client
        if not votes: continue
        num_clients = len(votes[0])
        client_votes = [sum(votes[m][i] for m in range(len(votes))) for i in range(num_clients)]
        
        # Threshold 3
        selected = [i for i, v in enumerate(client_votes) if v >= 3]
        
        # Stats
        benign_total = max(num_clients - num_corrupt, 0)
        malicious_total = max(min(num_corrupt, num_clients), 0)
        
        correct = sum(1 for i in selected if i >= num_corrupt)
        wrong = sum(1 for i in selected if i < num_corrupt)
        
        total_correct += correct
        total_wrong += wrong
        total_benign_rounds += benign_total
        total_malicious_rounds += malicious_total
        
    avg_tpr = total_correct / total_benign_rounds if total_benign_rounds > 0 else 0
    avg_fpr = total_wrong / total_malicious_rounds if total_malicious_rounds > 0 else 0
    print(f"Joint TPR: {avg_tpr:.4f}")
    print(f"Joint FPR: {avg_fpr:.4f}")
    print(f"Total Correct Selected: {total_correct}")
    print(f"Total Wrong Selected: {total_wrong}")



if __name__ == "__main__":
    main()

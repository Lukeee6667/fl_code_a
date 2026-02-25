import argparse
import ast
import math
import re
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from statistics import mean, median
from typing import Dict, List, Optional, Tuple


@dataclass(frozen=True)
class WeightConfig:
    benign_weight: Optional[float]
    suspicious_weight: Optional[float]
    malicious_weight: Optional[float]


@dataclass(frozen=True)
class SelectionEvent:
    order: int
    round_id: Optional[int]
    benign: List[int]
    suspicious: List[int]
    malicious: List[int]


@dataclass(frozen=True)
class TestEvent:
    order: int
    round_id: Optional[int]
    clean_acc: Optional[float]
    asr: Optional[float]
    backdoor_acc: Optional[float]


_RE_TS = re.compile(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),")
_RE_NUM_CORRUPT = re.compile(r"\bnum_corrupt=(\d+)\b")
_RE_NUM_AGENTS = re.compile(r"\bnum_agents=(\d+)\b")
_RE_BUILD = re.compile(r"\bbuild client:(\d+)\s+mal:(\d+)\s+data_num:(\d+)")
_RE_NAMESPACE = re.compile(r"\bNamespace\(")
_RE_BW = re.compile(r"\bbenign_weight=([0-9]*\.?[0-9]+)\b")
_RE_SW = re.compile(r"\bsuspicious_weight=([0-9]*\.?[0-9]+)\b")
_RE_MW = re.compile(r"\bmalicious_weight=([0-9]*\.?[0-9]+)\b")
_RE_ROUND = re.compile(r"\bRound\s+(\d+)\b")
_RE_SEL_B = re.compile(r"selected benign idx:\s*(\[[^\]]*\])")
_RE_SEL_S = re.compile(r"selected suspicious idx:\s*(\[[^\]]*\])")
_RE_SEL_M = re.compile(r"selected malicious idx:\s*(\[[^\]]*\])")
_RE_TEST = re.compile(r"---------Test\s+(\d+)\s+------------")
_RE_CLEAN = re.compile(r"\bClean ACC:\s*([0-9]*\.?[0-9]+)\b")
_RE_ASR = re.compile(r"\bAttack Success Ratio:\s*([0-9]*\.?[0-9]+)\b")
_RE_BA = re.compile(r"\bBackdoor ACC:\s*([0-9]*\.?[0-9]+)\b")


def _parse_ts(line: str) -> Optional[datetime]:
    m = _RE_TS.search(line)
    if not m:
        return None
    return datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S")


def _parse_list(s: str) -> List[int]:
    v = ast.literal_eval(s)
    if isinstance(v, list):
        return [int(x) for x in v]
    if isinstance(v, tuple):
        return [int(x) for x in v]
    raise ValueError(f"Expected list/tuple, got {type(v)}")


def _safe_mean(xs: List[float]) -> Optional[float]:
    return mean(xs) if xs else None


def _safe_median(xs: List[float]) -> Optional[float]:
    return median(xs) if xs else None


def _pearson_corr(xs: List[float], ys: List[float]) -> Optional[float]:
    if len(xs) != len(ys) or len(xs) < 2:
        return None
    mx = sum(xs) / len(xs)
    my = sum(ys) / len(ys)
    vx = sum((x - mx) ** 2 for x in xs)
    vy = sum((y - my) ** 2 for y in ys)
    if vx == 0 or vy == 0:
        return None
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    return cov / math.sqrt(vx * vy)


def parse_log(log_path: Path):
    text = log_path.read_text(errors="ignore")
    num_corrupt = int(_RE_NUM_CORRUPT.search(text).group(1)) if _RE_NUM_CORRUPT.search(text) else None
    num_agents = int(_RE_NUM_AGENTS.search(text).group(1)) if _RE_NUM_AGENTS.search(text) else None

    sizes: Dict[int, int] = {}
    namespaces: List[Tuple[Optional[datetime], WeightConfig]] = []
    selection_events: List[SelectionEvent] = []
    test_events: List[TestEvent] = []

    current_round: Optional[int] = None
    cur_sel_round: Optional[int] = None
    cur_sel_b: Optional[List[int]] = None
    cur_sel_s: Optional[List[int]] = None
    order = 0

    cur_test_round: Optional[int] = None
    cur_test: Optional[dict] = None

    for line in text.splitlines():
        order += 1

        mb = _RE_BUILD.search(line)
        if mb:
            sizes[int(mb.group(1))] = int(mb.group(3))

        if _RE_NAMESPACE.search(line):
            ts = _parse_ts(line)
            bw = float(_RE_BW.search(line).group(1)) if _RE_BW.search(line) else None
            sw = float(_RE_SW.search(line).group(1)) if _RE_SW.search(line) else None
            mw = float(_RE_MW.search(line).group(1)) if _RE_MW.search(line) else None
            namespaces.append((ts, WeightConfig(bw, sw, mw)))

        mr = _RE_ROUND.search(line)
        if mr:
            current_round = int(mr.group(1))

        msb = _RE_SEL_B.search(line)
        if msb:
            cur_sel_round = current_round
            cur_sel_b = _parse_list(msb.group(1))
            cur_sel_s = None
            continue

        mss = _RE_SEL_S.search(line)
        if mss and cur_sel_b is not None:
            cur_sel_s = _parse_list(mss.group(1))
            continue

        msm = _RE_SEL_M.search(line)
        if msm and cur_sel_b is not None and cur_sel_s is not None:
            cur_sel_m = _parse_list(msm.group(1))
            selection_events.append(
                SelectionEvent(
                    order=order,
                    round_id=cur_sel_round,
                    benign=cur_sel_b,
                    suspicious=cur_sel_s,
                    malicious=cur_sel_m,
                )
            )
            cur_sel_round = None
            cur_sel_b = None
            cur_sel_s = None
            continue

        mt = _RE_TEST.search(line)
        if mt:
            cur_test_round = int(mt.group(1))
            cur_test = {"clean": None, "asr": None, "ba": None, "order": order, "round": cur_test_round}
            continue

        if cur_test is not None:
            mc = _RE_CLEAN.search(line)
            if mc:
                cur_test["clean"] = float(mc.group(1))
            ma = _RE_ASR.search(line)
            if ma:
                cur_test["asr"] = float(ma.group(1))
            mba = _RE_BA.search(line)
            if mba:
                cur_test["ba"] = float(mba.group(1))

            if cur_test["clean"] is not None or cur_test["asr"] is not None or cur_test["ba"] is not None:
                test_events.append(
                    TestEvent(
                        order=cur_test["order"],
                        round_id=cur_test["round"],
                        clean_acc=cur_test["clean"],
                        asr=cur_test["asr"],
                        backdoor_acc=cur_test["ba"],
                    )
                )
                cur_test = None
                cur_test_round = None

    if num_agents is not None and not sizes:
        sizes = {i: 1 for i in range(num_agents)}

    return {
        "num_corrupt": num_corrupt,
        "num_agents": num_agents,
        "sizes": sizes,
        "namespaces": namespaces,
        "selection_events": selection_events,
        "test_events": test_events,
    }


def split_runs_by_round_order(selection_events: List[SelectionEvent], test_events: List[TestEvent]):
    sel_by_round: Dict[int, List[SelectionEvent]] = defaultdict(list)
    for ev in selection_events:
        if ev.round_id is not None:
            sel_by_round[ev.round_id].append(ev)
    for r in list(sel_by_round.keys()):
        sel_by_round[r].sort(key=lambda x: x.order)

    test_by_round: Dict[int, List[TestEvent]] = defaultdict(list)
    for ev in test_events:
        if ev.round_id is not None:
            test_by_round[ev.round_id].append(ev)
    for r in list(test_by_round.keys()):
        test_by_round[r].sort(key=lambda x: x.order)

    rounds = sorted(set(sel_by_round.keys()) & set(test_by_round.keys()))
    run0: List[Tuple[SelectionEvent, TestEvent]] = []
    run1: List[Tuple[SelectionEvent, TestEvent]] = []
    for r in rounds:
        if len(sel_by_round[r]) < 2 or len(test_by_round[r]) < 2:
            continue
        run0.append((sel_by_round[r][0], test_by_round[r][0]))
        run1.append((sel_by_round[r][1], test_by_round[r][1]))
    return run0, run1


def summarize_selection(selection_events: List[SelectionEvent], num_corrupt: int):
    def corrupt_to_frac(sel: List[int]):
        return sum(1 for i in sel if i < num_corrupt)

    corr_to_b = [corrupt_to_frac(ev.benign) / num_corrupt for ev in selection_events]
    corr_to_s = [corrupt_to_frac(ev.suspicious) / num_corrupt for ev in selection_events]
    corr_to_m = [corrupt_to_frac(ev.malicious) / num_corrupt for ev in selection_events]

    def purity(sel: List[int]):
        return (sum(1 for i in sel if i < num_corrupt) / len(sel)) if sel else 0.0

    pur_b = [purity(ev.benign) for ev in selection_events]
    pur_s = [purity(ev.suspicious) for ev in selection_events]
    pur_m = [purity(ev.malicious) for ev in selection_events]

    size_b = [len(ev.benign) for ev in selection_events]
    size_s = [len(ev.suspicious) for ev in selection_events]
    size_m = [len(ev.malicious) for ev in selection_events]

    true_m_in_m = [sum(1 for i in ev.malicious if i < num_corrupt) for ev in selection_events]

    return {
        "events": len(selection_events),
        "corrupt_flow_mean": {"benign": mean(corr_to_b), "suspicious": mean(corr_to_s), "malicious": mean(corr_to_m)},
        "corrupt_flow_median": {"benign": median(corr_to_b), "suspicious": median(corr_to_s), "malicious": median(corr_to_m)},
        "purity_mean": {"benign": mean(pur_b), "suspicious": mean(pur_s), "malicious": mean(pur_m)},
        "purity_median": {"benign": median(pur_b), "suspicious": median(pur_s), "malicious": median(pur_m)},
        "size_mean": {"benign": mean(size_b), "suspicious": mean(size_s), "malicious": mean(size_m)},
        "size_median": {"benign": median(size_b), "suspicious": median(size_s), "malicious": median(size_m)},
        "pct_rounds_malicious_has_zero_true_mal": sum(1 for x in true_m_in_m if x == 0) / len(true_m_in_m) if true_m_in_m else None,
    }


def weighted_malicious_fraction(ev: SelectionEvent, sizes: Dict[int, int], num_corrupt: int, bw: float, sw: float, mw: float):
    total = 0.0
    mal = 0.0

    def add(sel: List[int], w: float):
        nonlocal total, mal
        for idx in sel:
            mass = w * float(sizes.get(idx, 1))
            total += mass
            if idx < num_corrupt:
                mal += mass

    add(ev.benign, bw)
    add(ev.suspicious, sw)
    add(ev.malicious, mw)
    return mal / total if total > 0 else 0.0


def summarize_run(pairs: List[Tuple[SelectionEvent, TestEvent]], sizes: Dict[int, int], num_corrupt: int, weights: Optional[Tuple[float, float, float]]):
    sels = [p[0] for p in pairs]
    tests = [p[1] for p in pairs]
    sel_summary = summarize_selection(sels, num_corrupt)

    cleans = [t.clean_acc for t in tests if t.clean_acc is not None]
    asrs = [t.asr for t in tests if t.asr is not None]
    bas = [t.backdoor_acc for t in tests if t.backdoor_acc is not None]

    out = {
        "rounds": len(pairs),
        "clean_acc_mean": _safe_mean(cleans),
        "clean_acc_median": _safe_median(cleans),
        "asr_mean": _safe_mean(asrs),
        "asr_median": _safe_median(asrs),
        "backdoor_acc_mean": _safe_mean(bas),
        "backdoor_acc_median": _safe_median(bas),
        "selection": sel_summary,
    }

    if weights is not None:
        bw, sw, mw = weights
        mfs = [weighted_malicious_fraction(s, sizes, num_corrupt, bw, sw, mw) for s in sels]
        out["mal_frac_mean"] = _safe_mean(mfs)
        out["mal_frac_median"] = _safe_median(mfs)
        if asrs and len(asrs) == len(mfs):
            out["corr_mal_frac_asr"] = _pearson_corr(mfs, asrs)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("log_path")
    parser.add_argument("--simulate-weights", nargs=3, type=float, default=None, metavar=("BW", "SW", "MW"))
    args = parser.parse_args()

    parsed = parse_log(Path(args.log_path))
    num_corrupt = parsed["num_corrupt"]
    num_agents = parsed["num_agents"]
    sizes = parsed["sizes"]
    namespaces = parsed["namespaces"]
    selection_events = parsed["selection_events"]
    test_events = parsed["test_events"]

    if num_corrupt is None or num_agents is None:
        raise SystemExit("Could not parse num_corrupt/num_agents from log")
    if not selection_events:
        raise SystemExit("No selection events parsed; ensure the log contains 'selected benign idx' lines")
    if not test_events:
        raise SystemExit("No test events parsed; ensure the log contains '---------Test <round>------------' lines")

    run0, run1 = split_runs_by_round_order(selection_events, test_events)

    ns_sorted = sorted((t, w) for t, w in namespaces if w.benign_weight is not None or w.suspicious_weight is not None or w.malicious_weight is not None)
    run_weights: List[Optional[Tuple[float, float, float]]] = [None, None]
    if len(ns_sorted) >= 2:
        w0 = ns_sorted[0][1]
        w1 = ns_sorted[1][1]
        run_weights[0] = (float(w0.benign_weight), float(w0.suspicious_weight), float(w0.malicious_weight)) if w0.benign_weight is not None and w0.suspicious_weight is not None and w0.malicious_weight is not None else None
        run_weights[1] = (float(w1.benign_weight), float(w1.suspicious_weight), float(w1.malicious_weight)) if w1.benign_weight is not None and w1.suspicious_weight is not None and w1.malicious_weight is not None else None

    overall_sel = summarize_selection(selection_events, num_corrupt)
    print(f"num_agents={num_agents} num_corrupt={num_corrupt}")
    print(f"selection_events={len(selection_events)} test_events={len(test_events)}")
    print("overall:")
    print(f"  corrupt->benign mean/median {overall_sel['corrupt_flow_mean']['benign']:.3f}/{overall_sel['corrupt_flow_median']['benign']:.3f}")
    print(f"  corrupt->suspicious mean/median {overall_sel['corrupt_flow_mean']['suspicious']:.3f}/{overall_sel['corrupt_flow_median']['suspicious']:.3f}")
    print(f"  corrupt->malicious mean/median {overall_sel['corrupt_flow_mean']['malicious']:.3f}/{overall_sel['corrupt_flow_median']['malicious']:.3f}")
    print(
        "  corrupt count (out of %d) mean/median: benign %.2f/%.2f, suspicious %.2f/%.2f, malicious %.2f/%.2f"
        % (
            num_corrupt,
            overall_sel["corrupt_flow_mean"]["benign"] * num_corrupt,
            overall_sel["corrupt_flow_median"]["benign"] * num_corrupt,
            overall_sel["corrupt_flow_mean"]["suspicious"] * num_corrupt,
            overall_sel["corrupt_flow_median"]["suspicious"] * num_corrupt,
            overall_sel["corrupt_flow_mean"]["malicious"] * num_corrupt,
            overall_sel["corrupt_flow_median"]["malicious"] * num_corrupt,
        )
    )
    print(
        "  cluster size mean/median: benign %.2f/%.2f, suspicious %.2f/%.2f, malicious %.2f/%.2f"
        % (
            overall_sel["size_mean"]["benign"],
            overall_sel["size_median"]["benign"],
            overall_sel["size_mean"]["suspicious"],
            overall_sel["size_median"]["suspicious"],
            overall_sel["size_mean"]["malicious"],
            overall_sel["size_median"]["malicious"],
        )
    )
    print(
        "  purity mean/median: suspicious %.3f/%.3f, malicious %.3f/%.3f"
        % (
            overall_sel["purity_mean"]["suspicious"],
            overall_sel["purity_median"]["suspicious"],
            overall_sel["purity_mean"]["malicious"],
            overall_sel["purity_median"]["malicious"],
        )
    )
    print(f"  pct rounds malicious has 0 true-mal {overall_sel['pct_rounds_malicious_has_zero_true_mal']:.3f}")

    for idx, pairs in enumerate([run0, run1]):
        if not pairs:
            continue
        w = run_weights[idx]
        s = summarize_run(pairs, sizes, num_corrupt, w)
        print(f"run{idx}: rounds={s['rounds']}")
        if w is not None:
            print(f"  weights(b/s/m)={w[0]}/{w[1]}/{w[2]}")
        if s["clean_acc_mean"] is not None:
            print(f"  clean_acc mean/median {s['clean_acc_mean']:.4f}/{s['clean_acc_median']:.4f}")
        if s["asr_mean"] is not None:
            print(f"  asr mean/median {s['asr_mean']:.4f}/{s['asr_median']:.4f}")
        if s["backdoor_acc_mean"] is not None:
            print(f"  backdoor_acc mean/median {s['backdoor_acc_mean']:.4f}/{s['backdoor_acc_median']:.4f}")
        print(
            "  corrupt count (out of %d) mean/median: benign %.2f/%.2f, suspicious %.2f/%.2f, malicious %.2f/%.2f"
            % (
                num_corrupt,
                s["selection"]["corrupt_flow_mean"]["benign"] * num_corrupt,
                s["selection"]["corrupt_flow_median"]["benign"] * num_corrupt,
                s["selection"]["corrupt_flow_mean"]["suspicious"] * num_corrupt,
                s["selection"]["corrupt_flow_median"]["suspicious"] * num_corrupt,
                s["selection"]["corrupt_flow_mean"]["malicious"] * num_corrupt,
                s["selection"]["corrupt_flow_median"]["malicious"] * num_corrupt,
            )
        )
        print(
            "  cluster size mean/median: benign %.2f/%.2f, suspicious %.2f/%.2f, malicious %.2f/%.2f"
            % (
                s["selection"]["size_mean"]["benign"],
                s["selection"]["size_median"]["benign"],
                s["selection"]["size_mean"]["suspicious"],
                s["selection"]["size_median"]["suspicious"],
                s["selection"]["size_mean"]["malicious"],
                s["selection"]["size_median"]["malicious"],
            )
        )
        print(
            "  purity mean/median: suspicious %.3f/%.3f, malicious %.3f/%.3f"
            % (
                s["selection"]["purity_mean"]["suspicious"],
                s["selection"]["purity_median"]["suspicious"],
                s["selection"]["purity_mean"]["malicious"],
                s["selection"]["purity_median"]["malicious"],
            )
        )
        print(f"  pct rounds malicious has 0 true-mal {s['selection']['pct_rounds_malicious_has_zero_true_mal']:.3f}")
        if w is not None and s.get("mal_frac_mean") is not None:
            extra = f" corr(mal_frac,asr)={s['corr_mal_frac_asr']:.3f}" if s.get("corr_mal_frac_asr") is not None else ""
            print(f"  mal_frac mean/median {s['mal_frac_mean']:.4f}/{s['mal_frac_median']:.4f}{extra}")

    if args.simulate_weights is not None:
        bw, sw, mw = args.simulate_weights
        mfs = [weighted_malicious_fraction(ev, sizes, num_corrupt, bw, sw, mw) for ev in selection_events]
        print("simulate:")
        print(f"  weights(b/s/m)={bw}/{sw}/{mw}")
        print(f"  mal_frac mean/median {mean(mfs):.4f}/{median(mfs):.4f}")


if __name__ == "__main__":
    main()

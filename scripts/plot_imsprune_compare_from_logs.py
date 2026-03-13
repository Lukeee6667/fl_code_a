import argparse
import re
from pathlib import Path
from datetime import datetime

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm


_RE_POST_CLEAN = re.compile(r"Post-Prune\s+Clean\s+ACC:\s*([0-9.]+)")
_RE_POST_ASR = re.compile(r"Post-Prune\s+(Attack\s+Success\s+Ratio|ASR):\s*([0-9.]+)")
_RE_POST_BA = re.compile(r"Post-Prune\s+Backdoor\s+ACC:\s*([0-9.]+)")

_RE_FT_CLEAN = re.compile(r"FT\s+Epoch\s+(\d+)\s+Clean\s+ACC:\s*([0-9.]+)")
_RE_FT_ASR = re.compile(r"FT\s+Epoch\s+(\d+)\s+(Attack\s+Success\s+Ratio|ASR):\s*([0-9.]+)")
_RE_FT_BA = re.compile(r"FT\s+Epoch\s+(\d+)\s+Backdoor\s+ACC:\s*([0-9.]+)")


def _set_chinese_font():
    preferred_files = [
        "NotoSansCJK-Regular.ttc",
        "NotoSansCJK-Bold.ttc",
        "DroidSansFallbackFull.ttf",
        "NotoSerifCJK-Regular.ttc",
        "NotoSerifCJK-Bold.ttc",
    ]
    sys_fonts = []
    try:
        sys_fonts.extend(fm.findSystemFonts(fontext="ttf"))
    except Exception:
        pass
    try:
        sys_fonts.extend(fm.findSystemFonts(fontext="ttc"))
    except Exception:
        pass

    name_to_path = {}
    for path in sys_fonts:
        p = Path(path)
        name_to_path.setdefault(p.name, str(p))

    for file_name in preferred_files:
        path = name_to_path.get(file_name)
        if not path:
            continue
        try:
            fm.fontManager.addfont(path)
            name = fm.FontProperties(fname=path).get_name()
        except Exception:
            continue
        plt.rcParams["font.sans-serif"] = [name, "DejaVu Sans"]
        plt.rcParams["axes.unicode_minus"] = False
        return

    candidates = [
        "Noto Sans CJK SC",
        "Noto Sans CJK JP",
        "Noto Sans CJK",
        "Noto Sans SC",
        "Source Han Sans SC",
        "Source Han Sans CN",
        "Droid Sans Fallback",
        "SimHei",
        "Microsoft YaHei",
        "WenQuanYi Zen Hei",
        "PingFang SC",
    ]
    for name in candidates:
        try:
            fm.findfont(fm.FontProperties(family=name), fallback_to_default=False)
        except Exception:
            continue
        plt.rcParams["font.sans-serif"] = [name, "DejaVu Sans"]
        plt.rcParams["axes.unicode_minus"] = False
        return


def _parse_log(path: Path):
    text = path.read_text(errors="ignore").splitlines()

    post_clean = None
    post_asr = None
    post_ba = None
    ft = {}
    max_epoch = 0

    for line in text:
        m = _RE_POST_CLEAN.search(line)
        if m:
            post_clean = float(m.group(1))
        m = _RE_POST_ASR.search(line)
        if m:
            post_asr = float(m.group(2))
        m = _RE_POST_BA.search(line)
        if m:
            post_ba = float(m.group(1))

        m = _RE_FT_CLEAN.search(line)
        if m:
            ep = int(m.group(1))
            ft.setdefault(ep, {})["clean"] = float(m.group(2))
            max_epoch = max(max_epoch, ep)
        m = _RE_FT_ASR.search(line)
        if m:
            ep = int(m.group(1))
            ft.setdefault(ep, {})["asr"] = float(m.group(3))
            max_epoch = max(max_epoch, ep)
        m = _RE_FT_BA.search(line)
        if m:
            ep = int(m.group(1))
            ft.setdefault(ep, {})["ba"] = float(m.group(2))
            max_epoch = max(max_epoch, ep)

    if post_clean is None and not ft:
        raise ValueError(f"未在日志中找到 Post-Prune 或 FT Epoch 指标: {path}")

    stages = np.arange(0, max_epoch + 1, dtype=int)
    clean = np.full_like(stages, np.nan, dtype=float)
    asr = np.full_like(stages, np.nan, dtype=float)
    ba = np.full_like(stages, np.nan, dtype=float)

    if post_clean is not None:
        clean[0] = post_clean
    if post_asr is not None:
        asr[0] = post_asr
    if post_ba is not None:
        ba[0] = post_ba

    for ep in range(1, max_epoch + 1):
        d = ft.get(ep, {})
        if "clean" in d:
            clean[ep] = d["clean"]
        if "asr" in d:
            asr[ep] = d["asr"]
        if "ba" in d:
            ba[ep] = d["ba"]

    return stages, {"clean": clean, "asr": asr, "ba": ba}


def _plot_attack(ax, title, constrained, plain, stages_label):
    metrics = [
        ("CleanACC", "clean", "#1f77b4", "o"),
        ("ASR", "asr", "#d62728", "s"),
        ("BackdoorACC", "ba", "#2ca02c", "^"),
    ]

    for display, key, color, marker in metrics:
        # 受限微调：实线 + 实心 marker
        ax.plot(
            constrained[0],
            constrained[1][key],
            color=color,
            linewidth=2.4,
            linestyle='-',
            marker=marker,
            markersize=5.5,
            markerfacecolor=color,
            markeredgecolor=color,
            label=f"{display}（受限微调）",
        )

        # 普通微调：短虚线 + 空心 marker
        ax.plot(
            plain[0],
            plain[1][key],
            color=color,
            linewidth=2.2,
            linestyle=(0, (3, 2)),   # 比 '--' 更适合图例显示
            dash_capstyle='butt',
            marker=marker,
            markersize=5.5,
            markerfacecolor='white',
            markeredgecolor=color,
            markeredgewidth=1.6,
            label=f"{display}（普通微调）",
        )

    ax.set_title(title)
    ax.set_xlabel(stages_label)
    ax.set_ylabel("准确率")
    ax.set_ylim(0.0, 1.0)
    ax.set_xticks(constrained[0])
    ax.grid(True, alpha=0.25)


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--badnet_constrained",
        type=Path,
        default=Path(
            "/home/dell/fl_code/fl_code_260114/fl_code_a/logs/cifar10/attack_badnet_ar_0.00/defense_origin_alignins_clustering_prune_finetune/2026-03-02-12-40_iid_pr(0.3)_imsprune_constrained_ft/2026-03-02-12-40_iid_pr(0.3)_imsprune_constrained_ft.log"
        ),
    )
    p.add_argument(
        "--badnet_plain",
        type=Path,
        default=Path(
            "/home/dell/fl_code/fl_code_260114/fl_code_a/logs/cifar10/attack_badnet_ar_0.00/defense_origin_alignins_clustering_prune_finetune/2026-03-02-12-44_iid_pr(0.3)_imsprune_plain_ft/2026-03-02-12-44_iid_pr(0.3)_imsprune_plain_ft.log"
        ),
    )
    p.add_argument(
        "--dba_constrained",
        type=Path,
        default=Path(
            "/home/dell/fl_code/fl_code_260114/fl_code_a/logs/cifar10/attack_DBA_ar_0.00/defense_origin_alignins_clustering_prune_finetune/2026-03-02-12-47_iid_pr(0.3)_imsprune_constrained_ft/2026-03-02-12-47_iid_pr(0.3)_imsprune_constrained_ft.log"
        ),
    )
    p.add_argument(
        "--dba_plain",
        type=Path,
        default=Path(
            "/home/dell/fl_code/fl_code_260114/fl_code_a/logs/cifar10/attack_DBA_ar_0.00/defense_origin_alignins_clustering_prune_finetune/2026-03-02-12-50_iid_pr(0.3)_imsprune_plain_ft/2026-03-02-12-50_iid_pr(0.3)_imsprune_plain_ft.log"
        ),
    )
    p.add_argument(
        "--neurotoxin_constrained",
        type=Path,
        default=Path(
            "/home/dell/fl_code/fl_code_260114/fl_code_a/logs/cifar10/attack_neurotoxin_ar_0.00/defense_origin_alignins_clustering_prune_finetune/2026-03-02-12-53_iid_pr(0.3)_imsprune_constrained_ft/2026-03-02-12-53_iid_pr(0.3)_imsprune_constrained_ft.log"
        ),
    )
    p.add_argument(
        "--neurotoxin_plain",
        type=Path,
        default=Path(
            "/home/dell/fl_code/fl_code_260114/fl_code_a/logs/cifar10/attack_neurotoxin_ar_0.00/defense_origin_alignins_clustering_prune_finetune/2026-03-02-12-57_iid_pr(0.3)_imsprune_plain_ft/2026-03-02-12-57_iid_pr(0.3)_imsprune_plain_ft.log"
        ),
    )
    p.add_argument(
        "--pgd_constrained",
        type=Path,
        default=Path(
            "/home/dell/fl_code/fl_code_260114/fl_code_a/logs/cifar10/attack_pgd_ar_0.00/defense_origin_alignins_clustering_prune_finetune/2026-03-02-13-00_iid_pr(0.3)_imsprune_constrained_ft/2026-03-02-13-00_iid_pr(0.3)_imsprune_constrained_ft.log"
        ),
    )
    p.add_argument(
        "--pgd_plain",
        type=Path,
        default=Path(
            "/home/dell/fl_code/fl_code_260114/fl_code_a/logs/cifar10/attack_pgd_ar_0.00/defense_origin_alignins_clustering_prune_finetune/2026-03-02-13-03_iid_pr(0.3)_imsprune_plain_ft/2026-03-02-13-03_iid_pr(0.3)_imsprune_plain_ft.log"
        ),
    )
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()

    attack_pairs = [
        ("BadNet", args.badnet_constrained, args.badnet_plain),
        ("DBA", args.dba_constrained, args.dba_plain),
        ("Neurotoxin", args.neurotoxin_constrained, args.neurotoxin_plain),
        ("PGD", args.pgd_constrained, args.pgd_plain),
    ]

    parsed = []
    for name, c_path, p_path in attack_pairs:
        if not c_path.is_file():
            raise FileNotFoundError(c_path)
        if not p_path.is_file():
            raise FileNotFoundError(p_path)
        parsed.append((name, _parse_log(c_path), _parse_log(p_path)))

    try:
        plt.style.use("seaborn-v0_8-whitegrid")
    except Exception:
        plt.style.use("seaborn-whitegrid")

    _set_chinese_font()

    fig, axes = plt.subplots(2, 2, figsize=(12.8, 7.6), sharex=False, sharey=False)
    axes = axes.flatten()

    x_label = "阶段（0=剪枝后，1..E=微调Epoch）"
    for ax, (name, constrained, plain) in zip(axes, parsed):
        _plot_attack(ax, name, constrained, plain, x_label)

    handles, labels = axes[0].get_legend_handles_labels()
    if len(handles) == 6:
        handles = [handles[0], handles[2], handles[4], handles[1], handles[3], handles[5]]
        labels = [labels[0], labels[2], labels[4], labels[1], labels[3], labels[5]]
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=2,
        frameon=True,
        bbox_to_anchor=(0.5, -0.04),
        handlelength=4.2,
        columnspacing=1.6,
        handletextpad=0.6,
        markerscale=1.2,
    )
    fig.tight_layout(rect=(0, 0.12, 1, 1))

    if args.out is None:
        out = Path.cwd() / f"imsprune_compare_from_logs_fulllegend_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    else:
        out = args.out

    fig.savefig(out, dpi=300, bbox_inches="tight")
    print(out.as_posix())


if __name__ == "__main__":
    main()

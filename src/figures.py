from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
FIG = ROOT / "paper" / "figures"
FIG.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.size": 9, "axes.labelsize": 9, "axes.titlesize": 9,
    "legend.fontsize": 8, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 150, "savefig.bbox": "tight",
})
NAVY, RED, GREY = "#1f3b57", "#b2182b", "#888888"


def fig_flip_level():
    f = PROC / "flip_level.json"
    if not f.exists():
        print("  skip flip_level (no data)"); return
    rows = json.loads(f.read_text())
    dates = sorted({r["date"] for r in rows})
    fig, axes = plt.subplots(1, len(dates), figsize=(3.2 * len(dates), 2.8),
                             sharey=False)
    if len(dates) == 1:
        axes = [axes]
    for ax, d in zip(axes, dates):
        sub = [r for r in rows if r["date"] == d and r["wing"] == "C"]
        if not sub:
            continue
        sub.sort(key=lambda r: r["spot"])
        x = np.array([r["spot"] for r in sub])
        lo = np.array([r["lo"] for r in sub]) / 1e6
        hi = np.array([r["hi"] for r in sub]) / 1e6
        ax.fill_between(x, lo, hi, color=NAVY, alpha=0.22,
                        label="identified set")
        ax.plot(x, lo, color=NAVY, lw=1.0)
        ax.plot(x, hi, color=NAVY, lw=1.0)
        ax.axhline(0, color=RED, lw=1.2, ls="--", label="zero")
        ax.set_title(d)
        ax.set_xlabel("hypothetical spot")
        ax.set_ylabel("Pro gamma exposure (mn)")
        ax.ticklabel_format(axis="x", style="plain")
    axes[0].legend(frameon=False, loc="upper left")
    fig.suptitle("The identified set for dealer gamma straddles zero at every "
                 "spot level", y=1.04, fontsize=10)
    fig.savefig(FIG / "flip_level.pdf")
    plt.close(fig)
    print("  wrote flip_level.pdf")


def fig_disclosure():
    f = PROC / "disclosure.json"
    if not f.exists():
        print("  skip disclosure (no data)"); return
    rows = json.loads(f.read_text())
    keys = sorted({(r["date"], r["wing"]) for r in rows})[:4]
    fig, ax = plt.subplots(figsize=(5.2, 3.2))
    for (d, w) in keys:
        for rule, style in (("oi", "-"), ("gamma", "--")):
            sub = [r for r in rows if r["date"] == d and r["wing"] == w
                   and r.get("rule") == rule]
            if not sub:
                continue
            sub.sort(key=lambda r: r["k"])
            share = np.array([r["oi_share"] for r in sub]) * 100
            w0 = sub[0]["width"] or 1.0
            rel = np.array([r["width"] / w0 for r in sub])
            ax.plot(share, rel, style, lw=1.1, alpha=0.85,
                    label=f"{d} {w} ({rule})" if len(keys) <= 2 else None)
    ax.set_xlabel("% of open interest disclosed at participant-cell level")
    ax.set_ylabel("identified width / width at zero disclosure")
    ax.set_ylim(0, 1.05)
    ax.axvline(95, color=RED, lw=1.0, ls=":")
    ax.text(95.5, 0.75, "95% of OI disclosed:\nsign still undetermined",
            color=RED, fontsize=7.5, va="center")
    ax.set_title("Partial disclosure shrinks the set but does not identify the sign",
                 fontsize=9)
    fig.savefig(FIG / "disclosure.pdf")
    plt.close(fig)
    print("  wrote disclosure.pdf")


def fig_panel_widths():
    f = PROC / "gamma_panel.json"
    if not f.exists():
        print("  skip panel (no data)"); return
    rows = [r for r in json.loads(f.read_text())
            if r["participant"] in ("Client", "FII", "Pro")]
    if not rows:
        return
    fig, ax = plt.subplots(figsize=(5.2, 3.0))
    parts = ["Client", "FII", "Pro"]
    for j, p in enumerate(parts):
        sub = [r for r in rows if r["participant"] == p]
        lo = np.array([r["lo1"] for r in sub]) / 1e6
        hi = np.array([r["hi1"] for r in sub]) / 1e6
        x = np.arange(len(sub)) + j * 0.0
        ax.vlines(x + j * 0.25, lo, hi, color=[NAVY, GREY, RED][j],
                  alpha=0.55, lw=1.0, label=p)
    ax.axhline(0, color="black", lw=1.2)
    ax.set_xlabel("participant-wing-day (ordered)")
    ax.set_ylabel("gamma exposure (mn)")
    ax.set_title("Every identified interval contains zero (96 of 96)", fontsize=9)
    ax.legend(frameon=False, ncol=3)
    fig.savefig(FIG / "panel_widths.pdf")
    plt.close(fig)
    print("  wrote panel_widths.pdf")


if __name__ == "__main__":
    print("generating figures ...")
    fig_flip_level()
    fig_disclosure()
    fig_panel_widths()
    print(f"figures in {FIG.relative_to(ROOT)}")

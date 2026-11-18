# Figuras del articulo (300 dpi)
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
from .model import phase_model, tones

plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                     "grid.color": "#e6e6e6", "grid.linewidth": 0.6, "axes.edgecolor": "#888", "savefig.dpi": 300})
RAMP = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281", "#0d366b"]
BLUE, ORANGE, AQUA, GREY, INK = "#2a78d6", "#eb6834", "#1baf7a", "#b7b6ad", "#333"


def fig1(path, f, temps, P, p, loto):
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 3.1), gridspec_kw={"width_ratios": [2.2, 1]})
    F = f / 1e6
    M = np.rad2deg(phase_model(p, temps, f))
    for k, t in enumerate(temps):
        c = np.polyfit(F, P[k], 1)
        r = P[k] - np.polyval(c, F)
        rm = M[k] - np.polyval(c, F)
        rm = rm - np.mean(rm - r)
        ax[0].plot(F, r - 12 * k, "o", ms=1.6, color=RAMP[k], label=f"{int(t)} °C")
        ax[0].plot(F, rm - 12 * k, "-", lw=1.0, color=INK, alpha=0.8)
    ax[0].set_xlabel("Frequency (MHz)")
    ax[0].set_ylabel("Phase − linear trend (°)\n(curves offset by 12°)")
    ax[0].legend(ncol=6, fontsize=6.5, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.22),
                 markerscale=3, handletextpad=0.2, columnspacing=0.8)
    ax[0].set_title("(a) Measured ripple (dots) vs delay-line + echo model (lines)", fontsize=8.5, loc="left")
    x = np.arange(len(loto))
    ax[1].bar(x - 0.19, loto[:, 1], 0.38, color=GREY, label="Ideal delay line")
    ax[1].bar(x + 0.19, loto[:, 2], 0.38, color=BLUE, label="Line + 1 echo")
    ax[1].set_xticks(x, [str(int(t)) for t in loto[:, 0]])
    ax[1].set_xlabel("Held-out temperature (°C)")
    ax[1].set_ylabel("Held-out RMS error (°)")
    ax[1].set_ylim(0, 13)
    ax[1].legend(fontsize=6.5, frameon=False, loc="upper center")
    ax[1].set_title("(b) Leave-one-temperature-out", fontsize=8.5, loc="left")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def fig2(path, f0s, T0s, S):
    fig, ax = plt.subplots(figsize=(4.6, 3.0))
    im = ax.pcolormesh(f0s, T0s, np.abs(S), cmap="Blues", shading="auto", vmin=0, vmax=5)
    fig.colorbar(im, ax=ax).set_label("|∂φ/∂T| (°/°C)")
    ax.set_xlabel("Operating frequency f₀ (MHz)")
    ax.set_ylabel("Ambient temperature T₀ (°C)")
    ax.grid(False)
    ax.set_title("Local phase sensitivity (measured, rigid shift)", fontsize=8.5, loc="left")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def fig3(path, vals):
    # vals: {nombre del metodo: [m=1, m=2, m=3]}
    fig, ax = plt.subplots(figsize=(5.2, 3.1))
    x = np.arange(3)
    for j, (k, v) in enumerate(vals.items()):
        ax.bar(x + (j - 1) * 0.26, v, 0.26, color=[ORANGE, GREY, BLUE][j], label=k)
        for xi, vi in zip(x + (j - 1) * 0.26, v):
            ax.text(xi, vi + 0.05, f"{vi:.2f}", ha="center", fontsize=6.5, color=INK)
    ax.axhline(1, color="#888", lw=0.8)
    ax.set_xticks(x, ["m = 1", "m = 2", "m = 3 (triple transit)"])
    ax.set_xlabel("Catalyst crossings of the dominant echo")
    ax.set_ylabel("Gain variation over T₀ = 25–60 °C\n(max/min, median over f₀)")
    ax.set_ylim(0, 4.4)
    ax.legend(fontsize=6.5, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.25), ncol=3, handlelength=1)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def fig4(path, f, temps, P_aligned, period):
    sp = [CubicSpline(f, q) for q in P_aligned]
    mid = (temps[1:] + temps[:-1]) / 2
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.7), sharey=True)
    for j, fc in enumerate([97.70e6, 98.00e6, 98.25e6]):
        for a, K, ttl in [(ax[0], 1, "(a) Single tone at f₀"), (ax[1], 4, "(b) Mean of 4 tones over one ripple period")]:
            ph = np.array([s(tones(fc, K, period)).mean() for s in sp])
            S = np.abs(np.diff(ph) / np.diff(temps))
            a.plot(mid, S, "-o", ms=4, lw=2, color=[BLUE, ORANGE, AQUA][j], label=f"f₀ = {fc / 1e6:.2f} MHz")
            a.set_title(ttl, fontsize=8.5, loc="left")
            a.set_xlabel("Temperature interval midpoint (°C)")
    ax[0].set_ylabel("Sensitivity (°/°C)")
    ax[0].legend(fontsize=7, frameon=False)
    ax[0].set_ylim(0, 6)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def fig5(path, out, dT, T0):
    lab = {"1T ideal": "1 tone, ideal detector", "1T AD8302": "1 tone, AD8302", "1T AD8302+NN": "1 tone, AD8302 + network",
           "4T ideal": "4 tones, ideal detector", "4T AD8302": "4 tones, AD8302 (no ML)", "4T AD8302+NN": "4 tones, AD8302 + network"}
    col = {"1T ideal": GREY, "1T AD8302": GREY, "1T AD8302+NN": GREY, "4T ideal": AQUA, "4T AD8302": ORANGE, "4T AD8302+NN": BLUE}
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 3.0), gridspec_kw={"width_ratios": [1.15, 1]})
    b = dT >= 1
    keys = list(lab)[::-1]
    for i, k in enumerate(keys):
        rel = 100 * np.abs(out[k][b] - dT[b]) / dT[b]
        md, p95 = np.nanmedian(rel), np.nanpercentile(rel, 95)
        ax[0].plot([md, p95], [i, i], color=col[k], lw=2)
        ax[0].plot(md, i, "o", color=col[k], ms=6)
        ax[0].text(p95 * 1.15, i, f"{md:.1f}% / {p95:.0f}%", va="center", fontsize=6.5, color=INK)
    ax[0].set_yticks(range(len(keys)), [lab[k] for k in keys], fontsize=7.5)
    ax[0].set_xscale("log")
    ax[0].set_xlim(0.5, 2e4)
    ax[0].set_xlabel("Relative error in ΔT (%), ΔT ≥ 1 °C\n(dot = median, bar end = 95th percentile)")
    ax[0].set_title("(a) Accuracy over new units, T₀ = 20–60 °C", fontsize=8.5, loc="left")
    bins = np.arange(20, 61, 5)
    mid = (bins[1:] + bins[:-1]) / 2
    for k, c, nm in (("1T ideal", GREY, "1 tone, ideal"), ("4T ideal", AQUA, "4 tones, ideal"),
                     ("4T AD8302+NN", BLUE, "4 tones, AD8302 + network")):
        e = 100 * np.abs(out[k] - dT) / dT
        q = [np.percentile(e[b & (T0 >= lo) & (T0 < hi)], 95) for lo, hi in zip(bins[:-1], bins[1:])]
        ax[1].plot(mid, q, "-o", ms=4, lw=2, color=c, label=nm)
    ax[1].set_xlabel("Ambient temperature T₀ (°C)")
    ax[1].set_ylabel("95th-percentile |error| (%)")
    ax[1].legend(fontsize=7, frameon=False)
    ax[1].set_ylim(0, None)
    ax[1].set_title("(b) Robustness to ambient temperature", fontsize=8.5, loc="left")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)

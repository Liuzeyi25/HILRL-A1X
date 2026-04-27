"""
CRISP Figure 1 — Zone 2 (Q-value plot) reproduction.
Targets the visual style of the rendered preview the user shared.
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

# ---------- Reproducibility ----------
np.random.seed(42)

# ---------- Global style ----------
mpl.rcParams.update({
    "font.family": "Times New Roman",
    "font.size": 13,
    "axes.linewidth": 0.8,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "lines.dash_capstyle": "round",
    "lines.solid_capstyle": "round",
})

# ---------- Color palette (locked to spec) ----------
C_GT       = "#2C3E50"
C_HILSERL  = "#E67E22"
C_CRISP    = "#2E86AB"
C_RED      = "#E74C3C"
C_DARKRED  = "#9E2F22"
C_GREEN    = "#27AE60"
C_PURPLE   = "#8E44AD"
C_TEAL     = "#16A085"
C_TEXT     = "#1A1A1A"
C_TEXT2    = "#555555"
C_GRID     = "#E5E5E5"
C_GUIDE    = "#DDDDDD"

# ---------- Time axis ----------
t = np.arange(0, 157)  # 157 frames total

# ============================================================
# DATA SOURCE SWITCH
# True  → use real measured Q-values read from the actual plot
# False → use hand-crafted spec anchors
# ============================================================
USE_REAL_DATA = True

# ---- Ground truth (discounted return) — same for both modes ----
gt_t = np.array([0,   25,   50,   75,   100,  125,  156])
gt_y = np.array([0.22, 0.29, 0.37, 0.47, 0.61, 0.78, 1.00])
gt   = np.interp(t, gt_t, gt_y)

# ---- Spec / hand-crafted anchors (USE_REAL_DATA = False) ----
spec_hil_t = np.array([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150])
spec_hil_y = np.array([0.36, 0.39, 0.41, 0.42, 0.43, 0.45, 0.50, 0.55, 0.50, 0.42, 0.40, 0.43, 0.55, 0.72, 0.88, 0.99])

spec_crisp_t = np.array([0,  5, 10, 15, 20, 25, 30, 40, 50, 60, 70, 80, 85, 90, 95, 100, 110, 120, 130, 140, 150])
spec_crisp_y = np.array([0.37, 0.34, 0.30, 0.20, 0.05, -0.05, -0.10, -0.15, -0.17, -0.18, -0.17, -0.13, -0.05, 0.10, 0.30, 0.42, 0.45, 0.50, 0.70, 0.88, 0.99])

# ---- Real measured data — full 157 points digitized from plot image ----
# Orange = Q-B (0423_baseline_1/checkpoint_8000)  → maps to HIL-SERL
# Blue   = Q-A (0423_2_multi-bc/checkpoint_8000)  → maps to CRISP (Ours)
# Intervention (pink dashed) at t ≈ 102
real_hil = np.array([
    +0.363, +0.363, +0.363, +0.367, +0.371, +0.369, +0.350, +0.358, +0.350, +0.356,  # t=0..9
    +0.347, +0.354, +0.360, +0.360, +0.361, +0.371, +0.374, +0.376, +0.387, +0.396,  # t=10..19
    +0.398, +0.394, +0.383, +0.380, +0.385, +0.367, +0.356, +0.347, +0.343, +0.349,  # t=20..29
    +0.347, +0.343, +0.341, +0.338, +0.347, +0.339, +0.334, +0.332, +0.334, +0.341,  # t=30..39
    +0.339, +0.338, +0.349, +0.369, +0.372, +0.380, +0.378, +0.367, +0.389, +0.418,  # t=40..49
    +0.448, +0.418, +0.431, +0.444, +0.428, +0.424, +0.433, +0.440, +0.455, +0.431,  # t=50..59
    +0.411, +0.439, +0.429, +0.472, +0.494, +0.497, +0.505, +0.517, +0.530, +0.517,  # t=60..69
    +0.495, +0.484, +0.505, +0.512, +0.536, +0.560, +0.567, +0.538, +0.532, +0.505,  # t=70..79
    +0.455, +0.426, +0.400, +0.393, +0.383, +0.369, +0.361, +0.352, +0.349, +0.347,  # t=80..89
    +0.352, +0.352, +0.354, +0.349, +0.354, +0.358, +0.363, +0.367, +0.360, +0.358,  # t=90..99
    +0.354, +0.356, +0.360, +0.360, +0.360, +0.363, +0.367, +0.371, +0.380, +0.393,  # t=100..109
    +0.396, +0.403, +0.415, +0.429, +0.442, +0.451, +0.484, +0.486, +0.501, +0.541,  # t=110..119
    +0.573, +0.591, +0.607, +0.617, +0.637, +0.659, +0.664, +0.672, +0.681, +0.688,  # t=120..129
    +0.705, +0.710, +0.719, +0.732, +0.734, +0.741, +0.762, +0.778, +0.796, +0.813,  # t=130..139
    +0.817, +0.829, +0.844, +0.857, +0.870, +0.877, +0.879, +0.919, +0.940, +0.945,  # t=140..149
    +0.962, +0.971, +0.98, +0.982, +0.988, +0.988, 0.997                                 # t=150..155
])
 
# CRISP / Ours (blue / Q-A in the original plot)
real_crisp = np.array([
    +0.382, +0.382, +0.396, +0.394, +0.369, +0.344, +0.319, +0.238, +0.211, +0.238,  # t=0..9
    +0.205, +0.299, +0.341, +0.331, +0.321, +0.257, +0.297, +0.343, +0.363, +0.336,  # t=10..19
    +0.350, +0.376, +0.355, +0.334, +0.209, +0.086, +0.005, +0.038, +0.143, +0.112,  # t=20..29
    +0.081, +0.114, +0.114, +0.084, +0.062, +0.026, -0.009, -0.063, -0.086, -0.118,  # t=30..39
    -0.103, -0.105, -0.138, -0.143, -0.130, -0.130, -0.151, -0.189, -0.175, -0.169,  # t=40..49
    -0.158, -0.156, -0.160, -0.141, -0.149, -0.151, -0.167, -0.171, -0.163, -0.158,  # t=50..59
    -0.152, -0.176, -0.176, -0.176, -0.193, -0.211, -0.219, -0.219, -0.211, -0.213,  # t=60..69
    -0.200, -0.197, -0.200, -0.209, -0.215, -0.211, -0.211, -0.193, -0.189, -0.187,  # t=70..79
    -0.154, -0.136, -0.134, -0.099, -0.074, -0.041, -0.002, +0.042, +0.128, +0.176,  # t=80..89
    +0.215, +0.261, +0.327, +0.418, +0.431, +0.429, +0.437, +0.435, +0.435, +0.442,  # t=90..99
    +0.439, +0.435, +0.431, +0.431, +0.428, +0.424, +0.422, +0.420, +0.420, +0.413,  # t=100..109
    +0.413, +0.412, +0.410, +0.409, +0.409, +0.413, +0.420, +0.424, +0.433, +0.446,  # t=110..119
    +0.451, +0.462, +0.473, +0.486, +0.503, +0.512, +0.517, +0.528, +0.547, +0.562,  # t=120..129
    +0.574, +0.591, +0.602, +0.622, +0.628, +0.644, +0.661, +0.675, +0.681, +0.690,  # t=130..139
    +0.701, +0.705, +0.719, +0.732, +0.741, +0.754, +0.771, +0.789, +0.811, +0.831,  # t=140..149
    +0.861, +0.894, +0.92, +0.945, +0.964, +0.985, +0.999                                 # t=150..155
])

# ---- Smooth helper (Gaussian, preserves length) ----
from scipy.ndimage import gaussian_filter1d
def smooth(arr, sigma=2.0):
    return gaussian_filter1d(arr.astype(float), sigma=sigma)

# ---- Select active dataset ----
if USE_REAL_DATA:
    hil   = smooth(real_hil,   sigma=2.5)
    crisp = smooth(real_crisp, sigma=2.5)
else:
    hil   = np.interp(t, spec_hil_t,   spec_hil_y)
    crisp = np.interp(t, spec_crisp_t, spec_crisp_y)


# ---------- Figure ----------
fig, ax = plt.subplots(figsize=(16, 4.6), dpi=120)

# Phase bands (drawn first, behind everything)
phases = [
    ((0,   25),  C_GREEN,  0.07),
    ((25,  85),  C_RED,    0.10),
    ((85,  120), C_PURPLE, 0.07),
    ((120, 156), C_TEAL,   0.07),
]
for (lo, hi), color, alpha in phases:
    ax.axvspan(lo, hi, color=color, alpha=alpha, zorder=0)

# Vertical alignment guides (very faint)
for tk in [25, 50, 75, 100, 125, 150]:
    ax.axvline(tk, color=C_GUIDE, lw=0.4, ls="--", zorder=0.5)

# Hatched gap fill in the suboptimal segment — more visible
mask = (t >= 25) & (t <= 95)
ax.fill_between(
    t[mask], hil[mask], crisp[mask],
    facecolor=C_RED, alpha=0.18,
    hatch="////", edgecolor=C_DARKRED, linewidth=0.4,
    zorder=2,
)

# Zero line
ax.axhline(0, color="#888888", lw=0.5, zorder=1.5)

# Curves — dashed lines, small per-point markers
ax.plot(t, gt,    color=C_GT,      lw=2.0, zorder=4, ls="-",
        marker="o", markersize=4, markevery=1,
        markerfacecolor=C_GT,      markeredgewidth=0,
        label="Discounted Return (ground truth)")
ax.plot(t, hil,   color=C_HILSERL, lw=2.0, zorder=4, ls="-",
        marker="s", markersize=4, markevery=1,
        markerfacecolor=C_HILSERL, markeredgewidth=0,
        label="HIL-SERL (baseline)")
ax.plot(t, crisp, color=C_CRISP,   lw=2.0, zorder=4, ls="-",
        marker="^", markersize=4, markevery=1,
        markerfacecolor=C_CRISP,   markeredgewidth=0,
        label="CRISP (Ours)")

# Axes
ax.set_xlim(0, 156)
ax.set_ylim(-0.30, 1.08)
ax.set_xticks([0, 25, 50, 75, 100, 125, 150])
ax.set_yticks([-0.2, 0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
ax.set_ylabel("Q value / Return", fontsize=14, fontweight="bold", color=C_TEXT)
ax.set_xlabel("Timestep", fontsize=14, fontweight="bold", color=C_TEXT)
ax.tick_params(axis="both", colors=C_TEXT2, labelsize=12)

# Bold tick labels
for label in ax.get_xticklabels() + ax.get_yticklabels():
    label.set_fontweight("bold")

# Spines: only left + bottom
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
for s in ("left", "bottom"):
    ax.spines[s].set_color(C_TEXT2)
    ax.spines[s].set_linewidth(0.7)

ax.yaxis.grid(True, linestyle=":", linewidth=0.4, color=C_GRID)
ax.set_axisbelow(True)

# Legend
leg = ax.legend(
    loc="upper left", fontsize=12, framealpha=0.92,
    edgecolor="#CCCCCC", fancybox=False, borderpad=0.7,
    handlelength=2.2, handletextpad=0.7, labelspacing=0.45,
)
leg.get_frame().set_linewidth(0.5)
for txt in leg.get_texts():
    if "CRISP" in txt.get_text():
        txt.set_fontweight("bold")

# ---------- Annotations ----------
# 1) "0.7 Q-value gap" key callout with curved arrow
ax.text(60, 0.92, "≈ 0.7 Q-value gap",
        fontsize=19, fontweight="bold", color=C_DARKRED,
        ha="center", va="bottom")
ax.text(60, 0.83, "Systematic overestimation by HIL-SERL",
        fontsize=13, color=C_TEXT, ha="center", va="bottom")
ax.annotate("",
    xy=(60, 0.32), xytext=(60, 0.81),
    arrowprops=dict(arrowstyle="-|>", color=C_DARKRED, lw=1.8,
                    connectionstyle="arc3,rad=-0.18",
                    mutation_scale=20),
    zorder=4,
)

# 2) HIL-SERL annotation (top-right area)
ax.plot(75, hil[75], "o", color=C_HILSERL, ms=7, mec="white", mew=1.0, zorder=5)
ax.annotate(
    "HIL-SERL: Q remains high\n— credit propagates to\nsuboptimal actions",
    xy=(75, hil[75]), xytext=(98, 0.95),
    fontsize=12, ha="left", va="top", color=C_TEXT, fontweight="semibold",
    arrowprops=dict(arrowstyle="-", color=C_TEXT2, lw=0.6,
                    connectionstyle="arc3,rad=0.0"),
    zorder=4,
)

# 3) CRISP annotation (moved up and left to avoid x-axis overlap)
ax.plot(55, crisp[55], "o", color=C_CRISP, ms=7, mec="white", mew=1.0, zorder=5)
ax.annotate(
    "Ours (CRISP): corrected Q value by\nsuboptimal segment identification",
    xy=(55, crisp[55]), xytext=(7, -0.20),
    fontsize=12, ha="left", va="center", color=C_TEXT, fontweight="semibold",
    arrowprops=dict(arrowstyle="-", color=C_TEXT2, lw=0.6),
    zorder=4,
)

# 4) Convergence note (top-right corner)
ax.text(150, 1.04, "*Both converge at success (done)",
        fontsize=12, fontstyle="italic", color=C_TEXT2,
        ha="right", va="center")

plt.tight_layout()
plt.savefig("paper_figure/figure/q_value_plot.png", dpi=200, bbox_inches="tight", pad_inches=0.06)
plt.savefig("paper_figure/figure/q_value_plot.pdf", bbox_inches="tight", pad_inches=0.06)
print("Saved.")

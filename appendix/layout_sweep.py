"""Layout sweep over 2x3 gridworld MDPs (companion to gbec_two_mdps.ipynb).

Samples 200 layouts of the 2x3 grid by varying the number of red cells
(k = 1..5, 40 layouts per k) and the terminal-state position (6 choices,
drawn independently of cell color — the terminal may land on a red cell),
then for each layout:
  - runs Value Iteration,
  - derives the gBEC constraints from all-optimal-demonstrations (same
    machinery as gbec_two_mdps.ipynb),
  - renders the layout with arrows for every tied optimal action,
  - renders the resulting feasible reward region.

Run with: venv/bin/python3 appendix/layout_sweep.py
"""

import io
import os
import sys
import itertools
from contextlib import redirect_stdout

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Polygon

SEED = 42
np.random.seed(SEED)

_here = os.path.abspath(os.path.dirname(__file__))
_root = os.path.dirname(_here) if os.path.basename(_here) == "appendix" else _here
if _root not in sys.path:
    sys.path.insert(0, _root)

from agent.q_learning_agent import ValueIteration
from experiments.gridworld_env_layout import GridWorldMDPFromLayoutEnv
from utils import (
    compute_successor_features_iterative_from_q,
    derive_constraints_from_q_ties,
    remove_redundant_constraints,
)
from utils.halfspace_plot import _intersection_polygon_2d

GAMMA = 0.99
ROWS, COLS = 2, 3
N_STATES = ROWS * COLS
UP, DOWN, LEFT, RIGHT = 0, 1, 2, 3
ACTION_DXY = {UP: (0, 0.32), DOWN: (0, -0.32), LEFT: (-0.32, 0), RIGHT: (0.32, 0)}

PER_K = 40                       # layouts per red-cell count -> 5 * PER_K = 200 total
N_LAYOUTS = 5 * PER_K

_w_raw = np.array([-0.9701425, -0.24253563])
TRUE_REWARD = _w_raw / np.linalg.norm(_w_raw)
COLOR_TO_FEATURE = {"red": np.array([1.0, 0.0]), "blue": np.array([0.0, 1.0])}

OUT_DIR = os.path.join(_here, "layout_sweep_outputs")
os.makedirs(OUT_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Design space: 40 layouts per k in {1,...,5}; terminal position is drawn
# independently of the red/blue coloring, so it may fall on a red cell.
# ---------------------------------------------------------------------------

def sample_layouts_balanced(per_k=PER_K, seed=SEED):
    rng = np.random.default_rng(seed)
    layouts = []
    for k in range(1, 6):
        combos = [
            (term, subset)
            for term in range(N_STATES)
            for subset in itertools.combinations(range(N_STATES), k)
        ]
        rng.shuffle(combos)
        if len(combos) >= per_k:
            chosen = combos[:per_k]
        else:
            chosen = list(combos)
            pad_idx = rng.integers(0, len(combos), size=per_k - len(combos))
            chosen += [combos[i] for i in pad_idx]
        layouts.extend((k, term, subset) for term, subset in chosen)
    return layouts


def build_grid(red_cells):
    grid = [["blue"] * COLS for _ in range(ROWS)]
    for s in red_cells:
        r, c = divmod(s, COLS)
        grid[r][c] = "red"
    return grid


# ---------------------------------------------------------------------------
# Per-layout analysis (VI + successor features + gBEC constraints)
# ---------------------------------------------------------------------------

_cache = {}


def analyze_layout(term, red_cells):
    key = (term, tuple(sorted(red_cells)))
    if key in _cache:
        return _cache[key]

    grid = build_grid(red_cells)
    env = GridWorldMDPFromLayoutEnv(
        gamma=GAMMA,
        layout=grid,
        color_to_feature_map=COLOR_TO_FEATURE,
        terminal_states=[term],
        custom_feature_weights=TRUE_REWARD,
    )
    q = ValueIteration(env).get_q_values()
    with redirect_stdout(io.StringIO()):
        mu_sa, _, _, _ = compute_successor_features_iterative_from_q(env, q)
    raw = derive_constraints_from_q_ties(mu_sa, q, env)
    vecs = remove_redundant_constraints([v for v, *_ in raw])

    opt_actions = {}
    for s in range(N_STATES):
        if s == term:
            continue
        row = q[s]
        m = row.max()
        opt_actions[s] = [a for a in range(4) if abs(row[a] - m) < 1e-8]

    result = {"grid": grid, "term": term, "vecs": vecs, "opt_actions": opt_actions}
    _cache[key] = result
    return result


def gbec_area(vecs, box=1.0, res=300):
    if len(vecs) == 0:
        return 1.0
    xs = np.linspace(-box, box, res)
    W1, W2 = np.meshgrid(xs, xs)
    feasible = np.ones_like(W1, dtype=bool)
    for v in vecs:
        feasible &= v[0] * W1 + v[1] * W2 >= 0
    return feasible.mean()


# ---------------------------------------------------------------------------
# Panel drawing
# ---------------------------------------------------------------------------

def draw_layout_panel(ax, grid, term, opt_actions):
    for r in range(ROWS):
        for c in range(COLS):
            s = r * COLS + c
            y = ROWS - 1 - r  # row 0 drawn at the top
            face = "#767676" if grid[r][c] == "red" else "white"
            ax.add_patch(Rectangle((c, y), 1, 1, facecolor=face,
                                    edgecolor="black", linewidth=1.3))
            if s == term:
                ax.text(c + 0.5, y + 0.5, "T", ha="center", va="center",
                        fontsize=9, fontweight="bold")
            else:
                for a in opt_actions[s]:
                    dx, dy = ACTION_DXY[a]
                    ax.annotate("", xy=(c + 0.5 + dx, y + 0.5 + dy),
                                xytext=(c + 0.5, y + 0.5),
                                arrowprops=dict(arrowstyle="-|>", color="black",
                                                lw=1.1, mutation_scale=7))
    ax.set_xlim(0, COLS)
    ax.set_ylim(0, ROWS)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def draw_feasible_panel(ax, vecs, box=1.0):
    xs = np.array([-box, box])
    for vx, vy in vecs:
        if abs(vy) < 1e-12:
            ax.axvline(0, color="#d81b60", lw=1.1)
        else:
            ax.plot(xs, -(vx / vy) * xs, color="#d81b60", lw=1.1)

    poly = _intersection_polygon_2d(vecs, box=box)
    if poly.shape[0] > 0:
        ax.add_patch(Polygon(poly, closed=True, facecolor="#f5bd23", alpha=0.9,
                              edgecolor="none", hatch="///"))

    ax.axhline(0, color="black", lw=0.5)
    ax.axvline(0, color="black", lw=0.5)
    ax.set_xlim(-box, box)
    ax.set_ylim(-box, box)
    ax.set_aspect("equal", "box")
    ax.set_xticks([])
    ax.set_yticks([])


# ---------------------------------------------------------------------------
# Gallery figures (grouped into 5 row-blocks, one per k)
# ---------------------------------------------------------------------------

GALLERY_NCOLS = 20
GALLERY_ROWS_PER_K = PER_K // GALLERY_NCOLS
GALLERY_NROWS = 5 * GALLERY_ROWS_PER_K


def _grid_position(idx):
    k_block = idx // PER_K
    pos_in_block = idx % PER_K
    r = k_block * GALLERY_ROWS_PER_K + pos_in_block // GALLERY_NCOLS
    c = pos_in_block % GALLERY_NCOLS
    return r, c


def plot_layout_gallery(layouts, results):
    fname = os.path.join(OUT_DIR, "layout_gallery_terminal_random.pdf")
    fig, axes = plt.subplots(GALLERY_NROWS, GALLERY_NCOLS,
                              figsize=(GALLERY_NCOLS * 1.05, GALLERY_NROWS * 0.78))
    for idx, ((k, term, subset), res) in enumerate(zip(layouts, results)):
        r, c = _grid_position(idx)
        ax = axes[r, c]
        draw_layout_panel(ax, res["grid"], res["term"], res["opt_actions"])
        if c == 0 and r % GALLERY_ROWS_PER_K == 0:
            ax.set_ylabel(f"k={k}", fontsize=12, fontweight="bold", rotation=0,
                          labelpad=24, va="center")

    fig.suptitle(
        "2×3 gridworld layouts and optimal policies\n"
        "(gray and white cells denote the two features, T = terminal — may sit on either color, "
        "arrows = every tied optimal action) — 40 sampled layouts per red-cell count k",
        fontsize=13, y=1.004,
    )
    plt.tight_layout()
    fig.savefig(fname, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {fname}")


def plot_feasible_gallery(layouts, results):
    fname = os.path.join(OUT_DIR, "feasible_region_gallery_terminal_random.pdf")
    fig, axes = plt.subplots(GALLERY_NROWS, GALLERY_NCOLS,
                              figsize=(GALLERY_NCOLS * 0.9, GALLERY_NROWS * 0.9))
    for idx, ((k, term, subset), res) in enumerate(zip(layouts, results)):
        r, c = _grid_position(idx)
        ax = axes[r, c]
        draw_feasible_panel(ax, res["vecs"])
        if c == 0 and r % GALLERY_ROWS_PER_K == 0:
            ax.set_ylabel(f"k={k}", fontsize=12, fontweight="bold", rotation=0,
                          labelpad=24, va="center")

    fig.suptitle(
        "gBEC feasible reward regions for 200 sampled 2×3 layouts\n"
        "(gold hatch = feasible half-space intersection, red line = derived constraint boundary)",
        fontsize=13, y=1.004,
    )
    plt.tight_layout()
    fig.savefig(fname, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {fname}")


def plot_area_summary(ks, areas):
    fname = os.path.join(OUT_DIR, "area_vs_red_count_terminal_random.pdf")
    ks = np.asarray(ks)
    fig, ax = plt.subplots(figsize=(6, 4.5))

    box_data = [areas[ks == k] for k in range(1, 6)]
    ax.boxplot(box_data, positions=range(1, 6), widths=0.5,
               patch_artist=True, showfliers=False,
               medianprops=dict(color="#0b3d91", linewidth=2),
               boxprops=dict(facecolor="#cfe0f3", edgecolor="#4a4a4a", linewidth=1),
               whiskerprops=dict(color="#4a4a4a", linewidth=1),
               capprops=dict(color="#4a4a4a", linewidth=1))

    jitter_rng = np.random.default_rng(SEED)
    for k in range(1, 6):
        y = areas[ks == k]
        x = k + jitter_rng.uniform(-0.12, 0.12, size=len(y))
        ax.scatter(x, y, s=14, color="#1f77b4", alpha=0.6, zorder=3, linewidths=0)

    ax.set_xticks(range(1, 6))
    ax.set_xlabel("Number of red cells (k)")
    ax.set_ylabel("gBEC area (fraction of [-1,1]² box)")
    ax.set_title("Feasible-region size vs. red-cell coverage\n(200 sampled 2×3 layouts, 40 per k)")
    ax.set_ylim(0, 1.05)
    ax.yaxis.grid(True, linestyle="--", alpha=0.3)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    fig.savefig(fname, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {fname}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    layouts = sample_layouts_balanced()
    print(f"Sampled {len(layouts)} layouts ({PER_K} per red-cell count k=1..5)")

    unique_keys = {(term, tuple(sorted(subset))) for _, term, subset in layouts}
    print(f"Unique (terminal, red-cell-set) combinations: {len(unique_keys)} / {len(layouts)}")

    results = [analyze_layout(term, subset) for _, term, subset in layouts]
    ks = np.array([k for k, _, _ in layouts])

    areas = np.array([gbec_area(r["vecs"]) for r in results])
    n_constraints = np.array([len(r["vecs"]) for r in results])
    true_reward_ok = np.array([
        all(np.dot(v, TRUE_REWARD) >= -1e-8 for v in r["vecs"]) for r in results
    ])

    print(f"True reward feasible in all layouts: {true_reward_ok.all()} "
          f"({true_reward_ok.sum()}/{len(results)})")
    print(f"Non-redundant constraint count — min {n_constraints.min()}, "
          f"max {n_constraints.max()}, mean {n_constraints.mean():.2f}")

    print("\nFeasible-region area (fraction of [-1,1]² box) by red-cell count k:")
    for k in range(1, 6):
        mask = ks == k
        print(f"  k={k}: mean={areas[mask].mean():.4f}  std={areas[mask].std():.4f}  n={mask.sum()}")

    plot_layout_gallery(layouts, results)
    plot_feasible_gallery(layouts, results)
    plot_area_summary(ks, areas)


if __name__ == "__main__":
    main()

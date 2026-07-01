"""
Visualize all 50 MiniGrid LavaWorld environments for each seed × feedback type.

Output: results_chpc/mini_viz/seed_XXXX/<feedback_label>.png
        One 5×10 grid per (seed, feedback) showing all 50 env layouts.
        - Title per env: E{i} [W/R] L={n}  r={regret:.3f}
        - SCOT-selected envs: bright orange border
        - Held-out envs: cyan border
        - Train envs with non-zero regret: red-to-yellow gradient border
"""

import os, sys, glob, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, REPO)

from utils.minigrid_lava_generator import generate_lavaworld

OUT_BASE  = os.path.join(HERE, "mini_viz")
N_ENVS    = 50
GRID_SIZE = 8
GAMMA     = 0.99

# Cell colours
C_WALL = "#1a1a2e"   # dark navy
C_LAVA = "#ff3c3c"   # vivid red
C_GOAL = "#00e676"   # vivid green
C_FREE = "#ffffff"   # white

# Border colours
C_SCOT    = "#ff8c00"   # bright orange — SCOT selected
C_HELDOUT = "#00ccff"   # cyan — held-out env
C_GRID    = "#cccccc"   # light silver grid lines

# Display names for feedback folders
FEEDBACK_LABELS = {
    "demo":        "Demo",
    "pairwise":    "Comparison",
    "estop":       "E-stop",
    "correction":  "Correction",
}


def draw_env(ax, mdp, title="", border_color=None, border_lw=2.5):
    size   = mdp["size"]
    wall   = mdp["wall_mask"]
    lava   = mdp["lava_mask"]
    gy, gx = mdp["goal_yx"]

    img = np.zeros((size, size, 3), dtype=float)
    for y in range(size):
        for x in range(size):
            if wall[y, x]:
                c = matplotlib.colors.to_rgb(C_WALL)
            elif lava[y, x]:
                c = matplotlib.colors.to_rgb(C_LAVA)
            elif (y, x) == (gy, gx):
                c = matplotlib.colors.to_rgb(C_GOAL)
            else:
                c = matplotlib.colors.to_rgb(C_FREE)
            img[y, x] = c

    ax.imshow(img, origin="upper", interpolation="nearest")
    for i in range(size + 1):
        ax.axhline(i - 0.5, color=C_GRID, lw=0.3, alpha=0.6)
        ax.axvline(i - 0.5, color=C_GRID, lw=0.3, alpha=0.6)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title, fontsize=6, pad=2)

    if border_color is not None:
        for spine in ax.spines.values():
            spine.set_edgecolor(border_color)
            spine.set_linewidth(border_lw)
    else:
        for spine in ax.spines.values():
            spine.set_edgecolor("#dddddd")
            spine.set_linewidth(0.5)


def load_result(feedback_folder, seed):
    pattern = os.path.join(
        HERE, "mini_single", feedback_folder, f"seed_{seed}", "*.json"
    )
    files = sorted(glob.glob(pattern))
    if not files:
        return None
    return json.load(open(files[-1]))


def get_train_heldout_idxs(seed, n_envs=50, heldout_frac=0.2):
    rng = np.random.default_rng(seed)
    n_train = int(n_envs * (1 - heldout_frac))
    train_idxs   = sorted(rng.choice(n_envs, n_train, replace=False))
    heldout_idxs = sorted(set(range(n_envs)) - set(train_idxs))
    return train_idxs, heldout_idxs


def plot_seed_feedback(seed, feedback_folder, label, mdps, meta, out_dir):
    result = load_result(feedback_folder, seed)
    if result is None:
        print(f"  [{label}] no result — skipping")
        return

    train_idxs, heldout_idxs = get_train_heldout_idxs(seed)

    m = result["methods"]["hscot"]

    # regret map: original env idx -> (split, regret)
    regret_map = {}
    for i, orig in enumerate(train_idxs):
        r = m["regret"][i] if i < len(m["regret"]) else float("nan")
        regret_map[orig] = ("train", r)
    if m.get("heldout_regret"):
        for i, orig in enumerate(heldout_idxs):
            r = m["heldout_regret"][i] if i < len(m["heldout_regret"]) else float("nan")
            regret_map[orig] = ("heldout", r)

    # SCOT-selected envs (train indices → original)
    sel_train_idxs = set(m["selection_stats"].get("used_envs", []))
    scot_orig_idxs = {train_idxs[t] for t in sel_train_idxs if t < len(train_idxs)}

    max_regret = max((v for _, v in regret_map.values() if not np.isnan(v)), default=1.0)
    mean_r = np.nanmean([v for _, v in regret_map.values()])

    nrows, ncols = 5, 10
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(ncols * 1.5, nrows * 1.7),
                             constrained_layout=True)
    fig.suptitle(
        f"{label}   mean regret = {mean_r:.4f}",
        fontsize=13, fontweight="bold",
    )

    for i in range(N_ENVS):
        r, c = divmod(i, ncols)
        ax = axes[r, c]
        layout = meta["layout_type"][i]
        n_lava = int(mdps[i]["lava_mask"].sum())

        split, reg = regret_map.get(i, ("train", float("nan")))
        reg_str = f"{reg:.3f}" if not np.isnan(reg) else "—"
        title = f"E{i} [{layout[0].upper()}] L={n_lava}\nr={reg_str}"

        if i in scot_orig_idxs:
            border, lw = C_SCOT, 3.5
        elif split == "heldout":
            border, lw = C_HELDOUT, 2.5
        elif not np.isnan(reg) and reg > 1e-4:
            intensity = min(reg / max(max_regret, 1e-8), 1.0)
            border = matplotlib.colors.to_hex((1.0, 1.0 - intensity * 0.8, 0.0))
            lw = 2.5
        else:
            border, lw = None, 1.0

        draw_env(ax, mdps[i], title=title, border_color=border, border_lw=lw)

    # legend
    legend_patches = [
        mpatches.Patch(facecolor=C_LAVA,    label="Lava"),
        mpatches.Patch(facecolor=C_GOAL,    edgecolor="#aaa", label="Goal"),
        mpatches.Patch(facecolor=C_FREE,    edgecolor="#aaa", label="Free"),
        mpatches.Patch(facecolor="none",    edgecolor=C_SCOT,    linewidth=2.5, label="HSCOT selected"),
        mpatches.Patch(facecolor="none",    edgecolor=C_HELDOUT, linewidth=2.5, label="Held-out"),
        mpatches.Patch(facecolor="#ffdd00", edgecolor="#aaa", label="High regret (train)"),
    ]
    fig.legend(handles=legend_patches, loc="lower center", ncol=6,
               bbox_to_anchor=(0.5, -0.03), fontsize=9, framealpha=0.9)

    fname = f"{label.lower().replace(' ', '_').replace('-', '_')}.png"
    path = os.path.join(out_dir, fname)
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {path}")


def main():
    feedbacks = [
        ("demo",        "Demo"),
        ("pairwise",    "Comparison"),
        ("estop",       "E-stop"),
        ("correction",  "Correction"),
    ]

    seed_dirs = sorted(glob.glob(
        os.path.join(HERE, "mini_single", "demo", "seed_*")
    ))
    seeds = [int(os.path.basename(d).replace("seed_", "")) for d in seed_dirs]
    print(f"Found {len(seeds)} seeds: {seeds}")

    for seed in seeds:
        out_dir = os.path.join(OUT_BASE, f"seed_{seed}")
        os.makedirs(out_dir, exist_ok=True)
        print(f"\nSeed {seed}")

        _, mdps, meta = generate_lavaworld(
            n_envs=N_ENVS, size=GRID_SIZE, seed=seed, gamma=GAMMA
        )

        for folder, label in feedbacks:
            plot_seed_feedback(seed, folder, label, mdps, meta, out_dir)

    print(f"\nDone. Images in {OUT_BASE}/")


if __name__ == "__main__":
    main()

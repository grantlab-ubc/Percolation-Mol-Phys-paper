import numpy as np
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import warnings
import os
import csv

plt.rcParams['font.sans-serif'] = ['Helvetica', 'Arial', 'sans-serif']
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.linewidth'] = 3.0

FIGSIZE = (6, 5)
AXIS_LINE_WIDTH = 3.0
TICK_WIDTH = 2.0
TICK_LEN_MAJOR = 8
TICK_LEN_MINOR = 4
LABEL_FONT_SIZE = 18
TICK_FONT_SIZE = 14
TEXT_LABEL_SIZE = 24
MARKER_SIZE_PTS = 6
MARKER_EDGE_W = 1.2
DATA_COLOR = '#3b86c4'
DATA_ALPHA = 0.95

FIT_COLOR = 'k'
FIT_LW = 2.0
FIT_LS = '--'
FIT_DASHES = (3, 3)

SHOW_GRID = False

def _ensure_dir_for_file(path):
    if path is None:
        return
    d = os.path.dirname(os.path.abspath(path))
    if d and (not os.path.exists(d)):
        os.makedirs(d, exist_ok=True)

def save_xy_csv(path, x, y, header=("x", "y")):
    if path is None:
        return
    _ensure_dir_for_file(path)
    x = np.asarray(x).ravel()
    y = np.asarray(y).ravel()
    if x.size != y.size:
        raise ValueError(f"x and y must have same length. Got {x.size} vs {y.size}.")
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(list(header))
        for xi, yi in zip(x, y):
            w.writerow([float(xi), float(yi)])

def save_multi_series_xy_csv(path, series_list, header=("series", "x", "y")):
    if path is None:
        return
    _ensure_dir_for_file(path)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(list(header))
        for s in series_list:
            name = str(s["series"])
            x = np.asarray(s["x"]).ravel()
            y = np.asarray(s["y"]).ravel()
            if x.size != y.size:
                raise ValueError(f"Series {name}: x and y length mismatch.")
            for xi, yi in zip(x, y):
                w.writerow([name, float(xi), float(yi)])

def save_counts_csv(path, counts, header=("cluster_size", "count")):
    if path is None:
        return
    _ensure_dir_for_file(path)
    counts = np.asarray(counts)
    s = np.nonzero(counts)[0]
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(list(header))
        for si in s:
            w.writerow([int(si), int(counts[si])])

def style_axes(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(AXIS_LINE_WIDTH)
    ax.spines['bottom'].set_linewidth(AXIS_LINE_WIDTH)

    ax.tick_params(which='both', direction='in', width=TICK_WIDTH)
    ax.tick_params(which='major', length=TICK_LEN_MAJOR, labelsize=TICK_FONT_SIZE)
    ax.tick_params(which='minor', length=TICK_LEN_MINOR)

    ax.minorticks_on()

    if SHOW_GRID:
        ax.grid(True, which='major', linestyle='--', linewidth=0.8, alpha=0.4)

def flood_fill_allowed(grid, x, y, z, visited, allowed_values):
    N = grid.shape[0]
    allowed = set(allowed_values)

    stack = [(x, y, z)]
    size = 0

    while stack:
        cx, cy, cz = stack.pop()

        if visited[cx, cy, cz]:
            continue
        if grid[cx, cy, cz] not in allowed:
            continue

        visited[cx, cy, cz] = True
        size += 1

        for dx, dy, dz in [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]:
            nx, ny, nz = (cx + dx) % N, (cy + dy) % N, (cz + dz) % N
            if (not visited[nx, ny, nz]) and (grid[nx, ny, nz] in allowed):
                stack.append((nx, ny, nz))

    return size, visited

def count_all_clusters(grid, allowed_values):
    N = grid.shape[0]
    visited = np.zeros((N, N, N), dtype=bool)
    counts = np.zeros(N**3 + 1, dtype=int)
    allowed = set(allowed_values)

    for x in range(N):
        for y in range(N):
            for z in range(N):
                if (not visited[x, y, z]) and (grid[x, y, z] in allowed):
                    sz, visited = flood_fill_allowed(grid, x, y, z, visited, allowed)
                    if sz > 0:
                        counts[sz] += 1
    return counts

def count_all_P_clusters(grid):
    return count_all_clusters(grid, allowed_values=[2])


def count_all_RP_clusters(grid):
    return count_all_clusters(grid, allowed_values=[1, 2])

def max_cluster_size_from_counts(counts):
    idx = np.nonzero(counts)[0]
    return int(idx.max()) if idx.size else 0

def initialize_random_grid(N, R_perc, P_perc, V_perc, rng=None):
    if abs(R_perc + P_perc + V_perc - 1.0) > 1e-2:
        raise ValueError("Percs must sum to 1 (within tolerance).")

    if rng is None:
        rng = np.random.default_rng()

    total = N**3
    R_count = int(round(R_perc * total))
    P_count = int(round(P_perc * total))
    V_count = int(round(V_perc * total))

    diff = total - (R_count + P_count + V_count)
    R_count += diff

    if R_count < 0 or P_count < 0 or V_count < 0:
        raise ValueError("Invalid counts after rounding. Check percentages.")

    flat = np.empty(total, dtype=int)
    flat[:V_count] = 0
    flat[V_count:V_count + R_count] = 1
    flat[V_count + R_count:] = 2

    rng.shuffle(flat)
    return flat.reshape((N, N, N))

def simulate_run(N, n_iters, Q, T, H, R_perc, P_perc, V_perc, sample_iters, seed=None, verbose=False):
    rng = np.random.default_rng(seed)
    grid = initialize_random_grid(N, R_perc, P_perc, V_perc, rng=rng)

    cum_counts_P = np.zeros(N**3 + 1, dtype=int)
    cum_counts_RP = np.zeros(N**3 + 1, dtype=int)

    cluster_nums_P = []

    maxP_each, maxRP_each = [], []
    occP_each, occRP_each = [], []
    Pcount_each, RPcount_each, Rcount_each = [], [], []

    dt = 1.0

    for it in range(1, n_iters + 1):
        counts_P_now = count_all_P_clusters(grid)
        counts_RP_now = count_all_RP_clusters(grid)

        cluster_nums_P.append(np.count_nonzero(counts_P_now))

        maxP = max_cluster_size_from_counts(counts_P_now)
        maxRP = max_cluster_size_from_counts(counts_RP_now)
        maxP_each.append(maxP)
        maxRP_each.append(maxRP)

        Pcount = int(np.count_nonzero(grid == 2))
        RPcount = int(np.count_nonzero((grid == 1) | (grid == 2)))
        Rcount = int(np.count_nonzero(grid == 1))

        Pcount_each.append(Pcount)
        RPcount_each.append(RPcount)
        Rcount_each.append(Rcount)

        occP_each.append(Pcount / grid.size)
        occRP_each.append(RPcount / grid.size)

        if it in sample_iters:
            cum_counts_P += counts_P_now
            cum_counts_RP += counts_RP_now

            if verbose:
                print(f"[N={N} sample it={it}] Pcount={Pcount} RPcount={RPcount} "
                      f"Rcount={Rcount} maxP={maxP} maxRP={maxRP}")

        P_mask = (grid == 2)
        neighborP = ((np.roll(P_mask, 1, axis=0)).astype(int) +
                     (np.roll(P_mask, -1, axis=0)).astype(int) +
                     (np.roll(P_mask, 1, axis=1)).astype(int) +
                     (np.roll(P_mask, -1, axis=1)).astype(int) +
                     (np.roll(P_mask, 1, axis=2)).astype(int) +
                     (np.roll(P_mask, -1, axis=2)).astype(int))
        S_field = neighborP.astype(float)

        newg = grid.copy()
        P_orig_mask = P_mask.copy()

        r_mask = (grid == 1)
        if np.any(r_mask):
            p_RtoP = 1.0 - np.exp(-Q * S_field * dt)
            u = rng.random(grid.shape)
            toP = r_mask & (u < p_RtoP)
            newg[toP] = 2

        if np.any(P_orig_mask):
            p_PtoR = 1.0 - np.exp(-T * S_field * dt)
            u = rng.random(grid.shape)
            toR = P_orig_mask & (u < p_PtoR)
            newg[toR] = 1

        if np.any(P_orig_mask):
            still_P_mask = P_orig_mask & (newg == 2)
            if np.any(still_P_mask):
                p_PtoV = 1.0 - np.exp(-H * dt)
                u = rng.random(grid.shape)
                toV = still_P_mask & (u < p_PtoV)
                newg[toV] = 0

        grid = newg

    return (cum_counts_P, cum_counts_RP,
            cluster_nums_P,
            maxP_each, maxRP_each,
            occP_each, occRP_each,
            Pcount_each, RPcount_each, Rcount_each)


def _compute_global_size_range(counts_list):
    all_sizes = np.concatenate([np.nonzero(c)[0] for c in counts_list])
    all_sizes = all_sizes[all_sizes > 0]
    if all_sizes.size == 0:
        raise ValueError("No cluster sizes found (all counts are zero).")
    return int(all_sizes.min()), int(all_sizes.max())

def _sanitize_range(rng, s_min_global, s_max_global, name="range"):
    if rng is None:
        return s_min_global, s_max_global
    a, b = rng
    if a is None:
        a = s_min_global
    if b is None:
        b = s_max_global
    a = max(a, s_min_global)
    b = min(b, s_max_global)
    if a >= b:
        raise ValueError(f"{name} must satisfy min < max after clamping. Got {rng} -> {(a, b)}")
    return a, b

def plot_cumulative_clusters_log(
    counts_list,
    out_png='cumulative_clusters_log.png',
    bins_per_decade=10,
    plot_range=None,
    fit_range=None,
    xlabel='Avalanche size (arb.)',
    ylabel='Normalized probability',
    override_exponent=None,
    annotate_fit=True,
    panel_label=None,
    out_csv_points=None,
    out_csv_raw_prefix=None
):
    s_min_global, s_max_global = _compute_global_size_range(counts_list)
    s_plot_min, s_plot_max = _sanitize_range(plot_range, s_min_global, s_max_global, name="plot_range")
    s_fit_min, s_fit_max = _sanitize_range(fit_range, s_min_global, s_max_global, name="fit_range")

    decades = np.log10(s_plot_max) - np.log10(s_plot_min)
    nbins = int(np.ceil(decades * bins_per_decade))
    nbins = max(nbins, 5)
    edges = np.logspace(np.log10(s_plot_min), np.log10(s_plot_max), nbins + 1)
    mids = np.sqrt(edges[:-1] * edges[1:])

    fig, ax = plt.subplots(figsize=FIGSIZE)

    mids_fit_all, y_fit_all = [], []
    series_points = []

    for i, c in enumerate(counts_list):
        if out_csv_raw_prefix is not None:
            save_counts_csv(f"{out_csv_raw_prefix}_run{i}.csv", c)

        s_i = np.nonzero(c)[0]
        s_i = s_i[(s_i >= s_plot_min) & (s_i <= s_plot_max)]
        if s_i.size == 0:
            continue

        f_i = c[s_i].astype(float)
        hist_i, _ = np.histogram(s_i, bins=edges, weights=f_i)

        mask = hist_i > 0
        if not np.any(mask):
            continue

        mids_i = mids[mask]
        y_counts_i = hist_i[mask].astype(float)

        total = float(np.sum(y_counts_i))
        if total <= 0:
            continue
        y_i = y_counts_i / total

        series_points.append({"series": f"run{i}", "x": mids_i, "y": y_i})

        ax.loglog(
            mids_i, y_i, 'o',
            markerfacecolor=DATA_COLOR,
            markeredgecolor=DATA_COLOR,
            markeredgewidth=MARKER_EDGE_W,
            markersize=MARKER_SIZE_PTS,
            linestyle='None'
        )

        fit_mask = (mids_i >= s_fit_min) & (mids_i <= s_fit_max)
        if np.any(fit_mask):
            mids_fit_all.append(mids_i[fit_mask])
            y_fit_all.append(y_i[fit_mask])

    if out_csv_points is not None:
        save_multi_series_xy_csv(out_csv_points, series_points)

    if len(mids_fit_all) == 0:
        warnings.warn("No data points fell inside fit_range. Fitting over ALL plotted points instead.")
        mids_fit_all = []
        y_fit_all = []
        for s in series_points:
            mids_fit_all.append(np.asarray(s["x"]))
            y_fit_all.append(np.asarray(s["y"]))

    mids_fit_all = np.concatenate(mids_fit_all)
    y_fit_all = np.concatenate(y_fit_all)

    if mids_fit_all.size < 2:
        raise ValueError("Not enough points to fit. Expand fit_range or plot_range.")

    a_fit, b_fit = np.polyfit(np.log10(mids_fit_all), np.log10(y_fit_all), 1)

    if override_exponent is None:
        a = a_fit
        K = 10**b_fit
    else:
        a = float(override_exponent)
        logK = np.mean(np.log10(y_fit_all) - a * np.log10(mids_fit_all))
        K = 10**logK

    x1 = max(s_fit_min, s_plot_min)
    x2 = min(s_fit_max, s_plot_max)
    ax.loglog([x1, x2], [K * x1**a, K * x2**a],
              FIT_LS, color=FIT_COLOR, linewidth=FIT_LW, dashes=FIT_DASHES)

    ax.set_xlim(s_plot_min, s_plot_max)

    ax.set_xlabel(xlabel, fontweight='bold', fontsize=LABEL_FONT_SIZE)
    ax.set_ylabel(ylabel, fontweight='bold', fontsize=LABEL_FONT_SIZE)

    if panel_label is not None:
        ax.text(0.1, 1.05, panel_label, transform=ax.transAxes,
                fontsize=TEXT_LABEL_SIZE, fontweight='bold')

    style_axes(ax)
    plt.tight_layout()
    fig.savefig(out_png, dpi=300)
    plt.close(fig)

def plot_cluster_numbers(cluster_nums_list, out_png='clusters_vs_iteration.png'):
    fig, ax = plt.subplots(figsize=FIGSIZE)
    for nums in cluster_nums_list:
        ax.plot(range(1, len(nums) + 1), nums,
                marker='o', markersize=4.5,
                linewidth=2.5, color=DATA_COLOR, alpha=0.9)
    ax.set_xlabel('Iteration', fontweight='bold', fontsize=LABEL_FONT_SIZE)
    ax.set_ylabel('Number of P-clusters', fontweight='bold', fontsize=LABEL_FONT_SIZE)
    style_axes(ax)
    plt.tight_layout()
    fig.savefig(out_png, dpi=300)
    plt.close(fig)

def plot_max_cluster_sizes(maxP_runs, maxRP_runs, out_png='max_cluster_size_vs_iteration.png'):
    fig, ax = plt.subplots(figsize=FIGSIZE)

    for maxP, maxRP in zip(maxP_runs, maxRP_runs):
        ax.plot(range(1, len(maxP) + 1), maxP, linewidth=2.5, alpha=0.85, label='P')
        ax.plot(range(1, len(maxRP) + 1), maxRP, linewidth=2.5, alpha=0.85, label='R+P')

    ax.set_xlabel('Iteration', fontweight='bold', fontsize=LABEL_FONT_SIZE)
    ax.set_ylabel('Largest cluster size', fontweight='bold', fontsize=LABEL_FONT_SIZE)

    handles, labels = ax.get_legend_handles_labels()
    uniq = {}
    for h, lab in zip(handles, labels):
        if lab not in uniq:
            uniq[lab] = h
    ax.legend(uniq.values(), uniq.keys(), frameon=False, fontsize=13)

    style_axes(ax)
    plt.tight_layout()
    fig.savefig(out_png, dpi=300)
    plt.close(fig)

def plot_histogram_linear(values, out_png, bins=80,
                          xlabel='Value', ylabel='Count',
                          xlim=None, ylim=None,
                          out_csv_points=None):
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        raise ValueError("No finite values for histogram.")

    fig, ax = plt.subplots(figsize=FIGSIZE)
    counts, edges, _ = ax.hist(v, bins=bins)

    mids = 0.5 * (edges[:-1] + edges[1:])
    if out_csv_points is not None:
        save_xy_csv(out_csv_points, mids, counts, header=("x_mid", "y_count"))

    if xlim is not None:
        ax.set_xlim(xlim[0], xlim[1])
    if ylim is not None:
        ax.set_ylim(ylim[0], ylim[1])

    ax.set_xlabel(xlabel, fontweight='bold', fontsize=LABEL_FONT_SIZE)
    ax.set_ylabel(ylabel, fontweight='bold', fontsize=LABEL_FONT_SIZE)
    style_axes(ax)
    plt.tight_layout()
    fig.savefig(out_png, dpi=300)
    plt.close(fig)

def plot_histogram_log_binned(values, out_png, bins_per_decade=10,
                              plot_range=None,
                              xlabel='Value', ylabel='Binned count',
                              out_csv_points=None):
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    v = v[v > 0]

    if v.size == 0:
        raise ValueError("No positive values for log-binned histogram.")

    vmin = float(v.min())
    vmax = float(v.max())

    if plot_range is not None:
        a, b = plot_range
        if a is None:
            a = vmin
        if b is None:
            b = vmax
        v = v[(v >= a) & (v <= b)]
        if v.size == 0:
            raise ValueError("No values inside plot_range for log-binned histogram.")
        vmin, vmax = float(v.min()), float(v.max())

    decades = np.log10(vmax) - np.log10(vmin)
    nbins = int(np.ceil(decades * bins_per_decade))
    nbins = max(nbins, 5)

    edges = np.logspace(np.log10(vmin), np.log10(vmax), nbins + 1)
    mids = np.sqrt(edges[:-1] * edges[1:])
    hist, _ = np.histogram(v, bins=edges)

    mask = hist > 0
    if not np.any(mask):
        raise ValueError("All bins empty in log-binned histogram (unexpected).")

    if out_csv_points is not None:
        save_xy_csv(out_csv_points, mids[mask], hist[mask].astype(float), header=("x_mid", "y_binned_count"))

    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.loglog(
        mids[mask], hist[mask], 'o',
        markerfacecolor=DATA_COLOR,
        markeredgecolor=DATA_COLOR,
        markeredgewidth=MARKER_EDGE_W,
        markersize=MARKER_SIZE_PTS,
        linestyle='None'
    )

    ax.set_xlabel(xlabel, fontweight='bold', fontsize=LABEL_FONT_SIZE)
    ax.set_ylabel(ylabel, fontweight='bold', fontsize=LABEL_FONT_SIZE)

    style_axes(ax)
    plt.tight_layout()
    fig.savefig(out_png, dpi=300)
    plt.close(fig)

def plot_histogram_binned_linear_x(values, out_png, bins=40,
                                  xlabel='Value', ylabel='Binned count',
                                  loglog=False,
                                  xlim=None, ylim=None,
                                  add_powerlaw_ref=False,
                                  ref_exponent=-1.4,
                                  ref_anchor='max',
                                  normalize_to_prob=False,
                                  out_csv_points=None):
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        raise ValueError("No finite values for binned histogram.")

    if loglog:
        v = v[v > 0]
        if v.size == 0:
            raise ValueError("No positive values available for log-x plot.")

    vmin = float(v.min())
    vmax = float(v.max())
    if vmin == vmax:
        if not loglog:
            vmin -= 0.5
            vmax += 0.5
        else:
            vmin *= 0.95
            vmax *= 1.05

    edges = np.linspace(vmin, vmax, int(bins) + 1)
    mids = 0.5 * (edges[:-1] + edges[1:])
    hist, _ = np.histogram(v, bins=edges)

    mask = hist > 0
    if not np.any(mask):
        raise ValueError("All bins have zero count (unexpected).")

    yvals = hist[mask].astype(float)
    if normalize_to_prob:
        total = float(np.sum(yvals))
        if total <= 0:
            raise ValueError("Cannot normalize: total count is not positive.")
        yvals = yvals / total

    xpts = mids[mask]

    if out_csv_points is not None:
        save_xy_csv(out_csv_points, xpts, yvals, header=("x_mid", "y"))

    fig, ax = plt.subplots(figsize=FIGSIZE)

    if loglog:
        ax.loglog(
            xpts, yvals, 'o',
            markerfacecolor=DATA_COLOR,
            markeredgecolor=DATA_COLOR,
            markeredgewidth=MARKER_EDGE_W,
            markersize=MARKER_SIZE_PTS,
            linestyle='None'
        )
    else:
        ax.plot(
            xpts, yvals, 'o',
            markerfacecolor=DATA_COLOR,
            markeredgecolor=DATA_COLOR,
            markeredgewidth=MARKER_EDGE_W,
            markersize=MARKER_SIZE_PTS,
            linestyle='None'
        )

    if xlim is not None:
        ax.set_xlim(xlim[0], xlim[1])
    if ylim is not None:
        ax.set_ylim(ylim[0], ylim[1])
    else:
        if loglog:
            ymin = float(np.min(yvals)) * 0.8
            ymax = float(np.max(yvals)) * 1.3
            ax.set_ylim(max(ymin, 1e-12), ymax)

    if loglog and add_powerlaw_ref:
        a = float(ref_exponent)

        if xlim is not None:
            x1, x2 = float(xlim[0]), float(xlim[1])
        else:
            x1, x2 = float(xpts.min()), float(xpts.max())

        x_anchor = np.sqrt(x1 * x2)

        if ref_anchor == 'median':
            y_anchor = float(np.median(yvals))
        else:
            y_anchor = float(np.max(yvals))
        y_anchor = max(y_anchor, 1e-12)

        K = y_anchor / (x_anchor ** a)

        ax.loglog([x1, x2], [K * (x1 ** a), K * (x2 ** a)],
                  FIT_LS, color=FIT_COLOR, linewidth=FIT_LW, dashes=FIT_DASHES)

    ax.set_xlabel(xlabel, fontweight='bold', fontsize=LABEL_FONT_SIZE)
    ax.set_ylabel(ylabel, fontweight='bold', fontsize=LABEL_FONT_SIZE)

    style_axes(ax)
    plt.tight_layout()
    fig.savefig(out_png, dpi=300)
    plt.close(fig)

def _pad_counts_to_len(c, L):
    if len(c) == L:
        return c.astype(np.float64, copy=False)
    out = np.zeros(L, dtype=np.float64)
    out[:len(c)] = c.astype(np.float64, copy=False)
    return out

def build_three_cases_varlen(counts_list):
    if len(counts_list) == 0:
        raise ValueError("counts_list is empty.")

    max_len = max(len(c) for c in counts_list)
    old_list = [_pad_counts_to_len(c, max_len) for c in counts_list]

    pooled = np.zeros(max_len, dtype=np.float64)
    for c in old_list:
        pooled += c

    averaged = pooled / float(len(old_list))
    return old_list, [pooled], [averaged]

if __name__ == '__main__':
    OUTDIR = "export_xy"
    os.makedirs(OUTDIR, exist_ok=True)

    grid_sizes = [40]

    n_iters = 100
    sample_iters = range(5, 9)
    n_reps = 1

    Q, T, H = 0.01, 0.01, 0.001

    initials = [
        (0.124, 0.438, 0.438),
    ]

    all_counts_P, all_counts_RP = [], []
    all_cluster_nums = []
    all_maxP_runs, all_maxRP_runs = [], []

    all_Pcount_each_runs = []
    all_maxP_each_runs = []
    all_RPcount_each_runs = []
    all_maxRP_each_runs = []
    all_Rcount_each_runs = []

    for N in grid_sizes:
        for (R_perc, P_perc, V_perc) in initials:
            for rep in range(n_reps):
                (cumP, cumRP,
                 cluster_nums_P,
                 maxP_each, maxRP_each,
                 occP_each, occRP_each,
                 Pcount_each, RPcount_each, Rcount_each) = simulate_run(
                    N, n_iters, Q, T, H,
                    R_perc, P_perc, V_perc,
                    sample_iters,
                    seed=1000 + rep + 17 * N,
                    verbose=False
                )

                all_counts_P.append(cumP)
                all_counts_RP.append(cumRP)

                all_cluster_nums.append(cluster_nums_P)
                all_maxP_runs.append(maxP_each)
                all_maxRP_runs.append(maxRP_each)

                all_Pcount_each_runs.append(np.asarray(Pcount_each, dtype=float))
                all_maxP_each_runs.append(np.asarray(maxP_each, dtype=float))

                all_RPcount_each_runs.append(np.asarray(RPcount_each, dtype=float))
                all_maxRP_each_runs.append(np.asarray(maxRP_each, dtype=float))

                all_Rcount_each_runs.append(np.asarray(Rcount_each, dtype=float))

    plot_range_P = (1, 4000)
    fit_range_P = (1, 3500)
    plot_range_RP = (1, 4000)
    fit_range_RP = (1, 3500)

    P_old, P_pooled, P_avg = build_three_cases_varlen(all_counts_P)
    RP_old, RP_pooled, RP_avg = build_three_cases_varlen(all_counts_RP)

    plot_cumulative_clusters_log(
        P_old,
        out_png='cumulative_clusters_log_P_old_multiN.png',
        bins_per_decade=20,
        plot_range=plot_range_P,
        fit_range=fit_range_P,
        xlabel='Avalanche size (arb.)',
        ylabel='Normalized probability',
        override_exponent=None,
        annotate_fit=True,
        panel_label="(b)",
        out_csv_points=os.path.join(OUTDIR, "cumulative_clusters_log_P_old_points.csv"),
        out_csv_raw_prefix=os.path.join(OUTDIR, "cumulative_clusters_log_P_old_rawcounts")
    )
    plot_cumulative_clusters_log(
        RP_old,
        out_png='cumulative_clusters_log_RP_old_multiN.png',
        bins_per_decade=20,
        plot_range=plot_range_RP,
        fit_range=fit_range_RP,
        xlabel='Avalanche size (arb.)',
        ylabel='Normalized probability (R+P clusters)',
        override_exponent=None,
        annotate_fit=True,
        panel_label="(d)"
    )

    plot_cumulative_clusters_log(
        P_pooled,
        out_png='cumulative_clusters_log_P_pooled_multiN.png',
        bins_per_decade=20,
        plot_range=plot_range_P,
        fit_range=fit_range_P,
        xlabel='Avalanche size (arb.)',
        ylabel='Normalized probability (P clusters) — POOLED',
        override_exponent=None,
        annotate_fit=True
    )
    plot_cumulative_clusters_log(
        RP_pooled,
        out_png='cumulative_clusters_log_RP_pooled_multiN.png',
        bins_per_decade=20,
        plot_range=plot_range_RP,
        fit_range=fit_range_RP,
        xlabel='Avalanche size (arb.)',
        ylabel='Normalized probability (R+P clusters) — POOLED',
        override_exponent=None,
        annotate_fit=True
    )

    plot_cumulative_clusters_log(
        P_avg,
        out_png='cumulative_clusters_log_P_avg_multiN.png',
        bins_per_decade=20,
        plot_range=plot_range_P,
        fit_range=fit_range_P,
        xlabel='Avalanche size (arb.)',
        ylabel='Normalized probability (P clusters) — AVERAGED',
        override_exponent=None,
        annotate_fit=True
    )
    plot_cumulative_clusters_log(
        RP_avg,
        out_png='cumulative_clusters_log_RP_avg_multiN.png',
        bins_per_decade=20,
        plot_range=plot_range_RP,
        fit_range=fit_range_RP,
        xlabel='Avalanche size (arb.)',
        ylabel='Normalized probability (R+P clusters) — AVERAGED',
        override_exponent=None,
        annotate_fit=True
    )

    plot_cluster_numbers(all_cluster_nums, out_png='clusters_vs_iteration_P_multiN.png')
    plot_max_cluster_sizes(all_maxP_runs, all_maxRP_runs, out_png='max_cluster_size_vs_iteration_multiN.png')

    Pcount_all = np.concatenate(all_Pcount_each_runs) if len(all_Pcount_each_runs) else np.array([])
    maxP_all = np.concatenate(all_maxP_each_runs) if len(all_maxP_each_runs) else np.array([])
    RPcount_all = np.concatenate(all_RPcount_each_runs) if len(all_RPcount_each_runs) else np.array([])
    maxRP_all = np.concatenate(all_maxRP_each_runs) if len(all_maxRP_each_runs) else np.array([])

    plot_histogram_linear(Pcount_all, out_png='hist_total_P_cells_per_iteration_linear.png',
                          bins=100, xlabel='Total P cells (per iteration, pooled over runs)', ylabel='Count')
    plot_histogram_binned_linear_x(Pcount_all, out_png='hist_total_P_cells_per_iteration_binned_loglog_refline.png',
                                   bins=80, xlabel='Total P cells (per iteration, pooled over runs)',
                                   ylabel='Binned count', loglog=True,
                                   add_powerlaw_ref=True, ref_exponent=-1.4,
                                   normalize_to_prob=False)

    plot_histogram_linear(maxP_all, out_png='hist_max_P_cluster_per_iteration_linear.png',
                          bins=120, xlabel='Largest P-cluster size (per iteration, pooled over runs)', ylabel='Count')
    plot_histogram_binned_linear_x(maxP_all, out_png='hist_max_P_cluster_per_iteration_binned_loglog_refline.png',
                                   bins=80, xlabel='Largest P-cluster size (per iteration, pooled over runs)',
                                   ylabel='Binned count', loglog=True,
                                   add_powerlaw_ref=True, ref_exponent=-1.4,
                                   normalize_to_prob=False)

    plot_histogram_linear(RPcount_all, out_png='hist_total_RP_cells_per_iteration_linear.png',
                          bins=100, xlabel='Total (R+P) cells (per iteration, pooled over runs)', ylabel='Count')
    plot_histogram_binned_linear_x(RPcount_all, out_png='hist_total_RP_cells_per_iteration_binned_loglog_refline.png',
                                   bins=80, xlabel='Total (R+P) cells (per iteration, pooled over runs)',
                                   ylabel='Binned count', loglog=True,
                                   add_powerlaw_ref=True, ref_exponent=-1.4,
                                   normalize_to_prob=False)

    plot_histogram_linear(maxRP_all, out_png='hist_max_RP_cluster_per_iteration_linear.png',
                          bins=120, xlabel='Largest (R+P) cluster size (per iteration, pooled over runs)', ylabel='Count')
    plot_histogram_binned_linear_x(maxRP_all, out_png='hist_max_RP_cluster_per_iteration_binned_loglog_refline.png',
                                   bins=80, xlabel='Largest (R+P) cluster size (per iteration, pooled over runs)',
                                   ylabel='Binned count', loglog=True,
                                   add_powerlaw_ref=True, ref_exponent=-1.4,
                                   normalize_to_prob=False)

    hist_iters = [9, 8, 10, 11, 7]
    R_XLIM = (7250, 16150)

    raw_R_table_path = os.path.join(OUTDIR, "Rcount_each_run_by_iteration.csv")
    with open(raw_R_table_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["run_index"] + [f"iter_{i}" for i in range(1, n_iters + 1)])
        for ri, arr in enumerate(all_Rcount_each_runs):
            w.writerow([ri] + [float(v) for v in arr])

    for it0 in hist_iters:
        if it0 < 1 or it0 > n_iters:
            raise ValueError(f"hist_iters contains {it0}, but valid range is 1..{n_iters}")

        R_at_it0 = np.array([arr[it0 - 1] for arr in all_Rcount_each_runs], dtype=float)

        plot_histogram_linear(
            R_at_it0,
            out_png=f'hist_total_R_cells_iteration_{it0}_linear.png',
            bins=50,
            xlabel='Total Count',
            ylabel='Count',
            xlim=R_XLIM,
            out_csv_points=os.path.join(OUTDIR, f"hist_total_R_cells_iteration_{it0}_linear_points.csv")
        )

        plot_histogram_binned_linear_x(
            R_at_it0,
            out_png=f'hist_total_R_cells_iteration_{it0}_binned.png',
            bins=50,
            xlabel='Total Count',
            ylabel='Binned count',
            loglog=False,
            xlim=R_XLIM,
            out_csv_points=os.path.join(OUTDIR, f"hist_total_R_cells_iteration_{it0}_binned_points.csv")
        )

        plot_histogram_binned_linear_x(
            R_at_it0,
            out_png=f'hist_total_R_cells_iteration_{it0}_binned_loglog_refline.png',
            bins=40,
            xlabel='Total Count',
            ylabel='Binned count',
            loglog=True,
            xlim=R_XLIM,
            add_powerlaw_ref=True,
            ref_exponent=-1.4,
            ref_anchor='max',
            normalize_to_prob=False,
            out_csv_points=os.path.join(OUTDIR, f"hist_total_R_cells_iteration_{it0}_binned_loglog_points.csv")
        )

    print("\nSaved figures with reference styling.")
    print(f"CSV outputs folder: {OUTDIR}")
    print(f"- P_old cumulative points: {os.path.join(OUTDIR, 'cumulative_clusters_log_P_old_points.csv')}")
    print(f"- P_old raw counts: {os.path.join(OUTDIR, 'cumulative_clusters_log_P_old_rawcounts_run0.csv')} etc.")
    print(f"- Rcount table: {raw_R_table_path}")

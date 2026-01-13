import numpy as np
import matplotlib.pyplot as plt

def main():
   
    x_raw = np.array([
        16.5222, 20.0133, 24.26, 29.3801, 35.5894, 43.1003, 52.1922, 63.2681,
        76.6315, 92.8274, 112.538, 136.259, 165.087, 199.954, 242.18, 293.297,
        355.488, 430.728, 521.786, 765.61, 927.431, 1122.86, 1360.62, 1648.37,
        1996.96, 2420.49
    ], dtype=float)

    y_counts = np.array([
        17, 61, 72, 94, 86, 103, 109, 67,
        48, 38, 21, 23, 14, 8, 8, 5,
        2, 1, 1, 1, 1, 1, 1, 3,
        2, 1
    ], dtype=float)

    total = float(np.sum(y_counts))
    if total <= 0:
        raise ValueError("Cannot normalize: total count is not positive.")
    y = y_counts / total  # sum(y)=1

  
    x = x_raw / np.min(x_raw)  
    X_NUDGE = 1.09
    x = x * X_NUDGE            

    plt.rcParams['font.sans-serif'] = ['Helvetica', 'Arial', 'sans-serif']
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['axes.linewidth'] = 3.0

    FIGSIZE = (6, 5)
    AXIS_LINE_WIDTH = 3.0
    TICK_WIDTH = 2
    TICK_LEN_MAJOR = 8
    TICK_LEN_MINOR = 4

    LABEL_FONT_SIZE = 18
    TICK_FONT_SIZE = 14
    TEXT_LABEL_SIZE = 24

    MARKER_SIZE = 6
    MARKER_EDGE_W = 1.2
    DATA_COLOR = '#3b86c4'

    XLABEL = "Avalanche signal (arb.)"
    YLABEL = "Normalized probability"

    Y_LO, Y_HI = 1e-4, 1.0
    X_LO, X_HI = 1.0, 100.0

    A_EXPONENT = -1.4
    REF_X0 = 1.0
    REF_Y0 = 0.45
    K = REF_Y0 / (REF_X0 ** A_EXPONENT)

    keep = (
        np.isfinite(x) & np.isfinite(y) & (y > 0) &
        (x >= X_LO) & (x <= X_HI) &
        (y >= Y_LO) & (y <= Y_HI)
    )

    idx_keep = np.where(keep)[0]

    print("\n=== PLOTTED POINTS (sorted by x) ===")
    if idx_keep.size == 0:
        print("No points are plotted with the current X/Y limits.")
    else:
        idx_sorted = idx_keep[np.argsort(x[idx_keep])]

       
        print(" i |   x_raw    x_scaled     y_count     y_norm")
        print("---+--------------------------------------------")
        for i in idx_sorted:
            print(f"{i:2d} | {x_raw[i]:8.3f}  {x[i]:8.3f}   {y_counts[i]:8.0f}   {y[i]:.8e}")

     
        N_LAST = 5
        last_idx = idx_sorted[-min(N_LAST, idx_sorted.size):]
        y_last = y[last_idx]

        print(f"\n=== LAST {min(N_LAST, idx_sorted.size)} PLOTTED POINTS (largest x) ===")
        for i in last_idx:
            print(f"i={i:2d}: x_scaled={x[i]:.6f}, y_count={y_counts[i]:.0f}, y_norm={y[i]:.12e}")

    
        exact_same = np.all(y_last == y_last[0])
        close_same = np.allclose(y_last, y_last[0], rtol=0, atol=0)

        print("\nAre last plotted y values exactly identical?")
        print("  exact check (==):", exact_same)
        print("  allclose(atol=0):", close_same)

        if not exact_same:
            print("\nIf you expected them identical, it means at least one of those points")
            print("does NOT have the same y_count as the others, or you were looking at")
            print("different points than the rightmost-by-x ones.\n")

    raw_threshold = (X_HI / X_NUDGE) * np.min(x_raw)
    print("\n=== NOTE ABOUT 'changing counts but nothing changes' ===")
    print(f"With X_HI={X_HI} and X_NUDGE={X_NUDGE}, any point with")
    print(f"x_raw > ~{raw_threshold:.2f} ends up with x_scaled > 100 and is NOT plotted.")
    print("So changing y_counts for those excluded points will not change the figure.")

   
    fig, ax = plt.subplots(figsize=FIGSIZE)

    x_plot = x[keep]
    y_plot = y[keep]

    ax.loglog(
        x_plot, y_plot, 'o',
        markerfacecolor=DATA_COLOR,
        markeredgecolor=DATA_COLOR,
        markeredgewidth=MARKER_EDGE_W,
        markersize=MARKER_SIZE,
        linestyle='None',
        clip_on=True
    )

    x_fit = np.logspace(np.log10(X_LO), np.log10(X_HI), 250)
    y_fit = K * (x_fit ** A_EXPONENT)
    ax.loglog(x_fit, y_fit, 'k--', linewidth=2, dashes=(3, 3), clip_on=True)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(AXIS_LINE_WIDTH)
    ax.spines['bottom'].set_linewidth(AXIS_LINE_WIDTH)

    ax.tick_params(which='both', direction='in', width=TICK_WIDTH)
    ax.tick_params(which='major', length=TICK_LEN_MAJOR, labelsize=TICK_FONT_SIZE)
    ax.tick_params(which='minor', length=TICK_LEN_MINOR)

    ax.set_xlabel(XLABEL, fontweight='bold', fontsize=LABEL_FONT_SIZE)
    ax.set_ylabel(YLABEL, fontweight='bold', fontsize=LABEL_FONT_SIZE)

    ax.set_xlim(X_LO, X_HI)
    ax.set_ylim(Y_LO, Y_HI)

    ax.text(0.1, 1.05, "(c)", transform=ax.transAxes,
            fontsize=TEXT_LABEL_SIZE, fontweight='bold')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()

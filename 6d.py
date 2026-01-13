import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

def main():
   
    x_raw = np.array([
        11365.64, 11392.2, 11405.48, 11418.76, 11432.04, 11445.32,
        11458.600000000000, 11471.880000000000, 11485.16, 11498.440000000000,
        11511.720000000000, 11525.0, 11538.280000000000, 11551.560000000000,
        11564.84, 11578.120000000000, 11591.400000000000, 11604.68, 11617.96,
        11631.24, 11644.52, 11657.8, 11671.08, 11684.36, 11697.64, 11710.92,
        11724.2, 11737.48, 11750.76, 11764.04, 11777.32, 11790.600000000000,
        11803.880000000000, 11817.16, 11830.440000000000, 11843.720000000000,
        11857.0, 11870.280000000000, 11883.560000000000, 11896.84,
        11910.120000000000, 11923.400000000000, 11936.68, 11949.96, 11963.24,
        11976.52, 12003.08, 12016.36
    ], dtype=float)

    y_counts = np.array([
        3.0, 2.0, 8.0, 6.0, 12.0, 10.0,
        23.0, 30.0, 65.0, 66.0, 91.0, 126.0, 179.0, 228.0, 223.0, 287.0, 368.0,
        411.0, 448.0, 496.0, 598.0, 571.0, 590.0, 571.0, 618.0, 484.0, 525.0,
        527.0, 430.0, 370.0, 289.0, 288.0, 236.0, 192.0, 151.0, 111.0, 114.0,
        69.0, 61.0, 31.0, 34.0, 22.0, 17.0, 5.0, 4.0, 4.0, 2.0, 2.0
    ], dtype=float)

    if x_raw.size != y_counts.size:
        raise ValueError(f"x_raw and y_counts must have same length. Got {x_raw.size} and {y_counts.size}.")

    x = x_raw 

    total = np.sum(y_counts)
    if total <= 0:
        raise ValueError("Cannot normalize: total count is not positive.")
    y = y_counts / total

    mask = y > 0
    x = x[mask]
    y = y[mask]

    
    FIGSIZE = (6, 5)
    DPI_SCREEN = 150
    DPI_SAVE = 300

    X_LIM = (1e3, 1e5)     
    Y_LIM = (1e-4, 1e0)     

    AXIS_LINE_WIDTH = 3.0
    TICK_WIDTH = 2.0
    TICK_LEN_MAJOR = 8
    TICK_LEN_MINOR = 4

    LABEL_FONT_SIZE = 18
    TICK_FONT_SIZE = 14
    TEXT_LABEL_SIZE = 24

    MARKER_SIZE = 6
    MARKER_EDGE_W = 1.2
    DATA_COLOR = "#3b86c4"

    plt.rcParams["figure.dpi"] = DPI_SCREEN
    plt.rcParams["savefig.dpi"] = DPI_SAVE
    plt.rcParams["font.sans-serif"] = ["Helvetica", "Arial", "sans-serif"]
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["axes.linewidth"] = AXIS_LINE_WIDTH
    plt.rcParams["xtick.major.width"] = TICK_WIDTH
    plt.rcParams["ytick.major.width"] = TICK_WIDTH
    plt.rcParams["xtick.minor.width"] = TICK_WIDTH
    plt.rcParams["ytick.minor.width"] = TICK_WIDTH

 
    fig, ax = plt.subplots(figsize=FIGSIZE)

    ax.loglog(
        x, y, "o",
        markerfacecolor=DATA_COLOR,
        markeredgecolor=DATA_COLOR,
        markeredgewidth=MARKER_EDGE_W,
        markersize=MARKER_SIZE,
        linestyle="None"
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(AXIS_LINE_WIDTH)
    ax.spines["bottom"].set_linewidth(AXIS_LINE_WIDTH)

    ax.tick_params(axis="both", which="major", direction="in",
                   width=TICK_WIDTH, length=TICK_LEN_MAJOR, labelsize=TICK_FONT_SIZE)
    ax.tick_params(axis="both", which="minor", direction="in",
                   width=TICK_WIDTH, length=TICK_LEN_MINOR)

    ax.set_xlabel("Electron signal (arb.)", fontweight="bold", fontsize=LABEL_FONT_SIZE)
    ax.set_ylabel("Normalized probability", fontweight="bold", fontsize=LABEL_FONT_SIZE)

    ax.set_xlim(*X_LIM)
    ax.set_ylim(*Y_LIM)

    ax.xaxis.set_major_locator(mticker.LogLocator(base=10.0, numticks=4))
    ax.xaxis.set_major_formatter(mticker.LogFormatterMathtext(base=10.0))
    ax.xaxis.set_minor_locator(mticker.LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1))
    ax.xaxis.set_minor_formatter(mticker.NullFormatter())

    ax.yaxis.set_major_locator(mticker.LogLocator(base=10.0, numticks=6))
    ax.yaxis.set_major_formatter(mticker.LogFormatterMathtext(base=10.0))
    ax.yaxis.set_minor_locator(mticker.LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1))
    ax.yaxis.set_minor_formatter(mticker.NullFormatter())

    ax.text(0.1, 1.05, "(d)", transform=ax.transAxes,
            fontsize=TEXT_LABEL_SIZE, fontweight="bold")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()

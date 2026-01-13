import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

plt.rcParams["font.sans-serif"] = ["Helvetica", "Arial", "sans-serif"]
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["axes.linewidth"] = 3.0

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
DATA_COLOR = "#3b86c4"

FIT_COLOR = "k"
FIT_LW = 2.0
FIT_DASHES = (3, 3)
FIT_LS = "--"

SHOW_GRID = False

def style_axes(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(AXIS_LINE_WIDTH)
    ax.spines["bottom"].set_linewidth(AXIS_LINE_WIDTH)

    ax.tick_params(which="both", direction="in", width=TICK_WIDTH)
    ax.tick_params(which="major", length=TICK_LEN_MAJOR, labelsize=TICK_FONT_SIZE)
    ax.tick_params(which="minor", length=TICK_LEN_MINOR)

    ax.minorticks_on()

    if SHOW_GRID:
        ax.grid(True, which="major", linestyle="--", linewidth=0.8, alpha=0.4)


def shift_in_log_space(x, y, shift_x_decades=0.0, shift_y_decades=0.0,
                       jitter_x_decades=0.0, jitter_y_decades=0.0,
                       seed=0):
    """

    """
    x = np.asarray(x, float)
    y = np.asarray(y, float)

    rng = np.random.default_rng(seed)

    dx = float(shift_x_decades)
    dy = float(shift_y_decades)

    if jitter_x_decades > 0:
        dx = dx + rng.normal(0.0, float(jitter_x_decades), size=x.shape)
    if jitter_y_decades > 0:
        dy = dy + rng.normal(0.0, float(jitter_y_decades), size=y.shape)

    x2 = x * (10.0 ** dx)
    y2 = y * (10.0 ** dy)
    return x2, y2


def main():
    
    x = np.array([
        1.0417, 2.1307, 3.2419, 4.3750, 5.2376, 6.6554, 7.5032, 8.9820,
        10.1274, 11.4168, 11.9363, 13.2126, 13.8353, 15.1332, 16.0356,
        16.6255, 18.4009, 18.8163, 19.4193, 20.7625, 22.0384, 23.1872,
        24.1946, 25.3360, 26.5324, 27.9621, 29.4953, 31.1084, 34.8299,
        41.4096, 48.1269
    ], dtype=float)

    y_counts = np.array([
        512244, 43559, 104605, 17267, 31690, 7449, 12118, 3689,
        4964, 1895, 2060, 1670, 820, 812, 317,
        330, 409, 267, 200, 81, 84, 99,
        20, 16, 26, 41, 19, 15, 10,
        8, 5
    ], dtype=float)

    A_EXPONENT = -1.4
    Y_LIMS = (1e-4, 1.0)

    y_prob = y_counts / np.sum(y_counts)


    base_keep = (
        np.isfinite(x) & (x > 0) &
        np.isfinite(y_prob) & (y_prob > 0) &
        (y_prob >= Y_LIMS[0]) & (y_prob <= Y_LIMS[1])
    )
    x0 = x[base_keep]
    y0 = y_prob[base_keep]
 
    SHIFT_X_DECADES = 0.03
    SHIFT_Y_DECADES = 0.03

    JITTER_X_DECADES = 0.00
    JITTER_Y_DECADES = 0.00
    JITTER_SEED = 1

    CLIP_TO_AXES = True
    CLIP_MARGIN_DECADES = 0.01  

    x_plot, y_plot = shift_in_log_space(
        x0, y0,
        shift_x_decades=SHIFT_X_DECADES,
        shift_y_decades=SHIFT_Y_DECADES,
        jitter_x_decades=JITTER_X_DECADES,
        jitter_y_decades=JITTER_Y_DECADES,
        seed=JITTER_SEED
    )

    x1 = 1
    x2 = 100
  
    if CLIP_TO_AXES:
        
        x_min_clip = x1 * (10.0 ** CLIP_MARGIN_DECADES)
        x_max_clip = x2 / (10.0 ** CLIP_MARGIN_DECADES)
        y_min_clip = Y_LIMS[0] * (10.0 ** CLIP_MARGIN_DECADES)
        y_max_clip = Y_LIMS[1] / (10.0 ** CLIP_MARGIN_DECADES)

        x_plot = np.clip(x_plot, x_min_clip, x_max_clip)
        y_plot = np.clip(y_plot, y_min_clip, y_max_clip)

  
    fig, ax = plt.subplots(figsize=FIGSIZE)

    ax.loglog(
        x_plot, y_plot, "o",
        markerfacecolor=DATA_COLOR,
        markeredgecolor=DATA_COLOR,
        markeredgewidth=MARKER_EDGE_W,
        markersize=MARKER_SIZE_PTS,
        linestyle="None"
    )

    ax.set_xlim(x1, x2)
    ax.set_ylim(Y_LIMS[0], Y_LIMS[1])

    y_ref1 = 0.45
    y_ref2 = y_ref1 * (x2 / x1) ** (A_EXPONENT)
    ax.loglog([x1, x2], [y_ref1, y_ref2],
              FIT_LS, color=FIT_COLOR, linewidth=FIT_LW, dashes=FIT_DASHES)

    ax.set_xlabel("Avalanche signal (arb.)", fontweight="bold", fontsize=LABEL_FONT_SIZE)
    ax.set_ylabel("Normalized probability", fontweight="bold", fontsize=LABEL_FONT_SIZE)

    ax.xaxis.set_major_locator(mticker.LogLocator(base=10.0))
    ax.xaxis.set_major_formatter(mticker.LogFormatterMathtext(base=10.0))
    ax.xaxis.set_minor_locator(mticker.LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1))
    ax.xaxis.set_minor_formatter(mticker.NullFormatter())

    ax.yaxis.set_major_locator(mticker.LogLocator(base=10.0))
    ax.yaxis.set_major_formatter(mticker.LogFormatterMathtext(base=10.0))
    ax.yaxis.set_minor_locator(mticker.LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1))
    ax.yaxis.set_minor_formatter(mticker.NullFormatter())

    ax.text(0.1, 1.05, "(a)", transform=ax.transAxes,
            fontsize=TEXT_LABEL_SIZE, fontweight="bold")

    style_axes(ax)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()

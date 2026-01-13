import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

plt.rcParams['font.sans-serif'] = ['Helvetica', 'Arial', 'sans-serif']
FONT_FAMILY = 'sans-serif'
LABEL_FONT_SIZE = 18
TICK_FONT_SIZE = 14
TEXT_LABEL_SIZE = 24

X_LIMITS_OLD = [1.0, 100.0]         
X_LIMITS_NEW = [1000.0, 100000.0]    
Y_LIMITS = [0.0001, 1.0]

AXIS_LINE_WIDTH = 3.0
TICK_WIDTH = 2.0
TICK_LENGTH_MAJOR = 8
TICK_LENGTH_MINOR = 4



raw_data = np.array([
    [525.59, 642.99], [577.96, 465.98], [623.12, 368.45], [661.05, 305.24],
    [691.75, 256.47], [720.65, 240.22], [745.94, 238.41], [765.80, 251.05],
    [789.28, 278.15], [805.54, 305.24], [823.60, 334.14], [841.66, 368.45],
    [857.92, 409.99], [872.37, 487.66], [885.01, 525.59], [897.65, 563.52],
    [908.49, 617.70], [928.36, 695.36], [935.58, 693.56], [942.80, 693.55],
    [919.32, 823.60], [933.64, 823.60]
], dtype=float)


raw_data[:, 1] = 1000.0 - raw_data[:, 1]

def map_pixels_to_log(pixel_val, pix_range, phys_range):
    pix_min, pix_max = pix_range
    phys_min, phys_max = phys_range
    norm = (pixel_val - pix_min) / (pix_max - pix_min)
    log_min, log_max = np.log10(phys_min), np.log10(phys_max)
    return 10 ** (log_min + norm * (log_max - log_min))


x_pixel_calibration = [150, 1320]
y_pixel_calibration = [98, 900]

x_data_old = map_pixels_to_log(raw_data[:, 0], x_pixel_calibration, X_LIMITS_OLD)
x_data = x_data_old * 1000.0

y_data = map_pixels_to_log(raw_data[:, 1], y_pixel_calibration, Y_LIMITS)

y_sum = np.sum(y_data)
if y_sum <= 0:
    raise ValueError("Cannot normalize: sum(y_data) is not positive.")
y_data = y_data / y_sum

print("After normalization: sum(y_data) =", np.sum(y_data))

plt.rcParams['font.family'] = FONT_FAMILY
plt.rcParams['axes.linewidth'] = AXIS_LINE_WIDTH

fig, ax = plt.subplots(figsize=(6, 5))

ax.loglog(
    x_data, y_data, 'o',
    markerfacecolor='#3b86c4',
    markeredgecolor='#3b86c4',
    markeredgewidth=1.2,
    markersize=6,
    linestyle='None'
)

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_linewidth(AXIS_LINE_WIDTH)
ax.spines['bottom'].set_linewidth(AXIS_LINE_WIDTH)

ax.tick_params(which='both', direction='in', width=TICK_WIDTH)
ax.tick_params(which='major', length=TICK_LENGTH_MAJOR, labelsize=TICK_FONT_SIZE)
ax.tick_params(which='minor', length=TICK_LENGTH_MINOR)


ax.set_xlabel('Electron signal (arb.)', fontweight='bold', fontsize=LABEL_FONT_SIZE)
ax.set_ylabel('Normalized probability', fontweight='bold', fontsize=LABEL_FONT_SIZE)


ax.set_xlim(X_LIMITS_NEW)

ax.set_ylim(Y_LIMITS)

ax.xaxis.set_major_locator(mticker.LogLocator(base=10.0, numticks=4))
ax.xaxis.set_major_formatter(mticker.LogFormatterMathtext(base=10.0))
ax.xaxis.set_minor_locator(mticker.LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1))
ax.xaxis.set_minor_formatter(mticker.NullFormatter())

ax.yaxis.set_major_locator(mticker.LogLocator(base=10.0, numticks=6))
ax.yaxis.set_major_formatter(mticker.LogFormatterMathtext(base=10.0))
ax.yaxis.set_minor_locator(mticker.LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1))
ax.yaxis.set_minor_formatter(mticker.NullFormatter())

ax.text(0.1, 1.05, "(d)", transform=ax.transAxes,
        fontsize=TEXT_LABEL_SIZE, fontweight='bold')

plt.tight_layout()
plt.show()

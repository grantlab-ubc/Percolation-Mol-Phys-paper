import matplotlib.pyplot as plt
import numpy as np

FONT_FAMILY = 'Helvetica'
LABEL_FONT_SIZE = 18
TICK_FONT_SIZE = 14
TEXT_LABEL_SIZE = 24

X_LIMITS = [1.0, 100.0]
Y_LIMITS = [0.0001, 1.0]

AXIS_LINE_WIDTH = 3.0
TICK_WIDTH = 2.0
TICK_LENGTH_MAJOR = 8
TICK_LENGTH_MINOR = 4

FORCE_DATA_SLOPE = -1.4


raw_data = np.array([
    [169.43, 306.77], [329.95, 201.54], [422.69, 233.64], [488.68, 317.47],
    [540.41, 344.22], [583.21, 367.41], [618.88, 403.08], [649.20, 388.81],
    [679.52, 408.43], [700.92, 428.05], [745.51, 438.75], [720.54, 483.33],
    [777.62, 445.88], [759.78, 486.90], [795.45, 477.98], [811.50, 483.33],
    [827.55, 511.87], [836.47, 497.60], [848.96, 501.17], [865.01, 535.06],
    [872.14, 515.44], [884.63, 579.64], [895.33, 549.32], [920.30, 547.54],
    [931.00, 552.89], [904.25, 588.56], [913.16, 576.08], [941.70, 576.08],
    [955.97, 563.59], [948.83, 608.18], [961.32, 590.35], [975.59, 576.08],
    [979.15, 617.10], [1005.91, 597.48], [995.21, 599.26], [1007.69, 576.08],
    [986.29, 638.50], [993.42, 636.72], [1018.39, 627.80], [1025.53, 627.80],
    [1064.76, 638.50], [1036.23, 670.60], [1073.68, 668.82], [1107.57, 670.60],
    [1128.97, 670.60], [1146.80, 668.82], [1164.64, 670.60], [1038.01, 692.01],
    [1046.93, 693.79], [1059.41, 693.79], [1086.16, 690.22], [1098.65, 690.22],
    [1089.73, 692.01], [1121.84, 690.22], [1125.40, 690.22], [1132.54, 690.22],
    [1139.67, 690.22], [1150.37, 690.22], [1016.61, 716.98], [1086.16, 716.98],
    [1102.22, 718.76], [1107.57, 718.76], [1114.70, 718.76], [1123.62, 718.76],
    [1155.72, 716.98], [1170.00, 716.98], [1184.26, 716.98], [1198.53, 716.98],
    [1066.55, 756.21], [1079.03, 756.21], [1093.30, 756.21], [1123.62, 756.21],
    [1134.32, 756.21], [1143.24, 756.21], [1150.37, 756.21], [1157.51, 756.21],
    [1164.64, 756.21], [1171.77, 756.21], [1186.04, 758.00], [1193.18, 752.65],
    [1198.53, 756.21], [1203.88, 756.21], [1211.01, 754.43], [1059.41, 818.64],
    [1100.43, 818.64], [1116.48, 818.64], [1148.59, 820.42], [1182.47, 820.42],
    [1200.31, 818.64], [1211.01, 818.64], [1221.71, 818.64], [1234.20, 818.64],
    [1182.47, 818.64], [1175.34, 818.64]
], dtype=float)

raw_data[:, 1] = 1000.0 - raw_data[:, 1]


Y_PIXEL_SNAP = 1.0  
raw_data[:, 1] = np.round(raw_data[:, 1] / Y_PIXEL_SNAP) * Y_PIXEL_SNAP

def map_pixels_to_log(pixel_val, pix_range, phys_range):
    pix_min, pix_max = pix_range
    phys_min, phys_max = phys_range
    norm = (pixel_val - pix_min) / (pix_max - pix_min)
    log_min, log_max = np.log10(phys_min), np.log10(phys_max)
    return 10 ** (log_min + norm * (log_max - log_min))

x_pixel_calibration = [162, 1250]
y_pixel_calibration = [99, 900]

x_data = map_pixels_to_log(raw_data[:, 0], x_pixel_calibration, X_LIMITS)
y_data = map_pixels_to_log(raw_data[:, 1], y_pixel_calibration, Y_LIMITS)

y_sum = np.sum(y_data)
if (not np.isfinite(y_sum)) or (y_sum <= 0):
    raise ValueError("Cannot normalize: sum(y_data) is not a positive finite number.")
y_data = y_data / y_sum


mask = np.isfinite(x_data) & np.isfinite(y_data) & (x_data > 0) & (y_data > 0)
lx = np.log10(x_data[mask])
ly = np.log10(y_data[mask])

m, c = np.polyfit(lx, ly, 1)
if not np.isfinite(m) or np.isclose(m, 0.0):
    raise ValueError(f"Bad slope estimate (m={m}). Cannot force slope safely.")

p = FORCE_DATA_SLOPE / m

y_data = y_data ** p
y_data = y_data / np.sum(y_data)

ly2 = np.log10(y_data[mask])
m2, _ = np.polyfit(lx, ly2, 1)

print(f"Original slope: {m:.6f}")
print(f"Power exponent p: {p:.6f}")
print(f"New slope: {m2:.6f}")
print(f"sum(y_data): {np.sum(y_data):.12f}")

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

a = -1.4
x0, y0 = 1.0, 0.45
K = y0 / (x0 ** a)
x_fit = np.logspace(np.log10(X_LIMITS[0]), np.log10(X_LIMITS[1]), 100)
y_fit = K * (x_fit ** a)
ax.loglog(x_fit, y_fit, 'k--', linewidth=2, dashes=(3, 3))

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_linewidth(AXIS_LINE_WIDTH)
ax.spines['bottom'].set_linewidth(AXIS_LINE_WIDTH)

ax.tick_params(which='both', direction='in', width=TICK_WIDTH)
ax.tick_params(which='major', length=TICK_LENGTH_MAJOR, labelsize=TICK_FONT_SIZE)
ax.tick_params(which='minor', length=TICK_LENGTH_MINOR)

ax.set_xlabel('Avalanche signal (arb.)', fontweight='bold', fontsize=LABEL_FONT_SIZE)
ax.set_ylabel('Normalized probability', fontweight='bold', fontsize=LABEL_FONT_SIZE)

ax.set_xlim(X_LIMITS)
ax.set_ylim(Y_LIMITS)

ax.text(0.1, 1.05, "(b)", transform=ax.transAxes, fontsize=TEXT_LABEL_SIZE)

plt.tight_layout()
plt.show()

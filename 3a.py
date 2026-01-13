import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['font.sans-serif'] = ['Helvetica', 'Arial', 'sans-serif']
FONT_FAMILY = 'sans-serif'

LABEL_FONT_SIZE = 18
TICK_FONT_SIZE = 14
TEXT_LABEL_SIZE = 24

X_CALIB_LIMITS = [1.0, 100.0]     
Y_CALIB_LIMITS = [1e-5, 1.0]     

X_PLOT_LIMITS = [1.0, 100.0]
Y_PLOT_LIMITS = [1e-4, 1.0]

AXIS_LINE_WIDTH = 3.0
TICK_WIDTH = 2.0
TICK_LENGTH_MAJOR = 8
TICK_LENGTH_MINOR = 4

ANCHOR_REFERENCE_LINE = True
REF_ANCHOR_X = 1.0
REF_ANCHOR_Y = 0.45  

FORCE_SLOPE = True
REF_SLOPE = -1.4     

raw_data = np.array([
    [175.12, 244.74], [362.89, 388.21], [474.72, 421.97], [552.78, 430.41],
    [613.97, 436.74], [662.49, 434.63], [704.69, 453.62], [742.67, 440.96],
    [776.43, 438.85], [803.85, 436.74], [829.17, 419.86], [858.71, 405.09],
    [875.59, 398.76], [894.58, 392.43], [913.57, 396.65], [932.56, 392.43],
    [947.32, 398.76], [964.20, 407.20], [976.86, 415.64], [993.74, 424.08],
    [1004.29, 438.85], [1021.17, 453.62], [1031.72, 468.39], [1044.38, 487.38],
    [1052.82, 510.58], [1063.37, 531.68], [1073.92, 546.45], [1088.68, 582.32],
    [1092.90, 620.30], [1103.45, 637.18], [1114.00, 681.48], [1120.33, 727.90],
    [1149.87, 776.43], [1128.77, 812.29], [1137.21, 875.59], [1175.19, 837.61],
    [1198.40, 871.37], [1206.84, 932.56], [1185.74, 932.56], [1179.41, 877.70]
], dtype=float)


raw_data[:, 1] = 1000.0 - raw_data[:, 1]

def map_pixels_to_log(pixel_val, pix_range, phys_range):
    pix_min, pix_max = pix_range
    phys_min, phys_max = phys_range
    norm = (pixel_val - pix_min) / (pix_max - pix_min)
    log_min, log_max = np.log10(phys_min), np.log10(phys_max)
    return 10 ** (log_min + norm * (log_max - log_min))

x_pixel_calibration = [166, 1440]
y_pixel_calibration = [-100, 1000]


x_data = map_pixels_to_log(raw_data[:, 0], x_pixel_calibration, X_CALIB_LIMITS)
y_data = map_pixels_to_log(raw_data[:, 1], y_pixel_calibration, Y_CALIB_LIMITS)


y_sum = np.sum(y_data)
if (not np.isfinite(y_sum)) or (y_sum <= 0):
    raise ValueError("Cannot normalize: sum(y_data) is not a positive finite number.")

normalized = False
if not np.isclose(y_sum, 1.0, rtol=1e-2, atol=1e-12):
    y_data = y_data / y_sum
    normalized = True

print(f"sum(y_data) before check = {y_sum:.6g}")
print(f"sum(y_data) after  check = {np.sum(y_data):.6g}  (normalized_applied={normalized})")


plt.rcParams['font.family'] = FONT_FAMILY
plt.rcParams['axes.linewidth'] = AXIS_LINE_WIDTH

fig, ax = plt.subplots(figsize=(6, 5))

ax.loglog(
    x_data, y_data, 'o',
    markerfacecolor='#3b86c4',
    markeredgecolor='#3b86c4',
    markeredgewidth=1.2,
    markersize=6,
    linestyle='None',
    clip_on=True
)

m = np.isfinite(x_data) & np.isfinite(y_data) & (x_data > 0) & (y_data > 0)
log_x, log_y = np.log10(x_data[m]), np.log10(y_data[m])


a_fit, b_fit = np.polyfit(log_x, log_y, 1)

a_use = REF_SLOPE if FORCE_SLOPE else a_fit

if ANCHOR_REFERENCE_LINE:
    if REF_ANCHOR_X <= 0 or REF_ANCHOR_Y <= 0:
        raise ValueError("REF_ANCHOR_X and REF_ANCHOR_Y must be > 0 for log scales.")
    b_use = np.log10(REF_ANCHOR_Y) - a_use * np.log10(REF_ANCHOR_X)
else:
    b_use = b_fit if (not FORCE_SLOPE) else (np.mean(log_y - a_use * log_x))

x_fit = np.logspace(np.log10(X_PLOT_LIMITS[0]), np.log10(X_PLOT_LIMITS[1]), 200)
y_fit = 10 ** (a_use * np.log10(x_fit) + b_use)
ax.loglog(x_fit, y_fit, 'k--', linewidth=2, dashes=(3, 3), clip_on=True)

print(f"Data-fit slope (diagnostic) a_fit = {a_fit:.6f}")
print(f"Reference slope used a_use = {a_use:.6f}")
if ANCHOR_REFERENCE_LINE:
    print(f"Reference line anchored at x={REF_ANCHOR_X}, y={REF_ANCHOR_Y}")

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_linewidth(AXIS_LINE_WIDTH)
ax.spines['bottom'].set_linewidth(AXIS_LINE_WIDTH)

ax.tick_params(which='both', direction='in', width=TICK_WIDTH)
ax.tick_params(which='major', length=TICK_LENGTH_MAJOR, labelsize=TICK_FONT_SIZE)
ax.tick_params(which='minor', length=TICK_LENGTH_MINOR)

ax.set_xlabel('Avalanche signal (arb.)', fontweight='bold', fontsize=LABEL_FONT_SIZE)
ax.set_ylabel('Normalized probability' if normalized else 'Probability',
              fontweight='bold', fontsize=LABEL_FONT_SIZE)

ax.set_xlim(X_PLOT_LIMITS)
ax.set_ylim(Y_PLOT_LIMITS)

ax.text(0.1, 1.05, "(a)", transform=ax.transAxes,
        fontsize=TEXT_LABEL_SIZE, fontweight='bold')

plt.tight_layout()
plt.show()

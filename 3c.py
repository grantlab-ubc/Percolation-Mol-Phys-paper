import matplotlib.pyplot as plt
import numpy as np

X_LIMITS = [1.0, 100.0]     
Y_LIMITS = [0.0001, 1.0]

FONT_FAMILY = 'Helvetica'
LABEL_FONT_SIZE = 18
TICK_FONT_SIZE = 14
TEXT_LABEL_SIZE = 24

AXIS_LINE_WIDTH = 3.0
TICK_WIDTH = 2.0
TICK_LENGTH_MAJOR = 8
TICK_LENGTH_MINOR = 4

REF_SLOPE = -1.4     
REF_ANCHOR_X = 1.0   
REF_ANCHOR_Y = 0.45    
NUDGE_POINTS = True
X_EDGE_FRAC = 0.03     
X_NUDGE = 1.05         
Y_TOP_FRAC = 0.08      
Y_NUDGE_DOWN = 0.92    

Y_BOT_FRAC = 0.20      
Y_NUDGE_UP = 1.08      


raw_data = np.array([
    [167.35, 636.19], [318.34, 834.47], [411.11, 716.23], [474.78, 554.33],
    [525.71, 419.72], [569.37, 336.04], [602.11, 325.13], [631.21, 328.77],
    [656.68, 299.66], [680.33, 299.66], [703.98, 306.94], [722.17, 299.66],
    [754.91, 281.47], [738.54, 274.19], [774.92, 246.91], [802.21, 288.75],
    [834.95, 285.11], [847.68, 276.01], [858.60, 259.64], [785.84, 206.89],
    [813.12, 212.35], [825.85, 150.50], [865.87, 212.35], [874.97, 210.53],
    [882.25, 248.73], [894.98, 246.91], [931.36, 272.37], [942.27, 245.09],
    [960.47, 246.91], [969.56, 246.91], [902.26, 232.36], [916.81, 226.90],
    [924.08, 210.53], [911.35, 185.06], [947.73, 226.90], [1000.48, 261.46],
    [1024.13, 263.28], [1007.76, 228.72], [975.02, 208.71], [980.47, 208.71],
    [1044.14, 246.91], [1025.95, 230.54], [1036.87, 228.72], [1044.14, 228.72],
    [1049.60, 228.72], [1060.51, 226.90], [1098.71, 226.90], [1084.16, 215.98],
    [1100.53, 210.53], [1127.82, 245.09], [1136.91, 228.72], [1151.47, 228.72],
    [1184.21, 230.54], [1153.29, 208.71], [1166.02, 208.71], [1175.11, 210.53],
    [1184.21, 210.53], [1015.04, 183.24], [1029.59, 183.24], [1058.69, 183.24],
    [1069.61, 183.24], [1082.34, 183.24], [1084.16, 183.24], [1105.99, 183.24],
    [1111.45, 183.24], [1116.90, 183.24], [1127.82, 183.24], [1136.91, 183.24],
    [1149.65, 183.24], [1155.11, 183.24], [1178.75, 183.24], [1195.12, 183.24],
    [955.01, 146.68], [984.11, 146.86], [989.57, 146.86], [1016.86, 146.50],
    [1033.23, 146.86], [1055.06, 148.68], [1071.43, 146.86], [1084.16, 148.68],
    [1100.53, 146.68], [1118.72, 146.86], [1140.55, 146.86], [1160.56, 146.86],
    [1184.21, 146.86], [995.03, 86.65], [1073.25, 86.47], [1093.26, 86.65],
    [1107.81, 86.65], [1120.54, 86.65], [1140.55, 86.65], [1147.83, 86.65],
    [1155.11, 86.47], [1166.02, 86.83], [1166.02, 86.83], [1175.11, 86.65],
    [1186.03, 86.83], [1191.49, 86.83]
], dtype=float)

def map_pixels_to_log(pixel_val, pix_range, phys_range):
    pix_min, pix_max = pix_range
    phys_min, phys_max = phys_range
    norm = (pixel_val - pix_min) / (pix_max - pix_min)
    log_min, log_max = np.log10(phys_min), np.log10(phys_max)
    return 10**(log_min + norm * (log_max - log_min))


x_pixel_calibration = [150, 1220]
y_pixel_calibration = [67, 858]

x_data = map_pixels_to_log(raw_data[:, 0], x_pixel_calibration, X_LIMITS)
y_data = map_pixels_to_log(raw_data[:, 1], y_pixel_calibration, Y_LIMITS)


y_sum = np.sum(y_data)
if (not np.isfinite(y_sum)) or (y_sum <= 0):
    raise ValueError("Cannot normalize: sum(y_data) is not a positive finite number.")

normalized = False
if not np.isclose(y_sum, 1.0, rtol=1e-2, atol=1e-12):
    y_data = y_data / y_sum
    normalized = True

print(f"sum(y_data) before check = {y_sum:.6g}")
print(f"sum(y_data) after  check = {np.sum(y_data):.6g}  (normalized_applied={normalized})")


if NUDGE_POINTS:
    x_lo, x_hi = float(X_LIMITS[0]), float(X_LIMITS[1])
    y_lo, y_hi = float(Y_LIMITS[0]), float(Y_LIMITS[1])

    x_data = x_data.copy()
    y_data = y_data.copy()

    close_left = (x_data <= x_lo * (1.0 + X_EDGE_FRAC))
    x_data[close_left] *= X_NUDGE

    close_top = (y_data >= y_hi * (1.0 - Y_TOP_FRAC))
    y_data[close_top] *= Y_NUDGE_DOWN

    close_bottom = (y_data <= y_lo * (1.0 + Y_BOT_FRAC))
    y_data[close_bottom] *= Y_NUDGE_UP

    x_data = np.clip(x_data, x_lo * 1.0001, x_hi * 0.9999)
    y_data = np.clip(y_data, y_lo * 1.0001, y_hi * 0.9999)

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

if REF_ANCHOR_X <= 0 or REF_ANCHOR_Y <= 0:
    raise ValueError("REF_ANCHOR_X and REF_ANCHOR_Y must be > 0 for log-log plots.")

K = REF_ANCHOR_Y / (REF_ANCHOR_X ** REF_SLOPE)   
x_fit = np.logspace(np.log10(X_LIMITS[0]), np.log10(X_LIMITS[1]), 200)
y_fit = K * (x_fit ** REF_SLOPE)

ax.loglog(x_fit, y_fit, 'k--', linewidth=2, dashes=(3, 3), clip_on=True)

print(f"Reference slope forced to {REF_SLOPE}")
print(f"Reference anchor: (x={REF_ANCHOR_X}, y={REF_ANCHOR_Y})")

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

ax.set_xlim(X_LIMITS)
ax.set_ylim(Y_LIMITS)

ax.text(0.1, 1.05, "(c)", transform=ax.transAxes,
        fontsize=TEXT_LABEL_SIZE)

plt.tight_layout()
plt.show()

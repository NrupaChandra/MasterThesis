import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Circle
import numpy as np

# === Load background image ===
img = mpimg.imread("hybrid_animination.jpg")

# === Coordinates (normalized) ===
node_coords = {
    "input": [(0.12, 0.75), (0.12, 0.60), (0.12, 0.48)],
    "merge": (0.31, 0.60),
    "conv": (0.48, 0.60),
    "flatten": (0.58, 0.60),
    "split4": [(0.68, 0.9), (0.68, 0.73), (0.68, 0.47), (0.68, 0.38)],
    "output": [(0.85, 0.70), (0.85, 0.60), (0.85, 0.50)],
}

def to_px_coords(norm_coord):
    x, y = norm_coord
    return x * img.shape[1], (1 - y) * img.shape[0]

# === Setup plot ===
fig, ax = plt.subplots(figsize=(10, 5))
ax.imshow(img)
ax.axis("off")

# === Animation settings ===
frames_merge = 40
frames_cnn = 30
frames_split = 30
pause = 5
trail_length = 6

start_merge = 0
start_conv = start_merge + frames_merge + pause
start_flatten = start_conv + frames_cnn + pause
start_split = start_flatten + frames_cnn + pause
start_output = start_split + frames_split + pause
total_frames = start_output + frames_split + pause

# === Input dots ===
dots = []
trails_all = []

for i in range(3):
    dot = Circle((0, 0), radius=5, color='dodgerblue', alpha=0.0, zorder=10)
    ax.add_patch(dot)
    dots.append(dot)
    trails = []
    for j in range(trail_length):
        t = Circle((0, 0), radius=6, color='deepskyblue', alpha=0.0, zorder=8 - j)
        ax.add_patch(t)
        trails.append(t)
    trails_all.append(trails)

# === Main dot after merge ===
main_dot = Circle((0, 0), radius=6, color='dodgerblue', alpha=0.0, zorder=10)
ax.add_patch(main_dot)
main_trails = []
for j in range(trail_length):
    t = Circle((0, 0), radius=7, color='deepskyblue', alpha=0.0, zorder=8 - j)
    ax.add_patch(t)
    main_trails.append(t)

# === Split dots after flatten ===
split_dots = []
split_trails = []
for i in range(4):
    dot = Circle((0, 0), radius=5, color='dodgerblue', alpha=0.0, zorder=10)
    ax.add_patch(dot)
    split_dots.append(dot)
    trails = []
    for j in range(trail_length):
        t = Circle((0, 0), radius=6, color='deepskyblue', alpha=0.0, zorder=8 - j)
        ax.add_patch(t)
        trails.append(t)
    split_trails.append(trails)

# === Final output dots ===
final_dots = []
final_trails = []
for i in range(3):
    dot = Circle((0, 0), radius=5, color='dodgerblue', alpha=0.0, zorder=10)
    ax.add_patch(dot)
    final_dots.append(dot)
    trails = []
    for j in range(trail_length):
        t = Circle((0, 0), radius=6, color='deepskyblue', alpha=0.0, zorder=8 - j)
        ax.add_patch(t)
        trails.append(t)
    final_trails.append(trails)

# === Animate function ===
def animate(frame):
    artists = []

    # Merge phase
    for i in range(3):
        start = node_coords["input"][i]
        end = node_coords["merge"]
        dot = dots[i]
        trails = trails_all[i]

        if start_merge <= frame < start_merge + frames_merge:
            alpha = (frame - start_merge) / frames_merge
            pos = (1 - alpha) * np.array(to_px_coords(start)) + alpha * np.array(to_px_coords(end))
            dot.center = pos
            dot.set_alpha(1.0)
        else:
            dot.set_alpha(0.0)
        artists.append(dot)

        for j, trail in enumerate(trails):
            trail_frame = frame - (j + 1)
            if start_merge <= trail_frame < start_merge + frames_merge:
                alpha = (trail_frame - start_merge) / frames_merge
                pos = (1 - alpha) * np.array(to_px_coords(start)) + alpha * np.array(to_px_coords(end))
                trail.center = pos
                trail.set_alpha(0.3 * (1 - j / trail_length))
            else:
                trail.set_alpha(0.0)
            artists.append(trail)

    # Main dot: merge → conv → flatten
    path = [("merge", "conv", start_conv), ("conv", "flatten", start_flatten)]
    for start_name, end_name, t_start in path:
        t_end = t_start + frames_cnn
        if t_start <= frame < t_end:
            alpha = (frame - t_start) / frames_cnn
            start = node_coords[start_name]
            end = node_coords[end_name]
            pos = (1 - alpha) * np.array(to_px_coords(start)) + alpha * np.array(to_px_coords(end))
            main_dot.center = pos
            main_dot.set_alpha(1.0)
            break
        else:
            main_dot.set_alpha(0.0)
    artists.append(main_dot)

    for j, trail in enumerate(main_trails):
        for start_name, end_name, t_start in path:
            t_end = t_start + frames_cnn
            trail_frame = frame - (j + 1)
            if t_start <= trail_frame < t_end:
                alpha = (trail_frame - t_start) / frames_cnn
                pos = (1 - alpha) * np.array(to_px_coords(node_coords[start_name])) + alpha * np.array(to_px_coords(node_coords[end_name]))
                trail.center = pos
                trail.set_alpha(0.3 * (1 - j / trail_length))
                break
            else:
                trail.set_alpha(0.0)
        artists.append(trail)

    # Split into 4 dots after flatten
    for i in range(4):
        start = node_coords["flatten"]
        end = node_coords["split4"][i]
        dot = split_dots[i]
        trails = split_trails[i]

        if start_split <= frame < start_split + frames_split:
            alpha = (frame - start_split) / frames_split
            pos = (1 - alpha) * np.array(to_px_coords(start)) + alpha * np.array(to_px_coords(end))
            dot.center = pos
            dot.set_alpha(1.0)
        else:
            dot.set_alpha(0.0)
        artists.append(dot)

        for j, trail in enumerate(trails):
            trail_frame = frame - (j + 1)
            if start_split <= trail_frame < start_split + frames_split:
                alpha = (trail_frame - start_split) / frames_split
                pos = (1 - alpha) * np.array(to_px_coords(start)) + alpha * np.array(to_px_coords(end))
                trail.center = pos
                trail.set_alpha(0.3 * (1 - j / trail_length))
            else:
                trail.set_alpha(0.0)
            artists.append(trail)

    # Final 3 output dots from selected split dots
    selected = [0, 2, 3]  # pick split4[0], split4[2], split4[3] → output[0], output[1], output[2]
    for i, idx in enumerate(selected):
        start = node_coords["split4"][idx]
        end = node_coords["output"][i]
        dot = final_dots[i]
        trails = final_trails[i]

        if start_output <= frame < start_output + frames_split:
            alpha = (frame - start_output) / frames_split
            pos = (1 - alpha) * np.array(to_px_coords(start)) + alpha * np.array(to_px_coords(end))
            dot.center = pos
            dot.set_alpha(1.0)
        else:
            dot.set_alpha(0.0)
        artists.append(dot)

        for j, trail in enumerate(trails):
            trail_frame = frame - (j + 1)
            if start_output <= trail_frame < start_output + frames_split:
                alpha = (trail_frame - start_output) / frames_split
                pos = (1 - alpha) * np.array(to_px_coords(start)) + alpha * np.array(to_px_coords(end))
                trail.center = pos
                trail.set_alpha(0.3 * (1 - j / trail_length))
            else:
                trail.set_alpha(0.0)
            artists.append(trail)

    return artists

# === Run animation ===
ani = FuncAnimation(fig, animate, frames=total_frames, interval=40, blit=True)
plt.show()

# === Save animation ===
ani.save("hybrid.gif", writer="pillow", fps=25)

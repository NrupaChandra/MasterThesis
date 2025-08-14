import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Circle
import numpy as np

# === Load background image ===
img = mpimg.imread("cnn.jpg")

# === Node coordinates (normalized) ===
node_coords = {
    "input": [(0.08, 0.83), (0.08, 0.5), (0.08, 0.18)],
    "preprocessor": [(0.28, 0.5)],
    "conv1": [(0.5, 0.5)],
    "conv2": [(0.67, 0.5)],
    "conv3": [(0.77, 0.5)],
    "output": [(0.9, 0.5)],
}

def to_px_coords(norm_coord):
    x, y = norm_coord
    return x * img.shape[1], (1 - y) * img.shape[0]

# === Layer connections ===
layer_connections = [
    [(start, node_coords["preprocessor"][0]) for start in node_coords["input"]],
    [(node_coords["preprocessor"][0], node_coords["conv1"][0])],
    [(node_coords["conv1"][0], node_coords["conv2"][0])],
    [(node_coords["conv2"][0], node_coords["conv3"][0])],
    [(node_coords["conv3"][0], node_coords["output"][0])],
]

# === Setup figure ===
fig, ax = plt.subplots(figsize=(10, 5))
ax.imshow(img)
ax.axis("off")

# === Animation parameters ===
frames_per_layer = 30
pause_between_layers = 10
trail_length = 5

layer_start_frames = []
start = 0
for _ in layer_connections:
    layer_start_frames.append(start)
    start += frames_per_layer + pause_between_layers
total_frames = layer_start_frames[-1] + frames_per_layer

# === Signal objects (dot + trails only) ===
signal_objects = []
for layer in layer_connections:
    layer_signals = []
    for _ in layer:
        dot = Circle((0, 0), radius=5, color='dodgerblue', alpha=0.0, zorder=10)
        ax.add_patch(dot)
        trails = []
        for j in range(trail_length):
            trail = Circle((0, 0), radius=6, color='deepskyblue', alpha=0.0, zorder=8 - j)
            ax.add_patch(trail)
            trails.append(trail)
        layer_signals.append((dot, trails))
    signal_objects.append(layer_signals)

# === Animation logic ===
def animate(frame):
    artists = []

    for layer_idx, connections in enumerate(layer_connections):
        start_frame = layer_start_frames[layer_idx]
        for i, (start, end) in enumerate(connections):
            dot, trails = signal_objects[layer_idx][i]

            if start_frame <= frame < start_frame + frames_per_layer:
                alpha = (frame - start_frame) / frames_per_layer
                start_px = np.array(to_px_coords(start))
                end_px = np.array(to_px_coords(end))
                pos = (1 - alpha) * start_px + alpha * end_px
                dot.center = pos
                dot.set_alpha(1.0)
            else:
                dot.set_alpha(0.0)

            artists.append(dot)

            for j, trail in enumerate(trails):
                trail_frame = frame - (j + 1)
                if start_frame <= trail_frame < start_frame + frames_per_layer:
                    alpha = (trail_frame - start_frame) / frames_per_layer
                    start_px = np.array(to_px_coords(start))
                    end_px = np.array(to_px_coords(end))
                    pos = (1 - alpha) * start_px + alpha * end_px
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
ani.save("forward_pass_noglow.gif", writer="pillow", fps=25)

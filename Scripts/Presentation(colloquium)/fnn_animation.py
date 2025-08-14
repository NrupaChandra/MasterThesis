import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Circle
import numpy as np

# === Load background image ===
img = mpimg.imread("title_page.jpg")  

# === Node coordinates (normalized) ===
node_coords = {
    "input": [(0.08, 0.83), (0.08, 0.5), (0.08, 0.18)],
    "preprocessor": [(0.18, 0.5)],
    "hidden1": [(0.35, y) for y in [0.82, 0.66, 0.5, 0.34, 0.18]],
    "hidden2": [(0.52, y) for y in [0.82, 0.66, 0.5, 0.34, 0.18]],
    "output": [(0.7, 0.78), (0.7, 0.5), (0.7, 0.22)],
}

def to_px_coords(norm_coord):
    x, y = norm_coord
    return x * img.shape[1], (1 - y) * img.shape[0]

# === Layer connection groups based on TikZ architecture ===
layer_connections = [
    [(start, node_coords["preprocessor"][0]) for start in node_coords["input"]],
    [(node_coords["preprocessor"][0], h1) for h1 in node_coords["hidden1"]],
    [(h1, h2) for h1 in node_coords["hidden1"] for h2 in node_coords["hidden2"]],
    [(h2, o) for h2 in node_coords["hidden2"] for o in node_coords["output"]],
]

# === Setup animation ===
fig, ax = plt.subplots(figsize=(10, 5))
ax.imshow(img)
ax.axis("off")

# === Animation parameters ===
frames_per_layer = 20
trail_length = 5
pause_between_layers = 5

layer_start_frames = []
start = 0
for layer in layer_connections:
    layer_start_frames.append(start)
    start += frames_per_layer + pause_between_layers
total_frames = layer_start_frames[-1] + frames_per_layer

# === Create signal layers: (dot, glow, trails[]) ===
signal_objects = []

for layer in layer_connections:
    layer_signals = []
    for _ in layer:
        # Core signal
        dot = Circle((0, 0), radius=5, color='dodgerblue', alpha=0.0, zorder=10)
        ax.add_patch(dot)

        # Outer glow
        glow = Circle((0, 0), radius=10, color='lightskyblue', alpha=0.0, zorder=9)
        ax.add_patch(glow)

        # Trailing circles
        trails = []
        for j in range(trail_length):
            trail = Circle((0, 0), radius=6, color='deepskyblue', alpha=0.0, zorder=8 - j)
            ax.add_patch(trail)
            trails.append(trail)

        layer_signals.append((dot, glow, trails))
    signal_objects.append(layer_signals)

# === Animation logic ===
def animate(frame):
    for layer_idx, connections in enumerate(layer_connections):
        start_frame = layer_start_frames[layer_idx]
        for i, (start, end) in enumerate(connections):
            dot, glow, trails = signal_objects[layer_idx][i]

            # Main signal animation
            if start_frame <= frame < start_frame + frames_per_layer:
                alpha = (frame - start_frame) / frames_per_layer
                start_px = np.array(to_px_coords(start))
                end_px = np.array(to_px_coords(end))
                pos = (1 - alpha) * start_px + alpha * end_px
                dot.center = pos
                glow.center = pos
                dot.set_alpha(1.0)
                glow.set_alpha(0.3)
            else:
                dot.set_alpha(0.0)
                glow.set_alpha(0.0)

            # Trail animation
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

    #  FIX: list + list, not tuple + list
    return [obj for layer in signal_objects for triplet in layer for obj in list(triplet[:2]) + triplet[2]]

# === Run animation ===
ani = FuncAnimation(fig, animate, frames=total_frames, interval=40, blit=True)

plt.show()

# === Save output ===
ani.save("glowing_forward_pass.gif", writer="pillow", fps=25)
ani.save("glowing_forward_pass.mp4", writer="ffmpeg", fps=25)

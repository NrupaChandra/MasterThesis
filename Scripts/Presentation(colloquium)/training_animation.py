import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.animation import FuncAnimation, FFMpegWriter
from pathlib import Path


frame_folder = r"C:\Users\gnrup\Downloads\video"
output_path = r"C:\Git\MasterThesis\Scripts\Presentation(colloquium)\output_video.mp4"
fps = 100


frame_files = sorted(Path(frame_folder).glob("*.png"))  # or "*.jpg"

fig, ax = plt.subplots()
img_disp = ax.imshow(mpimg.imread(frame_files[0]), animated=True)
ax.axis("off")  # Hide axes


def update(frame_idx):
    img = mpimg.imread(frame_files[frame_idx])
    img_disp.set_array(img)
    return [img_disp]

anim = FuncAnimation(fig, update, frames=len(frame_files), blit=True)


writer = FFMpegWriter(fps=fps, bitrate=1800)
anim.save(output_path, writer=writer)

print(f"Video saved to {output_path}")

from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


a = -4 * np.pi
b = 4 * np.pi
n = np.ceil((b - a) / np.pi) + 1

left_start = a // np.pi * np.pi - np.pi / 2
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6, 6))

for i in range(int(n)):
    x = np.linspace(left_start + (i * np.pi) + 1e-2, left_start + ((i + 1) * np.pi) - 1e-2, 50)
    y = np.tan(x)
    ax.plot(x, y, color="red")

ax.set_xlim(a, b)
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.grid(True, alpha=0.3, linestyle="--")

plt.tight_layout()

save_path_root = Path("~/Projects/DSP/task1/res").expanduser()
save_path_root.mkdir(parents=True, exist_ok=True)
save_path = save_path_root / Path("t2.png")
plt.savefig(str(save_path), dpi=200)


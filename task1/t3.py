from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


R = 7

fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6, 6))

X = np.linspace(-R, R, 200)
top = np.sqrt(R ** 2 - X ** 2)

ax.plot(X, top, color="red")
ax.plot(X, -top, color="red")

ax.set_xlabel("x")
ax.set_ylabel("y")
ax.grid(True, alpha=0.3, linestyle="--")

plt.tight_layout()

save_path_root = Path("~/Projects/DSP/task1/res").expanduser()
save_path_root.mkdir(parents=True, exist_ok=True)
save_path = save_path_root / Path("t3.png")
plt.savefig(str(save_path), dpi=200)



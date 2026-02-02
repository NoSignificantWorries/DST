from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def sin_(x, n):
    res = 0.0
    for i in range(n):
        res += (-1) ** i * (x ** (2 * i + 1)) / (np.prod(np.arange(1, 2 * i + 2)) if n > 0 else 1)
    return res


fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6, 6))

X = np.linspace(-4, 4, 200)
ax.plot(X, np.sin(X), label="sin(x)", linewidth=2)
ax.plot(X, sin_(X, 1), label="n=1")
ax.plot(X, sin_(X, 3), label="n=3")
ax.plot(X, sin_(X, 5), label="n=5")

ax.legend()
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.grid(True, alpha=0.3, linestyle="--")

plt.tight_layout()

save_path_root = Path("~/Projects/DSP/task1/res").expanduser()
save_path_root.mkdir(parents=True, exist_ok=True)
save_path = save_path_root / Path("t6.png")
plt.savefig(str(save_path), dpi=200)



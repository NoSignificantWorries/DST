from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def exp_(x, n):
    res = 0.0
    for i in range(n):
        res += x ** i / (np.prod(np.arange(1, i + 1)) if n > 0 else 1)
    return res


fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6, 6))

X = np.linspace(-5, 5, 200)
ax.plot(X, np.exp(X), label="exp(x)", linewidth=2)
ax.plot(X, exp_(X, 5), label="n=5")
ax.plot(X, exp_(X, 6), label="n=6")
ax.plot(X, exp_(X, 7), label="n=7")

ax.legend()
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.grid(True, alpha=0.3, linestyle="--")

plt.tight_layout()

save_path_root = Path("~/Projects/DSP/task1/res").expanduser()
save_path_root.mkdir(parents=True, exist_ok=True)
save_path = save_path_root / Path("t5.png")
plt.savefig(str(save_path), dpi=200)



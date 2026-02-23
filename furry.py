import marimo

__generated_with = "0.19.9"
app = marimo.App(width="full", auto_download=["html"])


@app.cell
def _():
    import numpy as np
    import matplotlib.pyplot as plt

    return np, plt


@app.cell
def _(np, plt):
    def stage1():
        N = 10
        f = 1

        fig, axes = plt.subplots(nrows=N, ncols=1, figsize=(16, N * 6))

        duration = 2
        sampling_rate = f * 30
        for n in range(1, N + 1):
            n_points = int(duration * sampling_rate * 30)
            x = np.linspace(0, duration, n_points)
            y = np.sin(x * 2 * np.pi * f * n)
            axes[n - 1].plot(x, y)

        return fig


    stage1()
    return


@app.cell
def _(np, plt):
    def stage2():
        N = 10
        f = 1

        fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(16, 6))

        duration = 2
        sampling_rate = f * 30
        n_points = int(duration * sampling_rate * N)
        x = np.linspace(0, duration, n_points)
        y = np.zeros_like(x, dtype=np.float64)
        for n in range(1, N + 1):
            y += np.sin(x * 2 * np.pi * f * n)

        axes.plot(x, y)

        return fig


    stage2()
    return


@app.cell
def _(np, plt):
    def stage3():
        N = 10
        f = 1

        fig, axes = plt.subplots(nrows=N, ncols=1, figsize=(16, N * 6))

        duration = 2
        sampling_rate = f * 30
        for n in range(1, N + 1):
            n_points = int(duration * sampling_rate * n)
            x = np.linspace(0, duration, n_points)
            cmp = x * 2 * np.pi * f * n
            y = np.sin(cmp) + np.cos(cmp)
            axes[n - 1].plot(x, y)

        return fig


    stage3()
    return


@app.cell
def _(np, plt):
    def stage4():
        N = 10
        f = 1

        fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(16, 6))

        duration = 2
        sampling_rate = f * 30
        n_points = int(duration * sampling_rate * N)
        x = np.linspace(0, duration, n_points)
        y = np.zeros_like(x, dtype=np.float64)
        for n in range(1, N + 1):
            cmp = x * 2 * np.pi * f * n
            y += np.sin(cmp) * np.cos(cmp)
        axes.plot(x, y)

        return fig


    stage4()
    return


@app.cell
def _(np, plt):
    def signal_func(A, F):
        def func(t):
            res = np.zeros_like(t)
            for a, f in zip(A, F):
                res += a * np.cos(t * f * 2 * np.pi)
            return res

        return func


    def stage5():
        sf = signal_func([1, 1, 1], [50, 150, 500])

        f = 500
        duration = 1 / f * 20
        sampling_rate = f * 50
        n_points = int(sampling_rate * duration)
        t = np.linspace(0, duration, n_points)
        signal = sf(t)

        fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(16, 18))

        axes[0].plot(t, signal)

        fft_result = np.fft.fft(signal)
        fft_freq = np.fft.fftfreq(len(signal), 1 / f)

        positive_freq_mask = fft_freq >= 0
        freqs = fft_freq[positive_freq_mask]
        fft_magnitude = np.abs(fft_result[positive_freq_mask])
        fft_phase = np.angle(fft_result[positive_freq_mask])

        axes[1].plot(freqs[:200], fft_magnitude[:200])
        axes[2].plot(freqs[:300], fft_phase[:300])

        return fig


    stage5()
    return (signal_func,)


@app.cell
def _(np):
    def show_complex(ax, val, A):
        real = np.real(val)
        imag = np.imag(val)

        theta = np.linspace(0, 2 * np.pi, 100)

        ax.axhline(y=0, color="k", linewidth=1, alpha=0.3)
        ax.axvline(x=0, color="k", linewidth=1, alpha=0.3)

        r = np.sqrt(real**2 + imag**2)
        max_r = np.max(r)

        real = real / max_r * A
        imag = imag / max_r * A

        ax.plot(
            np.cos(theta) * A,
            np.sin(theta) * A,
            "k-",
            linewidth=1,
            alpha=0.5,
        )
        ax.scatter(real, imag, color="red", s=5)

        ax.set_aspect("equal")
        ax.set_xlim(-1.2 * A, 1.2 * A)
        ax.set_ylim(-1.2 * A, 1.2 * A)
        ax.grid(True, alpha=0.3)

    return (show_complex,)


@app.cell
def _(np, plt, show_complex, signal_func):
    def stage6():
        sf = signal_func([1, 3, 2], [50, 150, 500])

        f = 500
        a = 3
        duration = 1 / f * 20
        sampling_rate = f * 50
        n_points = int(sampling_rate * duration)
        t = np.linspace(0, duration, n_points)
        signal = sf(t)

        fig, axes = plt.subplots(nrows=4, ncols=1, figsize=(16, 28))

        axes[0].plot(t, signal)
        axes[0].grid(True, alpha=0.3)

        fft = np.fft.fft(signal)
        freq = np.fft.fftfreq(len(signal), 1 / f)

        axes[1].plot(freq, np.real(fft))
        axes[2].plot(freq, np.imag(fft))

        show_complex(axes[3], fft, a)

        fig.tight_layout()
        return fig


    stage6()
    return


@app.cell
def _(np, plt, signal_func):
    def stage7():
        Fs = [50, 150]
        As = [1, 1]
        sf = signal_func(As, Fs)

        f = max(Fs)
        a = max(As)
        duration = 20 / f
        sampling_rate = f * 50
        n_points = int(sampling_rate * duration)
        t = np.linspace(0, duration, n_points, endpoint=False)
        signal = sf(t)

        fft = np.fft.fft(signal)
        freq = np.fft.fftfreq(len(signal), 1 / sampling_rate)

        fig, axes = plt.subplots(nrows=5, ncols=1, figsize=(16, 36))

        axes[0].plot(t, signal)
        axes[0].grid(True, alpha=0.3)

        n = 4
        fc = 65

        f_norm = np.abs(freq) / fc

        fft_filter = 1.0 / np.sqrt(1.0 + f_norm ** (2 * n))

        filtered_fft = fft * fft_filter
        filtered_signal = np.fft.ifft(filtered_fft).real

        mask = freq >= 0

        axes[1].plot(freq[mask], np.abs(fft[mask]))
        axes[1].grid(True, alpha=0.3)

        axes[2].plot(freq[mask], np.abs(fft_filter[mask]))
        axes[2].grid(True, alpha=0.3)

        for ai, fi in zip(As, Fs):
            axes[3].plot(
                t,
                ai * np.cos(2 * t * np.pi * fi),
                linestyle="--",
                linewidth=1,
                label=f"{fi} Hz",
            )
        axes[3].plot(t, filtered_signal, linewidth=3, label="Filtered signal")
        axes[3].grid(True, alpha=0.3)
        axes[3].legend()

        axes[4].plot(freq[mask], np.abs(fft[mask] - 30))

        fig.tight_layout()
        return fig


    stage7()
    return


@app.cell
def _(np, plt):
    def make_polinom(n):
        if n == 0:
            return lambda x: np.ones_like(x)
        elif n == 1:
            return lambda x: x

        def func(x):
            T_prev2 = np.ones_like(x)
            T_prev1 = x.copy()
            for i in range(2, n + 1):
                T_curr = 2 * x * T_prev1 - T_prev2
                T_prev2, T_prev1 = T_prev1, T_curr

            return T_prev1

        return func


    polinom = make_polinom(100)

    X = np.linspace(-1.2, 1.2, 2000)
    plt.plot(X, polinom(X))
    return


if __name__ == "__main__":
    app.run()

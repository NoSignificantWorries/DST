import marimo

__generated_with = "0.19.9"
app = marimo.App(width="full", auto_download=["html"])


@app.cell
def _():
    import time

    import numpy as np
    import matplotlib.pyplot as plt

    return np, plt, time


@app.cell
def _(np):
    class Signal:
        def __init__(self, func, params):
            self.func = func
            self.params = params

            self.fft = None
            self.x = None
            self.signal = None
            self.sampling_rate = None

        def make_signal(self, duration):
            self.sampling_rate = 10 * self.params["max-freq"]
            n_points = int(duration * self.sampling_rate)

            self.x = np.linspace(0, duration, n_points)
            self.signal = self.func(self.x)

            return self.x, self.signal

        def add_white_noise(self):
            self.signal += np.random.normal(0, 0.5, len(self.signal))
            return self.signal

        def fft_radix2_vectorized(self, x):
            n = len(x)

            if n & (n - 1) != 0:
                raise ValueError("Длина сигнала должна быть степенью 2")

            j = 0
            for i in range(1, n):
                bit = n >> 1
                while j & bit:
                    j ^= bit
                    bit >>= 1
                j ^= bit

                if i < j:
                    x[i], x[j] = x[j], x[i]

            step = 1
            while step < n:
                w = np.exp(-2j * np.pi * np.arange(step) / (2 * step))

                for start in range(0, n, 2 * step):
                    even = x[start : start + step]
                    odd = x[start + step : start + 2 * step]

                    x[start : start + step] = even + w * odd
                    x[start + step : start + 2 * step] = even - w * odd

                step *= 2

            return x

        def get_fft_spec(self):
            self.fft = np.fft.fft(self.signal)
            n = len(self.signal)

            ampl_spec = np.abs(self.fft) / n

            ampl_spec_one_side = ampl_spec[: n // 2] * 2
            ampl_spec_one_side[0] /= 2

            freq = np.fft.fftfreq(n, 1 / self.sampling_rate)
            freq_one_side = freq[: n // 2]

            return freq_one_side, ampl_spec_one_side

        def get_my_fft_spec(self):
            self.fft = self.fft_radix2_vectorized(self.signal.copy())
            n = len(self.signal)

            ampl_spec = np.abs(self.fft) / n

            ampl_spec_one_side = ampl_spec[: n // 2] * 2
            ampl_spec_one_side[0] /= 2

            freq = np.fft.fftfreq(n, 1 / self.sampling_rate)
            freq_one_side = freq[: n // 2]

            return freq_one_side, ampl_spec_one_side

        def restore_signal(self):
            signal = np.fft.ifft(self.fft)
            signal = np.real(signal)
            return signal

        def DFT_slow(self):
            n = len(self.signal)
            result = np.zeros(n, dtype=complex)

            # Прямая формула DFT: X[k] = sum(x[n] * exp(-2πi * k * n / N))
            for k in range(n):
                for i in range(n):
                    angle = 2 * np.pi * k * i / n
                    result[k] += self.signal[i] * np.exp(-1j * angle)

            ampl_spec = np.abs(result) / n

            ampl_spec_one_side = ampl_spec[: n // 2] * 2
            ampl_spec_one_side[0] /= 2

            freq = np.fft.fftfreq(n, 1 / self.sampling_rate)
            freq_one_side = freq[: n // 2]

            return freq_one_side, ampl_spec_one_side

    return (Signal,)


@app.cell
def _(np):
    def cos_signal(A, F):
        def func(t):
            res = 0
            for a, f in zip(A, F):
                w = 2 * np.pi * f
                res += a * np.cos(t * w)
            return res

        return func, max(F)

    return (cos_signal,)


@app.cell
def _(Signal, cos_signal, plt, time):
    def stage1():
        signal, max_freq = cos_signal([1, 1], [50, 150])
        signal = Signal(signal, {"max-freq": max_freq})

        duration = 2
        x, y = signal.make_signal(duration)

        a = time.time()
        fft_freq, fft_spec = signal.get_fft_spec()
        b = time.time()
        print("fft time:", b - a)
        y2 = signal.restore_signal()

        a = time.time()
        dft_freq, dft_spec = signal.DFT_slow()
        b = time.time()
        print("DFT slow time:", b - a)

        fig, axes = plt.subplots(nrows=4, ncols=1, figsize=(16, 18))

        axes[0].set_title("Signal")
        axes[0].plot(x, y, label="cos signal")
        axes[0].set_xlabel("Time")
        axes[0].set_ylabel("Value")
        axes[0].grid(True, alpha=0.3)
        axes[0].legend()

        axes[1].set_title("fft")
        axes[1].plot(fft_freq, fft_spec, label="fft spectrum")
        axes[1].set_xlabel("Frequency")
        axes[1].set_ylabel("Amplitude")
        axes[1].grid(True, alpha=0.3)
        axes[1].legend()

        axes[2].set_title("DFT slow")
        axes[2].plot(dft_freq, dft_spec, label="dst slow spectrum")
        axes[2].set_xlabel("Frequency")
        axes[2].set_ylabel("Amplitude")
        axes[2].grid(True, alpha=0.3)
        axes[2].legend()

        axes[3].set_title("Signal restored")
        axes[3].plot(x, y2, label="cos signal")
        axes[3].set_xlabel("Time")
        axes[3].set_ylabel("Value")
        axes[3].grid(True, alpha=0.3)
        axes[3].legend()

        fig.tight_layout()

        return fig


    stage1()
    return


@app.cell
def _(Signal, cos_signal, plt):
    def stage2():
        signal, max_freq = cos_signal([1, 1], [50, 150])
        signal = Signal(signal, {"max-freq": max_freq})

        duration = 2
        x, y = signal.make_signal(duration)

        fig, axes = plt.subplots(nrows=4, ncols=1, figsize=(16, 18))

        axes[0].set_title("Signal")
        axes[0].plot(x, y, label="cos signal")
        axes[0].set_xlabel("Time")
        axes[0].set_ylabel("Value")
        axes[0].grid(True, alpha=0.3)
        axes[0].legend()

        y_noise = signal.add_white_noise()

        fft_freq, fft_spec = signal.get_fft_spec()
        y2 = signal.restore_signal()

        axes[1].set_title("Signal noised")
        axes[1].plot(x, y_noise, label="cos signal")
        axes[1].set_xlabel("Time")
        axes[1].set_ylabel("Value")
        axes[1].grid(True, alpha=0.3)
        axes[1].legend()

        axes[2].set_title("fft")
        axes[2].plot(fft_freq, fft_spec, label="fft spectrum")
        axes[2].set_xlabel("Frequency")
        axes[2].set_ylabel("Amplitude")
        axes[2].grid(True, alpha=0.3)
        axes[2].legend()

        axes[3].set_title("Signal restored")
        axes[3].plot(x, y2, label="cos signal")
        axes[3].set_xlabel("Time")
        axes[3].set_ylabel("Value")
        axes[3].grid(True, alpha=0.3)
        axes[3].legend()

        fig.tight_layout()

        return fig


    stage2()
    return


@app.cell
def _(np):
    def period_func(A, T):
        def func(t):
            phase = (t % T) / T
            return np.where(phase < 0.5, 1.0, -1.0) * A

        return func, 1 / T

    return (period_func,)


@app.cell
def _(Signal, period_func, plt):
    def stage3():
        signal, max_freq = period_func(2, 2)
        signal = Signal(signal, {"max-freq": max_freq})

        duration = 4
        x, y = signal.make_signal(duration)

        fig, axes = plt.subplots(nrows=5, ncols=1, figsize=(16, 24))

        axes[0].set_title("Signal")
        axes[0].plot(x, y, label="cos signal")
        axes[0].set_xlabel("Time")
        axes[0].set_ylabel("Value")
        axes[0].grid(True, alpha=0.3)
        axes[0].legend()

        y_noise = signal.add_white_noise()

        fft_freq, fft_spec = signal.get_fft_spec()
        y2 = signal.restore_signal()

        dft_freq, dft_spec = signal.DFT_slow()

        axes[1].set_title("Signal noised")
        axes[1].plot(x, y_noise, label="cos signal")
        axes[1].set_xlabel("Time")
        axes[1].set_ylabel("Value")
        axes[1].grid(True, alpha=0.3)
        axes[1].legend()

        axes[2].set_title("fft")
        axes[2].plot(fft_freq, fft_spec, label="fft spectrum")
        axes[2].set_xlabel("Frequency")
        axes[2].set_ylabel("Amplitude")
        axes[2].grid(True, alpha=0.3)
        axes[2].legend()

        axes[3].set_title("DFT slow")
        axes[3].plot(dft_freq, dft_spec, label="dst slow spectrum")
        axes[3].set_xlabel("Frequency")
        axes[3].set_ylabel("Amplitude")
        axes[3].grid(True, alpha=0.3)
        axes[3].legend()

        axes[4].set_title("Signal restored")
        axes[4].plot(x, y2, label="cos signal")
        axes[4].set_xlabel("Time")
        axes[4].set_ylabel("Value")
        axes[4].grid(True, alpha=0.3)
        axes[4].legend()

        fig.tight_layout()

        return fig


    stage3()
    return


@app.cell
def _(Signal, cos_signal, plt):
    def stage4():
        signal, max_freq = cos_signal([1], [50])
        signal = Signal(signal, {"max-freq": max_freq})

        duration = 1024 / (10 * max_freq)
        x, y = signal.make_signal(duration)

        fft_freq, fft_spec = signal.get_my_fft_spec()

        fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(16, 12))

        axes[0].set_title("Signal")
        axes[0].plot(x, y, label="cos signal")
        axes[0].set_xlabel("Time")
        axes[0].set_ylabel("Value")
        axes[0].grid(True, alpha=0.3)
        axes[0].legend()

        axes[1].set_title("fft")
        axes[1].plot(fft_freq, fft_spec, label="fft spectrum")
        axes[1].set_xlabel("Frequency")
        axes[1].set_ylabel("Amplitude")
        axes[1].grid(True, alpha=0.3)
        axes[1].legend()

        fig.tight_layout()

        return fig


    stage4()
    return


if __name__ == "__main__":
    app.run()

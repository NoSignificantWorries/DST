import marimo

__generated_with = "0.19.9"
app = marimo.App(width="full", auto_download=["html"])


@app.cell
def _():
    import numpy as np
    import matplotlib.pyplot as plt

    return np, plt


@app.cell
def _(np):
    class SignalProcessor:
        def __init__(self, signal_function, F):
            # main signal
            self.signal_f = signal_function
            self.T = 1 / F
            self.F = F

            # sampled function
            self.sampling_rate = None
            self.t = None
            self.signal = None
            self.n_points = 0
            self.duration = 0
            self.resolution = 0

            # fft signals
            self.freq = None
            self.fft = None

        def func(self, t):
            return self.signal_f(t)

        def sample(self, duration, resolution=10):
            self.duration = duration
            self.resolution = resolution

            self.sampling_rate = self.resolution * self.F
            self.n_points = int(self.duration * self.sampling_rate)

            self.t = np.linspace(0, self.duration, self.n_points)
            self.signal = self.func(self.t)

        def get_signal(self):
            if self.signal is None:
                self.sample(10 * self.T)
            return self.t.copy(), self.signal.copy()

        def make_fft_spec(self):
            self.fft = np.fft.fft(self.signal)
            self.freq = np.fft.fftfreq(self.n_points, 1 / self.sampling_rate)

        def get_fft(self):
            if self.fft is None:
                self.make_fft_spec()
            return self.freq.copy(), self.fft.copy()

        def get_half_fft_spec(self):
            if self.fft is None:
                self.make_fft_spec()

            ampl_spec = np.abs(self.fft) / self.n_points

            ampl_spec_one_side = ampl_spec[: self.n_points // 2] * 2
            ampl_spec_one_side[0] /= 2

            freq_one_side = self.freq[: self.n_points // 2]

            return freq_one_side, ampl_spec_one_side

        def restore_signal(self):
            if self.fft is None:
                self.make_fft_spec()
            self.signal = np.fft.ifft(self.fft).real

        def copy(self):
            if self.signal_f is None:
                new_signal = SignalProcessor(None, self.F)
                new_signal.t, new_signal.signal = self.get_signal()
                new_signal.freq, new_signal.fft = self.get_fft()

                return new_signal

            new_signal = SignalProcessor(self.signal_f, self.F)
            new_signal.sample(self.duration, self.resolution)

            return new_signal

        def add_white_noise(self, ampl=1.0):
            self.signal += np.random.normal(0, ampl, len(self.signal))
            self.signal_f = None

        def apply_time_transform(self, transform):
            if self.signal is None:
                self.sample(self.T * 10)
            self.t, self.signal = transform(self.t, self.signal)
            self.make_fft_spec()
            self.signal_f = None

        def apply_fft_transform(self, transform):
            if self.fft is None:
                self.make_fft_spec()
            self.freq, self.fft = transform(self.freq, self.fft)
            self.restore_signal()
            self.signal_f = None

    return (SignalProcessor,)


@app.cell
def _(plt):
    class Plotter:
        def __init__(self, figsize, mosaic):
            self.fig = plt.figure(constrained_layout=True, figsize=figsize)
            self.axes = self.fig.subplot_mosaic(mosaic)

            for label, ax in self.axes.items():
                setattr(self, label, ax)

        def format(self, label, title, x_label, y_label):
            self.axes[label].set_title(title)
            self.axes[label].set_xlabel(x_label)
            self.axes[label].set_ylabel(y_label)
            self.axes[label].grid(True, alpha=0.3)
            self.axes[label].legend()

        def get_fig(self):
            self.fig.tight_layout()
            return self.fig

    return (Plotter,)


@app.cell
def _(np):
    def make_rect_signal(A, F, shift=0.0):
        def func(t):
            res = np.zeros_like(t, dtype=np.float64)
            for a, f in zip(A, F):
                T = 1 / f
                phase = ((t - shift) % T) / T
                res += np.where(phase < 0.5, 1.0, -1.0) * a
            return res

        return func


    def make_cos_signal(A, F, shift=0.0):
        def func(t):
            res = 0
            for a, f in zip(A, F):
                w = 2 * np.pi * f
                res += a * np.cos((t - shift) * w)
            return res

        return func


    def gauss(a, shift=0.0):
        def func(x):
            return a * np.exp(-((x - shift) ** 2))

        return func


    def linear(k, b, shift=0.0):
        def func(x):
            return k * (x - shift) + b

        return func


    def make_kernel(func, a, b, n):
        X = np.linspace(a, b, n)
        kernel = func(X)
        kernel /= kernel.sum()

        return X, kernel

    return gauss, linear, make_cos_signal, make_kernel, make_rect_signal


@app.cell
def _(np):
    def convolution_transform(kernel):
        kernel_size = len(kernel)
        pad = kernel_size - 1
        r_kernel = kernel[::-1]

        def func(x, y):
            convolved_signal = np.zeros_like(y, dtype=np.float64)

            padded_signal = np.zeros((len(y) + pad))
            padded_signal[pad // 2 : pad // 2 + len(y)] = y

            for i in range(len(y)):
                convolved_signal[i] = np.sum(
                    padded_signal[i : i + kernel_size] * r_kernel
                )

            return x, convolved_signal

        return func


    def convolution_transform_np(kernel):
        def func(x, y):
            convolved = np.convolve(y, kernel, mode="same")
            return x, convolved

        return func


    def fft_convolution_transform(kernel_fft):
        def func(x_freq, y_fft):
            convolved = y_fft * kernel_fft
            return x_freq, convolved

        return func

    return (
        convolution_transform,
        convolution_transform_np,
        fft_convolution_transform,
    )


@app.cell
def _(
    Plotter,
    SignalProcessor,
    convolution_transform,
    convolution_transform_np,
    gauss,
    linear,
    make_kernel,
    make_rect_signal,
):
    def stage1():
        P = Plotter((16, 12), [["A", "B"], ["C", "D"], ["E", "F"]])

        A = [1.0, 1.0, 1.0]
        F = [50, 60, 130]
        main_F = max(F)
        main_func = make_rect_signal(A, F)

        sig1 = SignalProcessor(main_func, main_F)
        sig1.sample(1 / main_F * 6, 20)

        P.A.plot(sig1.t, sig1.signal)
        P.format("A", "Signal", "Time", "Signal")

        kx, ky = make_kernel(gauss(1.0), -2, 2, 5)

        P.C.plot(kx, ky)
        P.format("C", "Kernel", "X", "Y")

        sig1_2 = sig1.copy()

        sig1.apply_time_transform(convolution_transform(ky))
        P.E.plot(
            sig1.t, sig1.signal, linewidth=2, color="blue", label="convolution"
        )
        sig1_2.apply_time_transform(convolution_transform_np(ky))
        P.E.plot(
            sig1_2.t,
            sig1_2.signal,
            linewidth=1,
            linestyle="--",
            color="red",
            label="numpy convolution",
        )
        P.format("E", "Convolved signal", "Time", "Signal")

        # ==================================================

        A = [1.0]
        F = [3]
        main_F = max(F)
        main_func = make_rect_signal(A, F, 1.5)

        sig2 = SignalProcessor(main_func, main_F)
        sig2.sample(1 / main_F * 1.4, 30)

        P.B.plot(sig2.t, sig2.signal)
        P.format("B", "Signal", "Time", "Signal")

        kx, ky = make_kernel(linear(-0.2, 1.0), 0, 5, 5)

        P.D.plot(kx, ky)
        P.format("D", "Kernel", "X", "Y")

        sig2_2 = sig2.copy()

        sig2.apply_time_transform(convolution_transform(ky))
        P.F.plot(
            sig2.t, sig2.signal, linewidth=2, color="blue", label="convolution"
        )
        sig2_2.apply_time_transform(convolution_transform_np(ky))
        P.F.plot(
            sig2_2.t,
            sig2_2.signal,
            linewidth=1,
            linestyle="--",
            color="red",
            label="numpy convolution",
        )
        P.format("F", "Convolved signal", "Time", "Signal")

        return P.get_fig()


    stage1()
    return


@app.cell
def _(
    Plotter,
    SignalProcessor,
    convolution_transform_np,
    fft_convolution_transform,
    gauss,
    make_cos_signal,
    make_kernel,
    np,
):
    def stage2():
        P = Plotter((16, 12), [["A", "B"], ["C", "D"], ["E", "E"]])

        A = [1.0, 1.0, 1.0]
        F = [50, 150, 450]
        main_F = max(F)
        main_function = make_cos_signal(A, F)

        sig1 = SignalProcessor(main_function, main_F)
        sig1.sample(1 / main_F * 20, 20)

        P.A.plot(sig1.t, sig1.signal, label="Signal")
        P.format("A", "Signal", "Time", "Signal")

        sig1.make_fft_spec()
        freq, fft = sig1.get_half_fft_spec()

        P.C.plot(freq, fft, label="freq")
        P.format("C", "Spectrum", "Freq", "Ampl")

        ker1 = SignalProcessor(None, 1)
        X = np.linspace(0, sig1.n_points, sig1.n_points)
        Y = gauss(1.0)(X)
        Y_norm = Y / Y.sum()
        ker1.t = X
        ker1.signal = Y_norm
        ker1.n_points = sig1.n_points
        ker1.sampling_rate = sig1.sampling_rate
        kx, ky = make_kernel(gauss(1.0), -2, 2, 5)

        P.B.plot(ker1.t, ker1.signal)
        P.B.plot(kx, ky)
        P.format("B", "Kernel", "X", "Y")

        ker1.make_fft_spec()
        k_freq, k_fft = ker1.get_half_fft_spec()
        P.D.plot(k_freq, k_fft)
        P.format("D", "Kernel specturm", "Freq", "Ampl")

        sig1_2 = sig1.copy()
        sig1.apply_fft_transform(fft_convolution_transform(ker1.fft))

        sig1_2.apply_time_transform(convolution_transform_np(ky))

        P.E.plot(
            sig1_2.t,
            sig1_2.signal,
            linewidth=2,
            color="blue",
            label="time convolution",
        )
        P.E.plot(
            sig1.t,
            sig1.signal,
            linewidth=1.3,
            color="red",
            linestyle="--",
            label="fft convolution",
        )
        P.format("E", "Convolution", "Time", "Signal")

        return P.get_fig()


    stage2()
    return


@app.cell
def _(np):
    def make_gauss_kernel(f, p):
        s = 2 * np.pi * f * (2 * np.pi - 1) / (4 * np.pi)

        def func(freq):
            g = np.exp(-0.5 * (freq - p) ** 2 / s**2)
            return g

        return func

    return (make_gauss_kernel,)


@app.cell
def _(
    Plotter,
    SignalProcessor,
    fft_convolution_transform,
    make_cos_signal,
    make_gauss_kernel,
    np,
):
    def stage3():
        P = Plotter((18, 12), [["A", "B"], ["C", "D"]])

        A = [1.0, 1.0, 1.0]
        F = [50, 150, 450]
        main_F = max(F)
        main_function = make_cos_signal(A, F)

        sig1 = SignalProcessor(main_function, main_F)
        sig1.sample(1 / main_F * 20, 20)
        sig1.add_white_noise(0.3)
        sig1.make_fft_spec()
        freq, fft_sig = sig1.get_half_fft_spec()

        gs = make_gauss_kernel(50, 0)
        g_fft = gs(freq)

        P.A.plot(freq, g_fft, label="Filter fft")
        P.A.plot(freq, fft_sig, label="Signal fft")
        P.format("A", "Spectrum", "Freq", "Ampl")

        sig1_2 = sig1.copy()
        sig1_2.apply_fft_transform(fft_convolution_transform(gs(sig1.freq)))

        def sub_cos(t, f):
            return np.cos(t * 2 * np.pi * f)

        for f in [50, 150]:
            P.B.plot(
                sig1.t,
                sub_cos(sig1.t, f),
                linewidth=1,
                linestyle="--",
                label=f"{f} Hz",
            )

        P.B.plot(sig1.t, sig1.signal, linewidth=2, label="Origin signal")
        P.B.plot(sig1_2.t, sig1_2.signal, linewidth=2, label="Filtered")
        P.format("B", "Signal", "Time", "Signal")

        # ================================================

        gs = make_gauss_kernel(20, 450)
        g_fft = gs(freq)

        P.C.plot(freq, g_fft, label="Filter fft")
        P.C.plot(freq, fft_sig, label="Signal fft")
        P.format("C", "Spectrum", "Freq", "Ampl")

        sig2_2 = sig1.copy()
        sig2_2.apply_fft_transform(fft_convolution_transform(gs(sig1.freq)))

        def sub_cos(t, f):
            return np.cos(t * 2 * np.pi * f)

        for f in [450]:
            P.D.plot(
                sig1.t,
                sub_cos(sig1.t, f),
                linewidth=1,
                linestyle="--",
                label=f"{f} Hz",
            )

        P.D.plot(sig1.t, sig1.signal, linewidth=2, label="Origin signal")
        P.D.plot(sig2_2.t, sig2_2.signal, linewidth=2, label="Filtered")
        P.format("D", "Signal", "Time", "Signal")

        return P.get_fig()


    stage3()
    return


@app.cell
def _(np):
    def make_plank_window(w, p, eps=0.3):
        def func(freq):
            start_point = p - w / 2
            start_idx = np.argmax(freq > start_point)
            end_point = p + w / 2
            end_idx = np.argmax(freq >= end_point)

            N = end_idx - start_idx + 1
            n = N - 1

            def za(k):
                return eps * n * (1 / k + 1 / (k - eps * n))

            def zb(k):
                return eps * n * (1 / (n - k) + 1 / ((1 - eps) * n - k))

            res = np.zeros((N,))
            for k in range(N):
                if k == 0 or k == n:
                    continue
                elif 0 < k < eps * n:
                    res[k] = 1 / (np.exp(za(k)) + 1)
                elif eps * n < k < (1 - eps) * n:
                    res[k] = 1
                else:
                    res[k] = 1 / (np.exp(zb(k)) + 1)

            res_freq = np.zeros_like(freq)
            res_freq[start_idx : end_idx + 1] = res

            return res_freq

        return func

    return (make_plank_window,)


@app.cell
def _(
    Plotter,
    SignalProcessor,
    fft_convolution_transform,
    make_cos_signal,
    make_plank_window,
    np,
):
    def stage4():
        P = Plotter((18, 12), [["A", "B"], ["C", "D"]])

        A = [1.0, 1.0, 1.0]
        F = [50, 150, 450]
        main_F = max(F)
        main_function = make_cos_signal(A, F)

        sig1 = SignalProcessor(main_function, main_F)
        sig1.sample(1 / main_F * 20, 20)
        sig1.add_white_noise(0.3)
        sig1.make_fft_spec()
        freq, fft_sig = sig1.get_half_fft_spec()

        pl = make_plank_window(400, 0, 0.01)

        P.A.plot(freq, pl(freq), label="Plank window")
        P.A.plot(freq, fft_sig, label="Signal fft")
        P.format("A", "Spectrum", "Freq", "Ampl")

        sig1_2 = sig1.copy()
        sig1_2.apply_fft_transform(fft_convolution_transform(pl(sig1.freq)))

        def sub_cos(t, f):
            return np.cos(t * 2 * np.pi * f)

        for f in [50, 150]:
            P.B.plot(
                sig1.t,
                sub_cos(sig1.t, f),
                linewidth=1,
                linestyle="--",
                label=f"{f} Hz",
            )

        P.B.plot(sig1.t, sig1.signal, linewidth=2, label="Origin signal")
        P.B.plot(sig1_2.t, sig1_2.signal, linewidth=2, label="Filtered")
        P.format("B", "Signal", "Time", "Signal")

        # =======================================================

        sig2_2 = sig1.copy()

        pl = make_plank_window(50, 450, 0.01)
        P.C.plot(freq, pl(freq), label="Plank window")
        P.C.plot(freq, fft_sig, label="Signal fft")
        P.format("C", "Spectrum", "Freq", "Ampl")

        sig2_2.apply_fft_transform(fft_convolution_transform(pl(sig1.freq)))

        def sub_cos(t, f):
            return np.cos(t * 2 * np.pi * f)

        for f in [450]:
            P.D.plot(
                sig1.t,
                sub_cos(sig1.t, f),
                linewidth=1,
                linestyle="--",
                label=f"{f} Hz",
            )

        P.D.plot(sig1.t, sig1.signal, linewidth=2, label="Origin signal")
        P.D.plot(sig2_2.t, sig2_2.signal, linewidth=2, label="Filtered")
        P.format("D", "Signal", "Time", "Signal")

        return P.get_fig()


    stage4()
    return


if __name__ == "__main__":
    app.run()

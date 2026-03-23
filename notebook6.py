import marimo

__generated_with = "0.19.9"
app = marimo.App(width="full", auto_download=["html"])


@app.cell
def _():
    import numpy as np
    import matplotlib.pyplot as plt
    import librosa

    return librosa, np, plt


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
            self.start = 0
            self.duration = 0
            self.resolution = 0

            # fft signals
            self.freq = None
            self.fft = None

        def func(self, t):
            return self.signal_f(t)

        def sample(self, start, duration, resolution=10, check_odd=False):
            self.start = start
            self.duration = duration
            self.resolution = resolution

            self.sampling_rate = self.resolution * self.F
            self.n_points = int(self.duration * self.sampling_rate)
            if check_odd and self.n_points % 2 == 0:
                self.n_points += 1

            self.t = np.linspace(
                self.start, self.start + self.duration, self.n_points
            )
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

        def get_half_fft_spec(self, k=10):
            if self.fft is None:
                self.make_fft_spec()

            ampl_spec = np.abs(self.fft) / self.n_points

            ampl_spec_one_side = ampl_spec[: self.n_points // 2] * 2
            ampl_spec_one_side[0] /= 2

            freq_one_side = self.freq[: self.n_points // 2]

            freq_k = int(self.F * k)

            if np.any(freq_one_side >= freq_k):
                idx = np.argwhere(freq_one_side >= freq_k).flatten()[0]

                freq_one_side = freq_one_side[:idx]
                ampl_spec_one_side = ampl_spec_one_side[:idx]

            return freq_one_side, ampl_spec_one_side

        def restore_signal(self):
            if self.fft is None:
                self.make_fft_spec()
            self.signal = np.fft.ifft(self.fft).real

        def copy(self):
            if self.signal_f is None:
                new_signal = SignalProcessor(None, self.F)
                new_signal.sampling_rate = self.sampling_rate
                new_signal.duration = self.duration
                new_signal.start = self.start
                new_signal.n_points = self.n_points
                new_signal.t, new_signal.signal = self.get_signal()
                new_signal.freq, new_signal.fft = self.get_fft()

                return new_signal

            new_signal = SignalProcessor(self.signal_f, self.F)
            new_signal.sample(self.start, self.duration, self.resolution)

            return new_signal

        def add_white_noise(self, ampl=1.0):
            self.signal += np.random.normal(0, ampl, len(self.signal))
            self.signal_f = None

        def convolve_wavelet(self, wavelet_func):
            if self.freq is None:
                self.make_fft_spec()
            wave = wavelet_func(self.freq)
            self.fft = self.fft * wave
            self.signal = np.fft.ifft(self.fft).real
            self.signal_f = None

        def make_spectrum(self, wfuncs):
            if self.freq is None:
                self.make_fft_spec()

            spectrum = []
            for fwave in wfuncs:
                wave_fft = fwave(self.freq)
                filtered = self.fft * wave_fft
                filtered_signal = np.fft.ifft(filtered)
                spectrum.append(np.abs(filtered_signal))

            return np.array(spectrum)

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

    return (make_cos_signal,)


@app.cell
def _(np):
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

    return


@app.cell
def _(Plotter, SignalProcessor, np):
    def morlet_core(alpha, f_0):
        w_0 = 2 * np.pi * f_0
        a2 = alpha**2

        def func(t):
            return np.exp(-(t**2) / a2) * np.exp(1j * w_0 * t)

        return func


    def morlet_fcore(alpha, f_0):
        w_0 = 2 * np.pi * f_0

        def func(freq):
            w = 2 * np.pi * freq
            phase = alpha**2 * (w - w_0) ** 2 / 4
            return alpha * np.sqrt(np.pi) * np.exp(-phase)

        return func


    def stage1():
        P = Plotter((18, 6), [["A", "B"]])

        def make_sig(F, a, i, k=1.25):
            mcr = morlet_core(a, F)

            morlet_func_fft = morlet_fcore(a, F)

            mcr_sig = SignalProcessor(mcr, F)
            mcr_sig.sample(-10, 20, 20)

            P.A.plot(
                mcr_sig.t, mcr_sig.signal, linewidth=1 / i * k, label=f"{F} Hz"
            )
            P.format("A", "Morlet time form", "Time", "Ampl")

            mcr_sig.make_fft_spec()
            freq, fft_sig = mcr_sig.get_half_fft_spec(5)

            P.B.plot(
                freq, morlet_func_fft(freq), linewidth=1 / i * k, label=f"{F} Hz"
            )
            P.B.axvline(x=F, linewidth=0.5, linestyle="--", color="red")
            P.format("B", "Morlet fft form", "Freq", "Ampl")

        params = [(0.2, 5.0), (0.5, 5.0), (1.0, 5.0), (2.0, 5.0)]
        for i, (F, a) in enumerate(params):
            make_sig(F, a, i + 1)

        return P.get_fig()


    stage1()
    return (morlet_fcore,)


@app.cell
def _(Plotter, SignalProcessor, np):
    def mexican_hat_core(sigma, f_0):
        w_0 = 2 * np.pi * f_0
        s2 = sigma**2

        def func(t):
            return (1 - t**2) * np.exp(-(t**2) / s2) * np.exp(1j * w_0 * t)
            # return (1 - t**2 / s2) * np.exp(-(t**2) / (2 * s2))

        return func


    def mhat_fcore(sigma, f_0):
        w_0 = 2 * np.pi * f_0
        s2 = sigma**2

        def func(freq):
            w = 2 * np.pi * freq
            ww = (w - w_0) ** 2
            # return ww * np.exp(-ww * s2 / 2)
            return (
                sigma
                * np.sqrt(np.pi)
                * np.exp(-s2 * ww / 4)
                * (1 - s2 / 2 + s2**2 * ww / 4)
            )

        return func


    def stage2():
        g = [["A", "B"]]
        P = Plotter((18, 6), [["A", "B"]])

        def make_sig(F, a, i, k=0.25):
            mhcr = mexican_hat_core(a, F)

            mhat_func_fft = mhat_fcore(a, F)

            mhcr_sig = SignalProcessor(mhcr, F)
            mhcr_sig.sample(-5, 10, 50)

            P.A.plot(mhcr_sig.t, mhcr_sig.signal, linewidth=i * k, label=f"{F} Hz")
            P.format("A", "Mexican Hat time form", "Time", "Ampl")

            mhcr_sig.make_fft_spec()
            freq, fft_sig = mhcr_sig.get_half_fft_spec()

            P.B.plot(freq, mhat_func_fft(freq), linewidth=i * k, label=f"{F} Hz")
            P.B.axvline(x=F, color="red", linestyle="--", linewidth=0.5)
            P.format("B", "Mexican Hat fft form", "Freq", "Ampl")

        params = [(0.1, 0.5), (0.2, 0.5), (0.5, 0.5), (1.0, 0.5)]
        for i, (F, a) in enumerate(params):
            make_sig(F, a, i + 1)

        return P.get_fig()


    stage2()
    return (mhat_fcore,)


@app.cell
def _(Plotter, SignalProcessor, np):
    def haar_core(a):
        def func(t):
            res = np.zeros_like(t)
            t_ = t / a
            res[(t_ >= 0) & (t_ < (a / 2))] = 1
            res[(t_ >= (a / 2)) & (t_ < a)] = -1
            return res / np.sqrt(a)

        return func


    def haar_fcore(a, alpha=1.0):
        def phi(w):
            # return np.sin(w / 4) / w * np.exp(-1j * alpha * w)
            mask = w <= 1e-16
            res = np.zeros_like(w, dtype=np.complex128)
            res[~mask] = 4 * np.sin(w[~mask] / 4) ** 2 / np.abs(w[~mask])
            return res

        def func(freq):
            w = 2 * np.pi * freq
            return np.sqrt(a) * phi(w * a)
            # return a * np.abs(phi(w * a)) ** 2

        return func


    def stage3():
        g = [["A", "B"]]
        P = Plotter((18, 6), [["A", "B"]])

        def make_sig(F, i, k=0.25):
            hcr = haar_core(0.5 / F)

            haar_func_fft = haar_fcore(0.5 / F, 1.0)

            hcr_sig = SignalProcessor(hcr, F)
            hcr_sig.sample(-2, 8, 50)

            P.A.plot(hcr_sig.t, hcr_sig.signal, linewidth=i * k, label=f"{F} Hz")
            P.format("A", "Haar time form", "Time", "Ampl")

            hcr_sig.make_fft_spec()
            freq, fft_sig = hcr_sig.get_half_fft_spec()

            P.B.plot(freq, haar_func_fft(freq), linewidth=i * k, label=f"{F} Hz")
            P.B.axvline(x=F, color="red", linestyle="--", linewidth=0.5)
            P.format("B", "Haar fft form", "Freq", "Ampl")

        params = [0.2, 0.3, 0.5, 1.0]
        for i, F in enumerate(params):
            make_sig(F, i + 1)

        return P.get_fig()


    stage3()
    return (haar_fcore,)


@app.cell
def _(
    Plotter,
    SignalProcessor,
    haar_fcore,
    make_cos_signal,
    mhat_fcore,
    morlet_fcore,
    np,
):
    def stage4():
        P = Plotter((18, 12), [["A"], ["B"]])

        A = [1.0, 1.0, 1.0]
        F = [5, 15, 30]
        main_F = max(F)
        main_function = make_cos_signal(A, F)

        sig1 = SignalProcessor(main_function, main_F)
        sig1.sample(0, 30 / main_F, 1000)
        sig1.add_white_noise(0.3)
        sig1.make_fft_spec()
        sig_freq, sig_fft = sig1.get_half_fft_spec(5)

        # morlet filter
        sig1_1 = sig1.copy()
        morlet_F = 30
        morlet_alpha = 4.0
        morlet_core_func = morlet_fcore(morlet_alpha, morlet_F)
        morlet_fft = morlet_core_func(sig_freq)
        sig1_1.convolve_wavelet(morlet_core_func)

        # mhat filter
        sig1_2 = sig1.copy()
        mhat_F = 15
        mhat_sigma = 2 / mhat_F
        mhat_core_func = mhat_fcore(mhat_sigma, mhat_F)
        mhat_fft = mhat_core_func(sig_freq)
        sig1_2.convolve_wavelet(mhat_core_func)

        # haar filter
        sig1_3 = sig1.copy()
        haar_F = 5
        haar_core_func = haar_fcore(1 / haar_F)
        haar_fft = haar_core_func(sig_freq)
        sig1_3.convolve_wavelet(haar_core_func)

        # visual
        P.A.plot(sig1.t, sig1.signal, color="gray", alpha=0.7, label="Signal")
        P.A.plot(
            sig1_1.t,
            sig1_1.signal / (np.max(np.abs(sig1_1.signal)) + 1e-12),
            color="green",
            linewidth=1.5,
            label="Morlet filtered",
        )
        P.A.plot(
            sig1_2.t,
            sig1_2.signal / (np.max(np.abs(sig1_2.signal)) + 1e-12),
            color="red",
            linewidth=1.5,
            label="MHAT filtered",
        )
        P.A.plot(
            sig1_3.t,
            sig1_3.signal / (np.max(np.abs(sig1_3.signal)) + 1e-12),
            color="blue",
            linewidth=1.5,
            label="Haar filtered",
        )
        x1 = np.linspace(0, 30 / main_F, 1000)
        P.A.plot(
            x1,
            np.cos(x1 * 2 * np.pi * morlet_F),
            color="black",
            linestyle="--",
            linewidth=1.0,
            label=f"Clear {morlet_F} Hz",
        )
        P.A.plot(
            x1,
            np.cos(x1 * 2 * np.pi * mhat_F),
            color="black",
            linestyle="--",
            linewidth=1.0,
            label=f"Clear {mhat_F} Hz",
        )
        P.A.plot(
            x1,
            np.cos(x1 * 2 * np.pi * haar_F),
            color="blue",
            linestyle="--",
            linewidth=1.0,
            label=f"Clear {haar_F} Hz",
        )
        P.format("A", "Signal", "Time", "Ampl")

        P.B.plot(sig_freq, sig_fft, color="gray", label="Signal fft")
        P.B.plot(sig_freq, morlet_fft, color="green", label="Morlet fft")
        P.B.plot(sig_freq, mhat_fft, color="red", label="MHAT fft")
        P.B.plot(sig_freq, haar_fft, color="blue", label="Haar fft")
        P.format("B", "fft", "Freq", "Ampl")

        return P.get_fig()


    stage4()
    return


@app.cell
def _(
    Plotter,
    SignalProcessor,
    haar_fcore,
    make_cos_signal,
    mhat_fcore,
    morlet_fcore,
    np,
):
    def make_fm_signal_integral(A, freq_func, duration, srate, shift=0.0):
        def func(t):
            t_shifted = t - shift

            instantaneous_freq = freq_func(t_shifted)

            dt = t[1] - t[0] if len(t) > 1 else 1 / srate
            phase = 2 * np.pi * np.cumsum(instantaneous_freq) * dt

            return A * np.sin(phase)

        return func


    def stage5():
        P = Plotter((32, 18), [["A", "B"], ["C", "D"]])

        A = [1.0, 1.0, 1.0]
        F = [5, 15, 30]
        main_F = max(F)
        main_function = make_cos_signal(A, F)

        sig1 = SignalProcessor(main_function, main_F)
        sig1.sample(0, 30 / main_F, 100)
        sig1.add_white_noise(0.2)
        sig1.make_fft_spec()
        sig_freq, sig_fft = sig1.get_half_fft_spec(2)

        frex = np.linspace(1, 35, 100)

        P.A.plot(sig1.t, sig1.signal, linewidth=2)
        P.format("A", "Signal", "Time", "Ampl")

        fmorlets = []
        fmhats = []
        fhaars = []
        for fc in frex:
            fmorlets.append(morlet_fcore(1.0, fc))
            fmhats.append(mhat_fcore(0.25, fc))
            fhaars.append(haar_fcore(0.5 / fc))

        sig2_1 = sig1.copy()
        morlet_spec = sig2_1.make_spectrum(fmorlets)
        sig2_2 = sig1.copy()
        mhat_spec = sig2_2.make_spectrum(fmhats)
        sig2_3 = sig1.copy()
        haar_spec = sig2_3.make_spectrum(fhaars)

        P.B.contourf(sig1.t, frex, morlet_spec, cmap="inferno")
        P.format("B", "Morlet wavelet", "Time", "Freq")
        P.C.contourf(sig1.t, frex, mhat_spec, cmap="inferno")
        P.format("C", "MHAT wavelet", "Time", "Freq")
        P.D.contourf(sig1.t, frex, haar_spec, cmap="inferno")
        P.format("D", "Haar wavelet", "Time", "Freq")

        return P.get_fig()


    stage5()
    return


@app.cell
def _(
    Plotter,
    SignalProcessor,
    haar_fcore,
    librosa,
    mhat_fcore,
    morlet_fcore,
    np,
):
    def stage6():
        P = Plotter((48, 32), [["A", "B"], ["C", "D"], ["E", "F"]])

        dur = 1
        audio_array, sample_rate = librosa.load(
            "music1.wav", sr=None, duration=dur
        )

        print(len(audio_array), sample_rate)

        sound = SignalProcessor(None, 1)
        sound.signal = audio_array
        sound.t = np.linspace(0, dur, len(audio_array))
        sound.sampling_rate = sample_rate
        sound.duration = dur
        sound.n_points = int(sound.duration * sound.sampling_rate)

        sound.make_fft_spec()

        print(sound.freq[np.argmax(np.abs(sound.fft))])

        freq, fft = sound.get_half_fft_spec(800)

        P.A.plot(sound.t, sound.signal)
        P.format("A", "Sound", "Time", "Ampl")
        P.B.plot(freq, fft)
        P.format("B", "Sound fft", "Freq", "Ampl")

        frex = np.linspace(1, 600, 400)

        fmorlets = []
        fmhats = []
        fhaars = []
        for fc in frex:
            fmorlets.append(morlet_fcore(1.0, fc))
            fmhats.append(mhat_fcore(0.25, fc))
            fhaars.append(haar_fcore(0.5 / fc))

        sig2_1 = sound.copy()
        morlet_spec = sig2_1.make_spectrum(fmorlets)
        sig2_2 = sound.copy()
        mhat_spec = sig2_2.make_spectrum(fmhats)
        sig2_3 = sound.copy()
        haar_spec = sig2_3.make_spectrum(fhaars)

        P.C.contourf(sound.t, frex, morlet_spec, cmap="inferno")
        P.format("C", "Morlet wavelet", "Time", "Freq")
        P.D.contourf(sound.t, frex, mhat_spec, cmap="inferno")
        P.format("D", "MHAT wavelet", "Time", "Freq")
        P.E.contourf(sound.t, frex, haar_spec, cmap="inferno")
        P.format("E", "Haar wavelet", "Time", "Freq")

        note_array, note_sample_rate = librosa.load("la-note.wav", sr=None)

        print(len(note_array))
        dur = len(note_array) / note_sample_rate
        print(dur)

        note = SignalProcessor(None, 1)
        note.signal = note_array
        note.t = np.linspace(0, dur, len(note_array))
        note.sampling_rate = note_sample_rate
        note.duration = dur
        note.n_points = int(note.duration * note.sampling_rate)

        note.make_fft_spec()

        sig3 = note.copy()
        morlet_spec = sig3.make_spectrum(fmorlets)

        P.F.contourf(note.t, frex, morlet_spec, cmap="inferno")
        P.format("F", "Morlet wavelet for 'La' note", "Time", "Freq")

        return P.get_fig()


    stage6()
    return


if __name__ == "__main__":
    app.run()

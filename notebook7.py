import marimo

__generated_with = "0.19.9"
app = marimo.App(width="full", auto_download=["html"])


@app.cell
def _():
    import requests

    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import scipy

    return np, pd, plt, requests, scipy


@app.cell
def _(np):
    class SignalProcessor:
        def __init__(self, signal_function=None, F=1):
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

        def make_from_signal(self, signal, t=None):
            self.n_points = len(signal)
            self.duration = self.n_points
            self.sampling_rate = 1
            self.start = 0
            self.t = t
            self.signal = signal

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

        def convolve_kernel(self, kernel_func, kernel_size):
            pad = kernel_size // 2

            padded_signal = np.pad(self.signal, pad)

            new_signal = np.zeros_like(self.signal, dtype=np.float64)
            for i in range(len(self.signal)):
                new_signal[i] = kernel_func(padded_signal[i : i + kernel_size])

            self.signal = new_signal

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
def _(np, scipy):
    def make_signal_by_points(pointsX, pointsY, start, duration, srate):
        n_points = int(duration * srate)
        x_points = np.linspace(start, start + duration, n_points)

        pointsX = np.array(pointsX, dtype=np.float64)
        pointsY = np.array(pointsY, dtype=np.float64)

        inds = np.argsort(pointsX)
        pointsX = pointsX[inds]
        pointsY = pointsY[inds]

        cs = scipy.interpolate.CubicSpline(
            pointsX, pointsY, extrapolate=False, bc_type="natural"
        )
        y_signal = cs(x_points)

        return x_points, y_signal

    return (make_signal_by_points,)


@app.cell
def _(Plotter, SignalProcessor, make_signal_by_points, np):
    def stage1():
        P = Plotter((24, 8), [["A", "B"]])

        dur = 3
        start = 0
        srate = 1000

        keyX = [0, 0.2, 0.4, 0.6, 1.0, 1.2, 1.4, 1.5, 1.7, 2.0, 2.2, 2.4, 2.6, 3.0]
        keyY = [10, 5, 13, 25, 8, 15, 22, 18, 4, 24, 2, 28, 15, 12]
        t, signal = make_signal_by_points(keyX, keyY, start, dur, srate)

        sig = SignalProcessor()
        sig.t = t.copy()
        sig.signal = signal.copy()
        sig.duration = dur
        sig.start = start
        sig.sampling_rate = srate
        sig.n_points = int(srate * dur)

        sig.add_white_noise(10)

        sig1 = sig.copy()
        sig1.convolve_kernel(np.mean, 21)

        P.A.plot(sig.t, sig.signal, color="blue", alpha=0.5, label="Signal")
        P.A.plot(sig1.t, sig1.signal, color="orange", label="Filtered signal")
        P.A.plot(t, signal, color="black", label="Origin")
        P.A.plot(keyX, keyY, linestyle="--", color="red", label="Key points")
        P.format("A", "Median Filter (Time domain)", "Time", "Value")

        sig.make_fft_spec()
        sig1.make_fft_spec()

        sfreq, sfft = sig.get_half_fft_spec(10)
        s1freq, s1fft = sig1.get_half_fft_spec(10)

        P.B.plot(sfreq, sfft, linewidth=3, color="blue", alpha=0.5, label="Signal")
        P.B.plot(
            s1freq,
            s1fft,
            linewidth=2,
            color="red",
            linestyle="--",
            label="Filtered signal",
        )
        P.format("B", "Median Filter (Freq domain)", "Freq", "Ampl")

        return P.get_fig()


    stage1()
    return


@app.cell
def _(Plotter, SignalProcessor, make_signal_by_points, np):
    def gauss_filter(w):
        def func(vec):
            a = (len(vec) - 1) // 2
            x = np.linspace(-a, a, len(vec), dtype=np.float64)
            tmp = -4 * np.log(2) * x**2
            g = np.exp(tmp / w**2)
            g /= g.sum()
            return np.sum(g * vec)

        return func


    def stage2():
        P = Plotter((24, 8), [["A", "B"]])

        dur = 3
        start = 0
        srate = 1000

        keyX = [0, 0.2, 0.4, 0.6, 1.0, 1.2, 1.4, 1.5, 1.7, 2.0, 2.2, 2.4, 2.6, 3.0]
        keyY = [10, 5, 13, 25, 8, 15, 22, 18, 4, 24, 2, 28, 15, 12]
        t, signal = make_signal_by_points(keyX, keyY, start, dur, srate)

        sig = SignalProcessor()
        sig.t = t.copy()
        sig.signal = signal.copy()
        sig.duration = dur
        sig.start = start
        sig.sampling_rate = srate
        sig.n_points = int(srate * dur)

        sig.add_white_noise(10)

        sig1 = sig.copy()
        sig1.convolve_kernel(gauss_filter(1.0), 11)
        sig2 = sig.copy()
        sig2.convolve_kernel(np.mean, 21)

        P.A.plot(sig.t, sig.signal, color="blue", alpha=0.5, label="Signal")
        P.A.plot(
            sig1.t,
            sig1.signal,
            color="green",
            alpha=0.7,
            label="Gauss Filtered signal",
        )
        P.A.plot(
            sig2.t,
            sig2.signal,
            color="orange",
            label="Median Filtered signal",
        )
        P.A.plot(t, signal, color="black", label="Origin")
        P.A.plot(keyX, keyY, linestyle="--", color="red", label="Key points")
        P.format("A", "Gauss x Median Filter (Time domain)", "Time", "Value")

        sig.make_fft_spec()
        sig1.make_fft_spec()
        sig2.make_fft_spec()

        sfreq, sfft = sig.get_half_fft_spec(10)
        s1freq, s1fft = sig1.get_half_fft_spec(10)
        s2freq, s2fft = sig2.get_half_fft_spec(10)

        P.B.plot(sfreq, sfft, linewidth=4, color="blue", alpha=0.5, label="Signal")
        P.B.plot(
            s1freq,
            s1fft,
            linewidth=3,
            color="green",
            linestyle="--",
            label="Gauss Filtered signal",
        )
        P.B.plot(
            s2freq,
            s2fft,
            linewidth=2,
            color="red",
            linestyle="--",
            label="Median Filtered signal",
        )
        P.format("B", "Gauss x Median Filter (Freq domain)", "Freq", "Ampl")

        return P.get_fig()


    stage2()
    return (gauss_filter,)


@app.cell
def _(Plotter, SignalProcessor, gauss_filter, np):
    def make_picks_signal(start, duration, srate, p=0.8):
        n_points = int(duration * srate)
        x_points = np.linspace(start, start + duration, n_points)

        y = np.zeros_like(x_points)

        for i in range(n_points):
            if i == 0 or y[i - 1] == 0:
                y[i] = np.random.random()
        y = np.where(y >= p, 1.0, 0.0)

        return x_points, y


    def stage3():
        P = Plotter((24, 8), [["A", "B"]])

        dur = 5
        start = 0
        srate = 100

        t, signal = make_picks_signal(start, dur, srate, 0.8)

        sig = SignalProcessor()
        sig.t = t.copy()
        sig.signal = signal.copy()
        sig.duration = dur
        sig.start = start
        sig.sampling_rate = srate
        sig.n_points = int(srate * dur)

        sig1 = sig.copy()
        sig1.convolve_kernel(gauss_filter(5.0), 51)

        P.A.plot(
            sig.t,
            sig.signal,
            color="black",
            linewidth=1,
            label="Signal",
        )
        P.A.plot(
            sig1.t,
            sig1.signal,
            color="red",
            linewidth=3,
            alpha=0.7,
            label="Gauss Filtered signal",
        )
        P.format("A", "Gauss Filter (Time domain)", "Time", "Value")

        sig.make_fft_spec()
        sig1.make_fft_spec()

        sfreq, sfft = sig.get_half_fft_spec(10)
        s1freq, s1fft = sig1.get_half_fft_spec(10)

        P.B.plot(
            sfreq, sfft, linewidth=3, color="black", alpha=0.5, label="Signal"
        )
        P.B.plot(
            s1freq,
            s1fft,
            linewidth=2,
            color="red",
            linestyle="--",
            label="Gauss Filtered signal",
        )
        P.format("B", "Gauss Filter (Freq domain)", "Freq", "Ampl")

        return P.get_fig()


    stage3()
    return (make_picks_signal,)


@app.cell
def _(Plotter, SignalProcessor, make_picks_signal, make_signal_by_points, np):
    def make_picks_signal_haotic(
        pointsX, pointsY, start, duration, srate, A=1.0, p=0.8
    ):
        x_points, basic_signal = make_signal_by_points(
            pointsX, pointsY, start, duration, srate
        )

        n_points = len(x_points)
        y = np.zeros_like(x_points)

        for i in range(n_points):
            if i == 0 or y[i - 1] == 0:
                y[i] = np.random.random()
        y *= A
        basic_signal += y

        return x_points, basic_signal


    def median_filter(threshold):
        def func(t):
            t[t > threshold] = t.mean()
            sorter = np.argsort(t)
            res = t[sorter]
            idx = len(t) // 2
            return res[idx]

        return func


    def stage4():
        P = Plotter((24, 8), [["A", "B"]])

        dur = 5
        start = 0
        srate = 50

        t, signal = make_picks_signal(start, dur, srate, 0.8)
        keyX = [0, 0.2, 0.4, 0.6, 1.0, 1.2, 1.4, 1.5, 1.7, 2.0, 2.2, 2.4, 2.6, 5.0]
        keyY = [7, 5, 6.5, 7, 7.5, 7.7, 7.5, 5, 4, 6, 3.5, 4.5, 5, 6]
        th, signalh = make_picks_signal_haotic(
            keyX, keyY, start, dur, srate, 30, 0.95
        )

        sigh = SignalProcessor()
        sigh.t = th.copy()
        sigh.signal = signalh.copy()
        sigh.duration = dur
        sigh.start = start
        sigh.sampling_rate = srate
        sigh.n_points = int(srate * dur)

        sig1 = sigh.copy()
        sig1.convolve_kernel(median_filter(8.0), 5)

        P.A.plot(
            sigh.t,
            sigh.signal,
            color="black",
            linewidth=1,
            label="Signal",
        )
        P.A.plot(
            sig1.t,
            sig1.signal,
            color="red",
            linewidth=2,
            alpha=0.7,
            label="Median Filtered signal",
        )
        P.format("A", "Median Filter (Time domain)", "Time", "Value")

        sigh.make_fft_spec()
        sig1.make_fft_spec()

        sfreq, sfft = sigh.get_half_fft_spec(10)
        s1freq, s1fft = sig1.get_half_fft_spec(10)

        P.B.plot(
            sfreq, sfft, linewidth=3, color="black", alpha=0.5, label="Signal"
        )
        P.B.plot(
            s1freq,
            s1fft,
            linewidth=2,
            color="red",
            linestyle="--",
            label="Median Filtered signal",
        )
        P.format("B", "Median Filter (Freq domain)", "Freq", "Ampl")

        return P.get_fig()


    stage4()
    return


@app.cell
def _(pd, requests):
    def get_data_by_url():
        url = "https://query1.finance.yahoo.com/v7/finance/chart/BTC-USD?range=max&interval=1d&includePrePost=false&events=history"

        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
            "Accept": "application/json,text/plain,*/*",
            "Accept-Language": "en-US,en;q=0.9",
            "Connection": "keep-alive",
        }

        r = requests.get(url, headers=headers, timeout=30)
        r.raise_for_status()
        data = r.json()

        result = data["chart"]["result"][0]
        ts = result["timestamp"]
        quote = result["indicators"]["quote"][0]

        df = pd.DataFrame(
            {
                "datetime": pd.to_datetime(ts, unit="s"),
                "open": quote["open"],
                "high": quote["high"],
                "low": quote["low"],
                "close": quote["close"],
                "volume": quote["volume"],
            }
        )

        df = df.dropna(subset=["close"])

        return df

    return (get_data_by_url,)


@app.cell
def _(Plotter, get_data_by_url, np):
    def stage5():
        P = Plotter((18, 6), [["A"]])

        data = get_data_by_url()

        price = data["close"].to_numpy()[60:]
        n = len(price)
        time = np.linspace(0, n - 1, n)

        b, a = np.polyfit(time, price, 1)
        trend = a + b * time

        err = np.sum((trend - price) ** 2) / n
        k = n * np.log(err) + 2 * np.log(n)

        detrended_price = price - trend

        P.A.plot(
            time,
            a + time * b,
            linestyle="--",
            alpha=0.7,
            label="trend",
        )
        P.A.plot(time, price, label="Bitcoin price")
        P.A.plot(time, detrended_price, label="Bitcoin detrended price")
        P.format("A", "Bitcoin price", "Time", "Price (USD)")

        return P.get_fig()


    stage5()
    return


@app.cell
def _(Plotter, SignalProcessor, np, scipy):
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


    def stage6():
        P = Plotter(
            (18, 22),
            [
                ["A", "B", "C"],
                ["D", "D", "D"],
                ["E1", "E2", "E3"],
                ["F1", "F2", "F3"],
            ],
        )

        matdat = scipy.io.loadmat("EEG_example.mat")
        EEGdat = matdat["EEGdat"]
        eyedat = matdat["eyedat"]
        timevec = matdat["timevec"][0]

        times, values = eyedat.shape

        X = np.column_stack([np.ones(len(eyedat)), eyedat])
        B, *_ = np.linalg.lstsq(X, EEGdat, rcond=None)
        Residual = EEGdat - X @ B

        P.A.imshow(EEGdat.T, cmap="inferno")
        P.B.imshow(eyedat.T, cmap="inferno")
        P.C.imshow(Residual.T, cmap="inferno")
        P.format("A", "EEG", "Time", "Signal")
        P.format("B", "EOG", "Time", "Signal")
        P.format("C", "Residual", "Time", "Signal")

        channel = 0
        signal = EEGdat[:, channel]
        artefact = eyedat[:, channel]
        resid = Residual[:, channel]

        P.D.plot(timevec, signal, label="EEG")
        P.D.plot(timevec, artefact, label="EOG")
        P.D.plot(timevec, resid, label="Residual")
        P.format("D", "Cleeaned signal", "Time", "Signal")

        # ==================================================================

        eeg = SignalProcessor(None, 1)
        eeg.make_from_signal(signal, timevec)
        eeg_freq, eeg_fft = eeg.get_half_fft_spec(100)
        P.E1.plot(eeg_freq, eeg_fft)
        P.format("E1", "EEG Ampl spectrum", "Freq", "Ampl")

        eog = SignalProcessor(None, 1)
        eog.make_from_signal(artefact, timevec)
        eog_freq, eog_fft = eog.get_half_fft_spec(100)
        P.E2.plot(eog_freq, eog_fft)
        P.format("E2", "EOG Ampl spectrum", "Freq", "Ampl")

        res = SignalProcessor(None, 1)
        res.make_from_signal(resid, timevec)
        res_freq, res_fft = res.get_half_fft_spec(100)
        P.E3.plot(res_freq, res_fft)
        P.format("E3", "Residual Ampl spectrum", "Freq", "Ampl")

        frex = np.linspace(0.01, 1, 100)

        fmorlets = []
        for fc in frex:
            fmorlets.append(morlet_fcore(0.01, fc))

        eeg1 = eeg.copy()
        eeg_spec = eeg1.make_spectrum(fmorlets)
        eog1 = eog.copy()
        eog_spec = eog1.make_spectrum(fmorlets)
        res1 = res.copy()
        res_spec = res1.make_spectrum(fmorlets)

        P.F1.contourf(eeg.t, frex, eeg_spec, cmap="inferno")
        P.format("F1", "EEG Spectrum", "Time", "Freq")
        P.F2.contourf(eog.t, frex, eog_spec, cmap="inferno")
        P.format("F2", "EOG Spectrum", "Time", "Freq")
        P.F3.contourf(res.t, frex, res_spec, cmap="inferno")
        P.format("F3", "Residual Spectrum", "Time", "Freq")

        return P.get_fig()


    stage6()
    return


if __name__ == "__main__":
    app.run()

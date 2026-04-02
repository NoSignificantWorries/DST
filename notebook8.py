import marimo

__generated_with = "0.21.1"
app = marimo.App(width="medium", auto_download=["html"])


@app.cell
def _():
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import scipy

    return np, plt, scipy


@app.cell
def _(np, scipy):
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

        def sample(self, start, duration, resolution=2, check_odd=False):
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

        def downsample(self, resolution, filter=True):
            new_sampling_rate = resolution * self.F

            if filter:
                self.butter_filter(new_sampling_rate / 2)

            k = int(self.sampling_rate / new_sampling_rate)
            filter = np.argwhere(np.arange(self.n_points) % k == 0)

            self.sampling_rate = new_sampling_rate
            self.signal = self.signal[filter]
            self.t = self.t[filter]

            self.resolution = resolution
            self.n_points = int(self.duration * new_sampling_rate)

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

        def add_white_noise(self, ampl=1.0, part=1.0):
            n = int(len(self.signal) * part)
            xs = np.random.choice(self.n_points, size=n, replace=False)
            self.signal[xs] += np.random.normal(0, ampl, n)
            self.signal_f = None

        def butter_filter(self, F, alpha=10):
            sos = scipy.signal.butter(
                alpha, F, btype="low", fs=self.sampling_rate, output="sos"
            )

            self.signal = scipy.signal.sosfiltfilt(sos, self.signal)

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

        def filter_by_points(self, filter):
            self.t, self.signal, deleted, mask = filter(self.t, self.signal)
            return deleted, mask

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


    def make_picks_signal(
        start, duration, srate, p=0.05, random=True, rounded=False, scaler=1.0
    ):
        n_points = int(duration * srate)
        x_points = np.linspace(start, start + duration, n_points)
        y = np.zeros(n_points)

        # Количество пиков
        n_peaks = int(n_points * p)

        if random:
            peak_indices = np.random.choice(n_points, size=n_peaks, replace=False)
        else:
            if n_peaks > 0:
                step = n_points / n_peaks
                peak_indices = np.floor(np.arange(0, n_points, step)).astype(int)
                peak_indices = peak_indices[:n_peaks]

        if rounded:
            y[peak_indices] = 1.0
        else:
            y[peak_indices] = np.random.random(size=n_peaks)

        y *= scaler

        return x_points, y


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

    return make_picks_signal, make_signal_by_points


@app.cell
def _(Plotter, SignalProcessor, make_picks_signal, np):
    def make_std_filter():
        def func(t, y):
            E = np.mean(y)
            S = np.std(y)
            mask = np.abs(y - E) <= S
            return t[mask], y[mask], np.argwhere(mask), mask

        return func


    def stage1():
        P = Plotter((18, 6), [["A", "B"]])

        start = 0
        dur = 1
        srate = 1000
        t, signal = make_picks_signal(start, dur, srate, p=0.9, scaler=6.0)
        _, drops = make_picks_signal(start, dur, srate, p=0.03, scaler=60)
        signal += drops

        sig = SignalProcessor()
        sig.t = t.copy()
        sig.signal = signal.copy()
        sig.duration = dur
        sig.start = start
        sig.sampling_rate = srate
        sig.n_points = int(srate * dur)

        P.A.plot(
            sig.t, sig.signal, marker="s", color="black", label="Signal with drops"
        )
        E = sig.signal.mean()
        S = sig.signal.std()
        s3 = 1.0 * S
        P.A.axhline(y=sig.signal.mean(), color="red", linestyle="--", label="Mean")
        P.A.fill_between(t, E - s3, E + s3, alpha=0.3, color="gray")
        P.format("A", "Signal", "Time", "Value")

        sig1 = sig.copy()
        sig1.filter_by_points(make_std_filter())
        P.B.plot(
            sig.t, sig.signal, marker="s", color="black", label="Signal with drops"
        )
        P.B.plot(
            sig1.t,
            sig1.signal,
            marker="s",
            color="red",
            label="Filtered signal drops",
        )
        P.format("B", "Filtered Signal", "Time", "Value")

        return P.get_fig()


    stage1()
    return


@app.cell
def _(Plotter, SignalProcessor, make_signal_by_points, np):
    def make_noise(n, ampl=(0, 1.0)):
        return np.random.normal(ampl[0], ampl[1], n)


    def make_window_RMS(k=3):
        def func(t, y):
            n = len(y)
            pad_y = np.pad(y, k // 2, mode="edge")
            keep = np.zeros(n, dtype=bool)

            for i in range(n):
                window = pad_y[i : i + k]
                E = window.mean()
                rms = np.sqrt(np.mean((E - window) ** 2))
                if rms < 3.0:
                    keep[i] = True

            idx = np.where(keep)[0]
            return t[idx], y[idx], idx, keep

        return func


    def stage2():
        P = Plotter((24, 8), [["A", "B"]])

        srate = 1.0

        keyX = [0, 150, 260, 600, 760, 850, 1100, 1400, 1500, 1700, 1850, 2000]
        keyY = [0, 25, 0, 18, 5, 28, 7, 10, 0, 20, 16, 15]
        dur = max(keyX)
        start = min(keyX)
        t, signal = make_signal_by_points(keyX, keyY, start, dur, srate)

        sig = SignalProcessor()
        sig.t = t.copy()
        sig.signal = signal.copy()
        sig.duration = dur
        sig.start = start
        sig.sampling_rate = srate
        sig.n_points = int(srate * dur)

        sig.add_white_noise(1.0)

        n1 = 80
        s1 = 1550
        sig.signal[s1 : s1 + n1] += make_noise(n1, (-5, 10)) + np.mean(
            sig.signal[s1 : s1 + n1]
        )

        n2 = 20
        s2 = 240
        sig.signal[s2 : s2 + n2] += make_noise(n2, (0, 10)) + np.mean(
            sig.signal[s2 : s2 + n2]
        )

        P.A.plot(sig.t, sig.signal, color="blue", label="Signal")
        P.format("A", "Signal", "Time", "Value")

        sig1 = sig.copy()
        idx, mask = sig1.filter_by_points(make_window_RMS(21))

        P.B.plot(sig.t, sig.signal, color="blue", label="Signal")
        P.B.plot(sig1.t, sig1.signal, color="red", label="Filtered")
        P.format("B", "Filtered Signal", "Time", "Value")

        return P.get_fig(), mask, sig, sig1


    fig, MASK, S, S1 = stage2()
    fig
    return MASK, S, S1


@app.cell
def _(MASK, Plotter, S, S1, np, scipy):
    def find_crops(t, mask):
        start_p = 0
        pairs = []
        flag = False
        for i in range(1, len(mask)):
            if mask[i - 1] and not mask[i] and not flag:
                start_p = i
                flag = True
                continue

            if not mask[i - 1] and mask[i]:
                pairs.append((start_p, i))
                flag = False

        if flag:
            pairs.append((start_p, len(mask) - 1))

        new_pairs = []
        acc = 0
        for p1, p2 in pairs:
            new_p = p1 - acc
            acc += p2 - p1 + 1
            new_pairs.append((p1, p2, new_p))

        return new_pairs


    def join_signal_fft(signal, crop, srate):
        p1, p2 = crop
        k = p2 - p1 + 1
        k1 = k - 2

        left = signal[p1 - k1 : p1].copy()
        right = signal[p2 + 1 : p2 + 1 + k1].copy()
        n0 = estimate_n0(left, right)

        # sos = scipy.signal.butter(10, 0.3, btype="low", fs=srate, output="sos")

        # left_filt = scipy.signal.sosfiltfilt(sos, left)
        # right_filt = scipy.signal.sosfiltfilt(sos, right)

        freq = np.fft.fftfreq(k1, 1 / srate)
        # f1 = np.fft.fft(left_filt)
        # f2 = np.fft.fft(right_filt)
        f1 = np.fft.fft(left)
        f2 = np.fft.fft(right)

        m = np.arange(k1)

        f2a = np.exp(+1j * m * 2 * np.pi * n0 / k1)
        alpha = np.vdot(f1, f2a) / np.vdot(f2a, f2a)
        f2a *= alpha

        wave = (f1 + f2a) * 0.5
        sig = np.fft.ifft(wave).real

        sig = np.pad(sig, 1)
        sig[0] = (left[-1] + sig[1]) / 2
        sig[-1] = (right[0] + sig[-2]) / 2

        return freq, f1, f2, wave, sig


    def refill_crop(t, sig, ref, p1, p2, idx):
        sig = np.concatenate([sig[:idx], ref, sig[idx + 1 :]])
        t = np.concatenate(
            [t[:idx], np.linspace(p1, p2, p2 - p1 + 1), t[idx + 1 :]]
        )
        return t, sig


    def estimate_n0(x1, x2):
        c = scipy.signal.correlate(x1 - x1.mean(), x2 - x2.mean(), mode="full")
        n0 = np.argmax(c) - (len(x2) - 1)
        return n0


    def align_spectrum(f1, f2):
        c = np.fft.ifft(f1 * np.conj(f2))
        shift = np.argmax(np.abs(c))
        k = len(f1)
        n = np.arange(k)
        f2a = f2 * np.exp(1j * 2 * np.pi * shift * n / k)
        return f2a, shift


    def stage3():
        P = Plotter((24, 8), [["A", "B"]])

        srate = 1.0

        crops = find_crops(S.t, MASK)
        p1, p2, pi = crops[1]
        freq, f1, f2, wave, sig = join_signal_fft(S1.signal, (p1, p2), srate)

        N = len(freq)
        idx = N // 2
        P.A.plot(freq[:idx], np.abs(f1)[:idx], linewidth=1.5, label="Left fft")
        P.A.plot(freq[:idx], np.abs(f2)[:idx], linewidth=1.5, label="Right fft")
        P.A.plot(freq[:idx], np.abs(wave)[:idx], linewidth=3, label="Mean fft")
        P.format("A", "Spectrum", "Freq", "Ampl")

        P.B.plot(S1.t, S1.signal, linewidth=4, label="Cropped Signal")
        sig2 = S1.copy()

        tmpX, tmp = refill_crop(sig2.t, sig2.signal, sig, p1, p2, pi)

        sig2.signal = tmp.copy()
        sig2.t = tmpX.copy()
        P.B.plot(sig2.t, sig2.signal, linewidth=1, label="Restored Signal")
        P.format("B", "Restored Signal", "Time", "Value")

        return P.get_fig()


    stage3()
    return


@app.cell
def _(Plotter, SignalProcessor, np):
    def cos_signal(A, F):
        def func(t):
            res = 0
            for a, f in zip(A, F):
                w = 2 * np.pi * f
                res += a * np.cos(t * w)
            return res

        return func, max(F)


    def stage4():
        P = Plotter((18, 6), [["A"]])

        A = [1, 1]
        F = [50, 100]
        signal_f, main_F = cos_signal(A, F)

        sig = SignalProcessor(signal_f, main_F)
        sig.sample(0, 10 / main_F, resolution=100)
        sig.add_white_noise(0.3)

        P.A.plot(sig.t, sig.signal, label="Original signal")

        sig1 = sig.copy()
        sig1.downsample(5)

        P.A.plot(sig1.t, sig1.signal, color="red", label="Downsampled signal")

        P.format("A", "Signal", "Time", "Value")

        return P.get_fig()


    stage4()
    return (cos_signal,)


@app.cell
def _(Plotter, SignalProcessor, cos_signal):
    def stage5():
        P = Plotter((18, 12), [["A"], ["B"]])

        def make_sig(A, F):
            signal_f, main_F = cos_signal([A], [F])
            sig = SignalProcessor(signal_f, main_F)
            sig.sample(0, 10 / 35, resolution=100)
            # sig.add_white_noise(0.1)
            return sig

        s1 = make_sig(1, 10)
        s2 = make_sig(1, 35)
        s3 = make_sig(1, 80)

        # 2 * 80 = 160
        #

        P.A.plot(s1.t, s1.signal, marker="s", label="10 Hz")
        P.A.plot(s2.t, s2.signal, marker="s", label="35 Hz")
        P.A.plot(s3.t, s3.signal, marker="s", label="80 Hz")
        P.format("A", "Signal", "Time", "Value")

        s11 = s1.copy()
        s11.downsample(40, filter=False)
        s21 = s2.copy()
        s21.downsample(11, filter=False)
        s31 = s3.copy()
        s31.downsample(5, filter=False)

        P.B.plot(s11.t, s11.signal, marker="s", label="10 Hz")
        P.B.plot(s21.t, s21.signal, marker="s", label="35 Hz")
        P.B.plot(s31.t, s31.signal, marker="s", label="80 Hz")
        P.format("B", "Signal", "Time", "Value")

        return P.get_fig()


    stage5()
    return


if __name__ == "__main__":
    app.run()

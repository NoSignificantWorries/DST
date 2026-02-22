import marimo

__generated_with = "0.19.9"
app = marimo.App(width="full", auto_download=["html"])


@app.cell
def _():
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy import signal

    return np, plt, signal


@app.cell
def _(np, signal):
    class Signal:
        def __init__(self, func, params):
            self.func = func
            self.params = params

            self.fft = None
            self.freq = None
            self.x = None
            self.signal = None
            self.sampling_rate = None

            self.low_filter = None
            self.high_filter = None

        def make_signal(self, duration, sampling_rate_koeff=10):
            self.sampling_rate = sampling_rate_koeff * self.params["max-freq"]
            n_points = int(duration * self.sampling_rate)

            self.x = np.linspace(0, duration, n_points)
            self.signal = self.func(self.x)

            return self.x, self.signal

        def add_white_noise(self):
            self.signal += np.random.normal(0, 0.5, len(self.signal))
            return self.signal

        def get_fft_spec(self):
            self.fft = np.fft.fft(self.signal)
            n = len(self.signal)

            ampl_spec = np.abs(self.fft) / n

            ampl_spec_one_side = ampl_spec[: n // 2] * 2
            ampl_spec_one_side[0] /= 2

            self.freq = np.fft.fftfreq(n, 1 / self.sampling_rate)
            freq_one_side = self.freq[: n // 2]

            return freq_one_side, ampl_spec_one_side

        def restore_signal(self):
            self.signal = np.fft.ifft(self.fft).real
            return self.signal

        def apply_butterworth_5th_order(self, fc):
            nyquist = self.sampling_rate / 2
            normalized_fc = fc / nyquist

            b, a = signal.butter(5, normalized_fc, btype="low", analog=False)
            self.signal = signal.filtfilt(b, a, self.signal)

            self.get_fft_spec()
            return self.signal

        def batterwort_low_filter(self, n, fc):
            poles = []
            for k in range(n):
                if n % 2 == 0:
                    angle = np.pi / n * (0.5 + k)
                else:
                    angle = np.pi * k / n

                pole = np.exp(1j * angle)

                poles.append(pole)
                poles.append(1 / pole)

            poles = np.array(poles, dtype=complex)

            H = np.ones_like(self.freq, dtype=complex)

            for i, f in enumerate(self.freq):
                s = 1j * f / fc
                H[i] = 1.0 / np.prod(s - poles)

            self.low_filter = H

            return self.low_filter

        def batterwort_low_filter_2(self, fc):
            O = np.abs(self.freq) / fc
            self.low_filter = 1.0 / (-(O**2) + 1j * np.sqrt(2) * O + 1)
            return self.low_filter

        def batterwort_high_filter_2(self, fc):
            O = fc / (np.abs(self.freq) + 1e-8)
            self.high_filter = 1.0 / (-(O**2) + 1j * np.sqrt(2) * O + 1)
            return self.high_filter

        def cavity_filter_2(self, fc_low, fc_high):
            self.batterwort_low_filter_2(fc_low)
            self.batterwort_high_filter_2(fc_high)

            raw_filter = self.low_filter * self.high_filter

            fc_mid = (fc_low + fc_high) / 2
            mid_idx = np.argmin(np.abs(self.freq - fc_mid))

            mid_gain = np.abs(raw_filter[mid_idx])

            if mid_gain > 0:
                self.cavity_filter = raw_filter / mid_gain
            else:
                self.cavity_filter = raw_filter

            return self.cavity_filter

        def notch_filter_2(self):
            return self.low_filter + self.high_filter

        def apply_filter(self, fft_filter):
            self.fft *= fft_filter
            return self.restore_signal()

        def apply_filter_zero_phase(self, fft_filter):
            self.fft *= fft_filter
            filtered_forward = self.restore_signal()

            self.signal = filtered_forward[::-1]
            self.get_fft_spec()
            self.fft *= fft_filter
            filtered_backward = self.restore_signal()

            self.signal = filtered_backward[::-1]
            return self.signal

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
def _(Signal, cos_signal, np, plt):
    def stage1():
        ampls = [1, 1, 1]
        freqs = [50, 150, 450]
        signal, max_freq = cos_signal(ampls, freqs)
        signal = Signal(signal, {"max-freq": max_freq})

        duration = 1 / max_freq * 20
        x, y = signal.make_signal(duration, 25)

        fft_freq, fft_spec = signal.get_fft_spec()

        fig = plt.figure(constrained_layout=True, figsize=(30, 24))
        axes = fig.subplot_mosaic(
            [
                ["A", "B"],
                ["C", "D"],
            ]
        )

        axes["A"].set_title("Signal")
        for a, f in zip(ampls, freqs):
            axes["A"].plot(
                x,
                cos_signal([a], [f])[0](x),
                linewidth=1,
                linestyle="--",
                label=f"{f} Hz",
            )
        axes["A"].plot(x, y, color="blue", linewidth=2, label="cos signal")

        axes["A"].set_xlabel("Time")
        axes["A"].set_ylabel("Value")
        axes["A"].grid(True, alpha=0.3)
        axes["A"].legend()

        axes["B"].set_title("fft")
        mask = fft_freq <= int(max_freq * 1.2)
        axes["B"].plot(fft_freq[mask], fft_spec[mask], label="fft spectrum")

        for f in freqs:
            axes["B"].axvline(
                x=f, linestyle="--", color="r", linewidth=1, label=f"{f} Hz"
            )

        axes["B"].set_xlabel("Frequency")
        axes["B"].set_ylabel("Amplitude")
        axes["B"].grid(True, alpha=0.3)
        axes["B"].legend()

        low_filter_fft = signal.batterwort_low_filter_2(120)
        filtered_s = signal.apply_filter(low_filter_fft)

        axes["C"].set_title("Signal")
        axes["C"].plot(
            x,
            cos_signal([1], [50])[0](x),
            linewidth=1,
            linestyle="--",
            label="50 Hz",
        )
        axes["C"].plot(
            x, filtered_s, color="blue", linewidth=2, label="cos signal"
        )

        axes["C"].set_xlabel("Time")
        axes["C"].set_ylabel("Value")
        axes["C"].grid(True, alpha=0.3)
        axes["C"].legend()

        f_freq, f_fft = signal.get_fft_spec()
        axes["D"].set_title("fft")
        mask = f_freq <= int(max_freq * 1.2)
        axes["D"].plot(f_freq[mask], f_fft[mask], label="fft spectrum")

        axes["D"].axvline(
            x=50, linestyle="--", color="r", linewidth=1, label="50 Hz"
        )

        axes["D"].set_title("Filter fft")
        mask = (signal.freq <= int(max_freq * 1.2)) & (signal.freq >= 0)
        axes["D"].plot(
            signal.freq[mask], np.abs(low_filter_fft[mask]), color="orange"
        )
        axes["D"].set_xlabel("Frequency")
        axes["D"].set_ylabel("Amplitude")
        axes["D"].grid(True, alpha=0.3)
        axes["D"].legend()

        fig.tight_layout()

        return fig


    stage1()
    return


@app.cell
def _(Signal, cos_signal, np, plt):
    def stage2():
        ampls = [1, 1, 1]
        freqs = [50, 150, 450]
        signal, max_freq = cos_signal(ampls, freqs)
        signal = Signal(signal, {"max-freq": max_freq})

        duration = 1 / max_freq * 20
        x, y = signal.make_signal(duration, 25)

        fft_freq, fft_spec = signal.get_fft_spec()

        fig = plt.figure(constrained_layout=True, figsize=(30, 18))
        axes = fig.subplot_mosaic(
            [
                ["A", "B"],
                ["C", "D"],
            ]
        )

        axes["A"].set_title("Signal")
        for a, f in zip(ampls, freqs):
            axes["A"].plot(
                x,
                cos_signal([a], [f])[0](x),
                linewidth=1,
                linestyle="--",
                label=f"{f} Hz",
            )
        axes["A"].plot(x, y, color="blue", linewidth=2, label="cos signal")

        axes["A"].set_xlabel("Time")
        axes["A"].set_ylabel("Value")
        axes["A"].grid(True, alpha=0.3)
        axes["A"].legend()

        axes["B"].set_title("fft")
        mask = fft_freq <= int(max_freq * 1.2)
        axes["B"].plot(fft_freq[mask], fft_spec[mask], label="fft spectrum")

        for f in freqs:
            axes["B"].axvline(
                x=f, linestyle="--", color="r", linewidth=1, label=f"{f} Hz"
            )

        axes["B"].set_xlabel("Frequency")
        axes["B"].set_ylabel("Amplitude")
        axes["B"].grid(True, alpha=0.3)
        axes["B"].legend()

        high_filter_fft = signal.batterwort_high_filter_2(150)
        filtered_s = signal.apply_filter(high_filter_fft)

        axes["C"].set_title("Signal")
        axes["C"].plot(
            x,
            cos_signal([1], [450])[0](x),
            linewidth=1,
            linestyle="--",
            label="450 Hz",
        )
        axes["C"].plot(
            x, filtered_s, color="blue", linewidth=2, label="cos signal"
        )

        axes["C"].set_xlabel("Time")
        axes["C"].set_ylabel("Value")
        axes["C"].grid(True, alpha=0.3)
        axes["C"].legend()

        f_freq, f_fft = signal.get_fft_spec()
        axes["D"].set_title("fft")
        mask = f_freq <= int(max_freq * 1.2)
        axes["D"].plot(f_freq[mask], f_fft[mask], label="fft spectrum")

        axes["D"].axvline(
            x=450, linestyle="--", color="r", linewidth=1, label="450 Hz"
        )

        axes["D"].set_title("Filter fft")
        mask = (signal.freq <= int(max_freq * 1.2)) & (signal.freq >= 0)
        axes["D"].plot(
            signal.freq[mask], np.abs(high_filter_fft[mask]), color="orange"
        )
        axes["D"].set_xlabel("Frequency")
        axes["D"].set_ylabel("Amplitude")
        axes["D"].grid(True, alpha=0.3)
        axes["D"].legend()

        fig.tight_layout()

        return fig


    stage2()
    return


@app.cell
def _(Signal, cos_signal, np, plt):
    def stage3():
        ampls = [1, 1, 1]
        freqs = [50, 150, 450]
        signal, max_freq = cos_signal(ampls, freqs)
        signal = Signal(signal, {"max-freq": max_freq})

        duration = 1 / max_freq * 20
        x, y = signal.make_signal(duration, 25)

        fft_freq, fft_spec = signal.get_fft_spec()

        fig = plt.figure(constrained_layout=True, figsize=(30, 18))
        axes = fig.subplot_mosaic(
            [
                ["A", "B"],
                ["C", "D"],
            ]
        )

        axes["A"].set_title("Signal")
        for a, f in zip(ampls, freqs):
            axes["A"].plot(
                x,
                cos_signal([a], [f])[0](x),
                linewidth=1,
                linestyle="--",
                label=f"{f} Hz",
            )
        axes["A"].plot(x, y, color="blue", linewidth=2, label="cos signal")

        axes["A"].set_xlabel("Time")
        axes["A"].set_ylabel("Value")
        axes["A"].grid(True, alpha=0.3)
        axes["A"].legend()

        axes["B"].set_title("fft")
        mask = fft_freq <= int(max_freq * 1.2)
        axes["B"].plot(fft_freq[mask], fft_spec[mask], label="fft spectrum")

        for f in freqs:
            axes["B"].axvline(
                x=f, linestyle="--", color="r", linewidth=1, label=f"{f} Hz"
            )

        axes["B"].set_xlabel("Frequency")
        axes["B"].set_ylabel("Amplitude")
        axes["B"].grid(True, alpha=0.3)
        axes["B"].legend()

        cavity_filter_fft = signal.cavity_filter_2(120, 150)
        filtered_s = signal.apply_filter_zero_phase(cavity_filter_fft)

        axes["C"].set_title("Signal")
        axes["C"].plot(
            x,
            cos_signal([1], [150])[0](x),
            linewidth=1,
            linestyle="--",
            label="450 Hz",
        )
        axes["C"].plot(
            x, filtered_s, color="blue", linewidth=2, label="cos signal"
        )

        axes["C"].set_xlabel("Time")
        axes["C"].set_ylabel("Value")
        axes["C"].grid(True, alpha=0.3)
        axes["C"].legend()

        f_freq, f_fft = signal.get_fft_spec()
        axes["D"].set_title("fft")
        mask = f_freq <= int(max_freq * 1.2)
        axes["D"].plot(f_freq[mask], f_fft[mask], label="fft spectrum")

        axes["D"].axvline(
            x=150, linestyle="--", color="r", linewidth=1, label="150 Hz"
        )

        axes["D"].set_title("Filter fft")
        mask = (signal.freq <= int(max_freq * 1.2)) & (signal.freq >= 0)
        axes["D"].plot(
            signal.freq[mask], np.abs(cavity_filter_fft[mask]), color="orange"
        )
        axes["D"].set_xlabel("Frequency")
        axes["D"].set_ylabel("Amplitude")
        axes["D"].grid(True, alpha=0.3)
        axes["D"].legend()

        fig.tight_layout()

        return fig


    stage3()
    return


@app.cell
def _(Signal, cos_signal, np, plt):
    def stage4():
        ampls = [1, 1, 1]
        freqs = [50, 150, 450]
        signal, max_freq = cos_signal(ampls, freqs)
        signal = Signal(signal, {"max-freq": max_freq})

        duration = 1 / max_freq * 20
        x, y = signal.make_signal(duration, 25)

        fft_freq, fft_spec = signal.get_fft_spec()

        fig = plt.figure(constrained_layout=True, figsize=(30, 18))
        axes = fig.subplot_mosaic(
            [
                ["A", "B"],
                ["C", "D"],
            ]
        )

        axes["A"].set_title("Signal")
        for a, f in zip(ampls, freqs):
            axes["A"].plot(
                x,
                cos_signal([a], [f])[0](x),
                linewidth=1,
                linestyle="--",
                label=f"{f} Hz",
            )
        axes["A"].plot(x, y, color="blue", linewidth=2, label="cos signal")

        axes["A"].set_xlabel("Time")
        axes["A"].set_ylabel("Value")
        axes["A"].grid(True, alpha=0.3)
        axes["A"].legend()

        axes["B"].set_title("fft")
        mask = fft_freq <= int(max_freq * 1.2)
        axes["B"].plot(fft_freq[mask], fft_spec[mask], label="fft spectrum")

        for f in freqs:
            axes["B"].axvline(
                x=f, linestyle="--", color="r", linewidth=1, label=f"{f} Hz"
            )

        axes["B"].set_xlabel("Frequency")
        axes["B"].set_ylabel("Amplitude")
        axes["B"].grid(True, alpha=0.3)
        axes["B"].legend()

        low_filter_fft = signal.batterwort_low_filter_2(120)
        high_filter_fft = signal.batterwort_high_filter_2(150)
        notch_filter_fft = signal.notch_filter_2()
        filtered_s = signal.apply_filter(notch_filter_fft)

        axes["C"].set_title("Signal")
        axes["C"].plot(
            x,
            cos_signal([1, 1], [50, 450])[0](x),
            linewidth=1,
            linestyle="--",
            label="50 + 450 Hz",
        )
        axes["C"].plot(
            x, filtered_s, color="blue", linewidth=2, label="cos signal"
        )

        axes["C"].set_xlabel("Time")
        axes["C"].set_ylabel("Value")
        axes["C"].grid(True, alpha=0.3)
        axes["C"].legend()

        f_freq, f_fft = signal.get_fft_spec()
        axes["D"].set_title("fft")
        mask = f_freq <= int(max_freq * 1.2)
        axes["D"].plot(f_freq[mask], f_fft[mask], label="fft spectrum")

        axes["D"].axvline(
            x=50, linestyle="--", color="r", linewidth=1, label="50 Hz"
        )
        axes["D"].axvline(
            x=450, linestyle="--", color="r", linewidth=1, label="150 Hz"
        )

        axes["D"].set_title("Filter fft")
        mask = (signal.freq <= int(max_freq * 1.2)) & (signal.freq >= 0)
        axes["D"].plot(
            signal.freq[mask], np.abs(notch_filter_fft[mask]), color="orange"
        )
        axes["D"].set_xlabel("Frequency")
        axes["D"].set_ylabel("Amplitude")
        axes["D"].grid(True, alpha=0.3)
        axes["D"].legend()

        fig.tight_layout()

        return fig


    stage4()
    return


@app.cell
def _(Signal, cos_signal, np, plt):
    def stage5():
        ampls = [1, 1, 1]
        freqs = [50, 150, 450]
        signal, max_freq = cos_signal(ampls, freqs)
        signal = Signal(signal, {"max-freq": max_freq})

        duration = 1 / max_freq * 20
        x, y = signal.make_signal(duration, 25)

        fft_freq, fft_spec = signal.get_fft_spec()

        fig = plt.figure(constrained_layout=True, figsize=(30, 24))
        axes = fig.subplot_mosaic(
            [
                ["A", "B"],
                ["C", "D"],
            ]
        )

        axes["A"].set_title("Signal")
        for a, f in zip(ampls, freqs):
            axes["A"].plot(
                x,
                cos_signal([a], [f])[0](x),
                linewidth=1,
                linestyle="--",
                label=f"{f} Hz",
            )
        axes["A"].plot(x, y, color="blue", linewidth=2, label="cos signal")

        axes["A"].set_xlabel("Time")
        axes["A"].set_ylabel("Value")
        axes["A"].grid(True, alpha=0.3)
        axes["A"].legend()

        axes["B"].set_title("fft")
        mask = fft_freq <= int(max_freq * 1.2)
        axes["B"].plot(fft_freq[mask], fft_spec[mask], label="fft spectrum")

        for f in freqs:
            axes["B"].axvline(
                x=f, linestyle="--", color="r", linewidth=1, label=f"{f} Hz"
            )

        axes["B"].set_xlabel("Frequency")
        axes["B"].set_ylabel("Amplitude")
        axes["B"].grid(True, alpha=0.3)
        axes["B"].legend()

        low_filter_fft = signal.batterwort_low_filter(4, 65)
        filtered_s = signal.apply_filter(low_filter_fft)

        axes["C"].set_title("Signal")
        axes["C"].plot(
            x,
            cos_signal([1], [50])[0](x),
            linewidth=1,
            linestyle="--",
            label="50 Hz",
        )
        axes["C"].plot(
            x, filtered_s, color="blue", linewidth=2, label="cos signal"
        )

        axes["C"].set_xlabel("Time")
        axes["C"].set_ylabel("Value")
        axes["C"].grid(True, alpha=0.3)
        axes["C"].legend()

        f_freq, f_fft = signal.get_fft_spec()
        axes["D"].set_title("fft")
        mask = f_freq <= int(max_freq * 1.2)
        axes["D"].plot(f_freq[mask], f_fft[mask], label="fft spectrum")

        axes["D"].axvline(
            x=50, linestyle="--", color="r", linewidth=1, label="50 Hz"
        )

        axes["D"].set_title("Filter fft")
        mask = (signal.freq <= int(max_freq * 1.2)) & (signal.freq >= 0)
        axes["D"].plot(
            signal.freq[mask], np.abs(low_filter_fft[mask]), color="orange"
        )
        axes["D"].set_xlabel("Frequency")
        axes["D"].set_ylabel("Amplitude")
        axes["D"].grid(True, alpha=0.3)
        axes["D"].legend()

        fig.tight_layout()

        return fig


    stage5()
    return


@app.cell
def _(Signal, cos_signal, np, plt):
    def stage6():
        ampls = [1, 1, 1]
        freqs = [50, 150, 450]
        signal, max_freq = cos_signal(ampls, freqs)
        signal = Signal(signal, {"max-freq": max_freq})

        duration = 1 / max_freq * 20
        x, y = signal.make_signal(duration, 25)

        fft_freq, fft_spec = signal.get_fft_spec()

        fig = plt.figure(constrained_layout=True, figsize=(30, 24))
        axes = fig.subplot_mosaic(
            [
                ["A", "B"],
                ["C", "D"],
            ]
        )

        axes["A"].set_title("Signal")
        for a, f in zip(ampls, freqs):
            axes["A"].plot(
                x,
                cos_signal([a], [f])[0](x),
                linewidth=1,
                linestyle="--",
                label=f"{f} Hz",
            )
        axes["A"].plot(x, y, color="blue", linewidth=2, label="cos signal")

        axes["A"].set_xlabel("Time")
        axes["A"].set_ylabel("Value")
        axes["A"].grid(True, alpha=0.3)
        axes["A"].legend()

        axes["B"].set_title("fft")
        mask = fft_freq <= int(max_freq * 1.2)
        axes["B"].plot(fft_freq[mask], fft_spec[mask], label="fft spectrum")

        for f in freqs:
            axes["B"].axvline(
                x=f, linestyle="--", color="r", linewidth=1, label=f"{f} Hz"
            )

        axes["B"].set_xlabel("Frequency")
        axes["B"].set_ylabel("Amplitude")
        axes["B"].grid(True, alpha=0.3)
        axes["B"].legend()

        low_filter_fft = signal.batterwort_low_filter(5, 65)
        filtered_s = signal.apply_filter(low_filter_fft)

        axes["C"].set_title("Signal")
        axes["C"].plot(
            x,
            cos_signal([1], [50])[0](x),
            linewidth=1,
            linestyle="--",
            label="50 Hz",
        )
        axes["C"].plot(
            x, filtered_s, color="blue", linewidth=2, label="cos signal"
        )

        axes["C"].set_xlabel("Time")
        axes["C"].set_ylabel("Value")
        axes["C"].grid(True, alpha=0.3)
        axes["C"].legend()

        f_freq, f_fft = signal.get_fft_spec()
        axes["D"].set_title("fft")
        mask = f_freq <= int(max_freq * 1.2)
        axes["D"].plot(f_freq[mask], f_fft[mask], label="fft spectrum")

        axes["D"].axvline(
            x=50, linestyle="--", color="r", linewidth=1, label="50 Hz"
        )

        axes["D"].set_title("Filter fft")
        mask = (signal.freq <= int(max_freq * 1.2)) & (signal.freq >= 0)
        axes["D"].plot(
            signal.freq[mask], np.abs(low_filter_fft[mask]), color="orange"
        )
        axes["D"].set_xlabel("Frequency")
        axes["D"].set_ylabel("Amplitude")
        axes["D"].grid(True, alpha=0.3)
        axes["D"].legend()

        fig.tight_layout()

        return fig


    stage6()
    return


@app.cell
def _(Signal, cos_signal, plt):
    def stage7():
        ampls = [1, 1, 1]
        freqs = [50, 150, 450]
        signal, max_freq = cos_signal(ampls, freqs)
        signal = Signal(signal, {"max-freq": max_freq})

        duration = 1 / max_freq * 20
        x, y = signal.make_signal(duration, 25)

        fft_freq, fft_spec = signal.get_fft_spec()

        fig = plt.figure(constrained_layout=True, figsize=(30, 24))
        axes = fig.subplot_mosaic(
            [
                ["A", "B"],
                ["C", "D"],
            ]
        )

        axes["A"].set_title("Signal")
        for a, f in zip(ampls, freqs):
            axes["A"].plot(
                x,
                cos_signal([a], [f])[0](x),
                linewidth=1,
                linestyle="--",
                label=f"{f} Hz",
            )
        axes["A"].plot(x, y, color="blue", linewidth=2, label="cos signal")

        axes["A"].set_xlabel("Time")
        axes["A"].set_ylabel("Value")
        axes["A"].grid(True, alpha=0.3)
        axes["A"].legend()

        axes["B"].set_title("fft")
        mask = fft_freq <= int(max_freq * 1.2)
        axes["B"].plot(fft_freq[mask], fft_spec[mask], label="fft spectrum")

        for f in freqs:
            axes["B"].axvline(
                x=f, linestyle="--", color="r", linewidth=1, label=f"{f} Hz"
            )

        axes["B"].set_xlabel("Frequency")
        axes["B"].set_ylabel("Amplitude")
        axes["B"].grid(True, alpha=0.3)
        axes["B"].legend()

        signal.apply_butterworth_5th_order(65)

        axes["C"].set_title("Signal")
        axes["C"].plot(
            x,
            cos_signal([1], [50])[0](x),
            linewidth=1,
            linestyle="--",
            label="50 Hz",
        )
        axes["C"].plot(
            x, signal.signal, color="blue", linewidth=2, label="cos signal"
        )

        f_freq, f_fft = signal.get_fft_spec()
        axes["D"].set_title("fft")
        mask = f_freq <= int(max_freq * 1.2)
        axes["D"].plot(f_freq[mask], f_fft[mask], label="fft spectrum")

        axes["D"].axvline(
            x=50, linestyle="--", color="r", linewidth=1, label="50 Hz"
        )

        axes["D"].set_xlabel("Frequency")
        axes["D"].set_ylabel("Amplitude")
        axes["D"].grid(True, alpha=0.3)
        axes["D"].legend()

        fig.tight_layout()

        return fig


    stage7()
    return


@app.cell
def _(Signal, cos_signal, np, plt):
    def stage8():
        ampls = [1, 1, 1]
        freqs = [50, 150, 450]
        signal, max_freq = cos_signal(ampls, freqs)
        signal = Signal(signal, {"max-freq": max_freq})

        duration = 1 / max_freq * 20
        x, y = signal.make_signal(duration, 25)
        signal.add_white_noise()

        fft_freq, fft_spec = signal.get_fft_spec()

        fig = plt.figure(constrained_layout=True, figsize=(30, 24))
        axes = fig.subplot_mosaic(
            [
                ["A", "B"],
                ["C", "D"],
            ]
        )

        axes["A"].set_title("Signal")
        for a, f in zip(ampls, freqs):
            axes["A"].plot(
                x,
                cos_signal([a], [f])[0](x),
                linewidth=1,
                linestyle="--",
                label=f"{f} Hz",
            )
        axes["A"].plot(x, y, color="blue", linewidth=2, label="cos signal")

        axes["A"].set_xlabel("Time")
        axes["A"].set_ylabel("Value")
        axes["A"].grid(True, alpha=0.3)
        axes["A"].legend()

        axes["B"].set_title("fft")
        mask = fft_freq <= int(max_freq * 1.2)
        axes["B"].plot(fft_freq[mask], fft_spec[mask], label="fft spectrum")

        for f in freqs:
            axes["B"].axvline(
                x=f, linestyle="--", color="r", linewidth=1, label=f"{f} Hz"
            )

        axes["B"].set_xlabel("Frequency")
        axes["B"].set_ylabel("Amplitude")
        axes["B"].grid(True, alpha=0.3)
        axes["B"].legend()

        low_filter_fft = signal.batterwort_low_filter(8, 420)
        filtered_s = signal.apply_filter(low_filter_fft)

        axes["C"].set_title("Signal")
        axes["C"].plot(
            x,
            cos_signal([1, 1], [50, 150])[0](x),
            linewidth=1,
            linestyle="--",
            label="50 Hz",
        )
        axes["C"].plot(
            x, filtered_s, color="blue", linewidth=2, label="cos signal"
        )

        axes["C"].set_xlabel("Time")
        axes["C"].set_ylabel("Value")
        axes["C"].grid(True, alpha=0.3)
        axes["C"].legend()

        f_freq, f_fft = signal.get_fft_spec()
        axes["D"].set_title("fft")
        mask = f_freq <= int(max_freq * 2)
        axes["D"].plot(f_freq[mask], f_fft[mask], label="fft spectrum")

        axes["D"].axvline(
            x=50, linestyle="--", color="r", linewidth=1, label="50 Hz"
        )

        axes["D"].set_title("Filter fft")
        mask = (signal.freq <= int(max_freq * 2)) & (signal.freq >= 0)
        axes["D"].plot(
            signal.freq[mask], np.abs(low_filter_fft[mask]), color="orange"
        )
        axes["D"].set_xlabel("Frequency")
        axes["D"].set_ylabel("Amplitude")
        axes["D"].grid(True, alpha=0.3)
        axes["D"].legend()

        fig.tight_layout()

        return fig


    stage8()
    return


if __name__ == "__main__":
    app.run()

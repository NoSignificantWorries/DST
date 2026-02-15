import marimo

__generated_with = "0.19.9"
app = marimo.App(width="full", auto_download=["html", "ipynb"])


@app.cell
def _():
    import numpy as np
    import matplotlib.pyplot as plt
    from typing import Callable

    return Callable, np, plt


@app.cell
def _(Callable, np):
    def integrate(func: Callable, a: float, b: float, n: int = 1000) -> float:
        x = np.linspace(a, b, n)
        y = func(x)
        return np.trapezoid(y, x)


    class Function:
        def __init__(self, func: Callable, **kwargs):
            for name, val in kwargs.items():
                setattr(self, name, val)
                self.func = func

        def __call__(self, t: float) -> float:
            return self.func(t)


    class FourierSeries:
        def __init__(self, func: Function, N: int = 10):
            self.func = func
            self.T = self.func.T
            self.freq = 1 / self.T
            self.N_harmonics = N
            self.omega = 2 * np.pi / self.T

        def __call__(self, t: float) -> float:
            return self.func(t)

        def get_series(self, t_0: float = 0.0):
            a = t_0
            b = t_0 + self.func.T

            a_0 = 2 * integrate(self.func, a, b) / self.func.T

            coeffs_a = []
            coeffs_b = []

            omega = 2 * np.pi / self.func.T

            for n in range(1, self.N_harmonics + 1):
                integrand_cos = lambda t: self.func(t) * np.cos(n * omega * t)
                a_n = 2 * integrate(integrand_cos, a, b) / self.func.T
                coeffs_a.append(a_n)

                integrand_sin = lambda t: self.func(t) * np.sin(n * omega * t)
                b_n = 2 * integrate(integrand_sin, a, b) / self.func.T
                coeffs_b.append(b_n)

            def fourier_func(t):
                result = a_0 / 2
                for n in range(1, self.N_harmonics + 1):
                    result += coeffs_a[n - 1] * np.cos(n * omega * t) + coeffs_b[
                        n - 1
                    ] * np.sin(n * omega * t)
                return result

            print("an:", list(map(lambda x: round(float(x), 6), coeffs_a)))
            print("bn:", list(map(lambda x: round(float(x), 6), coeffs_b)))
            setattr(self, f"fourier_{self.N_harmonics}", fourier_func)
            return fourier_func, (a_0, coeffs_a, coeffs_b)

        def get_spectrum(self, signal, sampling_rate: int = 1000):
            N = len(signal)
            dt = 1 / sampling_rate

            fft_result = np.fft.fft(signal)

            frequencies = np.fft.fftfreq(N, dt)

            amplitude_spectrum = 2 * np.abs(fft_result) / N
            amplitude_spectrum[0] /= 2

            positive_freq_indices = frequencies >= 0
            frequencies = frequencies[positive_freq_indices]
            amplitude_spectrum = amplitude_spectrum[positive_freq_indices]

            return frequencies, amplitude_spectrum

    return FourierSeries, Function


@app.cell
def _(Callable, np):
    def period_func(T: float) -> Callable:
        def func(t):
            phase = (t % T) / T
            return np.where(phase < 0.5, 1.0, -1.0)

        return func

    return (period_func,)


@app.cell
def _(FourierSeries, Function, np, period_func, plt):
    T = 2
    f_x = Function(period_func(T=T), T=T)
    signal = FourierSeries(f_x)

    fourier_approx, (_, an_, bn_) = signal.get_series()

    X = np.linspace(0, 3 * T, 1000)
    y_original = signal(X)
    y_fourier = fourier_approx(X)

    freq, ampl_spec = signal.get_spectrum(y_original)
    harmonic_freqs_ = (
        np.linspace(1, signal.N_harmonics, signal.N_harmonics) * signal.freq
    )

    fig, axes = plt.subplots(3, 1, figsize=(12, 12))

    axes[0].plot(
        X, y_original, "b-", linewidth=2, label="Исходный сигнал", alpha=0.7
    )
    axes[0].plot(
        X, y_fourier, "r--", linewidth=2, label="Ряд Фурье (N=10)", alpha=0.9
    )
    axes[0].set_ylabel("Амплитуда")
    axes[0].set_title("Прямоугольный импульс и его разложение в ряд Фурье")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    error = y_original - y_fourier
    axes[1].plot(X, error, "g-", linewidth=1, label="Ошибка")
    axes[1].set_xlabel("Время")
    axes[1].set_ylabel("Ошибка")
    axes[1].set_title("Разность между исходной функцией и рядом Фурье")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(freq, ampl_spec)
    axes[2].plot(harmonic_freqs_, an_, "o", color="orange")
    axes[2].plot(harmonic_freqs_, bn_, "o", color="red")

    axes[2].plot(freq, ampl_spec, color="blue", label="Амплитудный спектр")
    axes[2].set_xlabel("Частота")
    axes[2].set_ylabel("Амплитуда")
    axes[2].set_title("Спектр")
    axes[2].plot(harmonic_freqs_, an_, "o", color="orange", label="an")
    axes[2].plot(harmonic_freqs_, bn_, "o", color="red", label="bn")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()
    return


@app.cell
def _(Callable, np):
    def period_func_sin(T: float, A: float) -> Callable:
        omega = 2 * np.pi / T

        def func(t):
            return A * np.cos(omega * t)

        return func

    return (period_func_sin,)


@app.cell
def _(FourierSeries, Function, np, period_func_sin, plt):
    T_sin = 2
    A_sin = 1
    f_x_sin = Function(period_func_sin(T=T_sin, A=A_sin), T=T_sin, A=A_sin)
    signal_sin = FourierSeries(f_x_sin)

    fourier_approx_sin, (_, an, bn) = signal_sin.get_series()

    X_sin = np.linspace(0, 3 * T_sin, 1000)
    y_original_sin = signal_sin(X_sin)
    y_fourier_sin = fourier_approx_sin(X_sin)

    freq_sin, ampl_spec_sin = signal_sin.get_spectrum(y_original_sin)
    harmonic_freqs = (
        np.linspace(1, signal_sin.N_harmonics, signal_sin.N_harmonics)
        * signal_sin.freq
    )

    fig_sin, axes_sin = plt.subplots(3, 1, figsize=(12, 12))

    axes_sin[0].plot(
        X_sin,
        y_original_sin,
        "b-",
        linewidth=2,
        label="Исходный сигнал",
        alpha=0.7,
    )
    axes_sin[0].plot(
        X_sin,
        y_fourier_sin,
        "r--",
        linewidth=2,
        label="Ряд Фурье (N=10)",
        alpha=0.9,
    )
    axes_sin[0].set_ylabel("Амплитуда")
    axes_sin[0].set_title("Косинусоидальный импульс и его разложение в ряд Фурье")
    axes_sin[0].legend()
    axes_sin[0].grid(True, alpha=0.3)

    error_sin = y_original_sin - y_fourier_sin
    axes_sin[1].plot(X_sin, error_sin, "g-", linewidth=1, label="Ошибка")
    axes_sin[1].set_xlabel("Время")
    axes_sin[1].set_ylabel("Ошибка")
    axes_sin[1].set_title("Разность между исходной функцией и рядом Фурье")
    axes_sin[1].legend()
    axes_sin[1].grid(True, alpha=0.3)

    axes_sin[2].plot(
        freq_sin, ampl_spec_sin, color="blue", label="Амплитудный спектр"
    )
    axes_sin[2].set_xlabel("Частота")
    axes_sin[2].set_ylabel("Амплитуда")
    axes_sin[2].set_title("Спектр")
    axes_sin[2].plot(harmonic_freqs, an, "o", color="orange", label="an")
    axes_sin[2].plot(harmonic_freqs, bn, "o", color="red", label="bn")
    axes_sin[2].legend()
    axes_sin[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()
    return


@app.cell
def _(np):
    def add_noise_to_function(original_func, noise_level=0.1, seed=None):
        if seed is not None:
            np.random.seed(seed)

        def noisy_func(t):
            t_array = np.asarray(t)
            original_values = original_func(t_array)

            noise = np.random.normal(0, noise_level, t_array.shape)

            return original_values + noise

        return noisy_func

    return (add_noise_to_function,)


@app.cell
def _(FourierSeries, Function, add_noise_to_function, np, period_func, plt):
    def stage4():
        T = 2
        f_x = Function(add_noise_to_function(period_func(T=T)), T=T)
        signal = FourierSeries(f_x)

        fourier_approx, (_, an_, bn_) = signal.get_series()

        X = np.linspace(0, 3 * T, 1000)
        y_original = signal(X)
        y_fourier = fourier_approx(X)

        freq, ampl_spec = signal.get_spectrum(y_original)
        harmonic_freqs_ = (
            np.linspace(1, signal.N_harmonics, signal.N_harmonics) * signal.freq
        )

        fig, axes = plt.subplots(3, 1, figsize=(12, 12))

        axes[0].plot(
            X, y_original, "b-", linewidth=2, label="Исходный сигнал", alpha=0.7
        )
        axes[0].plot(
            X, y_fourier, "r--", linewidth=2, label="Ряд Фурье (N=10)", alpha=0.9
        )
        axes[0].set_ylabel("Амплитуда")
        axes[0].set_title("Прямоугольный импульс и его разложение в ряд Фурье")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        error = y_original - y_fourier
        axes[1].plot(X, error, "g-", linewidth=1, label="Ошибка")
        axes[1].set_xlabel("Время")
        axes[1].set_ylabel("Ошибка")
        axes[1].set_title("Разность между исходной функцией и рядом Фурье")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        axes[2].plot(freq, ampl_spec)
        axes[2].plot(harmonic_freqs_, an_, "o", color="orange")
        axes[2].plot(harmonic_freqs_, bn_, "o", color="red")

        axes[2].plot(freq, ampl_spec, color="blue", label="Амплитудный спектр")
        axes[2].set_xlabel("Частота")
        axes[2].set_ylabel("Амплитуда")
        axes[2].set_title("Спектр")
        axes[2].plot(harmonic_freqs_, an_, "o", color="orange", label="an")
        axes[2].plot(harmonic_freqs_, bn_, "o", color="red", label="bn")
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()


    stage4()
    return


@app.cell
def _(
    FourierSeries,
    Function,
    add_noise_to_function,
    np,
    period_func_sin,
    plt,
):
    def stage5():
        T_sin = 2
        A_sin = 1
        f_x_sin = Function(
            add_noise_to_function(period_func_sin(T=T_sin, A=A_sin)),
            T=T_sin,
            A=A_sin,
        )
        signal_sin = FourierSeries(f_x_sin)

        fourier_approx_sin, (_, an, bn) = signal_sin.get_series()

        X_sin = np.linspace(0, 3 * T_sin, 1000)
        y_original_sin = signal_sin(X_sin)
        y_fourier_sin = fourier_approx_sin(X_sin)

        freq_sin, ampl_spec_sin = signal_sin.get_spectrum(y_original_sin)
        harmonic_freqs = (
            np.linspace(1, signal_sin.N_harmonics, signal_sin.N_harmonics)
            * signal_sin.freq
        )

        fig_sin, axes_sin = plt.subplots(3, 1, figsize=(12, 12))

        axes_sin[0].plot(
            X_sin,
            y_original_sin,
            "b-",
            linewidth=2,
            label="Исходный сигнал",
            alpha=0.7,
        )
        axes_sin[0].plot(
            X_sin,
            y_fourier_sin,
            "r--",
            linewidth=2,
            label="Ряд Фурье (N=10)",
            alpha=0.9,
        )
        axes_sin[0].set_ylabel("Амплитуда")
        axes_sin[0].set_title(
            "Косинусоидальный импульс и его разложение в ряд Фурье"
        )
        axes_sin[0].legend()
        axes_sin[0].grid(True, alpha=0.3)

        error_sin = y_original_sin - y_fourier_sin
        axes_sin[1].plot(X_sin, error_sin, "g-", linewidth=1, label="Ошибка")
        axes_sin[1].set_xlabel("Время")
        axes_sin[1].set_ylabel("Ошибка")
        axes_sin[1].set_title("Разность между исходной функцией и рядом Фурье")
        axes_sin[1].legend()
        axes_sin[1].grid(True, alpha=0.3)

        axes_sin[2].plot(
            freq_sin, ampl_spec_sin, color="blue", label="Амплитудный спектр"
        )
        axes_sin[2].set_xlabel("Частота")
        axes_sin[2].set_ylabel("Амплитуда")
        axes_sin[2].set_title("Спектр")
        axes_sin[2].plot(harmonic_freqs, an, "o", color="orange", label="an")
        axes_sin[2].plot(harmonic_freqs, bn, "o", color="red", label="bn")
        axes_sin[2].legend()
        axes_sin[2].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()


    stage5()
    return


if __name__ == "__main__":
    app.run()

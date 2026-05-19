import marimo

__generated_with = "0.21.1"
app = marimo.App(width="full", auto_download=["html"])


@app.cell
def _():
    import numpy as np
    import matplotlib.pyplot as plt
    import scipy
    from sklearn.decomposition import FastICA
    import scipy.linalg
    import scipy.io as sio
    from scipy.interpolate import griddata
    from matplotlib import patches
    from numpy.fft import fft, fftfreq

    np.random.seed(42)
    return FastICA, fft, fftfreq, griddata, np, patches, plt, scipy, sio


@app.function
def show_chanels(ax, x_data, channels_data, step=1, **kwargs):
    for i, data in enumerate(channels_data):
        ax.plot(x_data, data + step * i, **kwargs)


@app.cell
def _(np, plt):
    def stage1():
        N_CHANNELS = 10
        DURATION = 500
        SAMPLE_RATE = 1
        N_TIMEPOINTS = SAMPLE_RATE * DURATION
        TIME_AXIS = np.linspace(0, DURATION, N_TIMEPOINTS)
        AMP_1, FREQ_1 = 2.0, 0.05
        AMP_2, FREQ_2 = 1.0, 0.10
        NOISE_LEVEL = 0.5
        VERT_OFFSET = 2.5

        spatial_x = np.linspace(0, 2 * np.pi, N_CHANNELS)
        spatial_1 = np.sin(spatial_x)
        spatial_2 = np.cos(spatial_x)

        temporal_1 = AMP_1 * np.sin(2 * np.pi * FREQ_1 * TIME_AXIS)
        temporal_2 = AMP_2 * np.sin(2 * np.pi * FREQ_2 * TIME_AXIS)

        def generate_data(noise_level):
            clean = np.outer(spatial_1, temporal_1) + np.outer(
                spatial_2, temporal_2
            )
            noise = np.random.randn(N_CHANNELS, N_TIMEPOINTS) * noise_level
            return clean + noise, clean, noise

        noisy_data, clean_data, noise_data = generate_data(NOISE_LEVEL)

        # centering
        mean_per_channel = noisy_data.mean(axis=1, keepdims=True)
        centered_data = noisy_data - mean_per_channel
        # cov matrix
        cov_matrix = (centered_data @ centered_data.T) / (N_TIMEPOINTS - 1)
        # eigen vectors and values
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

        sorted_idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[sorted_idx]
        eigenvectors = eigenvectors[:, sorted_idx]

        print(f"Eigen values: {eigenvalues}")
        print(f"Eigen summ: {np.sum(eigenvalues):.4f}")

        explained_var_ratio = eigenvalues / np.sum(eigenvalues)
        for i, (ev, evr) in enumerate(zip(eigenvalues, explained_var_ratio)):
            print(f"\tPC{i + 1}: lam={ev:.4f}, p={evr:.3f} ({evr * 100:.1f}%)")
        principal_components = eigenvectors.T @ centered_data

        def normalize_component(
            spatial_weights, temporal_pc, ref_signal=None, target_amp=None
        ):
            scale = np.max(np.abs(spatial_weights))
            w_norm = spatial_weights / scale
            t_norm = temporal_pc * scale

            if ref_signal is not None:
                if np.corrcoef(t_norm, ref_signal)[0, 1] < 0:
                    w_norm = -w_norm
                    t_norm = -t_norm

            if target_amp is not None and np.max(np.abs(t_norm)) > 0:
                t_norm = t_norm / np.max(np.abs(t_norm)) * target_amp

            return w_norm, t_norm

        w1, t1 = normalize_component(
            eigenvectors[:, 0], principal_components[0], temporal_1, AMP_1
        )
        w2, t2 = normalize_component(
            eigenvectors[:, 1], principal_components[1], temporal_2, AMP_2
        )
        w3, t3 = normalize_component(eigenvectors[:, 2], principal_components[2])

        corr1_pca = np.corrcoef(t1, temporal_1)[0, 1]
        corr2_pca = np.corrcoef(t2, temporal_2)[0, 1]

        std_pc3 = np.std(t3)

        print(f"PC1 corelation: {corr1_pca:.4f}")
        print(f"PC2 corelation: {corr2_pca:.4f}")
        print(f"Noise std: {std_pc3:.4f}")

        fig, axes = plt.subplot_mosaic(
            [["noisy", "cov"], ["A", "B"], ["C", "C"]],
            figsize=(18, 14),
            width_ratios=[1, 1],
        )

        ax = axes["noisy"]
        show_chanels(ax, TIME_AXIS, noisy_data, step=3)
        show_chanels(
            ax,
            TIME_AXIS,
            clean_data,
            step=3,
            color="k",
            linestyle="--",
            linewidth=1,
        )
        # ax.imshow(noisy_data, cmap="rainbow", aspect="auto")
        ax.set_title("Data")
        ax.set_xlabel("Time")
        ax.set_ylabel("Channel")
        ax.grid(True, alpha=0.3)

        ax = axes["cov"]
        cov_im = ax.imshow(cov_matrix, cmap="RdBu_r", aspect="equal")
        ax.set_title("Covariance Matrix")
        ax.set_xlabel("Channel")
        ax.set_ylabel("Channel")
        plt.colorbar(cov_im, ax=ax)

        ax = axes["A"]
        ax.plot(
            range(N_CHANNELS),
            explained_var_ratio / np.max(explained_var_ratio),
            "k",
            marker="o",
        )
        ax.set_yticks(np.linspace(0, 1.0, 11))
        ax.set_xlabel("Components")
        ax.set_ylabel("Normilized var ratio")
        ax.set_title("Explained var ratio")
        ax.grid(True, alpha=0.3)

        ax = axes["B"]
        ax.plot(range(N_CHANNELS), w1, "g", marker="o", label="PC1")
        ax.plot(
            range(N_CHANNELS),
            spatial_1,
            "orange",
            linestyle="--",
            label="Origin 1",
        )
        ax.plot(range(N_CHANNELS), w2, "r", marker="o", label="PC2")
        ax.plot(range(N_CHANNELS), spatial_2, "b--", label="Origin 2")
        ax.plot(range(N_CHANNELS), w3, "k", marker="o", label="Noise")
        ax.set_xlabel("Channels")
        ax.set_ylabel("Weight")
        ax.set_title("Spatial weights")
        ax.grid(True, alpha=0.3)
        ax.legend()

        ax = axes["C"]
        TIME_AXIS = TIME_AXIS[:200]
        ax.axhline(y=NOISE_LEVEL, color="gray", linestyle="--", alpha=0.8)
        ax.axhline(y=-NOISE_LEVEL, color="gray", linestyle="--", alpha=0.8)
        ax.plot(TIME_AXIS, temporal_1[:200], "g", label="PC1")
        ax.plot(TIME_AXIS, t1[:200], "orange", linestyle="--", label="Origin 1")
        ax.plot(TIME_AXIS, temporal_2[:200], "r", label="PC2")
        ax.plot(TIME_AXIS, t2[:200], "b--", label="Origin 2")
        ax.plot(TIME_AXIS, t3[:200], "k", label="Noise")
        ax.set_xlabel("Time")
        ax.set_ylabel("Ampl")
        ax.set_title("Signal")
        ax.grid(True, alpha=0.3)
        ax.legend()

        fig.tight_layout()
        return fig


    stage1()
    return


@app.cell
def _(np, plt, scipy):
    def stage2():
        N_CHANNELS = 10
        N_TIMEPOINTS = 200
        TIME_AXIS = np.arange(N_TIMEPOINTS)
        AMP_1, FREQ_1 = 2.0, 0.05
        AMP_2, FREQ_2 = 1.0, 0.10
        NOISE_LEVEL = 0.5

        spatial_1 = np.sin(np.linspace(0, 2 * np.pi, N_CHANNELS))
        spatial_2 = np.cos(np.linspace(0, 2 * np.pi, N_CHANNELS))
        temporal_1 = AMP_1 * np.sin(2 * np.pi * FREQ_1 * TIME_AXIS)
        temporal_2 = AMP_2 * np.sin(2 * np.pi * FREQ_2 * TIME_AXIS)

        def generate_data(noise_level):
            clean = np.outer(spatial_1, temporal_1) + np.outer(
                spatial_2, temporal_2
            )
            noise = np.random.randn(N_CHANNELS, N_TIMEPOINTS) * noise_level
            return clean + noise, clean, noise

        noisy_data, clean_data, noise_data = generate_data(NOISE_LEVEL)

        data_centered = noisy_data - noisy_data.mean(axis=1, keepdims=True)
        noise_centered = noise_data - noise_data.mean(axis=1, keepdims=True)

        C_signal = (data_centered @ data_centered.T) / (N_TIMEPOINTS - 1)
        C_noise = (noise_centered @ noise_centered.T) / (N_TIMEPOINTS - 1)

        reg = 1e-8 * np.trace(C_noise) / N_CHANNELS
        C_noise_reg = C_noise + reg * np.eye(N_CHANNELS)

        evals, evecs = scipy.linalg.eigh(C_signal, C_noise_reg)

        evals = evals[::-1]
        evecs = evecs[:, ::-1]

        print("Eigen values GED (signal/noise ratio):")
        explained_var_ratio = evals / np.sum(evals)
        for i, ev in enumerate(evals):
            print(f"\tComponent {i + 1}: = {ev:.4f}")

        components_ged = evecs.T @ data_centered
        C_signal_diff = C_signal - C_noise

        def compute_forward(w, C_sig_diff, C_noise_reg):
            denom = w.T @ C_noise_reg @ w
            if np.abs(denom) < 1e-12:
                return np.zeros_like(w)
            return (C_sig_diff @ w) / denom

        forward_models = []
        for i in range(N_CHANNELS):
            fm = compute_forward(evecs[:, i], C_signal_diff, C_noise_reg)
            forward_models.append(fm)
        forward_models = np.array(forward_models)

        def normalize_component(
            spatial_weights, temporal_pc, ref_signal=None, target_amp=None
        ):
            scale = np.max(np.abs(spatial_weights))
            w_norm = spatial_weights / scale
            t_norm = temporal_pc * scale

            if ref_signal is not None:
                if np.corrcoef(t_norm, ref_signal)[0, 1] < 0:
                    w_norm = -w_norm
                    t_norm = -t_norm

            if target_amp is not None and np.max(np.abs(t_norm)) > 0:
                t_norm = t_norm / np.max(np.abs(t_norm)) * target_amp

            return w_norm, t_norm

        fw1, t1 = normalize_component(
            forward_models[0], components_ged[0], temporal_1, AMP_1
        )
        fw2, t2 = normalize_component(
            forward_models[1], components_ged[1], temporal_2, AMP_2
        )

        noise_idx = -1

        t_noise_raw = components_ged[noise_idx].copy()
        mean_noise_std = np.mean(np.std(noise_data, axis=1))
        t3 = t_noise_raw / np.std(t_noise_raw) * mean_noise_std

        fw_noise_raw = forward_models[noise_idx].copy()
        fw3 = fw_noise_raw / np.linalg.norm(fw_noise_raw) * NOISE_LEVEL

        corr1_ged = np.corrcoef(t1, temporal_1)[0, 1]
        corr2_ged = np.corrcoef(t2, temporal_2)[0, 1]

        std_gd3 = np.std(components_ged[2])

        print(f"GED1 corelation: {corr1_ged:.4f}")
        print(f"GED2 corelation: {corr2_ged:.4f}")
        print(f"Noise std: {std_gd3:.4f}")

        fig, axes = plt.subplot_mosaic(
            [["noisy", "cov"], ["A", "B"], ["C", "C"]],
            figsize=(18, 14),
            width_ratios=[1, 1],
        )

        ax = axes["noisy"]
        show_chanels(ax, TIME_AXIS, noisy_data, step=5)
        show_chanels(
            ax,
            TIME_AXIS,
            clean_data,
            step=5,
            color="k",
            linestyle="--",
            linewidth=1,
        )
        ax.set_title("Data")
        ax.set_xlabel("Time")
        ax.set_ylabel("Channel")
        ax.grid(True, alpha=0.3)

        ax = axes["cov"]
        cov_im = ax.imshow(C_signal, cmap="RdBu_r", aspect="equal")
        ax.set_title("Covariance Matrix")
        ax.set_xlabel("Channel")
        ax.set_ylabel("Channel")
        plt.colorbar(cov_im, ax=ax)

        ax = axes["A"]
        ax.plot(
            range(N_CHANNELS),
            explained_var_ratio / np.max(explained_var_ratio),
            "k",
            marker="o",
        )
        ax.set_yticks(np.linspace(0, 1.0, 11))
        ax.set_xlabel("Components")
        ax.set_ylabel("Normilized ratio")
        ax.set_title("Signal over noise ratio")
        ax.grid(True, alpha=0.3)

        ax = axes["B"]
        ax.plot(range(N_CHANNELS), fw1, "g", marker="o", label="GED1")
        ax.plot(
            range(N_CHANNELS),
            spatial_1,
            "orange",
            linestyle="--",
            label="Source 1",
        )
        ax.plot(range(N_CHANNELS), fw2, "r", marker="o", label="GED2")
        ax.plot(range(N_CHANNELS), spatial_2, "b--", label="Source 2")
        ax.plot(range(N_CHANNELS), fw3, "k", marker="o", label="Noise")
        ax.set_xlabel("Channels")
        ax.set_ylabel("Weight")
        ax.set_title("Spatial weights")
        ax.grid(True, alpha=0.3)
        ax.legend()

        ax = axes["C"]
        ax.axhline(y=NOISE_LEVEL, color="gray", linestyle="--", alpha=0.8)
        ax.axhline(y=-NOISE_LEVEL, color="gray", linestyle="--", alpha=0.8)
        ax.plot(TIME_AXIS, temporal_1, "g", label="GED1")
        ax.plot(TIME_AXIS, t1, "orange", linestyle="--", label="Source 1")
        ax.plot(TIME_AXIS, temporal_2, "r", label="GED2")
        ax.plot(TIME_AXIS, t2, "b--", label="Source 2")
        ax.plot(TIME_AXIS, t3, "k", label="Noise")
        ax.set_xlabel("Time")
        ax.set_ylabel("Ampl")
        ax.set_title("Signal")
        ax.grid(True, alpha=0.3)
        ax.legend()

        fig.tight_layout()
        return fig


    stage2()
    return


@app.cell
def _(FastICA, np, plt):
    def stage3():
        N_CHANNELS = 10
        N_TIMEPOINTS = 200
        TIME_AXIS = np.arange(N_TIMEPOINTS)
        AMP_1, FREQ_1 = 2.0, 0.05
        AMP_2, FREQ_2 = 1.0, 0.10
        NOISE_LEVEL = 0.5

        spatial_1 = np.sin(np.linspace(0, 2 * np.pi, N_CHANNELS))
        spatial_2 = np.cos(np.linspace(0, 2 * np.pi, N_CHANNELS))
        temporal_1 = AMP_1 * np.sin(2 * np.pi * FREQ_1 * TIME_AXIS)
        temporal_2 = AMP_2 * np.sin(2 * np.pi * FREQ_2 * TIME_AXIS)

        def generate_data(noise_level):
            clean = np.outer(spatial_1, temporal_1) + np.outer(
                spatial_2, temporal_2
            )
            noise = np.random.randn(N_CHANNELS, N_TIMEPOINTS) * noise_level
            return clean + noise, clean, noise

        noisy_data, clean_data, noise_data = generate_data(NOISE_LEVEL)

        data_centered = noisy_data - noisy_data.mean(axis=1, keepdims=True)

        cov_matrix = (data_centered @ data_centered.T) / (N_TIMEPOINTS - 1)

        X_ica = data_centered.T
        ica = FastICA(
            n_components=N_CHANNELS,
            random_state=42,
            whiten="arbitrary-variance",
            max_iter=2000,
        )

        sources = ica.fit_transform(X_ica)
        mixing = ica.mixing_

        sources = sources.T
        spatial_maps = mixing.T

        energy = np.sum(spatial_maps**2, axis=1)

        sorted_idx = np.argsort(energy)[::-1]
        spatial_maps = spatial_maps[sorted_idx]
        sources = sources[sorted_idx]
        energy = energy[sorted_idx]

        print("Maps enerdy:")
        for i, en in enumerate(energy):
            print(f"\tICA{i + 1}: energy = {en:.4f}")

        corr_with_source1 = np.array(
            [np.corrcoef(sources[i], temporal_1)[0, 1] for i in range(N_CHANNELS)]
        )
        corr_with_source2 = np.array(
            [np.corrcoef(sources[i], temporal_2)[0, 1] for i in range(N_CHANNELS)]
        )

        abs_corr1 = np.abs(corr_with_source1)
        abs_corr2 = np.abs(corr_with_source2)

        idx1 = np.argmax(abs_corr1)
        idx2 = np.argmax(abs_corr2)

        print(f"Source 1 -> ICA{idx1 + 1} (corr: {corr_with_source1[idx1]:.4f})")
        print(f"Source 2 -> ICA{idx2 + 1} (corr: {corr_with_source2[idx2]:.4f})")
        print(f"\nCorr matrix:")
        print(f"\t\t\tSource1\tSource2")
        for i in range(N_CHANNELS):
            print(f"\tICA{i + 1}:  {abs_corr1[i]:.4f}   {abs_corr2[i]:.4f}")

        def normalize_component(
            spatial_weights, temporal_pc, ref_signal=None, target_amp=None
        ):
            scale = np.max(np.abs(spatial_weights))
            w_norm = spatial_weights / scale
            t_norm = temporal_pc * scale

            if ref_signal is not None:
                if np.corrcoef(t_norm, ref_signal)[0, 1] < 0:
                    w_norm = -w_norm
                    t_norm = -t_norm

            if target_amp is not None and np.max(np.abs(t_norm)) > 0:
                t_norm = t_norm / np.max(np.abs(t_norm)) * target_amp

            return w_norm, t_norm

        sw1, st1 = normalize_component(
            spatial_maps[idx1], sources[idx1], temporal_1, AMP_1
        )
        sw2, st2 = normalize_component(
            spatial_maps[idx2], sources[idx2], temporal_2, AMP_2
        )

        noise_scores = abs_corr1 + abs_corr2
        noise_idx = np.argmin(noise_scores)
        sw_noise = (
            spatial_maps[noise_idx]
            / np.linalg.norm(spatial_maps[noise_idx])
            * NOISE_LEVEL
        )
        # st_noise = sources[noise_idx] / np.std(sources[noise_idx]) * NOISE_LEVEL
        mean_noise_std = np.mean(np.std(noise_data, axis=1))
        st_noise = sources[noise_idx] / np.std(sources[noise_idx]) * mean_noise_std

        corr1_ica = np.corrcoef(st1, temporal_1)[0, 1]
        corr2_ica = np.corrcoef(st2, temporal_2)[0, 1]
        std_noise_ica = np.std(st_noise)

        print(f"ICA{idx1 + 1} vs Source 1: {corr1_ica:.4f}")
        print(f"ICA{idx2 + 1} vs Source 2: {corr2_ica:.4f}")
        print(f"Noise component (ICA{noise_idx + 1}) std: {std_noise_ica:.4f}")

        fig, axes = plt.subplot_mosaic(
            [["noisy", "cov"], ["A", "B"], ["C", "C"]],
            figsize=(18, 14),
            width_ratios=[1, 1],
        )

        ax = axes["noisy"]
        show_chanels(ax, TIME_AXIS, noisy_data, step=5)
        show_chanels(
            ax,
            TIME_AXIS,
            clean_data,
            step=5,
            color="k",
            linestyle="--",
            linewidth=1,
        )
        ax.set_title("Data")
        ax.set_xlabel("Time")
        ax.set_ylabel("Channel")
        ax.grid(True, alpha=0.3)

        ax = axes["cov"]
        cov_im = ax.imshow(cov_matrix, cmap="RdBu_r", aspect="equal")
        ax.set_title("Covariance Matrix")
        ax.set_xlabel("Channel")
        ax.set_ylabel("Channel")
        plt.colorbar(cov_im, ax=ax)

        ax = axes["A"]
        ax.plot(
            range(N_CHANNELS),
            energy / np.max(energy),
            "k",
            marker="o",
        )
        ax.set_yticks(np.linspace(0, 1.0, 11))
        ax.set_xlabel("Components")
        ax.set_ylabel("Normalized energy")
        ax.set_title("Enenrgy")
        ax.grid(True, alpha=0.3)

        ax = axes["B"]
        ax.plot(range(N_CHANNELS), sw1, "g", marker="o", label="ICA1")
        ax.plot(
            range(N_CHANNELS),
            spatial_1,
            "orange",
            linestyle="--",
            label="Source 1",
        )
        ax.plot(range(N_CHANNELS), sw2, "r", marker="o", label="ICA2")
        ax.plot(range(N_CHANNELS), spatial_2, "b--", label="Source 2")
        ax.plot(range(N_CHANNELS), sw_noise, "k", marker="o", label="Noise")
        ax.set_xlabel("Channels")
        ax.set_ylabel("Weight")
        ax.set_title("Spatial weights")
        ax.grid(True, alpha=0.3)
        ax.legend()

        ax = axes["C"]
        ax.axhline(y=NOISE_LEVEL, color="gray", linestyle="--", alpha=0.8)
        ax.axhline(y=-NOISE_LEVEL, color="gray", linestyle="--", alpha=0.8)
        ax.plot(TIME_AXIS, temporal_1, "g", label="ICA1")
        ax.plot(TIME_AXIS, st1, "orange", linestyle="--", label="Source 1")
        ax.plot(TIME_AXIS, temporal_2, "r", label="ICA2")
        ax.plot(TIME_AXIS, st2, "b--", label="Source 2")
        ax.plot(TIME_AXIS, st_noise, "k", label="Noise")
        ax.set_xlabel("Time")
        ax.set_ylabel("Ampl")
        ax.set_title("Signal")
        ax.grid(True, alpha=0.3)
        ax.legend()

        fig.tight_layout()
        return fig


    stage3()
    return


@app.cell
def _(griddata, np, patches, plt):
    def topoplotIndie(Values, chanlocs, title="", ax=0):

        ## import and convert channel locations from EEG structure
        labels = []
        Th = []
        Rd = []
        x = []
        y = []

        #
        for ci in range(len(chanlocs[0])):
            labels.append(chanlocs[0]["labels"][ci][0])
            Th.append(np.pi / 180 * chanlocs[0]["theta"][ci][0][0])
            Rd.append(chanlocs[0]["radius"][ci][0][0])
            x.append(Rd[ci] * np.cos(Th[ci]))
            y.append(Rd[ci] * np.sin(Th[ci]))

        ## remove infinite and NaN values
        # ...

        # plotting factors
        headrad = 0.5
        plotrad = 0.6

        # squeeze coords into head
        squeezefac = headrad / plotrad
        # to plot all inside the head cartoon
        x = np.array(x) * squeezefac
        y = np.array(y) * squeezefac

        ## create grid
        xmin = np.min([-headrad, np.min(x)])
        xmax = np.max([headrad, np.max(x)])
        ymin = np.min([-headrad, np.min(y)])
        ymax = np.max([headrad, np.max(y)])
        xi = np.linspace(xmin, xmax, 67)
        yi = np.linspace(ymin, ymax, 67)

        # spatially interpolated data
        Xi, Yi = np.mgrid[xmin:xmax:67j, ymin:ymax:67j]
        Zi = griddata(np.array([y, x]).T, Values, (Yi, Xi))
        #     f  = interpolate.interp2d(y,x,Values)
        #     Zi = f(yi,xi)

        ## Mask out data outside the head
        mask = np.sqrt(Xi**2 + Yi**2) <= headrad
        Zi[mask == 0] = np.nan

        ## create topography
        # make figure
        if ax == 0:
            fig = plt.figure()
            ax = fig.add_subplot(111, aspect=1)
        clim = np.max(np.abs(Zi[np.isfinite(Zi)])) * 0.8
        ax.contourf(
            yi, xi, Zi, 60, cmap=plt.cm.jet, zorder=1, vmin=-clim, vmax=clim
        )

        # head ring
        circle = patches.Circle(
            xy=[0, 0], radius=headrad, edgecolor="k", facecolor="w", zorder=0
        )
        ax.add_patch(circle)

        # ears
        circle = patches.Ellipse(
            xy=[np.min(xi), 0],
            width=0.05,
            height=0.2,
            angle=0,
            edgecolor="k",
            facecolor="w",
            zorder=-1,
        )
        ax.add_patch(circle)
        circle = patches.Ellipse(
            xy=[np.max(xi), 0],
            width=0.05,
            height=0.2,
            angle=0,
            edgecolor="k",
            facecolor="w",
            zorder=-1,
        )
        ax.add_patch(circle)

        # nose (top, left, right)
        xy = [[0, np.max(yi) + 0.06], [-0.2, 0.2], [0.2, 0.2]]
        polygon = patches.Polygon(xy=xy, facecolor="w", edgecolor="k", zorder=-1)
        ax.add_patch(polygon)

        # add the electrode markers
        ax.scatter(y, x, marker="o", c="k", s=15, zorder=3)

        ax.set_xlim([-0.6, 0.6])
        ax.set_ylim([-0.6, 0.6])
        ax.axis("off")
        ax.set_title(title)
        ax.set_aspect("equal")

    return (topoplotIndie,)


@app.cell
def _(fft, fftfreq, np, plt, scipy, sio, topoplotIndie):
    def stage4():
        MATFILE_NAME = "emptyEEG.mat"
        matfile = sio.loadmat(MATFILE_NAME)
        lf = matfile["lf"][0, 0]
        EEG = matfile["EEG"][0, 0]

        Gain_all = lf["Gain"]
        n_chan, n_orient, n_dip = Gain_all.shape

        srate = 500
        n_trials = 200
        n_pnts = 1000
        times = np.arange(n_pnts) / srate

        DIPOLE_LOC1 = 108
        DIPOLE_LOC2 = 134

        FREQ1 = 15
        FREQ2 = 10
        AMP1 = 2.0
        AMP2 = 1.0
        NOISE_LEVEL = 0.5

        tidx = n_pnts // 2
        n_pre = tidx
        n_pst = n_pnts - tidx
        t_post = times[tidx:]

        print(f"Channels: {n_chan}, Orientations: {n_orient}, Dipoles: {n_dip}")

        # ============ Dipole orientations ============
        orient1 = np.array([1.0, 0.5, 0.3])
        orient1 = orient1 / np.linalg.norm(orient1)
        orient2 = np.array([0.2, 1.0, 0.4])
        orient2 = orient2 / np.linalg.norm(orient2)

        true_topo_dip1 = np.zeros(n_chan)
        true_topo_dip2 = np.zeros(n_chan)
        for o in range(3):
            true_topo_dip1 += Gain_all[:, o, DIPOLE_LOC1] * orient1[o]
            true_topo_dip2 += Gain_all[:, o, DIPOLE_LOC2] * orient2[o]

        # ============ Data generation ============
        EEG_data = np.zeros((n_chan, n_pnts, n_trials))
        true_dip1 = np.zeros((n_trials, n_pnts))
        true_dip2 = np.zeros((n_trials, n_pnts))

        for trial in range(n_trials):
            phase1 = np.random.uniform(0, 2 * np.pi)
            phase2 = np.random.uniform(0, 2 * np.pi)

            dipole_activity = np.zeros((3, n_dip, n_pnts))

            signal1 = AMP1 * np.sin(2 * np.pi * FREQ1 * t_post + phase1)
            signal2 = AMP2 * np.sin(2 * np.pi * FREQ2 * t_post + phase2)

            for o in range(3):
                dipole_activity[o, DIPOLE_LOC1, tidx:] = orient1[o] * signal1
                dipole_activity[o, DIPOLE_LOC2, tidx:] = orient2[o] * signal2

            true_dip1[trial, tidx:] = signal1
            true_dip2[trial, tidx:] = signal2

            scalp_clean = np.zeros((n_chan, n_pnts))
            for o in range(3):
                scalp_clean += Gain_all[:, o, :] @ dipole_activity[o]

            noise = np.random.randn(n_chan, n_pnts) * NOISE_LEVEL
            EEG_data[:, :, trial] = scalp_clean + noise

        # ============ GED ============
        EEG_centered = EEG_data - EEG_data.mean(axis=1, keepdims=True)

        C_signal = np.zeros((n_chan, n_chan))
        C_noise = np.zeros((n_chan, n_chan))

        for trial in range(n_trials):
            pre_data = EEG_centered[:, :tidx, trial]
            C_noise += (pre_data @ pre_data.T) / n_pre
            pst_data = EEG_centered[:, tidx:, trial]
            C_signal += (pst_data @ pst_data.T) / n_pst

        C_signal /= n_trials
        C_noise /= n_trials

        reg = 1e-8 * np.trace(C_noise) / n_chan
        C_noise_reg = C_noise + reg * np.eye(n_chan)

        evals, evecs = scipy.linalg.eigh(C_signal, C_noise_reg)
        evals = evals[::-1]
        evecs = evecs[:, ::-1]

        # Projection
        components_ged = np.zeros((n_chan, n_pnts, n_trials))
        for trial in range(n_trials):
            components_ged[:, :, trial] = evecs.T @ EEG_centered[:, :, trial]

        # Forward model
        C_signal_diff = C_signal - C_noise

        def compute_forward(w, C_sig_diff, C_noise_reg):
            denom = w.T @ C_noise_reg @ w
            if np.abs(denom) < 1e-12:
                return np.zeros_like(w)
            return (C_sig_diff @ w) / denom

        forward_models = []
        for i in range(n_chan):
            fm = compute_forward(evecs[:, i], C_signal_diff, C_noise_reg)
            forward_models.append(fm)
        forward_models = np.array(forward_models)

        # ============ Analisys ============
        ged_erp = components_ged.mean(axis=2)
        true_dip1_erp = true_dip1.mean(axis=0)
        true_dip2_erp = true_dip2.mean(axis=0)

        corr_1v1 = np.abs(
            np.corrcoef(ged_erp[0, tidx:], true_dip1_erp[tidx:])[0, 1]
        )
        corr_1v2 = np.abs(
            np.corrcoef(ged_erp[0, tidx:], true_dip2_erp[tidx:])[0, 1]
        )

        if corr_1v1 > corr_1v2:
            ged1_dip = 1
            ged2_dip = 2
        else:
            ged1_dip = 2
            ged2_dip = 1

        true_topo_for_ged1 = [true_topo_dip1, true_topo_dip2][ged1_dip - 1]
        true_topo_for_ged2 = [true_topo_dip1, true_topo_dip2][ged2_dip - 1]
        true_erp_for_ged1 = [true_dip1_erp, true_dip2_erp][ged1_dip - 1]
        true_erp_for_ged2 = [true_dip1_erp, true_dip2_erp][ged2_dip - 1]

        def align_sign(fwd_model, true_topo, ged_temporal, true_temporal):
            if np.corrcoef(fwd_model, true_topo)[0, 1] < 0:
                fwd_model = -fwd_model
                ged_temporal = -ged_temporal

            if np.corrcoef(ged_temporal, true_temporal)[0, 1] < 0:
                fwd_model = -fwd_model
                ged_temporal = -ged_temporal

            return fwd_model, ged_temporal

        # ПSign alignment
        fm1_aligned, ged1_aligned = align_sign(
            forward_models[0].copy(),
            true_topo_for_ged1,
            ged_erp[0].copy(),
            true_erp_for_ged1,
        )
        fm2_aligned, ged2_aligned = align_sign(
            forward_models[1].copy(),
            true_topo_for_ged2,
            ged_erp[1].copy(),
            true_erp_for_ged2,
        )

        # Correlations
        corr1_final = np.abs(
            np.corrcoef(ged1_aligned[tidx:], true_erp_for_ged1[tidx:])[0, 1]
        )
        corr2_final = np.abs(
            np.corrcoef(ged2_aligned[tidx:], true_erp_for_ged2[tidx:])[0, 1]
        )
        topo_corr1 = np.corrcoef(fm1_aligned, true_topo_for_ged1)[0, 1]
        topo_corr2 = np.corrcoef(fm2_aligned, true_topo_for_ged2)[0, 1]

        dip_freqs = [0, 15, 10]
        print(f"\nSummary:")
        print(f"  GED1 → Dipole {ged1_dip} ({dip_freqs[ged1_dip]} Hz)")
        print(f"    Temporal corr: {corr1_final:.4f}")
        print(f"    Topography corr: {topo_corr1:.4f}")
        print(f"  GED2 → Dipole {ged2_dip} ({dip_freqs[ged2_dip]} Hz)")
        print(f"    Temporal corr: {corr2_final:.4f}")
        print(f"    Topography corr: {topo_corr2:.4f}")

        # ============ Visualization ============
        fig, axes = plt.subplot_mosaic(
            [
                ["dip1", "dip2", "signal1", "signal1", "signal1"],
                ["dip1", "dip2", "signal2", "signal2", "signal2"],
                ["GED1", "GED2", "spec1", "spec2", "vals"],
                ["GED1", "GED2", "spec1", "spec2", "vals"],
            ],
            figsize=(26, 14),
            width_ratios=[1, 1, 1, 1, 1],
        )

        # True topography
        for i, (ax, topo, dip_num, freq) in enumerate(
            zip(
                [axes["dip1"], axes["dip2"]],
                [true_topo_dip1, true_topo_dip2],
                [1, 2],
                [15, 10],
            )
        ):
            topoplotIndie(
                topo,
                EEG["chanlocs"],
                ax=ax,
                title=f"True Dipole {dip_num} ({freq} Hz)",
            )

        # GED topography
        for i, (ax, topo, dip_num) in enumerate(
            zip(
                [axes["GED1"], axes["GED2"]],
                [fm1_aligned, fm2_aligned],
                [ged1_dip, ged2_dip],
            )
        ):
            topoplotIndie(
                topo,
                EEG["chanlocs"],
                ax=ax,
                title=f"GED{i + 1} → Dipole {dip_num}\n(topo corr: {[topo_corr1, topo_corr2][i]:.3f})",
            )

        # Eigenvalues
        ax = axes["vals"]
        colors_ev = ["steelblue" if i < 2 else "lightgray" for i in range(10)]
        ax.plot(
            range(1, 11), evals[:10] / np.max(evals[:10]), color="k", marker="o"
        )
        ax.axhline(0.1, color="r", linestyle="--", label="λ=1 (noise)")
        ax.set_xlabel("Components")
        ax.set_ylabel("Normilized energy")
        ax.set_title("GED Eigenvalues energy")
        ax.legend()
        ax.grid(alpha=0.3)

        # Signals
        def norm_signal(sig, ref):
            s = sig.copy()
            if np.corrcoef(s, ref)[0, 1] < 0:
                s = -s
            if np.max(np.abs(s)) > 0:
                s = s / np.max(np.abs(s)) * np.max(np.abs(ref))
            return s

        t1_norm = norm_signal(ged1_aligned, true_erp_for_ged1)
        t2_norm = norm_signal(ged2_aligned, true_erp_for_ged2)

        ax = axes["signal1"]
        ax.plot(
            times,
            true_dip1_erp,
            "orange",
            label="Dipole 1 (15 Hz)",
            lw=2,
            alpha=0.7,
        )
        ax.plot(
            times,
            t1_norm,
            "darkred",
            linestyle="--",
            label=f"GED1 (Dipole {ged1_dip})",
            lw=2,
        )
        ax.axvline(x=times[tidx], color="k", linestyle=":", alpha=0.5)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Amplitude")
        ax.set_title("Averaged Temporal Components")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

        ax = axes["signal2"]
        ax.plot(
            times,
            true_dip2_erp,
            "royalblue",
            label="Dipole 2 (10 Hz)",
            lw=2,
            alpha=0.7,
        )
        ax.plot(
            times,
            t2_norm,
            "darkgreen",
            linestyle="--",
            label=f"GED2 (Dipole {ged2_dip})",
            lw=2,
        )
        ax.axvline(x=times[tidx], color="k", linestyle=":", alpha=0.5)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Amplitude")
        ax.set_title("Averaged Temporal Components")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

        # Spectrum
        def compute_spectrum(signal):
            n = len(signal)
            fft_vals = fft(signal)
            power = np.abs(fft_vals[: n // 2]) ** 2 / n
            freqs = fftfreq(n, 1 / srate)[: n // 2]
            return freqs, power / np.max(power)

        for i, (ax, ged_temporal, dip_idx) in enumerate(
            zip(
                [axes["spec1"], axes["spec2"]],
                [ged1_aligned, ged2_aligned],
                [ged1_dip, ged2_dip],
            )
        ):
            freqs, spec_ged = compute_spectrum(ged_temporal[tidx:])
            _, spec_dip = compute_spectrum(
                [true_dip1_erp, true_dip2_erp][dip_idx - 1][tidx:]
            )

            dip_freq = dip_freqs[dip_idx]
            other_freq = dip_freqs[3 - dip_idx]

            ax.plot(
                freqs,
                spec_dip,
                "orange" if dip_idx == 1 else "royalblue",
                label=f"Dipole {dip_idx}",
                lw=2,
                alpha=0.7,
            )
            ax.plot(
                freqs,
                spec_ged,
                "darkred" if i == 0 else "darkgreen",
                linestyle="--",
                label=f"GED{i + 1}",
                lw=2,
            )
            ax.axvline(
                dip_freq,
                color="r",
                linestyle=":",
                alpha=0.7,
                label=f"{dip_freq} Hz",
            )
            ax.axvline(other_freq, color="gray", linestyle=":", alpha=0.3)
            ax.set_xlim(0, 30)
            ax.set_xlabel("Frequency (Hz)")
            ax.set_ylabel("Normalized Power")
            ax.set_title(f"Spectrum: GED{i + 1} (Dipole {dip_idx})")
            ax.legend(fontsize=8)
            ax.grid(alpha=0.3)

        fig.tight_layout()
        return fig


    stage4()
    return


if __name__ == "__main__":
    app.run()

import marimo

__generated_with = "0.21.1"
app = marimo.App(width="full", auto_download=["html"])


@app.cell
def _():
    import numpy as np
    import matplotlib.pyplot as plt

    return np, plt


@app.cell
def _(np):
    def RMSE(y, y_real):
        return np.sqrt(np.mean((y - y_real) ** 2.0))


    class SimpleKalman1D:
        def __init__(self, initial_x, initial_P, Q, R):
            self.x = initial_x
            self.P = initial_P
            self.Q = Q
            self.R = R

            self._history_x = []
            self._history_P = []

        @property
        def history_x(self):
            return self._history_x

        @property
        def history_P(self):
            return self._history_P

        def predict(self):
            self.x = self.x
            self.P = self.P + self.Q

        def update(self, z):
            y = z - self.x
            S = self.P + self.R
            K = self.P / S
            self.x = self.x + K * y
            self.P = (1 - K) * self.P

        def step(self, z):
            self.predict()
            self.update(z)

            self._history_x.append(self.x)
            self._history_P.append(self.P)

            return self.x

    return RMSE, SimpleKalman1D


@app.cell
def _(RMSE, SimpleKalman1D, np, plt):
    def step1():
        N = 200
        TRUE_VAUE = 5.0
        NOISE_SIGMA = 5.0

        time = np.arange(N)
        true_signal = np.full(N, TRUE_VAUE)
        real_signal = true_signal + np.random.normal(0, NOISE_SIGMA, N)

        Q = 0.1  # model
        R = 25  # sensor
        kf = SimpleKalman1D(initial_x=0.0, initial_P=100.0, Q=Q, R=R)

        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(18, 6))

        filtered_signal = []
        for point in real_signal:
            filtered_signal.append(kf.step(point))

        err = RMSE(filtered_signal, true_signal)

        ax.plot(time, true_signal, "k", label="True signal")
        ax.plot(time, real_signal, "b.", label="Real signal")
        ax.plot(
            time, filtered_signal, "r", label=f"Filtered signal (RMSE={err:.3f})"
        )

        ax2 = ax.twinx()
        ax2.plot(time, kf.history_P, "g--")
        ax2.set_ylabel("Error covariation (P)", color="green")

        ax.grid(True, alpha=0.3)
        ax.set_xlabel("Time")
        ax.set_ylabel("Value")
        ax.legend()

        fig.tight_layout()

        return fig


    step1()
    return


@app.cell
def _(np):
    class KalmanCV1D:
        def __init__(
            self, dt, initial_pos, initial_vel, initial_P, Q_pos, Q_vel, R
        ):
            self.dt = dt

            self.x = np.array([[initial_pos], [initial_vel]])
            self.P = np.array([[initial_P, 0.0], [0.0, initial_P]])
            self.F = np.array([[1.0, dt], [0.0, 1.0]])
            self.H = np.array([[1.0, 0.0]])
            self.Q = np.array([[Q_pos, 0.0], [0.0, Q_vel]])
            self.R = np.array([[R]])

            self._history_pos = []
            self._history_vel = []
            self._history_pos_P = []

        @property
        def history_pos(self):
            return self._history_pos

        @property
        def history_vel(self):
            return self._history_vel

        @property
        def history_pos_P(self):
            return self._history_pos_P

        def predict(self):
            self.x = self.F @ self.x
            self.P = self.F @ self.P @ self.F.T + self.Q

        def update(self, z):
            z = np.array([[z]])
            y = z - self.H @ self.x
            S = self.H @ self.P @ self.H.T + self.R
            K = self.P @ self.H.T @ np.linalg.inv(S)
            self.x = self.x + K @ y
            self.P = self.P - K @ self.H @ self.P

        def step(self, z):
            self.predict()
            self.update(z)

            self._history_pos.append(self.x[0, 0])
            self._history_vel.append(self.x[1, 0])
            self._history_pos_P.append(self.P[0, 0])

            return self.x[0, 0]

    return (KalmanCV1D,)


@app.cell
def _(KalmanCV1D, RMSE, np, plt):
    def step2():
        dt = 0.1
        N = 200
        NOISE_SIGMA = 2.0

        time = np.arange(N) * dt

        true_vel = np.zeros(N)
        true_vel[40:] = 2.5

        true_pos = np.zeros(N)
        for i in range(1, N):
            true_pos[i] = true_pos[i - 1] + true_vel[i - 1] * dt

        real_pos = true_pos + np.random.normal(0, NOISE_SIGMA, N)

        Q_pos = 0.01  # model
        Q_vel = 0.01
        R = 4.0  # sensor
        kf = KalmanCV1D(
            dt=dt,
            initial_pos=0.0,
            initial_vel=0.0,
            initial_P=10.0,
            Q_pos=Q_pos,
            Q_vel=Q_vel,
            R=R,
        )

        fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(18, 12))

        filtered_pos = []
        for point in real_pos:
            filtered_pos.append(kf.step(point))

        err = RMSE(filtered_pos, true_pos)

        ax[0].plot(time, true_pos, "k", label="True signal")
        ax[0].plot(time, real_pos, "b.", label="Real signal")
        ax[0].plot(
            time, filtered_pos, "r", label=f"Filtered signal (RMSE={err:.3f})"
        )

        ax2 = ax[0].twinx()
        ax2.plot(time, kf.history_pos_P, "g--")
        ax2.set_ylabel("Error covariation (P)", color="green")

        ax[0].grid(True, alpha=0.3)
        ax[0].set_xlabel("Time")
        ax[0].set_ylabel("Value")
        ax[0].legend()

        ax[1].plot(time, true_vel, "k--", label="True velocity")
        ax[1].plot(time, kf.history_vel, "r", label="Predicted velocity")

        ax[1].grid(True, alpha=0.3)
        ax[1].set_xlabel("Time")
        ax[1].set_ylabel("Velocity")
        ax[1].legend()

        fig.tight_layout()

        return fig


    step2()
    return


@app.cell
def _(np):
    class KalmanCV2D:
        def __init__(self, dt, initial_P, Q_pos, Q_vel, R):
            self.dt = dt

            self.x = np.zeros((4, 1))
            self.P = np.eye(4) * initial_P
            self.F_block = np.array([[1.0, dt], [0.0, 1.0]])
            self.F = np.block(
                [
                    [self.F_block, np.zeros((2, 2))],
                    [np.zeros((2, 2)), self.F_block],
                ]
            )
            self.H = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]])

            self.Q = np.diag([Q_pos, Q_vel, Q_pos, Q_vel])

            self.R = np.eye(2) * R

            self._history_x = []
            self._history_y = []
            self._history_vx = []
            self._history_vy = []

        @property
        def history_x(self):
            return self._history_x

        @property
        def history_y(self):
            return self._history_y

        @property
        def history_vx(self):
            return self._history_vx

        @property
        def history_vy(self):
            return self._history_vy

        def predict(self):
            self.x = self.F @ self.x
            self.P = self.F @ self.P @ self.F.T + self.Q

        def update(self, z):
            y = z - self.H @ self.x
            S = self.H @ self.P @ self.H.T + self.R
            K = self.P @ self.H.T @ np.linalg.inv(S)
            self.x = self.x + K @ y
            self.P = self.P - K @ self.H @ self.P

        def step(self, mx, my):
            z = np.array([[mx], [my]])
            self.predict()
            self.update(z)

            self.history_x.append(self.x[0, 0])
            self.history_y.append(self.x[2, 0])
            self.history_vx.append(self.x[1, 0])
            self.history_vy.append(self.x[3, 0])

            return self.x[0, 0], self.x[2, 0]

    return (KalmanCV2D,)


@app.cell
def _(KalmanCV2D, RMSE, np, plt):
    def step3():
        dt = 0.1
        N = 200
        NOISE_SIGMA = 1.0
        omega = 0.05
        R_circle = 10.0

        time = np.arange(N)

        true_x = R_circle * np.cos(omega * time)
        true_y = R_circle * np.sin(omega * time)

        true_vx = -R_circle * omega * np.sin(omega * time)
        true_vy = R_circle * omega * np.cos(omega * time)

        real_x = true_x + np.random.normal(0, NOISE_SIGMA, N)
        real_y = true_y + np.random.normal(0, NOISE_SIGMA, N)

        Q_pos = 0.01  # model
        Q_vel = 0.1
        R = 1.0  # sensor
        kf = KalmanCV2D(
            dt=dt,
            initial_P=10.0,
            Q_pos=Q_pos,
            Q_vel=Q_vel,
            R=R,
        )

        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(18, 6))

        filtered_x, filtered_y = [], []
        for point_x, point_y in zip(real_x, real_y):
            fx, fy = kf.step(point_x, point_y)
            filtered_x.append(fx)
            filtered_y.append(fy)

        err_x = RMSE(filtered_x, true_x)
        err_y = RMSE(filtered_y, true_y)

        # Main plot
        ax = axes[0]
        ax.plot(true_x, true_y, "k", label="True signal")
        ax.plot(real_x, real_y, "b.", label="Real signal")
        ax.plot(
            filtered_x,
            filtered_y,
            "r",
            label=f"Filtered signal (RMSE(x)={err_x:.3f} RMSE(y)={err_y:.3f})",
        )

        ax.grid(True, alpha=0.3)
        ax.set_xlabel("Time")
        ax.set_ylabel("Value")
        ax.legend()
        ax.axis("equal")

        # X velocity
        ax = axes[1]
        ax.plot(time, true_vx, "k--", label="True X velocity")
        ax.plot(time, kf.history_vx, "r", label="Predicted X velocity")

        # Y velocity
        ax.plot(time, true_vy, "b--", label="True Y velocity")
        ax.plot(time, kf.history_vy, "g", label="Predicted Y velocity")

        ax.grid(True, alpha=0.3)
        ax.set_xlabel("Time")
        ax.set_ylabel("Velocity")
        ax.legend()

        fig.tight_layout()

        return fig


    step3()
    return


@app.cell
def _(np):
    class EKF:
        def __init__(self, dt, initial_P, Q, R):
            self.dt = dt

            self.x = np.array([[0.0], [0.0], [0.0], [0.5]])

            self.P = np.eye(4) * initial_P
            self.Q = np.eye(4) * Q
            self.R = np.eye(2) * R
            self.H = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])

            # История
            self.history_x = []
            self.history_y = []
            self.history_psi = []
            self.history_v = []

        def predict(self):
            x, y, psi, v = self.x[0, 0], self.x[1, 0], self.x[2, 0], self.x[3, 0]

            x_new = x + v * np.cos(psi) * self.dt
            y_new = y + v * np.sin(psi) * self.dt
            psi_new = psi
            v_new = v

            self.x = np.array([[x_new], [y_new], [psi_new], [v_new]])

            F_jac = np.eye(4)
            F_jac[0, 2] = -v * np.sin(psi) * self.dt
            F_jac[0, 3] = np.cos(psi) * self.dt
            F_jac[1, 2] = v * np.cos(psi) * self.dt
            F_jac[1, 3] = np.sin(psi) * self.dt

            self.P = F_jac @ self.P @ F_jac.T + self.Q

        def update(self, z):
            z_pred = self.H @ self.x
            y = z - z_pred
            S = self.H @ self.P @ self.H.T + self.R
            K = self.P @ self.H.T @ np.linalg.inv(S)
            self.x = self.x + K @ y
            self.P = self.P - K @ self.H @ self.P

        def step(self, mx, my):
            z = np.array([[mx], [my]])
            self.predict()
            self.update(z)

            self.history_x.append(self.x[0, 0])
            self.history_y.append(self.x[1, 0])
            self.history_psi.append(self.x[2, 0])
            self.history_v.append(self.x[3, 0])

            return self.x[0, 0], self.x[1, 0]

    return (EKF,)


@app.cell
def _(EKF, RMSE, np, plt):
    def step4():
        dt = 0.1
        N = 250
        time = np.arange(N) * dt

        true_x = np.zeros(N)
        true_y = np.zeros(N)
        true_psi = np.zeros(N)
        true_v = np.zeros(N)

        true_v[50:] = 1.5

        true_psi[60:120] = np.linspace(0, 0.5, 60)
        true_psi[120:] = np.linspace(0.5, 0, N - 120)

        for i in range(1, N):
            true_x[i] = (
                true_x[i - 1] + true_v[i - 1] * np.cos(true_psi[i - 1]) * dt
            )
            true_y[i] = (
                true_y[i - 1] + true_v[i - 1] * np.sin(true_psi[i - 1]) * dt
            )

        NOISE_SIGMA = 0.4
        meas_x = true_x + np.random.normal(0, NOISE_SIGMA, N)
        meas_y = true_y + np.random.normal(0, NOISE_SIGMA, N)

        Q_val = 0.01
        R_val = 0.16

        ekf = EKF(dt=dt, initial_P=10.0, Q=Q_val, R=R_val)

        filtered_x, filtered_y = [], []
        for mx, my in zip(meas_x, meas_y):
            fx, fy = ekf.step(mx, my)
            filtered_x.append(fx)
            filtered_y.append(fy)

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Trajectory XY
        ax = axes[0, 0]
        ax.plot(true_x, true_y, "k-", label="Real trajectory", linewidth=2)
        ax.plot(meas_x, meas_y, "b.", label="GPS", alpha=0.3, markersize=3)
        ax.plot(filtered_x, filtered_y, "r-", label="EKF", linewidth=2)
        # Trajectory vector
        skip = 20
        ax.quiver(
            true_x[::skip],
            true_y[::skip],
            np.cos(true_psi[::skip]),
            np.sin(true_psi[::skip]),
            color="green",
            scale=25,
            width=0.004,
            label="Vector",
        )
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_title("Car trajectory")
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.axis("equal")

        # X(t)
        ax = axes[0, 1]
        ax.plot(time, true_x, "k-", label="True X", linewidth=2)
        ax.plot(time, meas_x, "b.", alpha=0.3, markersize=3)
        ax.plot(time, filtered_x, "r-", label="Filtered X", linewidth=2)
        ax.set_xlabel("Time")
        ax.set_ylabel("X")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Y(t)
        ax = axes[1, 0]
        ax.plot(time, true_y, "k-", label="True Y", linewidth=2)
        ax.plot(time, meas_y, "b.", alpha=0.3, markersize=3)
        ax.plot(time, filtered_y, "r-", label="Filtered Y", linewidth=2)
        ax.set_xlabel("Time")
        ax.set_ylabel("Y")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Vector and velocity
        ax = axes[1, 1]
        ax.plot(time, true_psi, "k-", label="True vector", linewidth=2)
        ax.plot(time, true_v, "k--", label="True velocity", linewidth=2)
        ax.plot(time, ekf.history_psi, "r-", label="Vector History", alpha=0.8)
        ax.plot(time, ekf.history_v, "b-", label="Velocity History", alpha=0.8)
        ax.set_xlabel("Time")
        ax.set_ylabel("phi / v")
        ax.set_title("Vector and Velocity")
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        rmse_x = RMSE(true_x, filtered_x)
        rmse_y = RMSE(true_y, filtered_y)
        rmse_meas = np.sqrt(
            np.mean((true_x - meas_x) ** 2 + (true_y - meas_y) ** 2)
        )
        rmse_ekf = np.sqrt(rmse_x**2 + rmse_y**2)
        print(f"RMSE GPS:      {rmse_meas:.3f} м")
        print(f"RMSE EKF:      {rmse_ekf:.3f} м")
        print(f"by X: {rmse_x:.3f}, by Y: {rmse_y:.3f}")

        return fig


    step4()
    return


if __name__ == "__main__":
    app.run()

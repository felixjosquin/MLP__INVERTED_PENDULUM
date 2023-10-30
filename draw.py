import numpy as np
import matplotlib.pyplot as plt
import math
from simulation import l_bar, dt
import time

fps_anim = 18


class Draw:
    def __init__(self, X):
        self.theta = X[:, 0]
        self.x = X[:, 2]
        self.time = np.array([i * dt for i in range(len(X))])
        self.run_simu = True

    def draw_cart(self, i=0):
        if self.run_simu:
            a = int(1 / (fps_anim * dt))
            plt.clf()
            self.plot_cart(i)
            pause = dt * a - 0.02 if dt * a > 0.02 else 0.001  # supp calcul time
            plt.pause(pause)
            i += a
            if i < len(self.time):
                self.draw_cart(i)

    def draw_graph(self):
        fig = plt.figure()
        fig.set_figheight(9)
        fig.set_figwidth(9)

        ax_theta = plt.subplot2grid(shape=(3, 2), loc=(0, 0), colspan=2)
        ax_dtheta = plt.subplot2grid((3, 2), (1, 0))
        ax_x = plt.subplot2grid((3, 2), (1, 1))
        ax_dx = plt.subplot2grid((3, 2), (2, 0))
        ax_d2x = plt.subplot2grid((3, 2), (2, 1))

        ax_theta.plot(self.time, np.rad2deg(self.theta))
        ax_theta.set_title("Theta [deg]")

        ax_dtheta.plot(self.time, np.rad2deg(self.dtheta))
        ax_dtheta.set_title("dTheta/dt [deg/s]")

        ax_x.plot(self.time, self.x)
        ax_x.set_title("x Function [m]")

        ax_dx.plot(self.time, self.dx)
        ax_dx.set_title("dx/dt [m/s]")

        ax_d2x.plot(self.time, self.d2x)
        ax_d2x.set_title("d2x/dt2 [m/s]")

        plt.gcf().canvas.mpl_connect(
            "key_release_event",
            lambda event: [plt.close("all") if event.key == "escape" else None],
        )
        plt.show()

    def plot_cart(self, i):
        xt, theta, time = self.x[i], self.theta[i], self.time[i]

        cart_w = 1.0  # longeur du card
        cart_h = 0.5  # hauteur du card
        radius = 0.1

        cx = np.array(
            [-cart_w / 2.0, cart_w / 2.0, cart_w / 2.0, -cart_w / 2.0, -cart_w / 2.0]
        )
        cy = np.array([0.0, 0.0, cart_h, cart_h, 0.0])
        cy += radius * 2.0

        cx = cx + xt

        bx = np.array([0.0, l_bar * math.sin(-theta)])
        bx += xt
        by = np.array([cart_h, l_bar * math.cos(-theta) + cart_h])
        by += radius * 2.0

        angles = np.arange(0.0, math.pi * 2.0, math.radians(3.0))
        ox = np.array([radius * math.cos(a) for a in angles])
        oy = np.array([radius * math.sin(a) for a in angles])

        rwx = np.copy(ox) + cart_w / 4.0 + xt
        rwy = np.copy(oy) + radius
        lwx = np.copy(ox) - cart_w / 4.0 + xt
        lwy = np.copy(oy) + radius

        wx = np.copy(ox) + bx[-1]
        wy = np.copy(oy) + by[-1]

        plt.plot(cx, cy, "-b")
        plt.plot(bx, by, "-k")
        plt.plot(rwx, rwy, "-k")
        plt.plot(lwx, lwy, "-k")
        plt.plot(wx, wy, "-k")
        plt.title(f"x: {xt:.2f} , theta: {math.degrees(theta):.2f}, time: {time:.2f}")

        # for stopping simulation with the esc key.
        plt.gcf().canvas.mpl_connect(
            "key_release_event",
            lambda event: [self.on_escape() if event.key == "escape" else None],
        )

        plt.ylim(cart_h - l_bar - 3 * radius - 0.1, cart_h + l_bar + 3 * radius + 0.1)
        plt.xlim(xt - l_bar - 3 * radius - 0.1, xt + l_bar + 3 * radius + 0.1)

    def on_escape(self):
        self.run_simu = False

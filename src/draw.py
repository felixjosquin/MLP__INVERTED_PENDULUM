import csv
import os
import sys

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from src.utils import CSV_HEADER

vit = 2


class Draw:
    def __init__(self, eps_number=None, simu_number=None):
        self.df_X, self.simu_number, self.eps_number = self._init_X(
            simu_number, eps_number
        )

    @staticmethod
    def _init_X(simu_number, eps_number):
        if simu_number is not None:
            df = pd.read_csv(f"./data/simu_{simu_number}.csv")
        else:
            i = 0
            file_path = f"./data/simu_0.csv"
            while os.path.exists(file_path):
                i += 1
                file_path = f"./data/simu_{i}.csv"
            df = pd.read_csv(f"./data/simu_{i-1}.csv")
        eps_cond = (
            df[CSV_HEADER.EPISODE] == eps_number
            if eps_number is not None
            else df[CSV_HEADER.EPISODE] == df[CSV_HEADER.EPISODE].max()
        )
        return (
            df.loc[eps_cond],
            simu_number if simu_number is not None else i - 1,
            eps_number if eps_number is not None else df[CSV_HEADER.EPISODE].max(),
        )

    def draw_animation(self):
        fig, ax = plt.subplots()
        df_X = self.df_X.loc[
            :, [CSV_HEADER.TIME.value, CSV_HEADER.THETA.value, CSV_HEADER.X.value]
        ]
        num_frames = len(df_X)

        def update(frame):
            plt.clf()
            X = df_X.iloc[frame * vit].to_numpy()
            plot_cart(
                X, f"Simulation n°{self.simu_number}  Episode n°{self.eps_number}"
            )

        animation = FuncAnimation(
            fig,
            update,
            frames=list(range(num_frames // vit)),
            interval=10,
        )

        def handle(event):
            if event.key == "escape":
                animation.event_source.stop()
                plt.close()
            if event.key == "r":
                animation.event_source.stop()
                animation.frame_seq = animation.new_frame_seq()
                animation.event_source.start()

        fig.canvas.mpl_connect("key_press_event", handle)
        plt.show()

    def draw_graph(self):
        fig = plt.figure()
        fig.set_figheight(9)
        fig.set_figwidth(9)

        time, theta, dtheta, X, dX, U = np.transpose(
            self.df_X.loc[
                :,
                [
                    CSV_HEADER.TIME,
                    CSV_HEADER.THETA,
                    CSV_HEADER.dTHETA,
                    CSV_HEADER.X,
                    CSV_HEADER.dX,
                    CSV_HEADER.U_command,
                ],
            ].to_numpy()
        )

        ax_theta = plt.subplot2grid(shape=(3, 2), loc=(0, 0), colspan=2)
        ax_dtheta = plt.subplot2grid((3, 2), (2, 0))
        ax_x = plt.subplot2grid((3, 2), (1, 0))
        ax_dx = plt.subplot2grid((3, 2), (1, 1))
        ax_forces = plt.subplot2grid((3, 2), (2, 1))

        ax_theta.plot(time, np.rad2deg(theta))
        ax_theta.set_title("Theta [deg]")

        ax_dtheta.plot(time, np.rad2deg(dtheta))
        ax_dtheta.set_title("dTheta/dt [deg/s]")

        ax_x.plot(time, X)
        ax_x.set_title("x Function [m]")

        ax_dx.plot(time, dX)
        ax_dx.set_title("dx/dt [m/s]")

        ax_forces.plot(time, U)
        ax_forces.set_title("tension [U]")

        plt.gcf().canvas.mpl_connect(
            "key_release_event",
            lambda event: [plt.close("all") if event.key == "escape" else None],
        )
        plt.show()


def plot_cart(X, title):
    time, theta, xt = X

    cart_w = 0.3  # longeur du card
    cart_h = 0.1  # hauteur du card
    l_bar = 0.3  # hauteur du card
    radius = 0.02

    cx = np.array(
        [-cart_w / 2.0, cart_w / 2.0, cart_w / 2.0, -cart_w / 2.0, -cart_w / 2.0]
    )
    cy = np.array([0.0, 0.0, cart_h, cart_h, 0.0])
    cy += radius * 2.0

    cx = cx + xt

    bx = np.array([0.0, l_bar * np.sin(-theta)])
    bx += xt
    by = np.array([cart_h, l_bar * np.cos(-theta) + cart_h])
    by += radius * 2.0

    angles = np.arange(0.0, np.pi * 2.0, np.radians(3.0))
    ox = np.array([radius * np.cos(a) for a in angles])
    oy = np.array([radius * np.sin(a) for a in angles])

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
    plt.title(
        f"{title}\nx: {xt:.2f} , theta: {np.degrees(theta):.2f}, time: {time:.2f}"
    )

    plt.ylim(cart_h - l_bar - 3 * radius - 0.1, cart_h + l_bar + 3 * radius + 0.1)
    plt.xlim(-0.2 - cart_w, 0.2 + cart_w)

import numpy as np
import matplotlib.pyplot as plt
import math
from simulation import l_bar,dt

class Draw:
    def __init__(self,X_historique):
        self.x=X_historique[0,:]
        self.dx=X_historique[1,:]
        self.theta=X_historique[2,:]
        self.dtheta=X_historique[3,:]        
        self.time=X_historique[4,:]
        self.d2x=X_historique[5,:]
        self.run_simu=True

    
    def draw_cart(self,i=0):
        if self.run_simu:
            plt.clf()
            plot_cart(self.x[i], self.theta[i],self.time[i],self)
            plt.pause(dt/10)
            if i+1<len(self.time):
                self.draw_cart(i+1)
    
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
        'key_release_event',
        lambda event: [plt.close('all') if event.key == 'escape' else None])
        plt.show()




def plot_cart(xt, theta,time,draw):
    cart_w = 1.0 # longeur du card
    cart_h = 0.5 # hauteur du card
    radius = 0.1

    plt.xlim([-2.0, 2.0])
    # Cordonée des point de la card
    cx = np.array([-cart_w / 2.0, cart_w / 2.0, cart_w /2.0, -cart_w / 2.0, -cart_w / 2.0])
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
        'key_release_event',
        lambda event: [on_escape(draw,event) if event.key == 'escape' else None])
    plt.axis("equal")

def on_escape(draw,event):
    draw.run_simu=False
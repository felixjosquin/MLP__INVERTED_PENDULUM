import math

import matplotlib.pyplot as plt
import numpy as np

################ Model parameters ###################
l_bar = 2.0  # length of bar
m = 0.3  # [kg]
g = 9.8  # [m/s^2]
#####################################################


################ simaulation parametre ##############
dt = 0.1  # time tick [s]
sim_time = 5.0  # simulation time
angle_max = 45 
show_animation=True
X0 = np.array([
        0.0, # x
        0.0, # dx/dt
        0.1, # tehta
        0.0  # dtheta/dt
    ])
#####################################################

################# fonction dérivé ###################
def F (t,X,d2x):
    d2theta=(g*math.sin(X[2])+d2x*math.cos(X[2]))/l_bar
    return np.array([X[1],d2x,X[3],d2theta])

#####################################################

def main():
    simu=simulation()
    bol_continue=True
    while bol_continue:
        bol_continue=simu.step(-0.5)
    simu.finish()

class simulation:
    def __init__(self,X=X0,show=show_animation):
        self.X=np.copy(X)
        self.X_historique=np.copy([X])
        self.show_animation=show
        self.time=0.0
    
    def step(self,dx2):
        self.X=calcul_Runge_Kutta(self.time,self.X,dx2)
        self.X_historique=np.append(self.X_historique,[self.X],axis = 0 )
        if self.show_animation:
            plt.clf()
            px = float(self.X[0])
            theta = float(self.X[2])
            plot_cart(px, theta)
            plt.pause(0.001)
        self.time += dt
        if sim_time < self.time or math.degrees(self.X[2])>angle_max or math.degrees(self.X[2])<-angle_max:
            return False
        return True
    
    def finish(self):
        print("Finish")
        print(f"x={float(self.X[0]):.2f} [m] , theta={math.degrees(self.X[2]):.2f} [deg]")
        if show_animation:
            plt.show()

def calcul_Runge_Kutta (time,X,d2x):
    k1=F(time,X,d2x)*dt
    k2=F(time+dt/2,X+k1/2,d2x)*dt
    k3=F(time+dt/2,X+k2/2,d2x)*dt
    k4=F(time+dt,X+k3,d2x)*dt
    return X+(k1+2*k2+2*k3+k4)/6

def plot_cart(xt, theta):
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
    plt.title(f"x: {xt:.2f} , theta: {math.degrees(theta):.2f}")

    # for stopping simulation with the esc key.
    plt.gcf().canvas.mpl_connect(
        'key_release_event',
        lambda event: [exit(0) if event.key == 'escape' else None])

    plt.axis("equal")
    plt.pause(0.001)

if __name__ == '__main__':
    main()
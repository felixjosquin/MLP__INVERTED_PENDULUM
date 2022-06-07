import math
import time
import matplotlib.pyplot as plt
import numpy as np

################ Model parameters ###################
l_bar = 2.0  # length of bar
m = 0.3  # [kg]
g = 9.8  # [m/s^2]
#####################################################

################ simaulation parametre ##############
dt = 0.05  # time tick [s]
sim_max = 20.0  # simulation time
angle_max = 150
#####################################################


################# fonction dérivé ###################
def F (t,X,d2x):
    d2theta=(g*math.sin(X[2])+d2x*math.cos(X[2]))/l_bar
    return np.array([X[1],d2x,X[3],d2theta])
#####################################################

class Simulation:
    def __init__(self,X0,register):
        self.X=np.copy(X0)
        self.time=0.0
        self.historique=np.append(np.copy(X0),[self.time,0.0],axis=0)
        self.register=register
        
    
    def step(self,dx2):
        self.X=calcul_Runge_Kutta(self.time,self.X,dx2,dt)
        self.time += dt
        arr=np.append(np.copy(self.X),[self.time,dx2],axis=0)
        self.historique=np.c_[self.historique,arr] 
        return (abs(self.X[2])<math.radians(angle_max)) and (self.time<=sim_max)
    
    def finish(self):
        if self.register:
            np.savetxt('data/historique.csv', self.historique, delimiter = ';') 
        print(f"theta 0={math.degrees(self.historique[2,0])} [deg] , theta={math.degrees(self.X[2]):.2f} [deg] , time={self.time:.2f} [s]")
    
    def get_input(self):
        return self.X

def calcul_Runge_Kutta (time,X,d2x,dt):
    k1=F(time,X,d2x)*dt
    k2=F(time+dt/2,X+k1/2,d2x)*dt
    k3=F(time+dt/2,X+k2/2,d2x)*dt
    k4=F(time+dt,X+k3,d2x)*dt
    return X+(k1+2*k2+2*k3+k4)/6


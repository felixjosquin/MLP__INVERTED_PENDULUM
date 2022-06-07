import time
import math
import simulation 

acc_x_const=40.
acc_theta_const=math.radians(170.)
theta_const=math.radians(90.)


class Trainer:
    def __init__(self, simu, NN):
        self.simu = simu
        self.network = NN
        self.training = None
        self.running = None
        self.alpha = []

    def train(self):
        X = self.simu.get_input()
        theta,dtheta=X[2],X[3]
        network_input = [theta/theta_const,dtheta/acc_theta_const]  
        while self.running:
            [output] = self.network.runNN(network_input)
            command=output*acc_x_const
            self.running=self.simu.step(command) 
            if self.training:
                X = self.simu.get_input()
                grad = [
                   -(X[2]*(simulation.dt)**2/(simulation.l_bar*math.cos(X[2])))
                ]
                self.network.backPropagate(grad,0.2, 0.1)
            X = self.simu.get_input()
            theta,dtheta=X[2],X[3]
            network_input = [theta/theta_const,dtheta/acc_theta_const] 
        self.simu.finish()

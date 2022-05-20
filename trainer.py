import time
import math
import simulation 

acc_x_const=10.
acc_theta_const=2.
theta_const=math.radians(10.)


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
        [output] = self.network.runNN(network_input)
        command=output*acc_x_const
        self.running=self.simu.step(command)  
        while self.running:
            [output] = self.network.runNN(network_input)
            command=output*acc_x_const
            self.running=self.simu.step(command) 
            if self.training:
                X = self.simu.get_input()
                grad = [
                    -X[2]*((simulation.dt)**2+math.cos(X[2]))/(2*simulation.l_bar)
                ]
                self.network.backPropagate(grad,0.1, 0)
            X = self.simu.get_input()
            theta,dtheta=X[2],X[3]
            network_input = [theta/theta_const,dtheta/acc_theta_const] 
        self.simu.finish()

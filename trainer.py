import time
import math

class Trainer:
    def __init__(self, simu, NN):
        self.simu = simu
        self.network = NN
        self.training = None
        self.running = None

    def train(self):
        while self.running:
            X = self.simu.get_input()
            network_input = [X[1],X[2],X[3]]
            [command] = self.network.runNN(network_input) # propage erreur et calcul vitesses roues instant t
            self.running=self.simu.step(command) 
            if self.training:
                grad = [
                
                ]
                self.network.backPropagate(grad, 0.05, 0)
        self.simu.finish()

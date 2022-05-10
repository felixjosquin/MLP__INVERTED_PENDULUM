from simulation import Simulation
from ReseauNeurone import NN
from trainer import Trainer

import json
import threading
import numpy as np


################ simaulation parametre ##############
dt = 0.05  # time tick [s]
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


simu = Simulation(X0,dt,True,angle_max)
HL_size= 10 # nbre neurons of Hiden layer
network = NN(3, HL_size, 1)

choice = input('Do you want to load previous network? (y/n) --> ')
if choice == 'y':
    with open('last_w.json') as fp:
        json_obj = json.load(fp)
    for i in range(3):
        for j in range(HL_size):
            network.wi[i][j] = json_obj["input_weights"][i][j]
    for i in range(HL_size):
        for j in range(2):
            network.wo[i][j] = json_obj["output_weights"][i][j]

trainer = Trainer(simu, network)

choice = ''
while choice!='y' and choice !='n':
    choice = input('Do you want to learn? (y/n) --> ')

if choice == 'y':
    trainer.training = True
elif choice == 'n':
    trainer.training = False

thread = threading.Thread(target=trainer.train)
trainer.running = True
thread.start()

while (trainer.running):
    trainer.simu.draw()

json_obj = {"input_weights": network.wi, "output_weights": network.wo}
with open('last_w.json', 'w') as fp:
    json.dump(json_obj, fp)

print("The last weights have been stored in last_w.json")

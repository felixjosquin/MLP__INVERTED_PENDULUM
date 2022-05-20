from simulation import Simulation
from ReseauNeurone import NN
from trainer import Trainer
from draw import Draw

import json
import threading
import numpy as np
import random
import math


# X0 = np.array([
#         0.0, # x
#         0.0, # dx/dt
#         0.06, # tehta
#         0.0  # dtheta/dt
#     ])

theta_max=math.radians(70)

HL_size= 10 # nbre neurons of Hiden layer
network = NN(2, HL_size, 1)

##################### AI training #####################
for k in range(100):
    X0=np.array([0,0,2*theta_max*(random.random()-0.5),0])
    simu = Simulation(X0,False)
    trainer = Trainer(simu, network)
    trainer.training = True
    trainer.running = True
    trainer.train()

################### AI on one exemple #####################
X0=np.array([0,0,2*theta_max*(random.random()-0.5),0])
simu = Simulation(X0,True)
trainer = Trainer(simu, network)
trainer.training = False
trainer.running = True
trainer.train()

################### afficher le résultat #####################
data = np.genfromtxt('data/historique.csv', delimiter=';')
draw = Draw(data)
draw.draw_cart()
draw.draw_graph()

json_obj = {"input_weights": network.wi, "output_weights": network.wo}
with open('data/last_w.json', 'w') as fp:
    json.dump(json_obj, fp)

print("The last weights have been stored in last_w.json")

# choice = input('Do you want to load previous network? (y/n) --> ')
# if choice == 'y':
#     with open('last_w.json') as fp:
#         json_obj = json.load(fp)
#     for i in range(2):
#         for j in range(HL_size):
#             network.wi[i][j] = json_obj["input_weights"][i][j]
#     for i in range(HL_size):
#         for j in range(1):
#             network.wo[i][j] = json_obj["output_weights"][i][j]


# choice = ''
# while choice!='y' and choice !='n':
#     choice = input('Do you want to learn? (y/n) --> ')

# if choice == 'y':
#     trainer.training = True
# elif choice == 'n':
#     trainer.training = False

# thread = threading.Thread(target=trainer.train)
# trainer.running = True
# thread.start()

# while (trainer.running):
#     trainer.simu.draw()
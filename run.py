from simulation import Simulation,sim_max
from ReseauNeurone import NN
from trainer import Trainer
from draw import Draw

import json
import numpy as np
import random
import math


# X0 = np.array([
#         0.0, # x
#         0.0, # dx/dt
#         0.06, # tehta
#         0.0  # dtheta/dt
#     ])

theta_max=math.radians(80)

HL_size= 10 # nbre neurons of Hiden layer
network = NN(2, HL_size, 1)

##################### AI training #####################
continue_train=True
nb_entrainement=0
nb_succes_affile=0  

while continue_train:
    X0=np.array([0,0,2*theta_max*(random.random()-0.5),0])
    simu = Simulation(X0,False)
    trainer = Trainer(simu, network)
    trainer.training = True
    trainer.running = True
    trainer.train()
    if trainer.simu.time>=sim_max:
        nb_succes_affile+=1
        if nb_succes_affile==10:
            continue_train=False
            print(f"Stabilité trouvé aprés {nb_entrainement} essaie")
    else:
        nb_succes_affile=0
    nb_entrainement+=1
    if nb_entrainement==200:
        continue_train=False
        print(f"Stabilité non trouvé aprés {nb_entrainement} essaie")

################### AI on one exemple #####################
X0=np.array([0,0,0.9*theta_max,0])
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

################### Register network  #####################
json_obj = {"input_weights": network.wi, "output_weights": network.wo}
with open('data/last_w.json', 'w') as fp:
    json.dump(json_obj, fp)

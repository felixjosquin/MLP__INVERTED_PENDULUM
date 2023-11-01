import simulation
import draw
import mlp
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim

dt_command = 0.04

optimizer = optim.Adam(mlp.model.parameters(), lr=0.005)

sucess = 0
for r in range(300):
    X0 = [random.random() * 0.3 + 0.2, 0.0, 0.0, 0.0]
    simu = simulation.Simulation(X0)
    stop = False
    for j in range(100):
        x = torch.tensor([x for x in simu.get_last()], dtype=torch.float32)
        output = mlp.model(x)
        loss = simulation.loss_simultation(output, x, dt_command)
        stop = not simu.step(output.item(), dt_command)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if stop:
            print("stop")
            break
    if not stop:
        sucess += 1
        if sucess == 5:
            break
    else:
        sucess = 0

    print("epoch ", r, "loss : ", loss.item(), "reel : ", simu.get_last()[0] ** 2)

# simu.step(output[0].item(), dt_command)
# theta = torch.tensor([simu.X[-1, 0]], requires_grad=True, dtype=torch.float32)

# loss = criterion(theta, x)


draw = draw.Draw(simu.X, simu.forces)
draw.draw_cart()
draw.draw_graph()

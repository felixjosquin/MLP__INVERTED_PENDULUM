from collections import deque
import random
import numpy as np
import torch

from src.simulation import Simulation
from src.mlp import Linear_QNet, QTrainer
from src.utils import (
    BATCH_SIZE,
    EPS_DECAY,
    EPS_END,
    EPS_START,
    HIDDEN_LAYER,
    MEMORY_DEQUE,
    NB_ACTION,
)


class Agent:
    def __init__(self):
        self.memory = deque(maxlen=MEMORY_DEQUE)
        self.steps_done = 0
        self.model = Linear_QNet(HIDDEN_LAYER)
        self.trainer = QTrainer(self.model)

    def get_action(self, state):
        self.steps_done += 1
        eps_threshold = EPS_END + (EPS_START - EPS_END) * np.exp(
            -self.steps_done / EPS_DECAY
        )
        if np.random.rand() > eps_threshold:
            state0 = torch.tensor(state, dtype=torch.float)
            with torch.no_grad():
                action = torch.argmax(self.model(state0)).item()
        else:
            action = np.random.randint(NB_ACTION - 1)
        return action

    def train_short(self, state, action, reward, next_state, termined):
        self.trainer.train_step([state], [action], [reward], [next_state], [termined])

    def train_batch(self):
        if len(self.memory) < BATCH_SIZE:
            mini_sample = mini_sample = self.memory
        else:
            mini_sample = random.sample(self.memory, BATCH_SIZE)  # list of tuples
        states, actions, rewards, next_states, termined = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, termined)

    def remember(self, state, action, reward, next_state, is_termined):
        self.memory.append((state, action, reward, next_state, is_termined))


def train():
    agent = Agent()
    simu = Simulation()
    while True:
        state_old = simu.get_state()
        action = agent.get_action(state_old)
        next_state, reward, is_termined, is_truncated = simu.step(action)
        agent.remember(state_old, action, reward, next_state, is_termined)
        agent.train_short(state_old, action, reward, next_state, is_termined)

        if is_termined or is_truncated:
            agent.train_batch()
            print(f"episode : {simu.episode} | time : {simu.time}")
            simu.reset()


if __name__ == "main":
    train()

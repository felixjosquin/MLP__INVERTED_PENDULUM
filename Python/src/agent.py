from collections import deque
import random
import numpy as np


from src.simulation import Simulation
from src.mlp import QTrainer
from src.utils import (
    BATCH_SIZE,
    EPS_DECAY,
    EPS_END,
    EPS_START,
    MEMORY_DEQUE,
    NB_ACTION,
)


class Agent:
    def __init__(self, path=None):
        self.memory = deque(maxlen=MEMORY_DEQUE)
        self.steps_done = 0
        self.trainer = QTrainer(path)
        self.times = []
        self.sum_max_time = 0.0

    def get_action(self, state):
        eps_threshold = EPS_END + (EPS_START - EPS_END) * np.exp(
            -self.steps_done / EPS_DECAY
        )
        if np.random.rand() > eps_threshold:
            action = self.trainer.do_a_prediction(state)
        else:
            action = np.random.randint(NB_ACTION - 1)
        self.steps_done += 1
        return action

    def train_short(self, *arg):
        self.trainer.train_step(*(np.array([a]) for a in arg))

    def train_batch(self):
        if len(self.memory) < BATCH_SIZE:
            mini_sample = list(self.memory)
        else:
            mini_sample = random.sample(self.memory, BATCH_SIZE)  # list of tuples
        batchs = [mini_sample[i : i + 50] for i in range(0, len(mini_sample), 50)]
        for batch in batchs:
            states, actions, rewards, next_states, termined = (
                np.array(x) for x in zip(*batch)
            )
            self.trainer.train_step(states, actions, rewards, next_states, termined)

    def remember(self, state, action, reward, next_state, is_termined):
        self.memory.append((state, action, reward, next_state, is_termined))

    def remember_batch(self, batch):
        self.memory.extend(batch)

    def add_time(self, time):
        self.times.append(time)

    def register__mlp(self, epsiode):
        self.trainer.register_model(epsiode)


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
            agent.add_time(simu.time)
            if simu.episode > 50 and agent.need_register_mlp():
                agent.register__mlp(simu.simu_number)

            agent.train_batch()
            print(f"episode : {simu.episode} | time : {simu.time}")
            simu.reset()


if __name__ == "main":
    train()

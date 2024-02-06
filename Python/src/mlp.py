import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from src.utils import GAMMA, LR, NB_ACTION


class Linear_QNet(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.linear1 = nn.Linear(4, hidden_size)
        self.linear2 = nn.Linear(hidden_size, NB_ACTION)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x


class QTrainer:
    def __init__(self, model):
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=LR)
        self.criterion = nn.SmoothL1Loss()

    def train_step(self, states, actions, rewards, next_states, termined):
        states = torch.tensor(states, dtype=torch.float)
        next_states = torch.tensor(next_states, dtype=torch.float)
        actions = torch.tensor(actions, dtype=torch.int64)
        rewards = torch.tensor(rewards, dtype=torch.float)
        done = torch.tensor(termined, dtype=torch.bool)
        # (n, x)
        predic = self.model(states)
        state_action_values = predic.gather(dim=1, index=actions.view(-1, 1))

        next_state_values = torch.zeros(states.shape[0])
        with torch.no_grad():
            next_state_values[~done] = self.model(next_states[~done]).max(1).values
        expected_state_action_values = (next_state_values * GAMMA) + rewards

        self.optimizer.zero_grad()
        loss = self.criterion(
            expected_state_action_values.view(-1, 1), state_action_values
        )
        loss.backward()

        self.optimizer.step()

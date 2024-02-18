import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_

from src.utils import GAMMA, LR, NB_ACTION, HIDDEN_LAYER, UPDATE_INTERVAL


class Linear_QNet(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.linear1 = nn.Linear(4, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, NB_ACTION)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return x


class QTrainer:
    def __init__(self):
        self.model = Linear_QNet(HIDDEN_LAYER)
        self.target_model = Linear_QNet(HIDDEN_LAYER)
        self.target_model.load_state_dict(self.model.state_dict())
        self.optimizer = optim.Adam(self.model.parameters(), lr=LR)
        self.criterion = nn.SmoothL1Loss()
        self.nb_step_unupdate = 0

    def do_a_prediction(self, state):
        state0 = torch.tensor(state, dtype=torch.float)
        with torch.no_grad():
            return torch.argmax(self.model(state0)).item()

    def train_step(self, states, actions, rewards, next_states, termined):
        states = torch.tensor(states, dtype=torch.float)
        next_states = torch.tensor(next_states, dtype=torch.float)
        actions = torch.tensor(actions, dtype=torch.int64)
        rewards = torch.tensor(rewards, dtype=torch.float)
        termined = torch.tensor(termined, dtype=torch.bool)

        # (n, x)
        predic = self.model(states)
        state_action_values = predic.gather(dim=1, index=actions.view(-1, 1))

        next_state_values = torch.zeros(states.shape[0])
        with torch.no_grad():
            next_state_values[~termined] = (
                self.target_model(next_states[~termined]).max(1).values
            )
        expected_state_action_values = (next_state_values * GAMMA) + rewards

        self.optimizer.zero_grad()
        loss = self.criterion(
            expected_state_action_values.view(-1, 1), state_action_values
        )
        loss.backward()
        clip_grad_norm_(self.model.parameters(), 1.0)

        self.optimizer.step()

        self.nb_step_unupdate += states.size(dim=0)
        if self._need_upadate_():
            self._update_target()

    def _need_upadate_(self):
        return self.nb_step_unupdate > UPDATE_INTERVAL

    def _update_target(self):
        self.target_model.load_state_dict(self.model.state_dict())
        self.nb_step_unupdate = 0

    def register_model(self, simu_number):
        folder_path = "./data/models"
        file_name = f"best_{simu_number}.pth"
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        file_path = os.path.join(folder_path, file_name)
        torch.save(self.model.state_dict(), file_path)

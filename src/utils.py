from enum import Enum

DT_COMMAND = 0.01


class CSV_HEADER(str, Enum):
    EPISODE = "EPISODE"
    TIME = "TIME"
    THETA = "THETA"
    dTHETA = "dTHETA/dt"
    X = "X"
    dX = "dX/dt"
    U_command = "U_command"


##### HYPERPAREMTER #####
NB_ACTION = 5
HIDDEN_LAYER = 500

GAMMA = 0.99
BATCH_SIZE = 250
MEMORY_DEQUE = 100_000

LR = 0.05

EPS_START = 0.9
EPS_END = 0.0
EPS_DECAY = 5000
#########################

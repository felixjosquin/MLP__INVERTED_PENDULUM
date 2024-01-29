from enum import Enum

DT_COMMAND = 0.05


class CSV_HEADER(str, Enum):
    EPISODE = "EPISODE"
    TIME = "TIME"
    THETA = "THETA"
    dTHETA = "dTHETA/dt"
    X = "X"
    dX = "dX/dt"
    U_command = "U_command"


##### HYPERPAREMTER #####
NB_ACTION = 11
BATCH_SIZE = 1000
HIDDEN_LAYER = 256

GAMMA = 0.99
MEMORY_DEQUE = 10_000

LR = 0.01

EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 500
#########################

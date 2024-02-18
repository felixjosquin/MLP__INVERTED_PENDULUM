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
NB_ACTION = 5
HIDDEN_LAYER = 256

GAMMA = 0.995
BATCH_SIZE = 1024
MEMORY_DEQUE = 50_000

LR = 0.001

UPDATE_INTERVAL = 1500

EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 4000

ACTIONS = [-10.0, -7.0, 0, 7.0, 10.0]

#########################

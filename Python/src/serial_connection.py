import csv
import json
import os
import serial
import numpy as np
from src.agent import Agent
from src.simulation import get_reward

from src.mlp import QTrainer
from src.utils import ACTIONS, CSV_HEADER


class connect_USB:
    def __init__(self):
        self.ser = serial.Serial(port="COM6", baudrate=115200, timeout=0.5)

    def start(self):
        output = ""
        while output != "start":
            self.ser.write(("start\n").encode("utf-8"))
            self.ser.flush()
            if self.ser.in_waiting:
                output = self.ser.readline().decode().strip()
        print("Connect -> start")

    def send_data(self, dic):
        json_string = json.dumps(dic)
        json_string += "\n"
        self.ser.write(json_string.encode("utf-8"))
        self.ser.flush()

    def read_data(self):
        json_string = ""
        while not json_string:
            json_string = self.ser.readline().decode("utf-8")
        return json.loads(json_string)

    def stop(self):
        self.ser.close()
        self.ser.open()
        self.ser.close()


class Middleware:
    def __init__(self, predict_function, learn_function=None):
        self.episode_states = []
        self.path = self._init_file()
        self.episode = 0
        self.predict_function = predict_function
        self.learn_function = learn_function

    def step(self, output):
        status = output.get("s", None)
        match status:
            case 1:  # rev data and need to send new command
                state = self.rcv_data(output)
                command = self.get_command(state)
                return {"c": command}
            case 2:  # end of episode
                state = self.rcv_data(output)
                self.end_episode(state)
                return None
            case 3:
                raise ArduinoException
            case _:
                raise NoResponceException

    def rcv_data(self, output):
        old_state = self.episode_states[-1] if self.episode_states else None
        new_state = {
            "time": output["t"] / 1.0e6,
            "x": output["x"] / 1.0e3,
            "theta": -output["th"] / 1.0e3,
            "command": output["c"],
        }
        new_state["dx/dt"] = (
            (new_state["x"] - old_state["x"]) / (new_state["time"] - old_state["time"])
            if old_state is not None
            else 0.0
        )
        new_state["dtheta/dt"] = (
            (new_state["theta"] - old_state["theta"])
            / (new_state["time"] - old_state["time"])
            if old_state is not None
            else 0.0
        )
        return new_state

    def end_episode(self, state):
        self.episode_states.append(state)
        if self.learn_function is not None:
            self.learn_function(self.episode_states, self.episode)
        print(
            f"-------- End of episode {self.episode} : {self.episode_states[-1]['time']:.2f} s --------"
        )
        self._register_data()
        self.episode += 1
        self.episode_states = []

    def get_command(self, state):
        state_np = np.array(
            [state[key] for key in ["theta", "dtheta/dt", "x", "dx/dt"]]
        )
        action = self.predict_function(state_np)
        state["action"] = action
        self.episode_states.append(state)
        return int(ACTIONS[action] * 255 / 12)

    def _init_file(self):
        os.makedirs("./data/real_data", exist_ok=True)
        file_name = "experience_0.csv"
        i = 0
        while os.path.exists(os.path.join("./data/real_test", file_name)):
            i += 1
            file_name = f"experience_{i}.csv"
        path = os.path.join("./data/real_test", file_name)
        with open(path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=[el.value for el in CSV_HEADER])
            writer.writeheader()
        return path

    def _register_data(self):
        lines_to_add = [
            {
                CSV_HEADER.EPISODE.value: self.episode,
                CSV_HEADER.TIME.value: data.get("time"),
                CSV_HEADER.THETA.value: data.get("theta"),
                CSV_HEADER.dTHETA.value: data.get("dtheta/dt"),
                CSV_HEADER.X.value: data.get("x"),
                CSV_HEADER.dX.value: data.get("dx/dt"),
                CSV_HEADER.U_command.value: data.get("command"),
            }
            for data in self.episode_states
        ]
        with open(self.path, "a", newline="") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=[el.value for el in CSV_HEADER])
            for line in lines_to_add:
                writer.writerow(line)


def predict(path):
    trainer = QTrainer(path)
    usbObject = connect_USB()

    def predic_function(state):
        return trainer.do_a_prediction(state)

    middleware = Middleware(predict_function=predic_function)
    try:
        usbObject.start()
        while True:
            data = usbObject.read_data()
            dic_to_send = middleware.step(data)
            if dic_to_send is not None:
                usbObject.send_data(dic_to_send)

    except KeyboardInterrupt:
        print("Exiting...")

    except ArduinoException:
        print("Arduino have a problem...")

    except NoResponceException:
        print("No status in arduino response...")

    finally:
        usbObject.stop()


def train_real(path):
    agent = Agent(path)
    usbObject = connect_USB()

    def predic_function(state):
        return agent.get_action(state)

    def learn(states, episode):
        agent.register__mlp(episode)
        states = [state for state in states if state["time"] > 1.0]
        states_np = [
            np.array([state[key] for key in ["theta", "dtheta/dt", "x", "dx/dt"]])
            for state in states
        ]
        nb_state = len(states)
        data = (
            (
                states_np[index],
                states[index]["action"],
                get_reward(states_np[index + 1], index == nb_state - 2),
                states_np[index + 1],
                index == nb_state - 2,
            )
            for index in range(nb_state - 1)
        )
        agent.remember_batch(data)
        agent.train_batch()

    middleware = Middleware(predict_function=predic_function, learn_function=learn)
    try:
        usbObject.start()
        while True:
            data = usbObject.read_data()
            dic_to_send = middleware.step(data)
            if dic_to_send is not None:
                usbObject.send_data(dic_to_send)

    except KeyboardInterrupt:
        print("Exiting...")

    except ArduinoException:
        print("Arduino have a problem...")

    except NoResponceException:
        print("No status in arduino response...")

    finally:
        usbObject.stop()


class ArduinoException(Exception):
    pass


class NoResponceException(Exception):
    pass

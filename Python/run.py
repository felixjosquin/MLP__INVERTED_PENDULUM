import argparse
from src.serial_connection import predict, train_real
from src.agent import train
from src.draw import Draw


parser = argparse.ArgumentParser()
parser.add_argument("-pl", "--plot", help="plot simulation", action="store_true")
parser.add_argument("-pr", "--predict", help="predict", action="store_true")
parser.add_argument("-tr", "--train_real", help="train model real", action="store_true")
parser.add_argument("-t", "--train", help="train model", action="store_true")

parser.add_argument("-pm", "--path_model", help="simu number to plot", type=str)
parser.add_argument("-sn", "--simu_number", help="simu number to plot", type=int)
parser.add_argument("-en", "--episode_number", help="episode to plot", type=int)
args = parser.parse_args()

if __name__ == "__main__":
    if args.plot:
        drawer = Draw(
            simu_number=args.simu_number,
            eps_number=args.episode_number,
        )
        drawer.draw_animation()
        drawer.draw_graph()
    if args.train:
        train()
    if args.predict:
        predict(args.path_model)
    if args.train_real:
        train_real(args.path_model)

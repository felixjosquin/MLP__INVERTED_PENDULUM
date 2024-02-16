import argparse
from src.agent import train
from src.draw import Draw


parser = argparse.ArgumentParser()
parser.add_argument("-p", "--plot", help="plot simulation", action="store_true")
parser.add_argument("-t", "--train", help="train model", action="store_true")
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

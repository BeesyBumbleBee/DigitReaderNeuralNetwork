from network import load_network
from time import perf_counter
from mnist_loader import load_data
import argparse

netpath = "./networks"

def main():
    args = parser.parse_args()
    _, test_data = load_data()
    netName = args.network_name
    net = load_network(f"{netpath}/{netName}", None)
    start = perf_counter()
    score = net.evaluate(test_data)
    end = perf_counter()
    print(f"{score:8d} / {len(test_data):8d}\nTook {end-start:4.2f} s")




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--network-name", action="store", default="digitreader")

    main()

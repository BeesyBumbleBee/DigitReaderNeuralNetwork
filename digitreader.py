from network import Network
from mnist_loader import load_data
from pickle import load, dump
import argparse

netpath = "./networks/"

def main():
    args = parser.parse_args(['--network-name', 'name', '--batch-size', 'batch', '--learning-rate', 'eta', '--epochs', 'epochs'])
    training_data, test_data = load_data()
    netName = args.name
    net = load_network(netName)
    if net is None:
        net = Network([784, 30, 10])

    net.SGD(training_data, args.epochs, args.batch, args.eta, test_data=test_data if args.testing else None)
    save_network(netName, net)


def load_network(name: str) -> Network | None:
    try:
        with open(f"{netpath}/{name}", "rb") as file:
            network = load(file)
    except FileNotFoundError:
        return None
    return network

def save_network(name: str, network: Network):
    with open(f"{netpath}/{name}", "wb") as file:
        dump(network, file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--epochs", default=30)
    parser.add_argument("-r", "--learning-rate", default=1)
    parser.add_argument("-b", "--batch-size", default=10)
    parser.add_argument("-n", "--network-name", default="digitreader")
    parser.add_argument("-t", "--testing", action="store_false")
    main()

from network import Network
from mnist_loader import load_data
from _pickle import load, dump
import gzip
import argparse

netpath = "./networks/"

def main():
    args = parser.parse_args()
    training_data, test_data = load_data()
    netName = args.network_name
    layers = ([784] + args.layers + [10])
    net = Network(layers)
    net = load_network(netName, net)
    if args.verbosity >= 2:
        print(f"Initial biases: {net.biases}\n\n")
        print(f"Initial weights: {net.weights}\n\n")
    if args.verbosity >= 1:
        print(f"Training network. Layers: {layers}")
    if args.testing is False:
        test_data = None

    net.SGD(training_data, args.epochs, args.batch_size, args.learning_rate, test_data=test_data)
    save_network(netName, net)


def load_network(name: str, net: Network) -> Network:
    try:
        fp = gzip.open(f"{netpath}/{name}.pkl.gz", "rb")
        print(f"Loading saved network: {name}");
        weights, biases = load(fp)
        fp.close()
        net.load_network(weights, biases)
    except FileNotFoundError:
        print(f"Creating network: {name}");
        pass
    return net

def save_network(name: str, network: Network):
    fp = gzip.open(f"{netpath}/{name}.pkl.gz", "wb")
    dump(network.get_network(), fp)
    fp.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--epochs", default=30, type=int)
    parser.add_argument("-r", "--learning-rate", default=1.0, type=float)
    parser.add_argument("-b", "--batch-size", default=10, type=int)
    parser.add_argument("-n", "--network-name", default="digitreader")
    parser.add_argument("-t", "--testing", action="store_false")
    parser.add_argument("-l", "--layers", nargs='+', type=int, default=[30])
    parser.add_argument("-v", "--verbosity", action="count", default=0)
    main()

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
    net = Network([784, 30, 10])
    net = load_network(netName, net)
    if args.testing is False:
        test_data = None

    net.SGD(training_data, args.epochs, args.batch_size, args.learning_rate, test_data=test_data)
    save_network(netName, net)


def load_network(name: str, net: Network) -> Network:
    try:
        fp = gzip.open(f"{netpath}/{name}.pkl.gz", "rb")
        weights, biases = load(fp)
        fp.close()
        net.load_network(weights, biases)
    except FileNotFoundError:
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
    main()

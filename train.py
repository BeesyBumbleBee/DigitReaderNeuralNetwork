from network import Network, load_network
from mnist_loader import load_data
from _pickle import load, dump
import gzip
import argparse

netpath = "./networks"

def main():
    args = parser.parse_args()
    training_data, test_data = load_data()
    netName = args.network_name
    layers = ([784] + args.layers + [10])
    net = load_network(f"{netpath}/{netName}", layers)

    if args.verbosity >= 2:
        print(f"Initial biases: {net.biases}\n\n")
        print(f"Initial weights: {net.weights}\n\n")
    if args.verbosity >= 1:
        print(f"Training network. Layers: {net.layers}")

    net.SGD(training_data, args.epochs, args.batch_size, args.learning_rate,
            lmbda=args.regularization,
            evaluation_data=test_data,
            monitor_eval_acc=args.eval_acc,
            monitor_eval_cost=args.eval_cost,
            monitor_train_acc=args.train_acc,
            monitor_train_cost=args.train_cost)
    net.save(f"{netpath}/{netName}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--epochs", default=30, type=int)
    parser.add_argument("-r", "--learning-rate", default=1.0, type=float)
    parser.add_argument("-R", "--regularization", default=0.0, type=float)
    parser.add_argument("-b", "--batch-size", default=10, type=int)
    parser.add_argument("-n", "--network-name", default="digitreader")
    parser.add_argument("--eval-acc", action="store_true")
    parser.add_argument("--eval-cost", action="store_true")
    parser.add_argument("--train-acc", action="store_true")
    parser.add_argument("--train-cost", action="store_true")
    parser.add_argument("-l", "--layers", nargs='+', type=int, default=[30])
    parser.add_argument("-v", "--verbosity", action="count", default=0)

    main()

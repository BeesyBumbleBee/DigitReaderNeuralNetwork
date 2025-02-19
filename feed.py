from network import load_network
from time import perf_counter
from mnist_loader import load_data
from json import load
import numpy as np
import argparse

netpath = "./networks"

def display(img : np.matrix, height=28, width=28):
    for row in range(height):
        for col in range(width):
            print(f"\033[48;2;{img.item(row*height+col)};{img.item(row*height+col)};{img.item(row*height+col)}m  \033[0m", end='')
        print("", end="\n")

def textToData(txt: str):
    result = []
    for char in txt:
        if char == '0':
            result.append(0)
        elif char == '1':
            result.append(255)
    return np.asmatrix(result).transpose()

def jsonToData(json_file):
    input_data = load(json_file)
    input_data = input_data['values']
    input_data = np.asmatrix(input_data)
    if input_data.shape[0] != 1:
        return input_data
    return input_data.transpose()

def main():
    args = parser.parse_args()
    netName = args.network_name
    net = load_network(f"{netpath}/{netName}", None)
    try:
        with open(args.input_file, "r") as f:
            if (args.input_json):
                input_data = jsonToData(f)
            else:
                input_data = textToData(f.read())
    except FileNotFoundError:
        print(f"File {args.input_file} not found")
        return 1
    
    display(input_data)
    result = net.feedforward(input_data)
    for i, x in enumerate(result):
        print(f"{i} : {float(x.item(0)):3.2f}")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--network-name", action="store", default="digitreader")
    parser.add_argument("-j", "--input-json", action="store_true");
    parser.add_argument("input_file")

    main()

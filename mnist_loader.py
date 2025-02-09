import numpy as np
import _pickle
import gzip

class Labels:
    def __init__(self, filename):
        tmp = filename.split("-")
        self.filename = "-".join(tmp[:2])
        self.extension = "-".join(tmp[2:])
        with open(f"{datapath}/{self.get_filename()}", "rb") as f:
            self.bin = f.read()
        self.magic_number = int.from_bytes(self.bin[0:4], 'big')
        self.number_of_items = int.from_bytes(self.bin[4:8], 'big')
        self.offset = 8

    def get_label(self, i):
        if self.number_of_items <= i:
            return -1
        return self.bin[self.offset + i]

    def get_filename(self):
        return '.'.join([self.filename, self.extension])


class Images:
    def __init__(self, filename):
        tmp = filename.split("-")
        self.filename = "-".join(tmp[:2])
        self.extension = "-".join(tmp[2:])

        # self.extract(datapath)
        with open(f"{datapath}/{self.get_filename()}", "rb") as f:
            self.bin = f.read()
        self.magic_number = int.from_bytes(self.bin[0:4], 'big')
        self.number_of_items = int.from_bytes(self.bin[4:8], 'big')
        self.number_of_rows = int.from_bytes(self.bin[8:12], 'big')
        self.number_of_cols = int.from_bytes(self.bin[12:16], 'big')
        self.offset = 16

    def get_image(self, i):
        if self.number_of_items <= i:
            return -1
        image = []
        for r in range(self.number_of_rows):
            for c in range(self.number_of_cols):
                image.append(self.bin[self.offset + c + (r * self.number_of_rows)
                                      + (self.number_of_cols * self.number_of_rows * i)])
        return np.asmatrix(image).transpose()

    def get_filename(self):
        return '.'.join([self.filename, self.extension])

    def get_resolution(self):
        return self.number_of_rows * self.number_of_cols


datapath = './dataset'
datasets = {
    'train-images': Images('train-images-idx3-ubyte'),
    'train-labels': Labels('train-labels-idx1-ubyte'),
    'test-images': Images('t10k-images-idx3-ubyte'),
    'test-labels': Labels('t10k-labels-idx1-ubyte')
}
datasets_url = 'http://yann.lecun.com/exdb/mnist/'

def pickle_data():
    # TODO: right now all datasets are stored locally, add downloading from a mirror site
    training_data = [(datasets['train-images'].get_image(x), vectorize_result(datasets['train-labels'].get_label(x))) for x in range(datasets['train-images'].number_of_items)]
    test_data = [(datasets['test-images'].get_image(x), datasets['test-labels'].get_label(x)) for x in range(datasets['test-images'].number_of_items)]
    fp = gzip.open(f"{datapath}/mnist_dataset.pkl.gz", "wb")
    _pickle.dump((training_data, test_data), fp)
    fp.close()

def unpickle_data():
    try:
        fp = gzip.open(f"{datapath}/mnist_dataset.pkl.gz", "rb")
        training_data, test_data = _pickle.load(fp)
        fp.close()
    except FileNotFoundError:
        pickle_data()
        fp = gzip.open(f"{datapath}/mnist_dataset.pkl.gz", "rb")
        training_data, test_data = _pickle.load(fp)
        fp.close()
    return (training_data, test_data)

def load_data():
    training_data, test_data = unpickle_data()
    return (training_data, test_data)

def vectorize_result(j):
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e

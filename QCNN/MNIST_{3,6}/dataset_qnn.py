import cirq
import tensorflow as tf
import tensorflow_quantum as tfq
from tensorflow.keras import datasets
import numpy as np
import collections
import encode

def generate_data(data_type, cl_en, qt_en, qubits):
    if data_type == "CIFAR10" :
        return generate_CIFAR10_data(cl_en, qt_en, qubits)
    if data_type == "MNIST" :
        return generate_mnist_data(cl_en, qt_en, qubits)
    if data_type == "Fashion-MNIST":
        return generate_fashion_mnist_data(cl_en, qt_en, qubits)

def generate_mnist_data(cl_en, qt_en, qubits):
    # 加载MNIST数据集
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    # Rescale the images from [0,255] to the [0.0,1.0] range.
    x_train, x_test = x_train[..., np.newaxis]/255.0, x_test[..., np.newaxis]/255.0

    print("Number of original training examples:", len(x_train))
    print("Number of original test examples:", len(x_test))
    print()
    
    def filter_36(x, y):
        keep = (y == 3) | (y == 6)
        x, y = x[keep], y[keep]
        y = y == 3
        return x,y
    
    x_train, y_train = filter_36(x_train, y_train)
    x_test, y_test = filter_36(x_test, y_test)

    print("Number of filtered(3,6) training examples:", len(x_train))
    print("Number of filtered(3,6) test examples:", len(x_test))
    print()

    # 经典编码
    x_train_small,x_test_small = encode.classical_encode(x_train, x_test, (4,4), cl_en)

    def remove_contradicting(xs, ys):
        mapping = collections.defaultdict(set)
        orig_x = {}
        # Determine the set of labels for each unique image:
        for x,y in zip(xs,ys):
            orig_x[tuple(x.flatten())] = x
            mapping[tuple(x.flatten())].add(y)

        new_x = []
        new_y = []
        for flatten_x in mapping:
            x = orig_x[flatten_x]
            labels = mapping[flatten_x]
            if len(labels) == 1:
                new_x.append(x)
                new_y.append(next(iter(labels)))
            else:
              # Throw out images that match more than one label.
                pass

        num_uniq_3 = sum(1 for value in mapping.values() if len(value) == 1 and True in value)
        num_uniq_6 = sum(1 for value in mapping.values() if len(value) == 1 and False in value)
        num_uniq_both = sum(1 for value in mapping.values() if len(value) == 2)
        print("Number of unique images:", len(mapping.values()))
        print("Number of unique 3s: ", num_uniq_3)
        print("Number of unique 6s: ", num_uniq_6)
        print("Number of unique contradicting labels (both 3 and 6): ", num_uniq_both)
        print("Initial number of images: ", len(xs))
        print("Remaining non-contradicting unique images: ", len(new_x))
        print()
        return np.array(new_x), np.array(new_y)
    
    print("Train dataset: remove contradict due to Downscale")
    x_train_nocon, y_train_nocon = remove_contradicting(x_train_small, y_train)
    print("Test dataset: remove contradict due to Downscale")
    x_test_nocon, y_test_nocon = remove_contradicting(x_test_small, y_test)
    
    # 量子编码
    x_train_circ ,x_test_circ= encode.quantum_encode(qubits,x_train_nocon,x_test_nocon,qt_en)

    # 转换为张量
    x_train_tfcirc = tfq.convert_to_tensor(x_train_circ)
    x_test_tfcirc = tfq.convert_to_tensor(x_test_circ)

    # 将标签[0,1]映射为[-1,1]以适应输出的期望值
    y_train_hinge = 2.0*y_train_nocon-1.0
    y_test_hinge = 2.0*y_test_nocon-1.0

    return x_train_tfcirc, y_train_hinge, x_test_tfcirc, y_test_hinge


def generate_fashion_mnist_data(cl_en, qt_en, qubits):
    # 加载Fashion-MNIST数据集
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()

    # Rescale the images from [0,255] to the [0.0,1.0] range.
    x_train, x_test = x_train[..., np.newaxis]/255.0, x_test[..., np.newaxis]/255.0

    print("Number of original training examples:", len(x_train))
    print("Number of original test examples:", len(x_test))

    def filter_03(x, y):
        keep = (y == 0) | (y == 3)
        x, y = x[keep], y[keep]
        y = y == 0
        return x,y
    
    x_train, y_train = filter_03(x_train, y_train)
    x_test, y_test = filter_03(x_test, y_test)
    
    print("Number of filtered training examples:", len(x_train))
    print("Number of filtered test examples:", len(x_test))

    # 经典编码
    x_train_small,x_test_small = encode.classical_encode(x_train, x_test, (4,4), cl_en)

    def remove_contradicting(xs, ys):
        mapping = collections.defaultdict(set)
        orig_x = {}
        # Determine the set of labels for each unique image:
        for x,y in zip(xs,ys):
            orig_x[tuple(x.flatten())] = x
            mapping[tuple(x.flatten())].add(y)

        new_x = []
        new_y = []
        for flatten_x in mapping:
            x = orig_x[flatten_x]
            labels = mapping[flatten_x]
            if len(labels) == 1:
                new_x.append(x)
                new_y.append(next(iter(labels)))
            else:
            # Throw out images that match more than one label.
                pass

        num_uniq_3 = sum(1 for value in mapping.values() if len(value) == 1 and True in value)
        num_uniq_6 = sum(1 for value in mapping.values() if len(value) == 1 and False in value)
        num_uniq_both = sum(1 for value in mapping.values() if len(value) == 2)
        print("Number of unique images:", len(mapping.values()))
        print("Number of unique 0s: ", num_uniq_3)
        print("Number of unique 3s: ", num_uniq_6)
        print("Number of unique contradicting labels (both 0 and 3): ", num_uniq_both)
        print("Initial number of images: ", len(xs))
        print("Remaining non-contradicting unique images: ", len(new_x))
        print()
        return np.array(new_x), np.array(new_y)
    
    print("Train dataset: remove contradict due to Downscale")
    x_train_nocon, y_train_nocon = remove_contradicting(x_train_small, y_train)
    print("Test dataset: remove contradict due to Downscale")
    x_test_nocon, y_test_nocon = remove_contradicting(x_test_small, y_test)

    # 量子编码
    x_train_circ ,x_test_circ= encode.quantum_encode(qubits,x_train_nocon,x_test_nocon,qt_en)

    # 转换为张量
    x_train_tfcirc = tfq.convert_to_tensor(x_train_circ)
    x_test_tfcirc = tfq.convert_to_tensor(x_test_circ)

    # 将标签[0,1]映射为[-1,1]以适应输出的期望值
    y_train_hinge = 2.0*y_train_nocon-1.0
    y_test_hinge = 2.0*y_test_nocon-1.0

    return x_train_tfcirc, y_train_hinge, x_test_tfcirc, y_test_hinge

def generate_CIFAR10_data(cl_en, qt_en, qubits):
    (x_train, y_train), (x_test, y_test) = datasets.cifar10.load_data()

    x_train = np.dot(x_train[..., :3], [0.299, 0.587, 0.114])
    x_test = np.dot(x_test[..., :3], [0.299, 0.587, 0.114])

    x_train, x_test = x_train[..., np.newaxis]/255.0, x_test[..., np.newaxis]/255.0

    y_train = y_train.flatten()
    y_test = y_test.flatten()

    print("Number of original training examples:", len(x_train))
    print("Number of original test examples:", len(x_test))

    def filter_35(x, y):
        keep = (y == 3) | (y == 5)
        x, y = x[keep], y[keep]
        y = y == 3
        return x,y
    
    x_train, y_train = filter_35(x_train, y_train)
    x_test, y_test = filter_35(x_test, y_test)

    print("Number of filtered training examples:", len(x_train))
    print("Number of filtered test examples:", len(x_test))
    print()

    # 经典编码
    x_train_small,x_test_small = encode.classical_encode(x_train, x_test, (4,4), cl_en)

    def remove_contradicting(xs, ys):
        mapping = collections.defaultdict(set)
        orig_x = {}
        # Determine the set of labels for each unique image:
        for x,y in zip(xs,ys):
            orig_x[tuple(x.flatten())] = x
            mapping[tuple(x.flatten())].add(y)

        new_x = []
        new_y = []
        for flatten_x in mapping:
            x = orig_x[flatten_x]
            labels = mapping[flatten_x]
            if len(labels) == 1:
                new_x.append(x)
                new_y.append(next(iter(labels)))
            else:
            # Throw out images that match more than one label.
                pass

        num_uniq_3 = sum(1 for value in mapping.values() if len(value) == 1 and True in value)
        num_uniq_6 = sum(1 for value in mapping.values() if len(value) == 1 and False in value)
        num_uniq_both = sum(1 for value in mapping.values() if len(value) == 2)
        print("Number of unique images:", len(mapping.values()))
        print("Number of unique 0s: ", num_uniq_3)
        print("Number of unique 3s: ", num_uniq_6)
        print("Number of unique contradicting labels (both 0 and 3): ", num_uniq_both)
        print("Initial number of images: ", len(xs))
        print("Remaining non-contradicting unique images: ", len(new_x))
        print()
        return np.array(new_x), np.array(new_y)
    
    print("Train dataset: remove contradict due to Downscale")
    x_train_nocon, y_train_nocon = remove_contradicting(x_train_small, y_train)
    print("Test dataset: remove contradict due to Downscale")
    x_test_nocon, y_test_nocon = remove_contradicting(x_test_small, y_test)
    
    # 量子编码
    x_train_circ ,x_test_circ= encode.quantum_encode(qubits,x_train_nocon,x_test_nocon,qt_en)

    # 转换为张量
    x_train_tfcirc = tfq.convert_to_tensor(x_train_circ)
    x_test_tfcirc = tfq.convert_to_tensor(x_test_circ)

    # 将标签[0,1]映射为[-1,1]以适应输出的期望值
    y_train_hinge = 2.0*y_train_nocon-1.0
    y_test_hinge = 2.0*y_test_nocon-1.0

    return x_train_tfcirc, y_train_hinge, x_test_tfcirc, y_test_hinge

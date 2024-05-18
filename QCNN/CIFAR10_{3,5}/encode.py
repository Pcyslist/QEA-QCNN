import tensorflow as tf
import numpy as np
import cirq

def classical_encode(x_train, x_test, size, method):
    if method == "BILINEAR":
        x_train_small = tf.image.resize(x_train, size, method = tf.image.ResizeMethod.BILINEAR).numpy()
        x_test_small = tf.image.resize(x_test, size, method = tf.image.ResizeMethod.BILINEAR).numpy()
    if method == "AREA":
        x_train_small = tf.image.resize(x_train, size, method = tf.image.ResizeMethod.AREA).numpy()
        x_test_small = tf.image.resize(x_test, size, method = tf.image.ResizeMethod.AREA).numpy()
    if method == "MINE":
        x_train_small = my_classical(x_train, size)
        x_test_small = my_classical(x_test, size)
    return x_train_small, x_test_small

def my_classical(images, size):
    pass





def quantum_encode(qubits, x_train, x_test, method):
    if method == "BASE_STATE":
        THRESHOLD = 0.5
        x_train_bin = np.array(x_train > THRESHOLD, dtype=np.float32)
        x_test_bin = np.array(x_test > THRESHOLD, dtype=np.float32)
        x_train_circ = [ base_state(x,qubits) for x in x_train_bin ]
        x_test_circ = [ base_state(x,qubits) for x in x_test_bin ]
    if method == "ANGLE":
        x_train_circ = [ angle(x,qubits) for x in x_train ]
        x_test_circ = [ angle(x,qubits) for x in x_test ]
    if method == "MINE":
        x_train_circ = [ my_quantum(x,qubits) for x in x_train ]
        x_test_circ = [ my_quantum(x,qubits) for x in x_test ]
    return x_train_circ, x_test_circ


def base_state(image,qubits):
    """Encode truncated classical image into quantum datapoint."""
    values = np.ndarray.flatten(image)
    circuit = cirq.Circuit()
    for i, value in enumerate(values):
        if value:
            circuit.append(cirq.X(qubits[i]))
    return circuit

def angle(image,qubits):
    """Encode truncated classical image into quantum datapoint."""
    values = np.ndarray.flatten(image)
    circuit = cirq.Circuit()
    for i, value in enumerate(values):
        circuit.append(cirq.rx(value * np.pi)(qubits[i])) # 此处将像素值从[0,1]映射到[0,pi]，以适应旋转角的pi周期
    return circuit

def my_quantum(image, qubits):
    pass

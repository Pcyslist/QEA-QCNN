import tensorflow as tf
import tensorflow_quantum as tfq

import cirq
import sympy
import numpy as np
import collections
import dataset
# 可视化工具
import matplotlib.pyplot as plt
from cirq.contrib.svg import SVGCircuit

# 持久化
import pickle

# 全局变量
# 用到的量子比特
cluster_state_bits = cirq.GridQubit.rect(4, 4)

train_excitations, train_labels, test_excitations, test_labels = dataset.generate_mnist_data(cluster_state_bits)

# 3个参数（可改为Rx/Ry/Rz）
def one_qubit_unitary(bit, symbols):
    """Make a Cirq circuit enacting a rotation of the bloch sphere about the X,
    Y and Z axis, that depends on the values in `symbols`.
    """
    return cirq.Circuit(
        cirq.X(bit)**symbols[0],
        cirq.Y(bit)**symbols[1],
        cirq.Z(bit)**symbols[2])

# 15个参数(12个单比特门参数、3个双比特门参数)(可改为Rx/Ry/Rz,已经通过one_qubit_unitary修改了)
def two_qubit_conv(bits, symbols):
    """即U"""
    circuit = cirq.Circuit()
    circuit += one_qubit_unitary(bits[0], symbols[0:3])
    circuit += one_qubit_unitary(bits[1], symbols[3:6])
    # 不同写法，意思一样
    circuit += [cirq.ZZ(bits[0], bits[1])**symbols[6]]
    circuit += [cirq.YY(bits[1], bits[0])**symbols[7]]
    circuit += [cirq.XX(*bits)**symbols[8]]
    
    circuit += one_qubit_unitary(bits[0], symbols[9:12])
    circuit += one_qubit_unitary(bits[1], symbols[12:])
    return circuit

def two_qubit_pool(source_qubit, sink_qubit, symbols):
    """含有6个参数（2个单比特门酉矩阵）和CNOT门组成 一个V"""
    pool_circuit = cirq.Circuit()
    
    sink_basis_selector = one_qubit_unitary(sink_qubit, symbols[0:3])
    source_basis_selector = one_qubit_unitary(source_qubit, symbols[3:6])
    
    pool_circuit.append(sink_basis_selector)
    pool_circuit.append(source_basis_selector)
    
    pool_circuit.append(cirq.CNOT(control=source_qubit, target=sink_qubit))
    
    pool_circuit.append(sink_basis_selector**-1)
    return pool_circuit

# 一层卷积由2层U组成，每个U共享训练参数
def quantum_conv_circuit(bits, symbols):
    """Quantum Convolution Layer following the above diagram.
    Return a Cirq circuit with the cascade of `two_qubit_unitary` applied
    to all pairs of qubits in `bits` as in the diagram above.
    利用U进行堆叠2层，每个U（two_qubit_unitary）共享相同的15个参数；
    卷积不减少量子比特
    """
    circuit = cirq.Circuit()
    for first, second in zip(bits[0::2], bits[1::2]):
        circuit += two_qubit_conv([first, second], symbols)
    for first, second in zip(bits[1::2], bits[2::2] + [bits[0]]):
        circuit += two_qubit_conv([first, second], symbols)
    return circuit
# 一层池化由1层V组成，每个V共享训练参数
def quantum_pool_circuit(source_bits, sink_bits, symbols):
    """A layer that specifies a quantum pooling operation.
    A Quantum pool tries to learn to pool the relevant information from two
    qubits onto 1.
    利用V进行堆叠1层，每个V（two_qubit_pool）共享相同的6个参数；
    池化减少量子比特数
    """
    circuit = cirq.Circuit()
    for source, sink in zip(source_bits, sink_bits):
        circuit += two_qubit_pool(source, sink, symbols)
    return circuit

def create_model_circuit_2layers(qubits):
    """Create sequence of alternating convolution and pooling operators 
    which gradually shrink over time."""
    model_circuit = cirq.Circuit()
    symbols_qconv = sympy.symbols('qconv0:60') # 4层卷积
    symbols_qpool = sympy.symbols('qpool0:24') # 4层池化
    # Cirq uses sympy.Symbols to map learnable variables. TensorFlow Quantum
    # scans incoming circuits and replaces these with TensorFlow variables.
    # 第1层卷积层+池化层
    model_circuit += quantum_conv_circuit(qubits[0::1], symbols_qconv[0:15])
    model_circuit += quantum_pool_circuit(qubits[0::2], qubits[1::2],symbols_qpool[0:6])
    # 第2层卷积层+池化层
    model_circuit += quantum_conv_circuit(qubits[1::2], symbols_qconv[15:30])
    model_circuit += quantum_pool_circuit(qubits[1::4], qubits[3::4],symbols_qpool[6:12])
    # 第3层卷积层+池化层
    # model_circuit += quantum_conv_circuit(qubits[3::4], symbols_qconv[30:45])
    # model_circuit += quantum_pool_circuit(qubits[3::8], qubits[7::8],symbols_qpool[12:18])
    # # 第4层卷积层+池化层
    # model_circuit += quantum_conv_circuit(qubits[7::8], symbols_qconv[45:60])
    # model_circuit += quantum_pool_circuit(qubits[7::16], qubits[15::16],symbols_qpool[18:24])
    return model_circuit

def train_qcnn():
    qcnn_circuit = create_model_circuit_2layers(cluster_state_bits)
    # 将Z观测于最后一个量子比特，用于读出[-1,1]的期望值
    readout_operators = [cirq.Z(bit) for bit in cluster_state_bits[3::4]]
    # 保存qcnn线路结构
    with open("qcnn_circuit.svg",'w') as f:
        f.write(SVGCircuit(qcnn_circuit)._repr_svg_())
    with open("qcnn_circuit.pkl", "wb")as f:
        pickle.dump(qcnn_circuit,f)
    with open("qcnn_obs.pkl","wb") as f:
        pickle.dump(readout_operators, f)
    
    # 数据集输入层
    excitation_input = tf.keras.Input(shape=(), dtype=tf.dtypes.string)
    # 将QCNN线路包装为PQC层，并在前面与输入数据点线路进行衔接，得到 quantum_model_output
    quantum_model_pqc = tfq.layers.PQC(qcnn_circuit,readout_operators)
    # 定义qcnn模型
    qcnn_model = tf.keras.Sequential([
    # The input is the data-circuit, encoded as a tf.string
    excitation_input,
    # The PQC layer returns the expected value of the readout gate, range [-1,1].
    quantum_model_pqc,
    tf.keras.layers.Dense(8),
    tf.keras.layers.Dense(1)
    ])

    # 自定义准确率以适应[-1,1]的期望输出范围.
    @tf.function
    def hinge_accuracy(y_true, y_pred):
        y_true = tf.squeeze(y_true)
        y_pred = tf.map_fn(lambda x: 1.0 if x >= 0 else -1.0, y_pred)
        return tf.keras.backend.mean(tf.keras.backend.equal(y_true, y_pred))
    
    # 模型编译设置损失函数、优化器
    qcnn_model.compile(optimizer=tf.keras.optimizers.Adam(),
                   loss=tf.keras.losses.mse,
                   metrics=[hinge_accuracy])
    # 用于查看QCNN结构和参数量
    qcnn_model.summary()
    # 超参数设置
    batch_size = 32
    epochs = 40

    # 开始训练
    history = qcnn_model.fit(x=train_excitations,
                            y=train_labels,
                            batch_size=batch_size,
                            epochs=epochs,
                            verbose=1,
                            validation_data=(test_excitations, test_labels))
    # 保存模型权重
    qcnn_model.save_weights('qcnn_model_weights.h5')
    # 保存准确率
    acc = qcnn_model.evaluate(test_excitations, test_labels)
    with open("acc.txt","w") as f:
        f.write(str(acc))

    # 绘图保存训练曲线
    plt.plot(history.history['hinge_accuracy'], label='Training')
    plt.plot(history.history['val_hinge_accuracy'], label='Validation')
    plt.title('MNIST HQCNN Training')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig('training_procedure_mnist_qcnn_16qubit_resize_cluster_origin_hinge_base.svg', format='svg')

def reload_model():
    # 用到的量子比特
    # cluster_state_bits = cirq.GridQubit.rect(4, 4)
    # 生成数据集（如果在另一个文件中打开，则需要重新加载数据集）
    # train_excitations, train_labels, test_excitations, test_labels = generate_data(cluster_state_bits)

    with open('qcnn_circuit.pkl', 'rb') as file:
        qcnn_circuit = pickle.load(file)
    with open('qcnn_obs.pkl', 'rb') as file:
        readout_operators = pickle.load(file)
    
    # 数据集输入层
    excitation_input = tf.keras.Input(shape=(), dtype=tf.dtypes.string)
    # 将QCNN线路包装为PQC层，并在前面与输入数据点线路进行衔接，得到 quantum_model_output
    quantum_model_pqc = tfq.layers.PQC(qcnn_circuit,readout_operators)
    # 定义qcnn模型
    qcnn_model = tf.keras.Sequential([
    # The input is the data-circuit, encoded as a tf.string
    excitation_input,
    # The PQC layer returns the expected value of the readout gate, range [-1,1].
    quantum_model_pqc,
    tf.keras.layers.Dense(8),
    tf.keras.layers.Dense(1)
    ])

    # 加载模型权重
    qcnn_model.load_weights("qcnn_model_weights.h5")
    qcnn_model.get_weights()
    # 自定义准确率以适应[-1,1]的期望输出范围.
    def hinge_accuracy(y_true, y_pred):
        y_true = tf.squeeze(y_true) > 0.0
        y_pred = tf.squeeze(y_pred) > 0.0
        result = tf.cast(y_true == y_pred, tf.float32)
        return tf.reduce_mean(result)
    
    # 模型编译设置损失函数、优化器
    qcnn_model.compile(optimizer=tf.keras.optimizers.Adam(),
                   loss=tf.keras.losses.mse,
                   metrics=[hinge_accuracy])
    # 用于查看QCNN结构和参数量
    qcnn_model.summary()
    # 再次验证模型的准确率
    qcnn_model.evaluate(test_excitations, test_labels)

# train_qcnn()
reload_model()

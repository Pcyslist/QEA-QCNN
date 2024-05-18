import tensorflow as tf
import tensorflow_quantum as tfq

import cirq
import sympy
import numpy as np
import collections
import dataset_qnn
# 可视化工具
import matplotlib.pyplot as plt
from cirq.contrib.svg import SVGCircuit

# 持久化
import pickle

# 全局变量
# 用到的量子比特
data_type = "FASHION-MNIST"
cl_en = "BILINEAR"
qt_en = "BASE_STATE"
cluster_state_bits = cirq.GridQubit.rect(4, 4)

# 生成数据集
train_excitations, train_labels, test_excitations, test_labels = dataset_qnn.generate_data(data_type,cl_en,qt_en,cluster_state_bits)

# qnn层构建器
class CircuitLayerBuilder():
    def __init__(self, data_qubits, readout):
        self.data_qubits = data_qubits
        self.readout = readout

    def add_layer(self, circuit, gate, prefix):
        for i, qubit in enumerate(self.data_qubits):
            symbol = sympy.Symbol(prefix + '-' + str(i))
            circuit.append(gate(qubit, self.readout)**symbol)

def create_model_circuit(data_qubits):
    """Create a QNN model circuit and readout operation to go along with it."""
#     data_qubits = cirq.GridQubit.rect(4, 4)  # a 4x4 grid.
    readout = cirq.GridQubit(-1, -1)         # a single qubit at [-1,-1]
    circuit = cirq.Circuit()

    # Prepare the readout qubit.
    circuit.append(cirq.X(readout))
    circuit.append(cirq.H(readout))

    builder = CircuitLayerBuilder(
        data_qubits = data_qubits,
        readout=readout)

    # Then add 2 layers (experiment by adding more).
    builder.add_layer(circuit, cirq.XX, "xx1")
    builder.add_layer(circuit, cirq.ZZ, "zz1")

    # Finally, prepare the readout qubit.
    circuit.append(cirq.H(readout))

    return circuit, cirq.Z(readout)
def train_qnn():
    # 创建qnn以及获得线路的Z测量算子
    qnn_circuit, readout_operators = create_model_circuit(cluster_state_bits)
    # 保存qnn线路结构
    with open("qnn_circuit.svg",'w') as f:
        f.write(SVGCircuit(qnn_circuit)._repr_svg_())
    with open("qnn_circuit.pkl", "wb")as f:
        pickle.dump(qnn_circuit,f)
    with open("qnn_obs.pkl","wb") as f:
        pickle.dump(readout_operators, f)
        
    # 将qnn线路用pqc包装为model
    qnn_pqc = tfq.layers.PQC(qnn_circuit, readout_operators)
    qnn_model = tf.keras.Sequential([
        # The input is the data-circuit, encoded as a tf.string
        tf.keras.layers.Input(shape=(), dtype=tf.string),
        # The PQC layer returns the expected value of the readout gate, range [-1,1].
        qnn_pqc])
    # 定义评价指标Acc
    def hinge_accuracy(y_true, y_pred):
        y_true = tf.squeeze(y_true) > 0.0
        y_pred = tf.squeeze(y_pred) > 0.0
        result = tf.cast(y_true == y_pred, tf.float32)
        return tf.reduce_mean(result)
    # 编译模型的损失函数和优化器，并指定评价指标
    qnn_model.compile(
        loss=tf.keras.losses.Hinge(),
        optimizer=tf.keras.optimizers.SGD(),
        metrics=[hinge_accuracy])
    # 总结模型结构和参数量
    qnn_model.summary()
    # 超参数
    epochs = 40
    batch_size = 32
    # 开始训练
    history = qnn_model.fit(x=train_excitations,
                        y=train_labels,
                        batch_size=batch_size,
                        epochs=epochs,
                        verbose=1,
                        validation_data=(test_excitations, test_labels))
    # 保存模型权重
    qnn_model.save_weights('qnn_model_weights.h5')
    # 保存准确率
    acc = qnn_model.evaluate(test_excitations, test_labels)
    with open("acc.txt","w") as f:
        f.write(str(acc))
    # 绘图保存训练曲线
    plt.plot(history.history['hinge_accuracy'], label='Training')
    plt.plot(history.history['val_hinge_accuracy'], label='Validation')
    plt.title('QNN {} Training'.format(data_type))
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig('training_qnn_{}.svg'.format(data_type), format='svg')

def reload_model():
    with open('qnn_circuit.pkl', 'rb') as file:
        qnn_circuit = pickle.load(file)
    with open('qnn_obs.pkl', 'rb') as file:
        readout_operators = pickle.load(file)
    # 将qnn线路用pqc包装为model
    qnn_pqc = tfq.layers.PQC(qnn_circuit, readout_operators)
    qnn_model = tf.keras.Sequential([
        # The input is the data-circuit, encoded as a tf.string
        tf.keras.layers.Input(shape=(), dtype=tf.string),
        # The PQC layer returns the expected value of the readout gate, range [-1,1].
        qnn_pqc])
    # 加载模型权重
    qnn_model.load_weights("qnn_model_weights.h5")
    # 定义评价指标Acc
    def hinge_accuracy(y_true, y_pred):
        y_true = tf.squeeze(y_true) > 0.0
        y_pred = tf.squeeze(y_pred) > 0.0
        result = tf.cast(y_true == y_pred, tf.float32)
        return tf.reduce_mean(result)
    # 编译模型的损失函数和优化器，并指定评价指标
    qnn_model.compile(
        loss=tf.keras.losses.Hinge(),
        optimizer=tf.keras.optimizers.Adam(),
        metrics=[hinge_accuracy])
    # 再次验证模型的准确率
    qnn_model.evaluate(test_excitations, test_labels)

train_qnn()
reload_model()

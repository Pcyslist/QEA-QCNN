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

train_excitations, train_labels, test_excitations, test_labels = dataset.generate_CIFAR10_data(cluster_state_bits)

def create_model_circuit_2layers():
    with open('best_qcnn_circuit.pkl', 'rb') as file:
        qcnn_circuit = pickle.load(file)
    with open('best_qcnn_obs.pkl', 'rb') as file:
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
    ])
    # 加载模型权重
    qcnn_model.load_weights("best_qcnn_model_weights.h5")
    print("weights:")
    print(qcnn_model.get_weights())
    print("weights dims:")
    print(qcnn_model.get_weights()[0].shape)
    print("PQC params:")
    print(quantum_model_pqc.symbol_values())

    def hinge_accuracy(y_true, y_pred):
        y_true = tf.squeeze(y_true) > 0.0
        y_pred = tf.squeeze(y_pred) > 0.0
        result = tf.cast(y_true == y_pred, tf.float32)
        return tf.reduce_mean(result)
    # 模型编译设置损失函数、优化器
    qcnn_model.compile(optimizer=tf.keras.optimizers.Adam(),
                   loss=tf.keras.losses.mse,
                   metrics=[hinge_accuracy])
    qcnn_model.summary()
    # 再次验证模型的准确率
    qcnn_model.evaluate(test_excitations, test_labels)

    return qcnn_circuit[:168]

def train_qcnn():
    qcnn_circuit = create_model_circuit_2layers()
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
    plt.title('CIFAR10 QEA-HQCNN Training')
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
    qcnn_model.summary()
    # 再次验证模型的准确率
    qcnn_model.evaluate(test_excitations, test_labels)

#train_qcnn()
reload_model()

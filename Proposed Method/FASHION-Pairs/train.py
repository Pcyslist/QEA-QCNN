import math
import pickle
import random
import cirq
import sympy
import dataset_qnn
import numpy as np
import collections
import tensorflow as tf
import tensorflow_quantum as tfq
import matplotlib.pyplot as plt
from itertools import combinations
from cirq.contrib.svg import SVGCircuit

# 用到的量子比特
qubits = cirq.GridQubit.rect(4, 4)
data_type = "FASHION-MNIST"
cl_en = "AREA"
qt_en = "ANGLE"


def train(i,j):
    with open('best_qcnn_circuit.pkl', 'rb') as file:
        qcnn_circuit = pickle.load(file)
    with open('best_qcnn_obs.pkl', 'rb') as file:
        readout_operators = pickle.load(file)
    with open("best_qcnn_circuit.svg",'w') as f:
        f.write(SVGCircuit(qcnn_circuit)._repr_svg_())
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
    # 再次验证模型的准确率
    qcnn_model.summary()
    # 超参数设置
    batch_size = 32
    epochs = 20
    # 开始训练
    history = qcnn_model.fit(x=train_excitations,
                            y=train_labels,
                            batch_size=batch_size,
                            epochs=epochs,
                            verbose=1,
                            validation_data=(test_excitations, test_labels))
    # 保存模型权重
    qcnn_model.save_weights('qcnn_model_weights({},{}).h5'.format(i,j))
    
    acc = qcnn_model.evaluate(test_excitations, test_labels)
    with open("acc.txt","a") as f:
        f.write("(" + str(i) + "," + str(j) + ") :" + str(acc) + "\n")
    # 绘图保存训练曲线
    plt.figure()
    plt.plot(history.history['hinge_accuracy'], label='Training')
    plt.plot(history.history['val_hinge_accuracy'], label='Validation')
    plt.title('QEA-QCNN {} Training'.format(data_type))
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig('training_qCnn_{}({},{}).svg'.format(data_type,i,j), format='svg')
for i in range(10):
    for j in range(i+1, 10):
        dataset_qnn.class0 = i
        dataset_qnn.class1 = j
        # 生成数据集
        train_excitations, train_labels, test_excitations, test_labels = dataset_qnn.generate_data(data_type,cl_en,qt_en,qubits)
        train(i,j)

import math
import pickle
import random
import cirq
import sympy
import numpy as np
import collections
import tensorflow as tf
import tensorflow_quantum as tfq
from itertools import combinations
from algorithm_params import algorithm_params as alp
from cirq.contrib.svg import SVGCircuit

# 用到的量子比特
qubits = cirq.GridQubit.rect(4, 4)

def generate_data(qubits):
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
    
    x_train_small = tf.image.resize(x_train, (4,4),method = tf.image.ResizeMethod.AREA).numpy()
    x_test_small = tf.image.resize(x_test, (4,4),method = tf.image.ResizeMethod.AREA).numpy()
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
    
    # THRESHOLD = 0.5(采用Angle编码，不进行二值化，此处用于基态编码)
    # x_train_bin = np.array(x_train_nocon > THRESHOLD, dtype=np.float32)
    # x_test_bin = np.array(x_test_nocon > THRESHOLD, dtype=np.float32)
    def convert_to_circuit(image):
        """Encode truncated classical image into quantum datapoint."""
        values = np.ndarray.flatten(image)
        # qubits = cirq.GridQubit.rect(4, 4) # 此处的qubits由外面的函数generate_data提供
        circuit = cirq.Circuit()
        for i, value in enumerate(values):
            # 如需基态编码则放开以下注释掉的代码
            # if value:
                # circuit.append(cirq.X(qubits[i]))
            circuit.append(cirq.rx(value * np.pi)(qubits[i])) # 此处将像素值从[0,1]映射到[0,pi]，以适应旋转角的pi周期
        return circuit
    # 用于基态编码
    # x_train_circ = [convert_to_circuit(x) for x in x_train_bin]
    # x_test_circ = [convert_to_circuit(x) for x in x_test_bin]
    
    # 用于角度编码
    x_train_circ = [convert_to_circuit(x) for x in x_train_nocon]
    x_test_circ = [convert_to_circuit(x) for x in x_test_nocon]
    # 转换为张量
    x_train_tfcirc = tfq.convert_to_tensor(x_train_circ)
    x_test_tfcirc = tfq.convert_to_tensor(x_test_circ)
    
    # 将标签[0,1]映射为[-1,1]以适应输出的期望值
    y_train_hinge = 2.0*y_train_nocon-1.0
    y_test_hinge = 2.0*y_test_nocon-1.0
    
    return x_train_tfcirc, y_train_hinge, x_test_tfcirc, y_test_hinge

# 生成数据集
train_excitations, train_labels, test_excitations, test_labels = generate_data(qubits)

def initialize_population(pop_size, dims, code_length):
    pop = []
    for _ in range(pop_size):
        individual = []
        for _ in range(dims * code_length):
            individual.append(1 if random.random() > 0.5 else 0)
        pop.append(individual)
    return pop

def one_qubit_unitary(genes, bit, symbol):
    """Make a Cirq circuit enacting a rotation of the bloch sphere about the X,
    Y and Z axis, that depends on the values in `symbols`.
    """
    gate_dict = {"[0, 0]":cirq.X,"[0, 1]":cirq.Z,"[1, 0]":cirq.Y,"[1, 1]":cirq.I}
    gate = gate_dict[str(genes)]
    return cirq.Circuit(gate(bit)**symbol)

def two_qubit_unitary(genes, bits, symbol):
    """即U"""
    gate_dict = {"[0, 0]":cirq.XX,"[0, 1]":cirq.ZZ,"[1, 0]":cirq.YY,"[1, 1]":cirq.I}
    gate = gate_dict[str(genes)]
    circuit = cirq.Circuit()
    if gate != cirq.I:
        circuit += [gate(*bits)**symbol]
        return circuit
    else:
        circuit += [gate(bits[0])**symbol]
        circuit += [gate(bits[1])**symbol]
        return circuit

def two_qubit_conv(genes, bits, symbols):
    # 5 个参数
    conv_circuit = cirq.Circuit()
    conv_circuit += one_qubit_unitary(genes[0:2],bits[0],symbols[0])
    conv_circuit += one_qubit_unitary(genes[2:4],bits[1],symbols[1])
    conv_circuit += two_qubit_unitary(genes[4:6],bits, symbols[2])
    conv_circuit += one_qubit_unitary(genes[6:8],bits[0],symbols[3])
    conv_circuit += one_qubit_unitary(genes[8:10],bits[1],symbols[4])
    return conv_circuit

def two_qubit_pool(genes, source_qubit, sink_qubit, symbols):
    """含有2个参数（2个单比特门酉矩阵）和CNOT门组成 一个V"""
    pool_circuit = cirq.Circuit()
    
    sink_basis_selector = one_qubit_unitary(genes[0:2],sink_qubit, symbols[0])
    source_basis_selector = one_qubit_unitary(genes[2:4],source_qubit, symbols[1])
    
    pool_circuit.append(sink_basis_selector)
    pool_circuit.append(source_basis_selector)
    
    pool_circuit.append(cirq.CNOT(control=source_qubit, target=sink_qubit))
    
    pool_circuit.append(sink_basis_selector**-1)
    return pool_circuit

# 一层卷积由2层U组成，每个U共享训练参数
def quantum_conv_circuit(genes, bits, layer_index):
    """Quantum Convolution Layer following the above diagram.
    Return a Cirq circuit with the cascade of `two_qubit_unitary` applied
    to all pairs of qubits in `bits` as in the diagram above.
    利用U进行堆叠2层，每个U（two_qubit_unitary）含有1个参数；
    卷积不减少量子比特
    """
    # genes基因长度为 双量子比特卷积数 * 5个量子门 * 2个基因控制
    symbols_nums = len(bits)
    symbols = [] # 1层的所有双量子比特卷积所需要的参数
    syms = [] # 每5个参数组成1个双量子比特卷积所需要的参数
    for i in range(symbols_nums * 5):
        symbol = sympy.Symbol("CL"+str(layer_index) + '-x' + str(i))
        syms.append(symbol)
        if len(syms)==5:
            symbols.append(syms)
            syms = []
    # 1层量子卷积有前后两层 2qbit conv
    symbols_pre = symbols[:symbols_nums//2]
    genes_pre = genes[:len(genes)//2]
    genes_pre_ = [genes_pre[i:i+10] for i in range(0,len(genes_pre),10)]

    symbols_aft = symbols[symbols_nums//2:]
    genes_aft = genes[len(genes)//2:]
    genes_aft_ = [genes_aft[i:i+10] for i in range(0,len(genes_aft),10)]

    circuit = cirq.Circuit()
    for first, second, syms, gene_2conv in zip(bits[0::2], bits[1::2],symbols_pre, genes_pre_):
        circuit += two_qubit_conv(gene_2conv, [first, second], syms)
    for first, second, syms, gene_2conv in zip(bits[1::2], bits[2::2] + [bits[0]], symbols_aft,genes_aft_):
        circuit += two_qubit_conv(gene_2conv, [first, second], syms)
    return circuit

# 一层池化由1层V组成，每个V共享训练参数
def quantum_pool_circuit(genes, source_bits, sink_bits, layer_index):
    """A layer that specifies a quantum pooling operation.
    A Quantum pool tries to learn to pool the relevant information from two
    qubits onto 1.
    利用V进行堆叠1层，每个V（two_qubit_pool）共享相同的6个参数；
    池化减少量子比特数
    """
    # genes基因长度为 双量子比特池化数 * 2个量子门 * 2个基因控制
    circuit = cirq.Circuit()
    symbols_nums = len(source_bits)
    symbols = []
    for i in range(symbols_nums * 2):
        symbol = sympy.Symbol("PL"+str(layer_index) + '-x' + str(i))
        symbols.append(symbol)
    genes_ = [genes[i:i+4] for i in range(0,len(genes),4)]
    for source, sink, symbol0, symbol1 ,gene_2pool in zip(source_bits, sink_bits, symbols[::2], symbols[1::2],genes_):
        circuit += two_qubit_pool(gene_2pool,source, sink, [symbol0,symbol1])
    return circuit

def create_model_circuit(indiv_code):
    model_circuit = cirq.Circuit()
    # 第1层卷积层+池化层
    s = 16*5*2 # 第一层有16个双量子比特卷积，每个双量子卷积有5个量子门，每个量子门由2个基因决定
    t = s + 8*2*2 # 第一层由8个双量子比特池化，每个双量子比特池化有3个量子门，其中2个是互逆门，因此实际上只有2个门，每个门由2个基因决定
    model_circuit += quantum_conv_circuit(indiv_code[0:s], qubits[0::1], 1)
    model_circuit += quantum_pool_circuit(indiv_code[s:t], qubits[0::2], qubits[1::2],1)
    # 第2层卷积层+池化层
    u = t + 8*5*2
    v = u + 4*2*2
    model_circuit += quantum_conv_circuit(indiv_code[t:u], qubits[1::2], 2)
    model_circuit += quantum_pool_circuit(indiv_code[u:v], qubits[1::4], qubits[3::4],2)
    # 第3层卷积层+池化层
    w = v + 4*5*2
    x = w + 2*2*2
    model_circuit += quantum_conv_circuit(indiv_code[v:w], qubits[3::4], 3)
    model_circuit += quantum_pool_circuit(indiv_code[w:x], qubits[3::8], qubits[7::8],3)
    # 第4层卷积层+池化层
    y = x + 2*5*2
    z = y + 1*2*2
    model_circuit += quantum_conv_circuit(indiv_code[x:y], qubits[7::8], 4)
    model_circuit += quantum_pool_circuit(indiv_code[y:z], qubits[7::16], qubits[15::16],4)
    return model_circuit

def fitness(indiv_code):
    qcnn_circuit = create_model_circuit(indiv_code)
    readout_operators = cirq.Z(qubits[-1])
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
    # 超参数设置
    batch_size = 32
    epochs = 3
    # 开始训练
    qcnn_model.fit(x=train_excitations,
                            y=train_labels,
                            batch_size=batch_size,
                            epochs=epochs,
                            verbose=1,
                            validation_data=(test_excitations, test_labels))
    # 保存准确率
    acc = qcnn_model.evaluate(test_excitations, test_labels)
    if(acc[1] > alp['best_fit']):
        alp['best_fit'] = acc[1]                                                     #记录最优值
        with open("best_qcnn_circuit.pkl", "wb")as f:
                pickle.dump(qcnn_circuit,f)
        with open("best_qcnn_obs.pkl","wb") as f:
            pickle.dump(readout_operators, f)
        qcnn_model.save_weights('best_qcnn_model_weights.h5')
        with open("best_acc.txt","w") as f:
            f.write(str(alp['best_fit']))
    return acc[1]

def get_fitness(pop):
    fits = []
    for indiv in pop:
        fits.append(fitness(indiv))
    return fits

def individual_to_key(indiv):
    temp = [str(i) for i in indiv]
    key = ''.join(temp)
    return key

def evaluate(pop):
    fits=get_fitness(pop)                                                     #计算适应度
    inds = list(map(individual_to_key, pop))                                        #将二进制列表转为二进制字符串串
    alp['inds_fits']=dict(zip(inds,fits))                                           #当代的个体_适应度键值对
    alp['bestfit_iters'].append(alp['best_fit'])

# 对种群进行分组，先将种群中每个个体进行随机打乱，然后分组。
def group_population(pop, n_group):
    assert len(pop) % n_group == 0, "pop_size must be a multiple of n_group."
    # 每组的个体数
    per_group = len(pop) // n_group
    group_index = list(range(0, len(pop)))
    random.shuffle(group_index)
    group_pop = []
    for i in range(n_group):
        temp_index = group_index[i * per_group: (i + 1) * per_group]
        temp_pop = []
        for j in temp_index:
            temp_pop.append(pop[j])
        group_pop.append(temp_pop)
    return group_pop

# 从当代种群中挑选n_select个个体，挑选几个个体就将种群分成几个组，从每个组中选择其中适应度最高（此处即rmse最小）的一个个体
def select(pop, n_select,inds_fits):
    # n_select==分组的个数n_group
    group_pop = group_population(pop, n_select)
    fitness_selected = []
    pop_selected = []
    for sub_group in group_pop:
        fits = []
        for indiv in sub_group:
            key = individual_to_key(indiv)
            fits.append(inds_fits[key])
        max_fitness = max(fits)
        pop_selected.append(sub_group[fits.index(max_fitness)])
        fitness_selected.append(max_fitness)
    return pop_selected, fitness_selected

# 获得交叉或变异的随机的几个断点，indiv_length就是一个个体的二进制串长度 time_steps*encode_length
def get_segment_ids(indiv_length):
    index = []
    while True:
        for i in range(indiv_length):
            if random.random() > 0.5:
                index.append(i)
        if len(index) > 0:
            break
    return index

# 变异
def mutation(indiv):
    indiv_length = len(indiv)
    # 随机确定变异的基因点
    index = get_segment_ids(indiv_length)
    for i in index:
        if indiv[i] == 0:
            indiv[i] = 1
        else:
            indiv[i] = 0
    return indiv


# 交叉并变异产生新的子代个体
def crossover(indiv1, indiv2):
    indiv_length = len(indiv1) # len(indiv1) == len(indiv2)
    # 获得父亲的基因断点位置
    a_index = get_segment_ids(indiv_length)
    # 母亲的基因断点位置就是不是父亲的基因断点位置的那些位置
    b_index = []
    for i in range(indiv_length):
        if i not in a_index:
            b_index.append(i)
    # 新的子代个体，初始化为0000...的长度为indiv_length的二进制序列串
    new = list()
    for i in range(indiv_length):
        new.append(0)
    # 从父亲那里得到遗传片段
    for i in a_index:
        new[i] = indiv1[i]
    # 从母亲那里得到遗传片段
    for i in b_index:
        new[i] = indiv2[i]
    # 只有很少的几率（此处为1-0.8的概率）得到的子代个体会发生变异
    if random.random() < alp['pm']:
        new = mutation(new)
    return new

# 重建种群
def reconstruct_population(pop_selected, pop_size):
    new_pop = list()
    # 保留挑选出来的个体
    # new_pop.extend(pop_selected)  #不执行保留策略会更好，就像袁天罡一样，优秀的父辈长存会导致后辈缺乏竞争优势
    pop_map = set()
    for i in range(len(new_pop)):
        pop_map.add(individual_to_key(new_pop[i]))
    # 从挑选出的个体中两两组成一对夫妇
    index = list(combinations(range(len(pop_selected)), 2))
    # 只要种群大小小于预定的种群大小就不停地让组成的夫妻生孩子
    random.shuffle(index)
    # print(index)
    while len(new_pop) < pop_size:
        for combi in index:
            new_indiv = crossover(pop_selected[combi[0]], pop_selected[combi[1]])
            # 保证产生的个体的独特性唯一性
            if not individual_to_key(new_indiv) in pop_map:
                new_pop.append(new_indiv)
                pop_map.add(individual_to_key(new_indiv))
            if len(new_pop) == pop_size:
                break
    return new_pop

def final_train():
    # 用到的量子比特
    # cluster_state_bits = cirq.GridQubit.rect(4, 4)
    # 生成数据集（如果在另一个文件中打开，则需要重新加载数据集）
    # train_excitations, train_labels, test_excitations, test_labels = generate_data(cluster_state_bits)

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
    # 加载模型权重
    qcnn_model.load_weights("best_qcnn_model_weights.h5")
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
    epochs = 5
    # 开始训练
    history = qcnn_model.fit(x=train_excitations,
                            y=train_labels,
                            batch_size=batch_size,
                            epochs=epochs,
                            verbose=1,
                            validation_data=(test_excitations, test_labels))
    acc = qcnn_model.evaluate(test_excitations, test_labels)
    if acc[1] > alp['best_fit']:
        # 保存模型权重
        qcnn_model.save_weights('best_qcnn_model_weights.h5')
        # 保存准确率
        with open("best_acc.txt","w") as f:
            f.write(str(acc[1]))
def reload():
    # 用到的量子比特
    # cluster_state_bits = cirq.GridQubit.rect(4, 4)
    # 生成数据集（如果在另一个文件中打开，则需要重新加载数据集）
    # train_excitations, train_labels, test_excitations, test_labels = generate_data(cluster_state_bits)

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
    # 加载模型权重
    qcnn_model.load_weights("best_qcnn_model_weights.h5")
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
    acc = qcnn_model.evaluate(test_excitations, test_labels)
reload()

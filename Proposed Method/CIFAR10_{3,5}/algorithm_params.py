'''
algorithm_params.py

    Author: 郝晓斌
'''
import math


algorithm_params={


    # 有关进化计算和注意力层的参数如下：
    # 目前表现最好的解
    # 'best_circuit':None, 不用，直接保存以及权重
    # 目前表现最好的值
    'best_fit': 0,
    # 历代的最高适应度值
    'bestfit_iters':[],
    # 当代的个体适应度键值对
    'inds_fits':None,

    # 迭代次数
    'iterations':20, # 20
    # 每一代的种群大小
    'pop_size':50,   # 60
    # 特征维度
    'dims': 180, # 32维度
    # 每个个体的每个特征的编码长度
    'encode_length':2,     # 10编码长度
    # 每次迭代后挑选出用于产生后代的个体数目,该参数就等于将种群划分的组的个数。
    'n_selected':10,   # 10精英
    # 变异概率
    'pm':0.25,    #0.25
}

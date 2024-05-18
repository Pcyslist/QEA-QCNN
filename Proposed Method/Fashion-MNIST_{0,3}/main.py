'''
main.py

        Author:郝晓斌
'''
from algorithm import *
from algorithm_params import algorithm_params as alp
import matplotlib.pyplot as plt
def round3(x):
    return round(x,8)

def main():
    pop=initialize_population(alp['pop_size'],alp['dims'],alp['encode_length'])     #初始化种群
    evaluate(pop)
    print('iter {} ,maxmum is :{}'.format(0,round3(alp['best_fit'])))
    for i in range(1, alp['iterations']):
        pop_selected, fitness_selected = select(pop, alp['n_selected'], alp['inds_fits'])
        pop = reconstruct_population(pop_selected, alp['pop_size'])
        evaluate(pop)
        print('iter {} ,maxmum is :{}'.format(i,round3(alp['best_fit'])))
if __name__=='__main__':
    main()
    final_train()
    plt.figure()
    plt.plot(alp['bestfit_iters'],label='Optimal Value: {}'.format(alp['best_fit']))
    plt.xlabel('Generations')
    plt.ylabel('Fitness')
    plt.title('Generations-Fitness')
    plt.legend()
    plt.savefig('EA_Search_Procedure.svg', format='svg')
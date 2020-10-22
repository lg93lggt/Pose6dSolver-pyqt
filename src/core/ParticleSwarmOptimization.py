from typing import List
import numpy as np
import random
import matplotlib.pyplot as plt
import math

def test_func(x, args=None):
    sum = 0
    length = len(x)
    for i in range(length):
        sum += (4*x[i]**3-5*x[i]**2+x[i]+6)**2
    return sum

class ParticleSwarmOptimization():
    def __init__(self, n_particles, n_dims, n_iters):
        #定义所需变量
        self.w = 0.8
        self.c1 = 2#学习因子
        self.c2 = 2

        self.r1 = 0.6#超参数
        self.r2 = 0.3

        self.n_particles = n_particles  # 粒子数量
        self.n_dims = n_dims  # 搜索维度
        self.n_iters = n_iters  # 迭代次数

        #定义各个矩阵大小
        self.X = np.zeros((self.n_particles, self.n_dims))  # 所有粒子的位置和速度矩阵
        self.V = np.zeros((self.n_particles, self.n_dims))
        self.pbest = np.zeros((self.n_particles, self.n_dims))  # 个体经历的最佳位置和全局最佳位置矩阵
        self.global_best = np.zeros((1, self.n_dims))
        self.p_fit = np.zeros(self.n_particles)  # 每个个体的历史最佳适应值
        self.theta = 1e10  # 全局最佳适应值
        return


    #目标函数，根据使用场景进行设置
    def set_objective_func(self, func_objective):
        self.func_objective = func_objective
        return

    def set_boundery(self, lower_bound=0, upper_bound=1):
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        return


    #初始化粒子群
    def init_population(self, args_of_func_objective: List):
        for i in range(self.n_particles):
            for j in range(self.n_dims):
                self.X[i][j] = self.lower_bound[j] + random.uniform(0, 1) * (self.upper_bound[j] - self.lower_bound[j]) #* np.array([1, 1, 1, 2*np.pi, 2*np.pi, 2*np.pi])
                self.V[i][j] = 2 * random.uniform(0, 1) #* np.array([1, 1, 1, 2*np.pi, 2*np.pi, 2*np.pi])
            self.pbest[i] = self.X[i]
            tmp = self.func_objective(self.X[i], args_of_func_objective)
            self.p_fit[i] = tmp
            if (tmp < self.theta):
                self.theta = tmp
                self.global_best = self.X[i]

    def run(self, *args_of_func_objective: List):
        self.init_population(args_of_func_objective)
        fitness = []
        print("\nPSO:")
        for i_iter in range(self.n_iters):
            for i in range(self.n_particles):  # 更新gbest\pbest
                loss = self.func_objective(self.X[i], args_of_func_objective)
                if (loss < self.p_fit[i]):  # 更新个体最优
                    self.p_fit[i] = loss
                    self.pbest[i] = self.X[i]
                    if (self.p_fit[i] < self.theta):  # 更新全局最优
                        self.global_best = self.X[i]
                        self.theta = self.p_fit[i]
            for i in range(self.n_particles):
                #粒子群算法公式
                self.V[i] = self.w * self.V[i] + self.c1 * self.r1 * (self.pbest[i] - self.X[i]) + \
                            self.c2 * self.r2 * (self.global_best - self.X[i])
                self.X[i] = self.X[i] + self.V[i]
            fitness.append(self.theta)
            
            # 输出
            n_step = self.n_iters // 100 
            if i_iter % n_step == 0: 
                print("iter {:0>4d}/{:0>4d}:\tloss: {:0>4f}".format(i_iter, self.n_iters, self.theta))
        return fitness

if __name__ == '__main__':
    pso_test = ParticleSwarmOptimization(30, 5, 100)
    pso_test.set_objective_func(test_func)
    fitness = pso_test.run()
    plt.figure(1)
    plt.title("Figure1")
    plt.xlabel("iterators", size=14)
    plt.ylabel("fitness", size=14)
    t = np.array([t for t in range(0, 100)])
    fitness = np.array(fitness)
    plt.plot(t, fitness, color='b', linewidth=3)
    plt.show()
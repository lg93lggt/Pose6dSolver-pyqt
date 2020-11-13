from typing import List
import numpy as np
import random
import matplotlib.pyplot as plt
import time

def test_func(x, args=None):
    sum = 0
    length = len(x)
    for i in range(length):
        sum += (4*x[i]**3-5*x[i]**2+x[i]+6)**2
    return sum

class ParticleSwarmOptimization():
    def __init__(self, n_pops, n_dims, n_iters):
        #定义所需变量
        self.w  = 0.9
        self.c1 = 2#学习因子
        self.c2 = 2

        self.r1 = 0.6#超参数
        self.r2 = 0.3

        self.n_pops  = n_pops  # 粒子数量
        self.n_dims  = n_dims       # 搜索维度
        self.n_iters = n_iters      # 迭代次数

        #定义各个矩阵大小
        self.X           = np.zeros((self.n_pops, self.n_dims))  # 所有粒子的值
        self.V           = np.zeros((self.n_pops, self.n_dims))  # 所有粒子的速度
        self.pop_best    = np.zeros((self.n_pops, self.n_dims))  # 个体经历的最佳位置和全局最佳位置矩阵
        self.global_best = np.zeros((1, self.n_dims))
        self.pop_loss    = np.zeros(self.n_pops)  # 每个个体的历史最佳适应值
        self.loss        = 1E10  # 全局最佳适应值
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
        for i in range(self.n_pops):
            for j in range(self.n_dims):
                self.X[i][j] = self.lower_bound[j] + random.uniform(0, 1) * (self.upper_bound[j] - self.lower_bound[j]) #* np.array([1, 1, 1, 2*np.pi, 2*np.pi, 2*np.pi])
                self.V[i][j] = random.uniform(0, 1) #* np.array([1, 1, 1, 2*np.pi, 2*np.pi, 2*np.pi])
            self.pop_best[i] = self.X[i]
            loss_tmp = self.func_objective(self.X[i], args_of_func_objective)
            self.pop_loss[i] = loss_tmp
            if (loss_tmp < self.loss):
                self.loss = loss_tmp
                self.global_best = self.X[i]
        return
 
    def _update_pop(self, j_pop):
        """
        更新算子：更新下一时刻的位置和速度
        """
        # for j_pop in range(self.n_pops):
            # 更新速度
        self.V[j_pop] = self.w * self.V[j_pop] + \
            self.c1 * self.r1* ( self.pop_best[j_pop] - self.X[j_pop]) + \
            self.c2 * self.r2* (self.global_best - self.X[j_pop])
        # 更新值
        self.X[j_pop] = self.X[j_pop] + self.V[j_pop]
        # 越界保护
        for k_dim in range(self.n_dims):
            if self.X[j_pop][k_dim] < self.lower_bound[k_dim]:
                self.X[j_pop][k_dim] = self.upper_bound[k_dim]
            if self.X[j_pop][k_dim] > self.upper_bound[k_dim]:
                self.X[j_pop][k_dim] = self.lower_bound[k_dim]
        return

    def run(self, *args_of_func_objective: List):

        t0       = time.time()
        log_loss = []
        self.init_population(args_of_func_objective)
        print("\nPSO:\tn_iters: {}\tw: {}\t c1: {}\t c2: {}".format(self.n_iters, self.w, self.c1, self.c2))
        for i_iter in range(self.n_iters):

            for j_pop in range(self.n_pops): # 更新gbest\pbest
                self._update_pop(j_pop) # 更新参数
                loss = self.func_objective(self.X[j_pop], args_of_func_objective)
                if (loss < self.pop_loss[j_pop]):  # 更新个体最优
                    self.pop_loss[j_pop] = loss.copy()
                    self.pop_best[j_pop] = self.X[j_pop].copy()
                    if (loss < self.loss):  # 更新全局最优
                        self.global_best = self.X[j_pop].copy()
                        self.loss        = self.pop_loss[j_pop].copy()
            log_loss.append(self.loss)
            
            # 输出
            n_step = self.n_iters // 100 
            if i_iter % n_step == 0: 
                t1 = time.time()
                print("iter {:0>4d}/{:0>4d}:\tloss: {:0>4f}\ttime: {:0>4f}".format(i_iter, self.n_iters, self.loss, t1 - t0))
                #print(self.func_objective(self.global_best, args_of_func_objective))
                t0 = t1
        return log_loss


if __name__ == '__main__':
    pso_test = ParticleSwarmOptimization(50, 5, 1000)
    pso_test.set_objective_func(test_func)
    pso_test.set_boundery(np.zeros(5), 10*np.ones(5))
    pso_test.set_boundery(lower_bound=-100*np.ones(5), upper_bound=100*np.ones(5))
    fitness = pso_test.run()
    plt.figure(1)
    plt.title("Figure1")
    plt.xlabel("iterators", size=14)
    plt.ylabel("fitness", size=14)
    t = np.array([t for t in range(0, 1000)])
    fitness = np.array(fitness)
    plt.plot(t, fitness, color='b', linewidth=3)
    plt.show()
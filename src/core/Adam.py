import math
import numpy as np
import cv2
import json

from . import geometry as geo




class Adam(object):

    def __init__(self, n_iters=1000, alpha=1E-3, beta1=0.9, beta2=0.999):
        self.n_iters = n_iters  
        self.alpha = alpha  # step size
        self.beta1 = beta1   # exponential decay rate 1 for moment, [0, 1)
        self.beta2 = beta2   # exponential decay rate 2 for moment, [0, 1)
        self.eps = 1E-8         # avoid 0 division
        self.m = 0 # initial 1st moment vec
        self.v = 0 # initial 2nd moment vec
        return

    def set_objective_func(self, func_objective):
        self.func_objective = func_objective
        return   

    def set_jacobian_func(self, func_jacobian):
        self.func_jacobian = func_jacobian
        return    

    def run(self, x0, *args_of_func_objective):
        """
            x0, args_of_func_objective
        """
        self.theta = x0
        loss = self.func_objective(self.theta, args_of_func_objective)
        log_loss = [loss]
        log_theta = [self.theta]
        
        is_converged = False
        print("\nAdam:")
        for i in range(self.n_iters):
            self.i_iters = i
            t = i + 1
            gt = self.func_jacobian(self.theta, self.func_objective, args_of_func_objective)
            self.m = self.beta1 * self.m + (1 - self.beta1) * gt
            self.v = self.beta2 * self.v + (1 - self.beta2) * (gt * gt)
            m_corrected = self.m / (1 - self.beta1 ** t)
            v_corrected = self.v / (1 - self.beta2 ** t)
            self.theta = self.theta - self.alpha * m_corrected / (np.sqrt(v_corrected) + self.eps)
            try:
                loss = self.func_objective(self.theta, args_of_func_objective)
            except :
                continue

            # is converged?
            n_step = self.n_iters // 100
            # cond = -1
            if self.i_iters % n_step == 0:
                # cond = np.std(np.array(log_loss[-n_step:]))
                # is_converged = False
                #print(self.alpha / (1 - self.beta1 ** t) * np.sqrt(1 - self.beta2 ** t))
                print("iter {:0>4d}/{:0>4d}:\tloss: {:0>4f}".format(self.i_iters, self.n_iters, loss))
                # print("iter {:0>4d}/{:0>4d}:\tloss: {:0>4f}\tstd_error: {:0>4f}".format(self.i_iters, self.n_iters, loss, cond))
                #cond = (cond - np.min(cond)) / (np.max(cond) - np.min(cond))
            # if ((0 < cond < 1E-4) and (loss < 2 ** 0.5)) and (not is_converged): #
            #     is_converged = True
            #     break
 
            # logging
            log_loss.append(loss)
            log_theta.append(self.theta)

        idx = np.argmin(log_loss)
        self.theta = log_theta[idx]
        return log_loss, log_theta

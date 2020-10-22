
from typing import List
import numpy as np



# calculating the derive of pointed parameter,whose shape is (num_data,1)
def cal_derivative(params: np.ndarray, i_param: int, func_objective, args_of_func_objective: List, delta: float=1E-6) -> np.ndarray:
    params1 = params.copy()
    params2 = params.copy()

    params1[i_param] += delta
    params2[i_param] -= delta
    data_est_output1 = func_objective(params1, args_of_func_objective)
    data_est_output2 = func_objective(params2, args_of_func_objective)
    deriv = np.linalg.norm(data_est_output1 - data_est_output2) / (2 * delta)
    return deriv

# calculating jacobian matrix,whose shape is (num_data,num_params)
def cal_jacobian(params: np.ndarray, func_objective, args_of_func_objective: List) -> np.ndarray:
    n_params = np.shape(params)[0]
    # num_data = np.shape(input_data)[0]
    J = np.zeros(n_params)
    for i_param in range(n_params):
        J[i_param] = cal_derivative(params, i_param, func_objective, args_of_func_objective)
    # print(J.shape)
    return J

# # calculating residual, whose shape is (num_data,1)
# def cal_residual(params: np.ndarray, cost_fun, args_cost_func: list) -> np.ndarray:
#     data_est_output = cost_fun(params, args_cost_func)
#     residual = np.linalg.norm(target - data_est_output)
#     return residual


class LevenbergMarquardt(object):
    def __init__(self, n_iters: int=1000, alpha: float=1E-3):
        self.alpha = alpha
        self.n_iters = n_iters  
        self.tao                = 10 ** -3
        self.threshold_stop     = 10 ** -15
        self.threshold_step     = 10 ** -15
        self.threshold_residual = 10 ** -15
        self.residual_memory    = []

    def set_residual_func(self, func_residual):
        self.func_residual = func_residual
        return

    def set_objective_func(self, func_objective):
        self.func_objective = func_objective
        return 

    def set_jacobian_func(self, func_jacobian=cal_jacobian):
        self.func_jacobian = func_jacobian
        return   
    
    # get the init u, using equation u=tao*max(Aii)
    def get_init_u(self, A, tao):
        if isinstance(A, float):
            m = 0
        else:
            m = np.shape(A)[0]
        Aii = []
        for i in range(0, m):
            Aii.append(A[i, i])
        u = tao * max(Aii)
        return u

    # LM algorithm
    def run(self, x0, *args_of_func_objective):
        """
            x0, args_of_func_objective
        """
        print("\nLM:")
        self.theta = x0
        k = 0  # set the init iter count is 0
        num_params = np.shape(self.theta)[0]  # the number of params
        # calculating the init residual
        residual = self.func_objective(self.theta, args_of_func_objective)
        # calculating the init Jocobian matrix
        jacobian = self.func_jacobian(self.theta, self.func_objective, args_of_func_objective)

        A = np.array([jacobian]).T.dot(np.array([jacobian]))  # calculating the init A
        g = jacobian.T.dot(residual)  # calculating the init gradient g
        stop = (np.linalg.norm(g, ord=np.inf) <= self.threshold_stop)  # set the init stop
        u = self.get_init_u(A, self.tao)  # set the init u
        v = 2  # set the init v=2

        log_theta = []
        log_loss  = []
        while ((not stop) and (k < self.n_iters)):
            while (1):
                #print(k, s)
                Hessian_LM = A + u * np.eye(num_params)  # calculating Hessian matrix in LM
                step = self.alpha * np.linalg.inv(Hessian_LM).dot(g)  # calculating the update step
                if (np.linalg.norm(step) <= self.threshold_step):
                    stop = True
                    break
                else:
                    new_params = self.theta - step  # update params using step
                    new_residual = self.func_objective(new_params, args_of_func_objective)  # get new residual using new params
                    rou = (np.linalg.norm(residual) ** 2 - np.linalg.norm(new_residual) ** 2) / (step.T.dot(u * step + g))
                    if rou > 0:
                        self.theta = new_params
                        residual   = new_residual
                        
                        # print (np.linalg.norm(new_residual)**2)
                        jacobian = self.func_jacobian(self.theta, self.func_objective, args_of_func_objective)  # recalculating Jacobian matrix with new params
                        A        = jacobian.T.dot(jacobian)  # recalculating A
                        g        = jacobian.T.dot(residual)  # recalculating gradient g
                        stop     = (np.linalg.norm(g, ord=np.inf) <= self.threshold_stop) or \
                            (np.linalg.norm(residual) ** 2 <= self.threshold_residual)
                        u = u * max(1 / 3, 1 - (2 * rou - 1) ** 3)
                        v = 2
                        log_theta.append(new_params)
                        log_loss.append(residual)
                    else:
                        u = u * v
                        v = 2 * v
                if (rou > 0 or stop):
                    break
            k += 1

            n_step = self.n_iters // 100
            # cond = -1
            if k % n_step == 0:
                # cond = np.std(np.array(log_loss[-n_step:]))
                # print(self.alpha / (1 - self.beta1 ** t) * np.sqrt(1 - self.beta2 ** t))
                print("iter {:0>4d}/{:0>4d}:\tloss: {:0>4f}".format(k, self.n_iters, residual))
                # print("iter {:0>4d}/{:0>4d}:\tloss: {:0>4f}\tstd_error: {:0>4f}".format(k, self.n_iters, residual, cond))

        return log_loss, log_theta


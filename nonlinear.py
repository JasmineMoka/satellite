import math

import numpy as np


class Subproblem_Quad_IP():
    def __init__(self, obj_coeff, constraints_coeff):
        self.n_dim = len(obj_coeff)
        self.obj_coeff = obj_coeff
        self.constraints_coeff = constraints_coeff

    def compute_obj(self, lamd):
        self.relaxed_obj_coeff = []
        for i in range(self.n_dim):
            self.relaxed_obj_coeff.append({'quad': obj_coeff[i], 'linear': (lamd * constraints_coeff[:, i]).sum()})

    def solve(self):
        self.opt_solution = np.zeros(self.n_dim)
        for i in range(self.n_dim):
            coeff = self.relaxed_obj_coeff[i]
            self.opt_solution[i] = round(-1 * coeff['linear'] / (2 * coeff['quad']))
        self.opt_solution = np.maximum(0, self.opt_solution)

    def solve_surrogate(self, dualproblem):
        self.compute_costfun(dualproblem)
        for i in range(self.n_dim):
            coeff = self.relaxed_obj_coeff[i]
            self.opt_solution[i] = round(-1 * coeff['linear'] / (2 * coeff['quad']))


    def compute_costfun(self, dualproblem):
        costfun_orig = np.dot((self.opt_solution * self.opt_solution).T, obj_coeff)
        cost_constraints = 0
        for i in range(len(dualproblem.lamd)):
            cost_constraints += self.lamd[i] * (np.dot(self.opt_solution.T, self.constraints_coeff[i, :]) - rhs[i])
        return costfun_orig + cost_constraints

    def report(self, i):
        print("iteration times:", i)
        print("current solution:", self.opt_solution)


class Dual_Problem():
    def __init__(self, n_constraints):
        self.lamd = np.zeros(n_constraints)
        self.m, self.r, self.alpha = 10, 0.01, 1.0
        self.step_init, self.step, self.iteration_time = 0.0, 0.0, 0
        self.subgradients_init = np.zeros(n_constraints)
        self.subgradients = np.zeros(n_constraints)
        self.subg_norm_iterations, self.step_iterations, self.costfun_iterations = [], [], []

    def compute_subgradients(self, subproblem_qip):
        for i in range(len(self.lamd)):
            self.subgradients[i] = np.dot(subproblem_qip.opt_solution.T, subproblem_qip.constraints_coeff[i, :]) - rhs[
                i]
        if self.iteration_time == 0:
            self.subgradients_init = self.subgradients.copy()

    def compute_stepsize(self, subproblem_qip):
        if self.iteration_time == 0:
            self.step_init = (417 - self.compute_costfun(subproblem_qip)) / np.linalg.norm(self.subgradients_init) ** 2
        else:
            p = 1 - 1 / (self.iteration_time ** self.r)
            self.alpha = self.alpha * (1 - 1 / (self.m * self.iteration_time ** p))
            if np.linalg.norm(self.subgradients) != 0:
                self.step = self.alpha * self.step_init * np.linalg.norm(self.subgradients_init) / np.linalg.norm(
                    self.subgradients)
            else:
                self.step = 0
                print("subgradients = 0")
        self.step_iterations.append(self.step)

    def compute_stepsize_simple(self, subproblem_qip):
        self.step = (417 - self.compute_costfun(subproblem_qip)) / np.linalg.norm(self.subgradients) ** 2
        self.step_iterations.append(self.step)
        if self.iteration_time == 0:
            self.step_init = self.step

    def update_lamd(self):
        if self.iteration_time == 0:
            self.lamd = self.lamd + self.step_init * self.subgradients_init
        else:
            self.lamd = self.lamd + self.step * self.subgradients
        self.lamd = np.maximum(self.lamd, 0)
        self.iteration_time += 1

    def compute_costfun(self, subproblem):
        costfun_orig = np.dot((subproblem.opt_solution * subproblem.opt_solution).T, obj_coeff)
        cost_constraints = 0
        for i in range(len(self.lamd)):
            cost_constraints += self.lamd[i] * (
                        np.dot(subproblem.opt_solution.T, subproblem.constraints_coeff[i, :]) - rhs[i])
        return costfun_orig + cost_constraints

    def report(self, subproblem):
        print("subgradients:", self.subgradients)
        print("lamd: ", self.lamd)
        print("step:", self.step, "  dual problem cost function:", self.compute_costfun(subproblem))
        print("")

# first try
    # 定义用户规模
    i=5
    j=5
    k=10
    users = np.zeros((i, j))
    # 定义任务
    task_o = np.linspace(0.5*pow(10, 6), 3*pow(10, 6), i*j)
    # L ij
    task_l = task_o.reshape(i,j)
    # C ij
    task_c = np.ones((i, j)) * 300
    # \beta
    bs_signal = np.zeros((i, j))
    # \alpha
    local_signal = np.zeros((i, j))
    # \gamma
    leo_signal = np.zeros((i, j))
    # \rho_{ijk}
    user_LEO_connection_signal = np.zeros((i, j, k))
    # \eta_{ijk}
    user_LEO_compute_signal = np.zeros((i, j, k))
    # \varepsilon_{ij}
    bandwidth_ratio_BS_user = np.zeros((i, j))
    # \delta_{ijk}
    bandwidth_ratio_LEO_user = np.zeros((i, j, k))
    # C k
    cache_capacity = np.ones(k) * 3 * pow(10, 4)

    # f ij local
    computecapacity_user = np.ones((i, j)) * pow(10, 5)
    # f ij BSi    变量
    compute_capacity_bs = np.zeros((i, j))
    # f ij LEOk  变量
    compute_capacity_leo = np.zeros((i, j, k))

    # communication model

    # 用户和BS的距离矩阵
    distance_BS_user = np.zeros((i, j))
    channel = np.zeros((i, j))

    noise_variance = np.zeros((i, j))
    trans_power = np.random.uniform(5, 10, size=(i, j))
    channel_BS_user = pow(distance_BS_user, -2)
    # 添加白噪声
    white_noise = np.random.normal(0, 7.9*pow(10,-13), (i,j))
    # 带宽设置
    band_C = 5*pow(10,8)
    band_Ka = 5*pow(10,8)
    # R ij BS
    capacity_BS_user = np.multiply(bandwidth_ratio_BS_user, math.log(1+trans_power*channel_BS_user/white_noise,2), band_C)
    # R ij BS total
    total_capacity_BS_user = np.sum(np.multiply(capacity_BS_user, bs_signal), axis=1)

    # \delta
    bandwidth_ratio_LEO_user = np.zeros((i, j,k))
    distance_LEO_user = np.zeros((i, j, k))
    channel_LEO_user = np.zeros((i, j, k))
    # R ij LEOk
    capacity_LEO_user = np.multiply(bandwidth_ratio_LEO_user, math.log(1 + trans_power * channel_LEO_user / white_noise, 2), band_C)
    # R ij LEOk total
    total_capacity_LEO_user = np.sum(np.multiply(capacity_LEO_user, leo_signal))
    # ISL communication
    hop_number = np.zeros((i,j))
    # ISL速率设置
    rate_ISL = pow(10,-5)
    time_ISL = hop_number * task_l / rate_ISL


    # computation model

    # T ij local
    time_local = task_l * task_c / computecapacity_user

    # T ij BSi
    time_BS = task_l / capacity_BS_user + task_l * task_c / compute_capacity_bs

    # T ij LEO
    time_LEO = user_LEO_connection_signal * task_l / capacity_LEO_user + user_LEO_compute_signal * task_l * task_c / compute_capacity_leo + time_ISL


    # energy model

    # P l
    power_compute_local = np.ones((i,j)) * 0.001
    # p l 具体值未定
    power_trans_local = np.ones((i,j)) * 0.001
    # E l ij
    energy_local = local_signal * power_compute_local * task_l * task_c / computecapacity_user + bs_signal * power_trans_local * task_l / capacity_BS_user + leo_signal * power_trans_local * task_l * np.sum(user_LEO_connection_signal / capacity_LEO_user, axis=1)
    total_energy_loacl = np.sum(energy_local)

    # p ij BSi
    power_trans_bs = np.ones((i,j))
    # P BS i
    power_compute_bs = np.ones((i,j))
    # E BSi ij
    energy_bs = bs_signal * [power_trans_bs * task_l / capacity_BS_user + power_compute_bs * task_l * task_c / compute_capacity_bs]
    total_energy_bs = np.sum(energy_bs)

    # P ij LEOk
    power_trans_leo = np.ones((i,j,k))
    # P LEO k
    power_compute_leo = np.ones(k)
    # P ISL
    power_ISL = 1
    # E LEO ij
    energy_leo = hop_number * power_ISL * task_l / rate_ISL + np.sum(user_LEO_connection_signal * power_trans_leo * task_l / capacity_LEO_user + user_LEO_compute_signal * power_compute_leo * task_l * task_c / compute_capacity_leo)
    # E LEO total
    total_energy_leo = np.sum(leo_signal * energy_leo)



n_constraints = 2
lamd = np.zeros(n_constraints)
obj_coeff = np.array([0.5, 0.1, 0.5, 0.1, 0.5, 0.1])
constraints_coeff = np.array([[-1, 0.2, -1, 0.2, -1, 0.2], [-5, 1, -5, 1, -5, 1]])
rhs = [-48, -250]
subproblem = Subproblem_Quad_IP(obj_coeff, constraints_coeff)
subproblem.compute_obj(lamd)
print("the dimension of decision variables:", subproblem.n_dim)
print("coefficients of objective function:", subproblem.obj_coeff)
print("coefficients of constraints:", subproblem.constraints_coeff)
print("relaxed problem objective coefficients:", subproblem.relaxed_obj_coeff)
subproblem.solve()
print("optimal solution:", subproblem.opt_solution)

dualproblem = Dual_Problem(n_constraints)
dualproblem.compute_subgradients(subproblem)
print("subgradients = ", dualproblem.subgradients_init)
dualproblem.compute_stepsize(subproblem)
print("step size = ", dualproblem.step)
print("the cost function of dual problem:", dualproblem.compute_costfun(subproblem))
dualproblem.update_lamd()
print("lamd = ", dualproblem.lamd)

max_itertimes = 30
for i in range(max_itertimes):
    subproblem.compute_obj(dualproblem.lamd)
    subproblem.solve()
    subproblem.report(i)
    dualproblem.compute_subgradients(subproblem)
    dualproblem.compute_stepsize(subproblem)
    dualproblem.update_lamd()
    dualproblem.report(subproblem)
print("optimal solution = ", subproblem.opt_solution)
print("subgradients = ", dualproblem.subgradients)
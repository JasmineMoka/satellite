import math
from msilib import Binary

import numpy as np
from pyomo.environ import *

from duality import Duality
from orlibary import ORLibary

data_set = ORLibary()

n_bs_number, n_per_cell_number, n_leo_number, deadline, bs_compute_capacity, leo_compute_capacity = data_set.gen_assign_prob(file_name='d05200.txt')

def initialize_deadline(model, i, j):
    return deadline[int(i - 1), int(j - 1)]


def init_bs_compute_capacity(model,i):
    return int(bs_compute_capacity[int(i - 1)])

def initialize_leo_compute_capacity(model, k):
    return int(leo_compute_capacity[int(k - 1)])

model = ConcreteModel()
# M bs  N 每个cell的用户量 O 卫星数量
model.M, model.N, model.O = RangeSet(1, int(n_bs_number)), RangeSet(1, int(n_per_cell_number)), RangeSet(1, int(n_leo_number))

# model.M, model.N, model.O = Set(initialize=range(int(n_bs_number))), Set(initialize=range(int(n_per_cell_number)),), Set(initialize=range(int(n_leo_number)))
# 定义参数
model.deadline = Param(model.M, model.N, within=NonNegativeReals, initialize=initialize_deadline)
model.bs_compute_capacity = Param(model.M, within=NonNegativeReals, initialize=init_bs_compute_capacity)
model.leo_compute_capacity = Param(model.O, within=NonNegativeReals, initialize=initialize_leo_compute_capacity)
model.Cache = Param(model.O, initialize=3 * pow(10, 4))
model.compute_capacity_user = Param(model.M, model.N, initialize=pow(10, 5))
# model.f_LEO = Param(model.O, initialize=lambda model, k: 100)
print(type(model.bs_compute_capacity))
# 定义变量
model.alpha = Var(model.M, model.N, within=Binary)
model.beta = Var(model.M, model.N, within=Binary)
model.gamma = Var(model.M, model.N, within=Binary)
model.rho = Var(model.M, model.N, model.O, within=Binary)
model.eta = Var(model.M, model.N, model.O, within=Binary)
model.epsilon = Var(model.M, model.N, within=NonNegativeReals, bounds=(0, 1), initialize=0.2)
model.delta = Var(model.M, model.N, model.O, within=NonNegativeReals, bounds=(0, 1), initialize=0.1)
model.f_bs = Var(model.M, model.N, within=NonNegativeReals,initialize=10)
model.f_leo = Var(model.M, model.N, model.O, within=NonNegativeReals,initialize=10)

model.z = Var(model.M, model.N, model.O, within=NonNegativeReals)
model.y = Var(model.M, model.N, model.O, within=NonNegativeReals)
# 用户和BS的距离矩阵
distance_BS_user_data = np.linspace(300, 3000, num=n_bs_number * n_per_cell_number)
distance_BS_user_dict = {}
index = 0
for m in model.M:
    for n in model.N:
        distance_BS_user_dict[(m, n)] = distance_BS_user_data[index]
        index += 1
model.distance_BS_user = Param(model.M, model.N, initialize=distance_BS_user_dict)
# model.channel = Param(model.M, model.N, initialize=0)

model.noise_variance = Param(model.M, model.N, initialize=0)
trans_power_data = {(i, j): np.random.uniform(5, 10) for i in model.M for j in model.N}
model.trans_power = Param(model.M, model.N, initialize=trans_power_data, within=NonNegativeReals)

def channel_BS_user(model, i, j):
    return pow(value(model.distance_BS_user[i, j]), -2)
# 带宽设置
model.band_C = Param(initialize=5 * pow(10, 8))
model.band_Ka = Param(initialize=5 * pow(10, 8))

# 添加白噪声
model.white_noise = Param(model.M, model.N, initialize=np.random.normal(0, 7.9 * pow(10, -13)))
model.channel_BS_user = Param(model.M, model.N, within=NonNegativeReals, initialize=channel_BS_user)
# R ij BS
# def capacity_BS_user(model, i, j):
#     return np.multiply(model.epsilon[i, j], math.log(1 + model.trans_power[i, j] * model.channel_BS_user[i, j] / model.white_noise[i, j], 2) * model.band_C)

def capacity_BS_user(model, i, j):
    # 计算输入 log 函数的值
    log_input = 1 + model.trans_power[i, j] * model.channel_BS_user[i, j] / model.white_noise[i, j]

    # 检查输入 log 函数的值是否为正数
    if log_input <= 0:
        # 如果输入值为负数或零，返回一个大的正数（或者您认为合适的值）
        return float('inf')
    else:
        return np.multiply(model.epsilon[i, j], math.log(log_input, 2) * model.band_C)


model.capacity_BS_user = Param(model.M, model.N, initialize=capacity_BS_user, within=NonNegativeReals)

# trans_power[i, j] 类似p ij

model.total_capacity_BS_user = Param(model.M, model.N, initialize=0)
# R ij BS total
def total_capacity_BS_user(model, i, j):
    model.total_capacity_BS_user[i, j] == np.sum(np.multiply(model.capacity_BS_user[i, j], model.beta[i, j]), axis=1)

distance_LEO_user_data ={(i, j, k): np.random.uniform(500, 300000) for i in model.M for j in model.N for k in model.O}
model.distance_LEO_user = Param(model.M, model.N, model.O, within=NonNegativeReals, initialize=distance_LEO_user_data)

def channel_LEO_user(model, i, j, k):
    return pow(value(model.distance_LEO_user[i, j, k]), -2)

model.channel_LEO_user = Param(model.M, model.N, model.O, within=NonNegativeReals, initialize=channel_LEO_user)


# R ij LEOk
def capacity_LEO_user(model, i, j):
    model.capacity_LEO_user[i, j] == np.multiply(model.delta[i, j, k], math.log(
        1 + model.trans_power[i, j] * model.channel_LEO_user[i, j, k] / model.white_noise[i, j], 2), model.band_C)


# R ij LEOk total
def total_capacity_LEO_user(model, i, j):
    model.total_capacity_LEO_user[i, j] == np.sum(np.multiply(model.capacity_LEO_user[i, j], model.gamma[i, j]))


# ISL communication
model.hop_number = Param(model.M, model.N, initialize=1)
# ISL速率设置
model.rate_ISL = Param(model.M, model.N, model.O, initialize=pow(10, -5))


def time_ISL(model):
    model.time_ISL[i, j] == model.hop_number[i, j] * model.task_l[i, j] / model.rate_ISL[i, j, k]


# computation model

# T ij local
def time_local(model, i, j):
    model.time_local[i, j] == model.task_l[i, j] * model.task_c[i, j] / model.compute_capacity_user[i, j]


# T ij BSi
def time_BS(model, i, j):
    model.time_BS[i, j] == model.task_l[i, j] / model.capacity_BS_user[i, j] + model.task_l[i, j] * model.task_c[i, j] / model.compute_capacity_user[i, j]


# T ij LEO
def time_LEO(model, i, j):
    (model.time_LEO[i, j] == model.epsilon[i, j] * model.task_l[i, j] / model.capacity_LEO_user[i, j]
     + model.delta[i, j, k] * model.task_l[i, j] * model.task_c[i, j] / model.leo_compute_capacity[i, j] + model.time_ISL[i, j])


# energy model
power_compute_local_data = {(i, j): 0.001 for i in model.M for j in model.N}
# P l
model.power_compute_local = Param(model.M, model.N, within=NonNegativeReals, initialize=power_compute_local_data)
# p l 具体值未定
model.power_trans_local = Param(model.M, model.N, within=NonNegativeReals, initialize=0.001)


# E l ij
def energy_local(model, i, j):
    (model.energy_local[i, j] == model.alpha[i, j] * model.power_compute_local[i, j] * model.task_l[i, j] * model.task_c[i, j] / model.compute_capacity_user[i, j]
     + model.beta[i, j] * model.power_trans_local[i, j] * model.task_l[i, j] / model.capacity_BS_user[i, j]
     + model.gamma[i, j] * model.power_trans_local[i, j] * model.task_l[i, j] * np.sum(model.epsilon[i, j] / model.capacity_LEO_user[i, j], axis=1))


def total_energy_loacl(model, i, j):
    model.total_energy_loacl[i, j] == np.sum(model.energy_local[i, j])


# p ij BSi
model.power_trans_bs = Param(model.M, model.N, initialize=1)
# P BS i
model.power_compute_bs = Param(model.M, initialize=1)


# E BSi ij
def energy_bs(model, i, j):
    (model.energy_bs[i, j] == model.beta[i, j] * model.power_trans_bs[i, j] * model.task_l[i, j] / model.capacity_BS_user[i, j] +
     model.power_compute_bs[i] * model.task_l[i, j] * model.task_c[i, j] / model.compute_capacity_user[i, j])
# power_trans_bs[i, j] 类似p BS ij

def total_energy_bs(model, i, j):
    model.total_energy_bs[i, j] == np.sum(model.energy_bs[i, j])


# P ij LEOk
model.power_trans_leo = Param(model.M, model.N, model.O, initialize=1)
# P LEO k
model.power_compute_leo = Param(model.O, initialize=1)
# P ISL
model.power_ISL = Param(model.O, initialize=1)


# E LEO ij
def energy_leo(model, i, j):
    model.energy_leo[i, j] == model.hop_number[i, j] * model.power_ISL[i, j] * model.task_l[i, j] / model.rate_ISL[i, j] + np.sum(
        model.epsilon[i, j] * model.power_trans_leo[i, j, k] * model.task_l[i, j] / model.capacity_LEO_user[i, j] + model.delta[i, j, k] *
        model.power_compute_leo[k] * model.task_l[i, j] * model.task_c[i, j] / model.compute_capacity_leo[i, j])


# E LEO total
def total_energy_leo(model, i, j):
    model.total_energy_leo[i, j] == np.sum(model.gamma[i, j] * model.energy_leo[i, j])


model.capacity_LEO_user = Param(model.M, model.N, within=NonNegativeReals, initialize=0)
model.total_capacity_LEO_user = Param(model.M, model.N, within=NonNegativeReals, initialize=0)
model.time_ISL = Param(model.M, model.N, within=NonNegativeReals, initialize=0)
model.time_local = Param(model.M, model.N, within=NonNegativeReals, initialize=0)
model.time_BS = Param(model.M, model.N, within=NonNegativeReals, initialize=0)
model.time_LEO = Param(model.M, model.N, within=NonNegativeReals, initialize=0)
model.energy_local = Param(model.M, model.N, within=NonNegativeReals, initialize=0)
model.total_energy_local = Param(model.M, model.N, within=NonNegativeReals, initialize=0)
model.energy_bs = Param(model.M, model.N, within=NonNegativeReals, initialize=0)
model.total_energy_bs = Param(model.M, model.N, within=NonNegativeReals, initialize=0)
model.energy_leo = Param(model.M, model.N, within=NonNegativeReals, initialize=0)
model.total_energy_leo = Param(model.M, model.N, within=NonNegativeReals, initialize=0)

def task_l_init(model):
    # 生成初始化数据
    task_o_data = np.linspace(0.5 * 10 ** 6, 3 * 10 ** 6, num=n_bs_number * n_per_cell_number)
    # 将数据转换为字典形式，以便 Pyomo 参数初始化
    task_o_dict = {}
    index = 0
    for m in model.M:
        for n in model.N:
            task_o_dict[(m, n)] = task_o_data[index]
            index += 1
    return task_o_dict

# 初始化参数
model.task_l = Param(model.M, model.N, initialize=task_l_init)
model.task_c = Param(model.M, model.N, initialize=300)





# communication model


def time_rule(model, i, j):
    return (model.alpha[i, j] * model.time_local[i, j] + model.beta[i, j] * model.time_BS[i, j] + model.gamma[i, j] *
            model.time_LEO[i, j]) <= model.deadline[i, j]


def task_assignment_rule(model, i, j):
    return model.alpha[i, j] + model.beta[i, j] + model.gamma[i, j] == 1


def satellite_connection_rule(model, m, n, o):
    return model.rho[m, n, o] <= model.gamma[m, n]


def satellite_computation_rule(model, m, n, o):
    return model.eta[m, n, o] <= model.gamma[m, n]


# 定义带宽限制约束
def bs_bandwidth_rule(model, i):
    return sum(model.beta[i, j] * model.epsilon[i, j] for j in model.N) <= 1


# 定义用户-LEO 带宽约束
# def leo_bandwidth_rule(model, k):
#     return sum(model.gamma[i, j] * model.rho[i, j, k] * model.delta[i, j, k] for i in model.M for j in model.N) <= 1

# 定义线性化约束
def leo_bandwidth_rule(model, k):
    return sum(model.z[i, j, k] for i in model.M for j in model.N) <= 1


# 线性化约束
def linearization_constraint1(model, i, j, k):
    return model.y[i, j, k] <= model.gamma[i, j]
model.linearization_constraint1 = Constraint(model.M, model.N, model.O, rule=linearization_constraint1)

def linearization_constraint2(model, i, j, k):
    return model.y[i, j, k] <= model.rho[i, j, k]
model.linearization_constraint2 = Constraint(model.M, model.N, model.O, rule=linearization_constraint2)

def linearization_constraint3(model, i, j, k):
    return model.y[i, j, k] <= model.delta[i, j, k]
model.linearization_constraint3 = Constraint(model.M, model.N, model.O, rule=linearization_constraint3)

def linearization_constraint4(model, i, j, k):
    return model.y[i, j, k] >= model.gamma[i, j] + model.rho[i, j, k] + model.delta[i, j, k] - 2
model.linearization_constraint4 = Constraint(model.M, model.N, model.O, rule=linearization_constraint4)

def linearization_constraint5(model, i, j, k):
    return model.z[i, j, k] == model.y[i, j, k]
model.linearization_constraint5 = Constraint(model.M, model.N, model.O, rule=linearization_constraint5)



# 定义卫星缓存容量约束
def cache_rule(model, k):
    return sum(model.gamma[i, j] * model.rho[i, j, k] * model.task_l[i, j] for i in model.M for j in model.N) <= \
        model.Cache[k]


# 定义基站计算能力约束
def bs_capacity_rule(model,i):
    return sum(model.beta[i, j] * model.f_bs[i, j] for j in model.N) <= model.bs_compute_capacity[i]


# 定义计算能力约束
def compute_capacity_constraint(model, k):
    return sum(model.gamma[i, j] * model.eta[i, j, k] * model.f_leo[i, j, k] for i in model.M for j in model.N) <= model.leo_compute_capacity[k]


# 定义用户只能连接一个卫星的约束
def user_single_connection_rule(model, i, j):
    return sum(model.eta[i, j, k] for k in model.O) <= 1


# 定义用户只能在一个卫星上完成计算的约束
def user_single_computation_rule(model, i, j):
    return sum(model.rho[i, j, k] for k in model.O) <= 1

def obj_expression(model):
    return sum(model.total_energy_local[i, j] + model.total_energy_bs[i, j] + model.total_energy_leo[i, j] for i in model.M for j in model.N)

# def relaxed_obj_rule(model, m, o, p, q, r, s, t, u):
def relaxed_obj_rule(model, m, o, p, q, r, s):
    return sum(model.alpha[i, j] * model.task_l[i, j] * model.task_c[i, j] * (model.m[i, j] + model.power_compute_local[i, j])
         + model.beta[i, j] * (
             (model.task_l[i, j] / model.capacity_BS_user[i, j] * (m[i, j] + model.trans_power[i, j] + model.power_trans_bs[i, j]))
             + (model.task_l[i, j] * model.task_c[i, j] / model.f_bs[i, j] * (m[i, j] + model.power_compute_bs[i]))
             + o[i] * model.epsilon[i, j]
             + s[i] * model.f_bs[i, j]
         )
         + model.gamma[i, j] *
             sum(
                 model.rho[i, j, k] * (model.task_l[i, j] / model.capacity_LEO_user[i, j] * (m[i, j] + model.trans_power[i, j] + model.power_trans_leo[i, j, k])
                                       + p[k] * model.delta[i, j, k] + r[k] * model.task_l[i, j])
                 + model.eta[i, j, k] * (model.task_l[i, j] * model.task_c[i, j] / model.f_leo[i, j, k] * (m[i, j] + model.power_compute_leo[k])
                                         + q[k] * model.f_leo[i, j, k])
                 + model.hop_number[i, j] * model.task_l[i, j] / model.rate_ISL[i, j, k] * (m[i, j] + model.power_ISL[k])
                 for k in model.O
             )
         for i in model.M for j in model.N
    )
    - sum(model.m[i, j] * model.deadline[i, j] for i in model.M for j in model.N)
    - sum(o[i] for i in model.M)
    - sum(p[k] for k in model.O)
    - sum(q[k] * model.leo_compute_capacity[k] for k in model.O)
    - sum(r[k] * model.cache[k] for k in model.O)
    - sum(s[i] * model.bs_compute_capacity[i] for i in model.M)

#
# def relaxed_obj_rule(model,m):
#     return (sum((model.alpha[i,j] * model.task_l[i,j] * model.task_c[i,j] * (m[i, j] + power_compute_local[i, j]))
#         + (model.beta[i,j] *
#                   (model.task_l[i,j] / model.capacity_BS_user[i, j] * (m[i,j] + trans_power[i, j] + power_trans_bs[i, j]))
#                   + (model.task_l[i,j] * model.task_c[i,j] / model.f_bs[i,j] * (m[i,j] + power_compute_bs[i]))
#                   + o[i] * model.epsilon[i,j]
#                   + s[i] * model.f_bs[i, j]
#                   )
#         + model.gamma[i,j] * (
#             model.rho[i, j, k] * (model.task_l[i, j] / model.capacity_LEO_user[i, j] * (m[i, j] + trans_power[i, j] + power_trans_leo[i, j, k])
#                                     + p[k] * model.delta[i, j, k] + r[k] * model.task_l[i,j])
#             + model.eta[i, j, k] * (model.task_l[i, j] * model.task_c[i, j] / model.f_leo[i, j, k] * (m[i, j] + power_compute_leo[k])
#                                     + q[k] * model.f_leo[i,j,k])
#             + hop_number[i,j] * model.task_l[i, j] / rate_ISL[i, j, k] * (m[i, j] + power_ISL[k])
#         for k in model.O)
#         for i in model.M for j in model.N)
#             - sum(m[i,j] * model.deadline[i, j]for i in model.M for j in model.N)
#             - sum(o[i] for i in model.M)
#             - sum(p[k] for k in model.O)
#             - sum(q[k] * model.leo_compute_capacity[k] for k in model.O)
#             - sum(r[k] * model.cache[k])
#             - sum(s[i] * model.bs_compute_capacity[i] for i in model.M))


    # return sum((model.alpha[i, j] * model.energy_local[i, j] + model.beta[i, j] * model.energy_bs[i, j] + model.gamma[
    #     i, j] * model.energy_leo[i, j])
    #            for i in model.M for j in model.N) - sum(model.m[i, j] *(model.alpha[i, j] * model.time_local[i, j] + model.beta[i, j] * model.time_BS[i, j] + model.gamma[
    #     i, j] * model.time_LEO[i, j]) for i in model.M for j in model.N)


def subgrad_rule(model, i, j):
    return (model.alpha[i, j] + model.beta[i, j] + model.gamma[i, j]) - 1

def set_m(model, m):
    for i in model.M:
        for j in model.N:
            model.m[i, j] = float(m[i-1][j-1])if m is not None else 0.01
def set_o(model, o):
    for i in model.M:
        model.o[i] = float(o[i-1])if o is not None else 0.01
def set_p(model, p):
    for k in model.O:
        model.p[k] = float(p[k-1])if p is not None else 0.01
def set_q(model, q):
    for k in model.O:
        model.q[k] = float(q[k - 1]) if q is not None else 0.01
def set_r(model, r):
    for k in model.O:
        model.r[k] = float(r[k - 1]) if r is not None else 0.01
def set_s(model, s):
    for i in model.M:
        model.s[i] = float(s[i - 1]) if s is not None else 0.01
def set_t(model, t):
    for i in model.M:
        for j in model.N:
            model.t[i, j] = float(t[i - 1][j - 1]) if t is not None else 0.01
def set_u(model, u):
    for i in model.M:
        for j in model.N:
            model.u[i, j] = float(u[i-1][j-1])if u is not None else 0.01
class SubgradientCalculator:
    def __init__(self):
        self.relaxed_model = None
        self.subgrad = None

    def get_subgrad(self, relaxed_model):
        self.relaxed_model = relaxed_model
        self.subgrad = np.array([[self.relaxed_model.subgrad[i, j]() for j in self.relaxed_model.N] for i in self.relaxed_model.M])
        return self.subgrad

    def get_step_size(self):
        # 定义步长计算方法
        pass

    def update(self):
        # 定义更新方法
        pass

    def print_status(self, iteration, max_iter):
        print(f"Iteration {iteration}/{max_iter}")
        print(f"Subgradient: {self.subgrad}")
        # 可以添加更多状态信息

model.time_rule = Constraint(model.M, model.N, rule=time_rule)
model.task_assignment_constraint = Constraint(model.M, model.N, rule=task_assignment_rule)
model.satellite_connection_constraint = Constraint(model.M, model.N, model.O, rule=satellite_connection_rule)
model.satellite_computation_constraint = Constraint(model.M, model.N, model.O, rule=satellite_computation_rule)
model.bs_bandwidth_constraint = Constraint(model.M, rule=bs_bandwidth_rule)
model.leo_bandwidth_constraint = Constraint(model.O, rule=leo_bandwidth_rule)
model.cache_constraint = Constraint(model.O, rule=cache_rule)
model.bs_capacity_constraint = Constraint(model.M, rule=bs_capacity_rule)
model.user_single_connection_constraint = Constraint(model.M, model.N, rule=user_single_connection_rule)
model.user_single_computation_constraint = Constraint(model.M, model.N, rule=user_single_computation_rule)





# 乘子
model.m = Param(model.M, model.N, initialize=0.0, mutable=True)
model.o = Param(model.M, initialize=0.0, mutable=True)
model.p = Param(model.O, initialize=0.0, mutable=True)
model.q = Param(model.O, initialize=0.0, mutable=True)
model.r = Param(model.O, initialize=0.0, mutable=True)
model.s = Param(model.M, initialize=0.0, mutable=True)
model.t = Param(model.M, model.N, initialize=0.0, mutable=True)
model.u = Param(model.M, model.N, initialize=0.0, mutable=True)


model.subgrad = Expression(model.M, model.N, rule = subgrad_rule)
model.obj = Objective(rule=relaxed_obj_rule, sense=minimize)

lagrangian_relaxation = Relaxation()
lagrangian_relaxation.relax_constrs(relaxed_constrs=model.time_rule)
lagrangian_relaxation.set_objective(model, relaxed_obj_rule)

results = solver.solve(model, tee=True)

m = np.zeros((n_bs_number, n_per_cell_number))
o = np.zeros((n_bs_number))
p = np.zeros((n_leo_number))
q = np.zeros((n_leo_number))
r = np.zeros((n_leo_number))
s = np.zeros((n_bs_number))
t = np.zeros((n_bs_number, n_per_cell_number))
u = np.zeros((n_bs_number, n_per_cell_number))

# duality_model = Duality(m_init=m)
solver = SolverFactory('gurobi')
subgrad_calculator = SubgradientCalculator()

duality_model = SubgradientCalculator()
# solver = SolverFactory("glpk", solver_io="python")
max_iter = 50

for z in range(max_iter):
    set_m(model, m)
    set_o(model, o)
    set_p(model, p)
    set_q(model, q)
    set_r(model, r)
    set_s(model, s)
    set_t(model, t)
    set_u(model, u)
    solution = solver.solve(model, tee=True)
    subgrad = duality_model.get_subgrad(relaxed_model=model)

    # 获取步长
    duality_model.get_step_size()

    # 更新参数
    m = duality_model.update()

    # 打印状态
    duality_model.print_status(z, max_iter)

    # duality_model.get_subgrad(relaxed_model=model)
    # duality_model.get_step_size()
    # m = duality_model.update()
    # duality_model.print_status(z, max_iter)



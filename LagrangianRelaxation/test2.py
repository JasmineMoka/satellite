
import math
from msilib import Binary

import numpy as np
from pyomo.environ import *

from LagrangianRelaxation.relaxation import Relaxation
from duality import Duality
from orlibary import ORLibary

data_set = ORLibary()

n_bs_number, n_per_cell_number, n_leo_number, deadline, bs_compute_capacity, leo_compute_capacity = data_set.gen_assign_prob(file_name='c05200.txt')

def initialize_deadline(model, i, j):
    return deadline[int(i - 1), int(j - 1)]


def init_bs_compute_capacity(model,i):
    return int(bs_compute_capacity[int(i - 1)])

def initialize_leo_compute_capacity(model, k):
    return int(leo_compute_capacity[int(k - 1)])
# 定义原始问题
model = ConcreteModel()

model.M, model.N, model.O = RangeSet(1, int(n_bs_number)), RangeSet(1, int(n_per_cell_number)), RangeSet(1, int(n_leo_number))
# 定义参数
model.deadline = Param(model.M, model.N, within=NonNegativeReals, initialize=initialize_deadline)
model.bs_compute_capacity = Param(model.M, within=NonNegativeReals, initialize=init_bs_compute_capacity)
model.leo_compute_capacity = Param(model.O, within=NonNegativeReals, initialize=initialize_leo_compute_capacity)
model.Cache = Param(model.O, initialize= pow(10, 4))
model.compute_capacity_user = Param(model.M, model.N, initialize=pow(10, 5))
# model.f_LEO = Param(model.O, initialize=lambda model, k: 100)

# 定义变量
model.alpha = Var(model.M, model.N, within=NonNegativeReals, bounds=(0, 1), initialize=0.3)
model.beta = Var(model.M, model.N, within=NonNegativeReals, bounds=(0, 1), initialize=0.5)
model.gamma = Var(model.M, model.N, within=NonNegativeReals, bounds=(0, 1), initialize=0.2)
# model.rho = Var(model.M, model.N, model.O, initialize=0, within=Binary)
model.rho = Var(model.M, model.N, model.O, within=NonNegativeReals, bounds=(0, 1), initialize=0.05)
# model.eta = Var(model.M, model.N, model.O, initialize=0, within=Binary)
model.eta = Var(model.M, model.N, model.O, within=NonNegativeReals, bounds=(0, 1), initialize=0.1)
model.epsilon = Var(model.M, model.N, within=NonNegativeReals, bounds=(0, 1), initialize=0.2)
model.delta = Var(model.M, model.N, model.O, within=NonNegativeReals, bounds=(0, 1), initialize=0.1)
model.f_bs = Var(model.M, model.N, within=NonNegativeReals,initialize=300)
model.f_leo = Var(model.M, model.N, model.O, within=NonNegativeReals,initialize=200)


# model.white_noise = Param(model.M, model.N, initialize=lambda model, i, j: abs(np.random.normal(0, 7.9 * pow(10, -12))))
model.white_noise = Param(model.M, model.N, initialize=pow(10, -12))
model.noise_variance = Param(model.M, model.N, initialize=0)
model.band_C = Param(initialize=pow(10, 8))
model.band_Ka = Param(initialize=pow(10, 8))

# 用户和BS的距离矩阵
# distance_BS_user_data = np.linspace(300, 1000, num=n_bs_number * n_per_cell_number)
distance_BS_user_data = np.linspace(100, 100, num=n_bs_number * n_per_cell_number)
distance_BS_user_dict = {(i+1, j+1): distance_BS_user_data[i*n_per_cell_number + j] for i in range(n_bs_number) for j in range(n_per_cell_number)}
model.distance_BS_user = Param(model.M, model.N, initialize=distance_BS_user_dict)
trans_power_data = {(i, j): np.random.uniform(1, 1) for i in model.M for j in model.N}
model.trans_power = Param(model.M, model.N, initialize=trans_power_data, within=NonNegativeReals)

def channel_BS_user(model, i, j):
    return pow(value(model.distance_BS_user[i, j]), -2)
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

# R ij BS total
def total_capacity_BS_user(model, i, j):
    return sum(model.capacity_BS_user[i, j] * model.beta[i, j] for j in model.N)
model.total_capacity_BS_user = Param(model.M, model.N, initialize=total_capacity_BS_user)

distance_LEO_user_data ={(i, j, k): int(np.random.uniform(5000, 5000)) for i in model.M for j in model.N for k in model.O}
model.distance_LEO_user = Param(model.M, model.N, model.O, within=NonNegativeReals, initialize=distance_LEO_user_data)
def channel_LEO_user(model, i, j):
    return pow(sum(value(model.distance_LEO_user[i, j, k]) * model.rho[i, j, k] for k in model.O), -2)
model.channel_LEO_user = Param(model.M, model.N, within=NonNegativeReals, initialize=channel_LEO_user)


# R ij LEOk
def capacity_LEO_user(model, i, j):
    log_input = 1 + model.trans_power[i, j] * model.channel_LEO_user[i, j] / model.white_noise[i, j]
    return sum(model.delta[i, j, k] * math.log(log_input, 2) * model.band_C for k in model.O) if log_input > 0 else float('inf')
model.capacity_LEO_user = Param(model.M, model.N, initialize=capacity_LEO_user, within=NonNegativeReals)


# R ij LEOk total
def total_capacity_LEO_user(model, i, j):
    return sum(model.capacity_LEO_user[i, j] * model.gamma[i, j] for k in model.O)
model.total_capacity_LEO_user = Param(model.M, model.N, initialize=total_capacity_LEO_user)


# 初始化任务长度和计算需求
def task_l_init(model):
    # task_o_data = np.linspace(0.5 * 10 ** 3, 3 * 10 ** 3, num=n_bs_number * n_per_cell_number)
    task_o_data = np.linspace(10 ** 3, 10 ** 3, num=n_bs_number * n_per_cell_number)
    task_o_dict = {(i+1, j+1): task_o_data[i*n_per_cell_number + j] for i in range(n_bs_number) for j in range(n_per_cell_number)}
    return task_o_dict
model.task_l = Param(model.M, model.N, initialize=task_l_init)

# model.task_c = Param(model.M, model.N, initialize=3000)
model.task_c = Param(model.M, model.N, initialize=10000)
# ISL communication
model.hop_number = Param(model.M, model.N, initialize=1)
# ISL速率设置
model.rate_ISL = Param(model.O, initialize=pow(10, 3))
def time_ISL(model, i, j):
    return model.hop_number[i, j] * model.task_l[i, j] / model.rate_ISL[1]
model.time_ISL = Param(model.M, model.N, initialize=time_ISL)
# 定义时间和能量计算的函数
# T ij local
def time_local(model, i, j):
    return model.task_l[i, j] * model.task_c[i, j] / model.compute_capacity_user[i, j]
model.time_local = Param(model.M, model.N, initialize=time_local)
# T ij BSi
def time_BS(model, i, j):
    return model.task_l[i, j] / model.capacity_BS_user[i, j] + model.task_l[i, j] * model.task_c[i, j] / model.f_bs[i, j]
model.time_BS = Param(model.M, model.N, initialize=time_BS)
# T ij LEO
def time_LEO(model, i, j):
    return model.epsilon[i, j] * model.task_l[i, j] / model.capacity_LEO_user[i, j] + sum(model.delta[i, j, k] * model.task_l[i, j] * model.task_c[i, j] / model.f_leo[i, j, k] + model.time_ISL[i, j] for k in model.O)
model.time_LEO = Param(model.M, model.N, initialize=time_LEO)

# 定义能量计算的参数和函数
power_compute_local_data = {(i, j): 0.001 for i in model.M for j in model.N}
# P l
model.power_compute_local = Param(model.M, model.N, within=NonNegativeReals, initialize=power_compute_local_data)
# p l 具体值未定
model.power_trans_local = Param(model.M, model.N, within=NonNegativeReals, initialize=0.001)
# p ij BSi
model.power_trans_bs = Param(model.M, model.N, initialize=1.1)
# P BS i
model.power_compute_bs = Param(model.M, initialize=1.3)
# P ij LEOk
model.power_trans_leo = Param(model.M, model.N, model.O, initialize=1.9)
# P LEO k
model.power_compute_leo = Param(model.O, initialize=1.7)
# P ISL
model.power_ISL = Param(model.O, initialize=1.2)

model.energy_local = Param(model.M, model.N, initialize=1.1)
model.energy_bs = Param(model.M, model.N, initialize=1.2)
model.energy_leo = Param(model.M, model.N, initialize=1.3)

# 定义能量计算约束
def energy_local_compute(model, i, j):
    return model.energy_local[i, j] == (
        model.alpha[i, j] * model.power_compute_local[i, j] * model.task_l[i, j] * model.task_c[i, j] / model.compute_capacity_user[i, j]
        + model.beta[i, j] * model.power_trans_local[i, j] * model.task_l[i, j] / model.capacity_BS_user[i, j]
        + model.gamma[i, j] * model.power_trans_local[i, j] * model.task_l[i, j] * (model.epsilon[i, j] / model.capacity_LEO_user[i, j])
    )


model.energy_local_compute = Constraint(model.M, model.N, rule=energy_local_compute)



def energy_bs_compute(model, i, j):
    return model.energy_bs[i, j]==(
        model.beta[i, j] * model.power_trans_bs[i, j] * model.task_l[i, j] / model.capacity_BS_user[i, j]
        + model.power_compute_bs[i] * model.task_l[i, j] * model.task_c[i, j] / model.compute_capacity_user[i, j]
    )



model.energy_bs_compute = Constraint(model.M, model.N, rule=energy_bs_compute)

# 定义辅助变量
model.z1 = Var(model.M, model.N, model.O, within=NonNegativeReals, initialize=0)
model.z2 = Var(model.M, model.N, model.O, within=NonNegativeReals, initialize=0)
model.z3 = Var(model.M, model.N, model.O, within=NonNegativeReals, initialize=0)
model.z4 = Var(model.M, model.N, model.O, within=NonNegativeReals, initialize=0)
# 定义约束
def energy_leo_compute(model, i, j):
    return model.energy_leo[i, j] == (
        sum(model.z2[i, j, k] for k in model.O) +
        sum(model.z4[i, j, k] for k in model.O)
    )
model.energy_leo_compute = Constraint(model.M, model.N, rule=energy_leo_compute)


# z1 = epsilon * power_trans_leo * task_l
def z1_rule(model, i, j, k):
    return model.z1[i, j, k] == model.epsilon[i, j] * model.power_trans_leo[i, j, k] * model.task_l[i, j]

model.z1_constr = Var(model.M, model.N, model.O, initialize=z1_rule)

# z2 * capacity_LEO_user = z1
def z2_rule(model, i, j, k):
    return model.z2[i, j, k] * model.capacity_LEO_user[i, j] == model.z1[i, j, k]

model.z2_constr = Var(model.M, model.N, model.O, initialize=z2_rule)

# z3 = delta * power_compute_leo * task_l * task_c
def z3_rule(model, i, j, k):
    return model.z3[i, j, k] == model.delta[i, j, k] * model.power_compute_leo[k] * model.task_l[i, j] * model.task_c[i, j]

model.z3_constr = Var(model.M, model.N, model.O, initialize=z3_rule)

# z4 * f_leo = z3
def z4_rule(model, i, j, k):
    return model.z4[i, j, k] * model.f_leo[i, j, k] == model.z3[i, j, k]

model.z4_constr = Var(model.M, model.N, model.O, initialize=z4_rule)




# 定义总能量计算
def total_energy_local(model):
    return sum(model.energy_local[i, j] for i in model.M for j in model.N)
model.total_energy_local = Expression(initialize=total_energy_local)

def total_energy_bs(model):
    return sum(model.energy_bs[i, j] for i in model.M for j in model.N)
model.total_energy_bs = Expression(rule=total_energy_bs)

def total_energy_leo(model):
    return sum(model.energy_leo[i, j] for i in model.M for j in model.N)
model.total_energy_leo = Expression(rule=total_energy_leo)

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


# 定义用户-LEO 带宽约束 拆分
# def leo_bandwidth_rule(model, k):
#     return sum(model.gamma[i, j] * model.rho[i, j, k] * model.delta[i, j, k] for i in model.M for j in model.N) <= 1

model.a = Var(model.M, model.N, model.O, within=NonNegativeReals)
model.w = Var(model.M, model.N, model.O, within=NonNegativeReals)

def a_definition_rule1(model, i, j, k):
    return model.a[i, j, k] >= model.gamma[i, j] + model.eta[i, j, k] - 1
def a_definition_rule2(model, i, j, k):
    return model.a[i, j, k] <= model.gamma[i, j]
def a_definition_rule3(model, i, j, k):
    return model.a[i, j, k] <= model.eta[i, j, k]
def a_definition_rule4(model, i, j, k):
    return model.a[i, j, k] >= 0
model.a_definition_constraint1 = Constraint(model.M, model.N, model.O, rule=a_definition_rule1)
model.a_definition_constraint2 = Constraint(model.M, model.N, model.O, rule=a_definition_rule2)
model.a_definition_constraint3 = Constraint(model.M, model.N, model.O, rule=a_definition_rule3)
model.a_definition_constraint4 = Constraint(model.M, model.N, model.O, rule=a_definition_rule4)

# 添加辅助约束来定义 w
def w_definition_rule1(model, i, j, k):
    return model.w[i, j, k] >= model.a[i, j, k] + model.f_leo[i, j, k] - 1
def w_definition_rule2(model, i, j, k):
    return model.w[i, j, k] <= model.a[i, j, k]
def w_definition_rule3(model, i, j, k):
    return model.w[i, j, k] <= model.f_leo[i, j, k]
def w_definition_rule4(model, i, j, k):
    return model.w[i, j, k] >= 0

model.w_definition_constraint1 = Constraint(model.M, model.N, model.O, rule=w_definition_rule1)
model.w_definition_constraint2 = Constraint(model.M, model.N, model.O, rule=w_definition_rule2)
model.w_definition_constraint3 = Constraint(model.M, model.N, model.O, rule=w_definition_rule3)
model.w_definition_constraint4 = Constraint(model.M, model.N, model.O, rule=w_definition_rule4)
def leo_compute_capacity_rule(model, k):
    return sum(model.w[i, j, k] for i in model.M for j in model.N) <= model.leo_compute_capacity[k]

model.z = Var(model.M, model.N, model.O, initialize=0, within=NonNegativeReals)
# 定义线性化约束
def leo_bandwidth_rule(model, k):
    return sum(model.z[i, j, k] for i in model.M for j in model.N) <= 1

model.y = Var(model.M, model.N, model.O, within=NonNegativeReals)
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
    return sum(model.gamma[i, j] * model.rho[i, j, k] * model.task_l[i, j] for i in model.M for j in model.N) <= model.Cache[k]


# 定义基站计算能力约束
def bs_capacity_rule(model,i):
    return sum(model.beta[i, j] * model.f_bs[i, j] for j in model.N) <= model.bs_compute_capacity[i]


# 定义计算能力约束 拆分约束
# def leo_compute_capacity_rule(model, k):
#     return sum(model.gamma[i, j] * model.eta[i, j, k] * model.f_leo[i, j, k] for i in model.M for j in model.N) <= model.leo_compute_capacity[k]
model.zy = Var(model.M, model.N, model.O, within=NonNegativeReals, initialize=0.3)
def linear_constraint1(model, i, j, k):
    return model.zy[i, j, k] <= model.gamma[i, j]
model.linear_constraint1 = Constraint(model.M, model.N, model.O, rule=linear_constraint1)

def linear_constraint2(model, i, j, k):
    return model.zy[i, j, k] <= model.eta[i, j, k]
model.linear_constraint2 = Constraint(model.M, model.N, model.O, rule=linear_constraint2)

def linear_constraint3(model, i, j, k):
    return model.zy[i, j, k] >= model.gamma[i, j] + model.eta[i, j, k] - 1
model.linear_constraint3 = Constraint(model.M, model.N, model.O, rule=linear_constraint3)
def leo_compute_capacity_rule(model, k):
    return sum(model.zy[i, j, k] * model.f_leo[i, j, k] for i in model.M for j in model.N) <= model.leo_compute_capacity[k]


# 定义用户只能连接一个卫星的约束
def user_single_connection_rule(model, i, j):
    return sum(model.eta[i, j, k] for k in model.O) <= 1


# 定义用户只能在一个卫星上完成计算的约束
def user_single_computation_rule(model, i, j):
    return sum(model.rho[i, j, k] for k in model.O) <= 1


model.time_constraint = Constraint(model.M, model.N, rule=time_rule)
model.task_assignment_constraint = Constraint(model.M, model.N, rule=task_assignment_rule)
model.satellite_connection_constraint = Constraint(model.M, model.N, model.O, rule=satellite_connection_rule)
model.satellite_computation_constraint = Constraint(model.M, model.N, model.O, rule=satellite_computation_rule)
model.bs_bandwidth_constraint = Constraint(model.M, rule=bs_bandwidth_rule)
model.leo_bandwidth_constraint = Constraint(model.O, rule=leo_bandwidth_rule)
model.cache_constraint = Constraint(model.O, rule=cache_rule)
model.bs_capacity_constraint = Constraint(model.M, rule=bs_capacity_rule)
model.leo_compute_capacity_constraint = Constraint(model.O, rule=leo_compute_capacity_rule)
model.user_single_connection_constraint = Constraint(model.M, model.N, rule=user_single_connection_rule)
model.user_single_computation_constraint = Constraint(model.M, model.N, rule=user_single_computation_rule)
def relaxed_obj_rule(model):
    # 定义你的目标函数规则
    return sum(model.energy_local[i, j] + model.energy_bs[i, j] + model.energy_leo[i, j] for i in model.M for
                j in model.N)
class Relaxation:
    def __init__(self, model, relaxed_constrs):
        self.model = model
        self.relaxed_constrs = relaxed_constrs
        self.lambdas = {constr: 1.0 for constr in relaxed_constrs}
        self.subgrad = None
        self.step_size = 1.0

    def relax_constrs(self):
        for constr in self.relaxed_constrs:
            constr.deactivate()

    def set_objective(self, relaxed_obj_rule):
        # 定义拉格朗日目标函数
        def lagrangian_obj_rule(model):
            lagrangian_term = sum(
                self.lambdas[constr] * (constr() - (constr.lower if constr.lower is not None else 0))
                for constr in self.relaxed_constrs
            )
            return relaxed_obj_rule(model) + lagrangian_term

        self.model.obj = Objective(rule=lagrangian_obj_rule, sense=minimize)
    def compute_subgrad(self):
        # 计算子梯度
        self.subgrad = {constr: constr() - (constr.lower if constr.lower is not None else 0) for constr in self.relaxed_constrs}

    def update_lambdas(self):
        # 更新拉格朗日乘子
        for constr in self.relaxed_constrs:
            self.lambdas[constr] += self.step_size * self.subgrad[constr]

    def get_step_size(self, iteration, max_iter):
        # 逐渐减小的步长
        self.step_size = 1 / (iteration + 1)
        return self.step_size




# 最大迭代次数
max_iter = 50
relaxed_constrs = (
            [model.time_constraint[i, j] for i in model.M for j in model.N]
            + [model.bs_bandwidth_constraint[i] for i in model.M]
            + [model.leo_bandwidth_constraint[k] for k in model.O]
            + [model.cache_constraint[k] for k in model.O]
            + [model.bs_capacity_constraint[i] for i in model.M]
            + [model.leo_compute_capacity_constraint[k] for k in model.O]
    )

def solve_subproblem_1(c_set, d_set, lamda_set):
    import copy
    import numpy as np
    sol_index_list = []
    sol_value_list = []
    A = copy.deepcopy(c_set)
    for i in range(len(A)):
        for j in range(len(A[0])):
            A[i][j] = A[i][j]*d_set[i] + lamda_set[i][j]
    A_matrix = np.array(A)
    #找到A矩阵每行的最小值对应的列索引
    for i in range(len(A)):
        sol_index_list.append(np.argmin(A[i]))
        sol_value_list.append(np.min(A[i]))
    sol_value = np.array(sol_value_list).sum()
    sol_matrix = np.zeros((len(A), len(A[0])))
    for i in range(len(A)):
        sol_matrix[i][sol_index_list[i]]=1
    return sol_matrix,sol_value

def obj_expression(model):
    return sum(
        model.energy_local[i, j] + model.energy_bs[i, j] + model.energy_leo[i, j] for i in model.M for
        j in model.N)
model.obj = Objective(rule=obj_expression, sense=minimize)


# 定义松弛后的目标函数
def relaxed_obj_rule(model, m, o, p, q, r, s):
    return sum(
        model.alpha[i, j] * model.task_l[i, j] * model.task_c[i, j] * (model.m[i, j] + model.power_compute_local[i, j])
        + model.beta[i, j] * (
                (model.task_l[i, j] / model.capacity_BS_user[i, j] * (
                            m[i, j] + model.trans_power[i, j] + model.power_trans_bs[i, j]))
                + (model.task_l[i, j] * model.task_c[i, j] / model.f_bs[i, j] * (m[i, j] + model.power_compute_bs[i]))
                + o[i] * model.epsilon[i, j]
                + s[i] * model.f_bs[i, j]
        )
        + model.gamma[i, j] *
        sum(
            model.rho[i, j, k] * (model.task_l[i, j] / model.capacity_LEO_user[i, j] * (
                        m[i, j] + model.trans_power[i, j] + model.power_trans_leo[i, j, k])
                                  + p[k] * model.delta[i, j, k] + r[k] * model.task_l[i, j])
            + model.eta[i, j, k] * (model.task_l[i, j] * model.task_c[i, j] / model.f_leo[i, j, k] * (
                        m[i, j] + model.power_compute_leo[k])
                                    + q[k] * model.f_leo[i, j, k])
            + model.hop_number[i, j] * model.task_l[i, j] / model.rate_ISL[i, j, k] * (m[i, j] + model.power_ISL[k])
            for k in model.O
        )
        for i in model.M for j in model.N
        )
    - sum(m[i, j] * model.deadline[i, j] for i in model.M for j in model.N)
    - sum(o[i] * model.bs_bandwidth_constraint[i] for i in model.M)
    - sum(p[k] * model.leo_bandwidth_constraint[k] for k in model.O)
    - sum(q[k] * model.leo_compute_capacity[k] for k in model.O)
    - sum(r[k] * model.cache[k] for k in model.O)
    - sum(s[i] * model.bs_compute_capacity[i] for i in model.M)

def function(model):
    function_beta = sum((model.task_l[i, j] / model.capacity_BS_user[i, j] * (
                            m[i, j] + model.trans_power[i, j] + model.power_trans_bs[i, j]))
                + (model.task_l[i, j] * model.task_c[i, j] / model.f_bs[i, j] * (m[i, j] + model.power_compute_bs[i]))
                + o[i] * model.epsilon[i, j]
                + s[i] * model.f_bs[i, j] for i in model.M for j in model.N)

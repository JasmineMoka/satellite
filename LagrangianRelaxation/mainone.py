
import numpy as np
from pyomo.environ import *

from duality import Duality
from orlibary import ORLibary
from relaxation import Relaxation

data_set = ORLibary()
# n_agent, n_ job, a, capacity, c = data_set.gen_assign_prob(file_name = 'b05200.txt')
# i j k
n_bs_number, n_per_cell_number, n_leo_number, a, capacity, c, bs_compute_capacity = data_set.gen_assign_prob(file_name = 'b05200.txt')

def bs_compute_capacity_rule(model,i):
    return sum(model.a[i,j]*model.y[i,j] for j in model.N) <= model.bs_compute_capacity[i]

def capacity_rule(model, i):
    return sum(model.a[i,j]*model.x[i,j] for j in model.N) <= model.capacity[i]

def one_rule(model, j):
    return sum(model.x[i,j] for i in model.M) == 1

def relaxed_obj_rule(model, u):
    return sum((model.c[i,j] + model.u[j]) * model.x[i,j] for i in model.M for j in model.N) - sum(model.u[j] for j in model.N)

def obj_rule(model):
    return sum(model.c[i,j] * model.x[i,j] for i in model.M for j in model.N)

def subgrad_rule(model, j):
    return sum(model.x[i,j] for i in model.M) - 1
# 定义约束规则

def init_c(model, i, j):
    return c[i,j]

def init_a(model, i, j):
    return a[i,j]

def set_u(model, u):
    for i in model.N:
        model.u[i] = u[i]

model = ConcreteModel()

# model.M, model.N = Set(initialize=range(n_agent)), Set(initialize=range(n_job))
# M bs  N 每个cell的用户量 O 卫星数量
model.M, model.N, model.O = Set(initialize=range(n_bs_number)), Set(initialize=range(n_per_cell_number),), Set(initialize=range(n_leo_number))
model.c, model.a = Param(model.M, model.N, initialize = init_c), Param(model.M, model.N, initialize = init_a)
model.capacity = Param(model.M, initialize = capacity)
# 添加
model.bs_compute_capacity = Param(model.M, initialize = bs_compute_capacity)

num_agents = {1: 5, 2: 7, 3: 4}
def x_init(model, i, j, k):
    if k in num_agents:
        return Binary
    else:
        return None
# 修改
model.x = Var(model.M, model.N, model.O, within=Binary)

# model.x = Var(model.M, model.N, within = Binary)



# 添加yz  通过xyz代表在local bs leo 计算
# model.y = Var(model.M, model.N, within = Binary)
# model.z = Var(model.O, within = Binary)



model.constrs1 = Constraint(model.M, rule = capacity_rule)
model.constrs2 = Constraint(model.N, rule = one_rule)
#
model.constrs9 = Constraint(model.M, model.N, rule = bs_compute_capacity_rule)
model.subgrad = Expression(model.N, rule = subgrad_rule)

# 添加约束到模型
model.task_assignment_constraint = Constraint(model.M, model.N, rule=task_assignment_rule)

model.u = Param(model.N, initialize = 0.0, mutable=True)
lagrangian_relaxation = Relaxation()
lagrangian_relaxation.relax_constrs(relaxed_constrs=model.constrs2)
lagrangian_relaxation.set_objective(model, relaxed_obj_rule)
solver = SolverFactory("gurobi", solver_io = "python")

u = np.ones(len(model.constrs2))
duality_model = Duality(u_init = u)
max_iter = 200
for k in range(max_iter):
    set_u(model, u)
    solution = solver.solve(model)
    duality_model.get_subgrad(relaxed_model = model)
    duality_model.get_step_size()
    u = duality_model.update()
    duality_model.print_status(k, max_iter)
